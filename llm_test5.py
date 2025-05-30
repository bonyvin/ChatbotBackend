import json
import re
import uuid
import traceback
import datetime
import operator # For GraphState messages typing

from typing import List, Optional, Literal, Any, Dict, Sequence, Annotated, AsyncIterator, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, EmailStr, field_validator, AliasChoices # Pydantic v1/v2 compatible
# from pydantic_core import PydanticCustomError # For more custom validation errors if needed

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage, AIMessageChunk, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver # Using MemorySaver as in user code
from typing_extensions import TypedDict
from llm_templates import template_Invoice_without_date,po_database_schema,po_extraction_prompt,invoice_extraction_prompt

# Assuming sqlalchemy and database setup (get_db, po_database_schema) are defined elsewhere
from sqlalchemy import text
from sqlalchemy.orm import Session


try:
    from database import get_db # Make sure get_db returns a context manager or generator yielding a Session
except ImportError:
    print("Warning: 'database.py' not found. Using mock 'get_db'.")
    from contextlib import contextmanager
    # Mock Session and get_db
    class MockSession:
        def execute(self, statement):
            class MockResult:
                def fetchall(self):
                    print(f"Mock DB executing: {statement}")
                    if "SHOW TABLES" in str(statement):
                         # Added more tables based on user output log
                         return [('invoicedetails',), ('invoiceheader',), ('itemdiffs',), ('itemmaster',), ('itemsupplier',), ('podetails',), ('poheader',), ('promotion_store_association',), ('promotiondetails',), ('promotionheader',), ('shipmentdetails',), ('shipmentheader',), ('storedetails',), ('suppliers',), ('users',)]
                    # Simulate finding items based on a simple query condition
                    if "WHERE im.brand = 'TestBrand'" in str(statement):
                        return [MockRow({'itemId': 'TEST001'}), MockRow({'itemId': 'TEST002'})]
                    if "WHERE isup.supplierCost < 50" in str(statement):
                         return [MockRow({'itemId': 'COST001', 'supplierCost': 45.0})]
                    # Simulate the yellow item query
                    if "WHERE im.brand = 'FashionX' AND im.itemDepartment LIKE 'T-Shirt%' AND (EXISTS" in str(statement) and "Yellow" in str(statement):
                         print("Mock DB: Simulating yellow FashionX T-shirt query.")
                         # Check if the query syntax is now correct (ends with ')')
                         if str(statement).strip().endswith(')'):
                             print("Mock DB: Query syntax appears correct.")
                             return [MockRow({'itemId': 'FASHION_YELLOW_TSHIRT_01'})]
                         else:
                             print("Mock DB: Query syntax error detected (missing closing parenthesis).")
                             # Simulate the syntax error the user saw
                             raise Exception('(pymysql.err.ProgrammingError) (1064, "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version for the right syntax to use near \'\' at line 1")')

                    return [] # Default empty result
            return MockResult()
        def close(self):
            pass # No-op for mock
    class MockRow:
        def __init__(self, data):
            self._mapping = data
        def __getitem__(self, key):
            # Allow access like row['itemId'] or row[0] if needed, primarily dict access
            if isinstance(key, str):
                return self._mapping[key]
            # Add index access if necessary, though _mapping makes dict access primary
            # elif isinstance(key, int):
            #     return list(self._mapping.values())[key]
            raise KeyError(key)


    @contextmanager
    def get_db():
        print("Using mock database session.")
        yield MockSession()
        print("Mock database session closed.")


# --- Configuration ---
load_dotenv()
# --- Environment Variables / Placeholders ---
# Ensure OPENAI_API_KEY is set in your environment or passed correctly
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY" # Replace with your actual key or load from env

# Placeholder for database schema and templates (as in user's code)
TABLE_SCHEMA = "system: Your database schema here..." # Replace with actual schema
# po_database_schema = "system: Your PO database schema here..." # Not directly used in snippet but part of context

# --- Pydantic Models ---
InvoiceType = Literal["Merchandise", "Non - Merchandise", "Debit Note", "Credit Note"]

class InvoiceItem(BaseModel):
    """
    Represents a single item within an invoice, with flexible field naming.
    """
    item_id: Optional[str] = Field(None, validation_alias=AliasChoices('Item ID', 'Product Code', 'SKU', 'Item No.'))
    quantity: Optional[int] = Field(None, validation_alias=AliasChoices('Quantity', 'Qty', 'Quantity Ordered', 'Units'))
    invoice_cost: Optional[float] = Field(None, validation_alias=AliasChoices('Invoice Cost', 'Item Cost', 'Total Cost per Item'))

    class Config:
        populate_by_name = True
        extra = 'ignore'
        # For Pydantic v2, use model_config dictionary:
        # model_config = {"populate_by_name": True, "extra": "ignore"}


class ExtractedInvoiceDetails(BaseModel):
    """
    Pydantic model to represent extracted invoice details,
    allowing for flexible field naming and normalization of 'Invoice Type'.
    """
    po_number: Optional[str] = Field(None, validation_alias=AliasChoices('PO Number', 'PO ID', 'Purchase Order Id', 'PO No.'))
    invoice_number: Optional[str] = Field(..., validation_alias=AliasChoices('Invoice Number', 'Invoice ID', 'Bill No.'))
    invoice_type: Optional[InvoiceType] = Field(None) # Made explicitly Optional with default None
    date: Optional[str] = Field(..., validation_alias=AliasChoices('Date', 'Invoice Date', 'Billing Date'))
    total_amount: Optional[float] = Field(..., validation_alias=AliasChoices('Total Amount', 'Invoice Total', 'Amount Due'))
    total_tax: Optional[float] = Field(None, validation_alias=AliasChoices('Total Tax', 'Tax Amount', 'VAT'))
    items: Optional[List[InvoiceItem]] = Field(default_factory=list, validation_alias=AliasChoices('Items', 'Invoice Items', 'Products'))
    supplier_id: Optional[str] = Field(None, validation_alias=AliasChoices('SupplierId', 'Supplier', 'Supp Id'))
    email: Optional[EmailStr] = Field(None, validation_alias=AliasChoices('Email', 'Email Address', 'Contact Email'))

    @field_validator('invoice_type', mode='before')
    @classmethod
    def normalize_invoice_type(cls, v: Optional[str]) -> Optional[str]:
        """
        Normalizes the invoice type string to one of the predefined values.
        Handles variations and shorthand inputs.
        """
        if v is None:
            return None
        if isinstance(v, str):
            lower_v = v.strip().lower()
            if lower_v in ["merchandise", "merch"]:
                return "Merchandise"
            elif lower_v in ["non - merchandise", "non-merchandise", "non merch", "nonmerchandise"]: # Added "nonmerchandise"
                return "Non - Merchandise"
            elif lower_v in ["debit note", "debit"]:
                return "Debit Note"
            elif lower_v in ["credit note", "credit"]:
                return "Credit Note"
        # If it doesn't match any known variations, let Pydantic's Literal validation handle it.
        return v

    class Config:
        populate_by_name = True
        extra = 'ignore'
        # For Pydantic v2:
        # model_config = {"populate_by_name": True, "extra": "ignore"}

# --- Tool Argument Schema ---
class GetPODetailsArgs(BaseModel):
    po_id: str = Field(description="The purchase order ID to fetch details for.")

# --- Database Schema and Helper Functions (from user code, with minor adjustments) ---
# Mocking get_db and text for standalone execution if needed
# In a real scenario, these would interact with a database
class MockDbSession:
    def execute(self, statement, params=None):
        class MockResult:
            def fetchall(self):
                if "SHOW TABLES" in str(statement):
                    return [("podetails",), ("other_table",)]
                if "SELECT * FROM podetails" in str(statement) and params and params.get("po_id") == "PO123":
                    return [
                        MockRow({"poId": "PO123", "supplierId": "SUP789", "itemId": "ITEM001", "quantity": 10, "cost": 5.0}),
                        MockRow({"poId": "PO123", "supplierId": "SUP789", "itemId": "ITEM002", "quantity": 5, "cost": 12.0}),
                    ]
                return []
            def mappings(self): # Added for podetails query
                if "SELECT * FROM podetails" in str(statement) and params and params.get("po_id") == "PO123":
                    return MockMappingResult([
                        {"poId": "PO123", "supplierId": "SUP789", "itemId": "ITEM001", "quantity": 10, "cost": 5.0, "_mapping": {"poId": "PO123", "supplierId": "SUP789", "itemId": "ITEM001", "quantity": 10, "cost": 5.0}},
                        {"poId": "PO123", "supplierId": "SUP789", "itemId": "ITEM002", "quantity": 5, "cost": 12.0, "_mapping": {"poId": "PO123", "supplierId": "SUP789", "itemId": "ITEM002", "quantity": 5, "cost": 12.0}},
                    ])
                return MockMappingResult([])


        class MockMappingResult:
            def __init__(self, data):
                self.data = data
            def all(self):
                return self.data


        class MockRow:
            def __init__(self, data):
                self._mapping = data
                self.data = data
            def __getitem__(self, key):
                return self.data[key]
            def _asdict(self): # if needed by `dict(row)`
                return self.data

        return MockResult()

    def close(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def get_db(): # Mock
    yield MockDbSession()

def text(query_string): # Mock
    return query_string


_table_names_cache = None
def get_table_names():
    global _table_names_cache
    if _table_names_cache is not None:
        return _table_names_cache
    try:
        with next(get_db()) as db:
            result = db.execute(text("SHOW TABLES")).fetchall()
            _table_names_cache = [row[0] for row in result]
            print(f"Fetched and cached table names: {_table_names_cache}")
            return _table_names_cache
    except Exception as e:
        print(f"Error fetching table names: {e}")
        return []

async def get_po_details(po_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all rows from podetails for the given purchase-order ID.
    Returns a list of dicts, one per row.
    """
    sql = text("SELECT * FROM podetails WHERE poId = :po_id")
    db_session: Optional[MockDbSession] = None # Type hint for clarity
    try:
        db_session = next(get_db())
        print(f"--- Executing SQL Query: {sql} with po_id={po_id} ---")
        # SQLAlchemy Core execute returns a CursorResult. .mappings().all() gives List[RowMapping]
        result_proxy = db_session.execute(sql, {"po_id": po_id})
        
        # Assuming result_proxy has .mappings().all() or similar
        # For older SQLAlchemy, it might be `result = [dict(row) for row in result_proxy]`
        # User code has .mappings().all()
        rows = result_proxy.mappings().all() 
        
        print(f"--- Query Execution Successful: Fetched {len(rows)} rows ---")
        # Convert RowMappings to simple dicts if they aren't already (they should be dict-like)
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"!!! Error executing SQL query for PO details: {e} !!!")
        traceback.print_exc()
        return []
    finally:
        if db_session:
            try:
                db_session.close()
            except Exception:
                pass

# --- Tool Definition ---
po_details_tool = Tool(
    name="get_po_details",
    func=get_po_details, # LangChain handles async func here
    description=(
        "Fetches detailed information for a given Purchase Order (PO) ID, "
        "including supplier and item specifics. Use this tool if you have a PO ID "
        "and need to enrich invoice data with details not present in the initial text, "
        "such as supplier ID or item breakdowns (item_id, quantity, cost)."
    ),
    args_schema=GetPODetailsArgs, # CRITICAL: Use Pydantic model for args_schema
)

# --- LLMs ---
# Main conversational LLM
chat_model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, streaming=True, temperature=0.7)
# LLM for SQL generation
sql_generator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0)
# LLM for detail extraction (non-streaming)
extractor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0)

# This is the LLM that will be used for function calling in extract_details
# It needs the tool definition.
# For Langchain >0.1, .bind_tools is preferred.
# For older versions, `functions` kwarg was used.
# User code uses `functions` kwarg for `ChatOpenAI` constructor.
extractor_agent_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0,
    streaming=False,
    # functions=[po_details_tool], # For older Langchain with OpenAI 'functions'
).bind_tools([po_details_tool]) # Modern way: bind the tool

# --- Extraction Prompt ---
# Dynamically include the schema in the prompt for the LLM
# Ensure ExtractedInvoiceDetails is defined before this line
schema_for_prompt = ExtractedInvoiceDetails.model_json_schema()
invoice_extraction_prompt_content = f"""
You are an AI assistant specialized in extracting information from invoice texts.
Your goal is to populate a JSON object based on the following Pydantic schema:

<schema>
{json.dumps(schema_for_prompt, indent=2)}
</schema>

From the user-provided text:
1. Extract all available fields directly from the text.
2. If a 'po_number' is identified in the text, AND details like 'supplier_id' or specific 'items' (item_id, quantity, invoice_cost) seem to be missing or could be elaborated by purchase order data, you MUST use the 'get_po_details' function/tool to fetch these details using the 'po_id' (which is the 'po_number' you found). The 'get_po_details' tool expects arguments like: {{"po_id": "string"}}.
3. After receiving data from the 'get_po_details' function (if called), integrate this information with what you initially extracted from the text. The tool returns a list of dictionaries, where each dictionary is an item from the PO (possibly including 'itemId', 'quantity', 'cost', 'supplierId'). Use these to populate the 'items' field in your JSON. If 'supplierId' is present in the tool's output, use it for the 'supplier_id' field. If there's an overlap in item details, prioritize information from the 'get_po_details' tool for fields it provides. Map 'itemId' to 'item_id', and 'cost' to 'invoice_cost' for each item.
4. Construct the final, complete JSON object according to the schema.
5. Your final response MUST be ONLY the JSON object. Do not include any markdown formatting (like ```json ```), conversational text, or explanations. Just the raw JSON string.
"""

# --- Graph State Definition ---
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: Optional[str]
    generated_sql: Optional[str]
    sql_results: Optional[Union[List[Dict], str]]
    llm_response_content: Optional[str]
    extracted_details: Optional[ExtractedInvoiceDetails]

# --- Helper for cleaning LLM's JSON output ---
def clean_json_string_from_llm(s: str) -> str:
    """
    Cleans a string to extract a valid JSON object.
    Handles cases where JSON is wrapped in markdown or has leading/trailing text.
    """
    if not isinstance(s, str):
        return "" # Or raise error

    # Try to find JSON within ```json ... ``` blocks
    match_markdown = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if match_markdown:
        return match_markdown.group(1)

    # Try to find a raw JSON object (greedy match for first '{' to last '}')
    # This is a common pattern for LLM JSON outputs.
    match_raw = re.search(r"(\{.*\})", s, re.DOTALL)
    if match_raw:
        return match_raw.group(0)
    
    # If no clear JSON structure is found, return original to let Pydantic try (and likely fail)
    # Or, you could raise an error here if strict JSON is expected.
    print(f"Warning: Could not find a clear JSON block in string: '{s[:200]}...'")
    return s


# --- Graph Nodes (preprocess_input, generate_sql, execute_sql, generate_response_from_sql, generate_direct_response are from user code) ---
# For brevity, I'm omitting the full node definitions from user code that are not directly part of the fix,
# but they would be here. I will include the modified `extract_details`.

async def preprocess_input(state: GraphState) -> Dict:
    print("--- Node: preprocess_input ---")
    last_message = state['messages'][-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("Last message is not a HumanMessage")
    user_query = last_message.content
    print(f"User Query: {user_query}")
    return {"user_query": user_query, "messages": []} # Return empty messages to avoid duplicating input to history via operator.add initially

# ... (generate_sql, execute_sql, etc. from user code) ...
# Assume other nodes (generate_sql, execute_sql, generate_response_from_sql, generate_direct_response) are defined as in the user's original code.
# I will paste them here for completeness of the runnable script if needed, but the focus is extract_details.

async def generate_sql(state: GraphState) -> Dict:
    print("--- Node: generate_sql ---")
    # ... (user's existing generate_sql logic) ...
    user_query = state.get("user_query")
    if not user_query:
        return {"generated_sql": None, "messages": []}
    # Simplified for this example
    # In real code, use user's SQL generation logic
    if "find po" in user_query.lower():
         # Example: "find po 123" -> SELECT * FROM purchase_orders WHERE id = '123'
        po_match = re.search(r"po\s*(\w+)", user_query.lower())
        if po_match:
            po_id_found = po_match.group(1)
            # This is just a placeholder, actual SQL generation is complex
            # return {"generated_sql": f"SELECT * FROM purchase_orders WHERE po_id = '{po_id_found.upper()}'"}
            # For the graph flow, let's assume it generates SQL that will lead to results
            return {"generated_sql": f"SELECT poId, supplierId, totalAmount FROM purchase_orders WHERE poId = '{po_id_found.upper()}'", "messages": []} 
    return {"generated_sql": None, "messages": []}


async def execute_sql(state: GraphState) -> Dict:
    print("--- Node: execute_sql ---")
    query = state.get("generated_sql")
    if not query:
        return {"sql_results": "Error: No SQL query generated.", "messages": []}
    
    # Mock execution
    if query and "PO123" in query: # Example check
         sql_results_data = [{"poId": "PO123", "supplierId": "SUPPLIER_FROM_SQL", "totalAmount": 150.75}]
         return {"sql_results": sql_results_data, "messages": []}
    elif query: # Generic mock for other SQL
        sql_results_data = [{"data_from_sql": "some value"}]
        return {"sql_results": sql_results_data, "messages": []}

    return {"sql_results": "Error: SQL execution failed or returned no data.", "messages": []}

async def generate_response_from_sql(state: GraphState) -> Dict:
    print("--- Node: generate_response_from_sql ---")
    # ... (user's existing generate_response_from_sql logic) ...
    # Simplified for this example
    sql_results = state.get("sql_results")
    llm_response = f"Based on the SQL query, I found: {json.dumps(sql_results)}"
    return {"llm_response_content": llm_response, "messages": [AIMessage(content=llm_response)]}

async def generate_direct_response(state: GraphState) -> Dict:
    print("--- Node: generate_direct_response ---")
    # ... (user's existing generate_direct_response logic) ...
    # Simplified for this example
    user_query = state.get("user_query", "your query")
    llm_response = f"I couldn't generate SQL for '{user_query}'. How else can I help? For example, I can try to extract invoice details if you provide some text."
    # Example: if user asks to extract from a specific text
    if "extract from this invoice" in user_query.lower(): # A simple trigger for testing extract_details
        # Simulate that the user provided some text that is now in llm_response_content
        # In a real flow, this text might come from user_query directly or a previous step
        sample_invoice_text = """
        Invoice ID: INV778899
        PO Number: PO123
        Date: 2024-05-29
        Total Amount: 150.75
        Email: test@example.com
        Items: ProductA, 2, 25.00
        """
        # This node (generate_direct_response) is *before* extract_details.
        # So, extract_details will pick up llm_response_content from here.
        # We need to ensure this node's output can be the input for extraction.
        llm_response = sample_invoice_text # The text to be extracted from
        print(f"--- Setting llm_response_content for extraction: {llm_response[:100]}... ---")


    return {"llm_response_content": llm_response, "messages": [AIMessage(content=llm_response)]}


async def extract_details(state: GraphState) -> Dict[str, Optional[ExtractedInvoiceDetails]]:
    """
    Extract invoice fields and enrich with PO details if present, using tool-calling.
    """
    print("--- Node: extract_details (tool-calling) ---")
    text_to_extract_from = state.get("llm_response_content") or "" # Input from previous LLM response

    if not text_to_extract_from:
        print("--- No text content found for extraction. Skipping. ---")
        return {"extracted_details": None, "messages": []}

    print(f"--- Text for extraction: '{text_to_extract_from[:200]}...' ---")

    # Prepare messages for the extractor LLM
    # The system prompt guides the LLM on schema, tool use, and JSON output.
    system_message = SystemMessage(content=invoice_extraction_prompt_content)
    human_message = HumanMessage(content=f"Extract invoice details from the following text:\n\n---\n{text_to_extract_from}\n---")
    
    messages_for_llm: List[BaseMessage] = [system_message, human_message]

    try:
        # First call to the LLM. It might decide to call the tool.
        # Using the `extractor_agent_llm` which has `po_details_tool` bound to it.
        # Langchain's .invoke() on a model bound with tools will handle the tool calling loop if one is initiated by the LLM.
        # The result of .invoke here will be an AIMessage, which might contain tool_calls.
        
        # The user's original code had a manual two-step call. Let's replicate that logic
        # with the bound LLM for clarity on how LangChain handles it.
        # However, a simpler .invoke might also work if the LLM is prompted correctly for a final JSON.
        # For this fix, we'll adapt the user's two-step explicit logic.

        # Step 1: Initial LLM call, see if it wants to use a tool
        print("--- extract_details: Making initial LLM call ---")
        # We use the `extractor_agent_llm` which is ChatOpenAI().bind_tools([po_details_tool])
        # For the user's explicit two-step:
        # The `functions` parameter in the original ChatOpenAI was for older OpenAI API.
        # The modern approach is `tools` and `tool_choice`.
        # Let's use a ChatOpenAI instance specifically for this node, configured with the tool.
        # This `extractor_agent_llm` is already configured.

        first_response_message: AIMessage = await extractor_agent_llm.ainvoke(messages_for_llm)
        messages_for_llm.append(first_response_message) # Add LLM's response to history for next call

        final_llm_content_str = ""

        # Step 2: Check if the LLM made a tool call
        if first_response_message.tool_calls and len(first_response_message.tool_calls) > 0:
            print(f"--- extract_details: LLM initiated tool call(s): {first_response_message.tool_calls} ---")
            # In this specific setup, we expect at most one call to get_po_details
            # LangChain's AIMessage.tool_calls is a list of `ToolCall` objects.
            
            tool_call = first_response_message.tool_calls[0] # Process the first tool call
            tool_name = tool_call['name']
            
            if tool_name == po_details_tool.name:
                tool_args = tool_call['args'] # Already a dict
                po_id_arg = tool_args.get('po_id')

                if not po_id_arg:
                    print(f"!!! Error: LLM called {tool_name} but 'po_id' arg was missing. Args: {tool_args} !!!")
                    # Fallback: try to get content directly from first response if tool call is malformed
                    final_llm_content_str = first_response_message.content if isinstance(first_response_message.content, str) else ""

                else:
                    print(f"--- extract_details: Calling tool '{tool_name}' with args: {tool_args} ---")
                    tool_output_data = await po_details_tool.ainvoke(tool_args) # Invoke the tool with its args
                    
                    # Create a ToolMessage with the output
                    tool_message = FunctionMessage( # Or ToolMessage in newer Langchain
                        name=tool_name,
                        content=json.dumps(tool_output_data) # Tool output must be JSON serializable string
                    )
                    messages_for_llm.append(tool_message)

                    # Step 3: Second LLM call, now with tool's output in history
                    print("--- extract_details: Making second LLM call with tool output ---")
                    second_response_message: AIMessage = await extractor_agent_llm.ainvoke(messages_for_llm)
                    final_llm_content_str = second_response_message.content if isinstance(second_response_message.content, str) else ""
            else:
                print(f"--- extract_details: LLM called unknown tool '{tool_name}'. Ignoring. ---")
                final_llm_content_str = first_response_message.content if isinstance(first_response_message.content, str) else ""
        else:
            # No tool call was made by the LLM
            print("--- extract_details: No tool call made by LLM. Using its direct content. ---")
            final_llm_content_str = first_response_message.content if isinstance(first_response_message.content, str) else ""


        # Step 4: Parse the final LLM content string (should be JSON)
        if not final_llm_content_str:
            print("--- extract_details: Final LLM content is empty. Cannot parse. ---")
            return {"extracted_details": None, "messages": []}

        print(f"--- extract_details: Final LLM content string (raw): '{final_llm_content_str[:300]}...' ---")
        cleaned_json_str = clean_json_string_from_llm(final_llm_content_str)
        
        if not cleaned_json_str:
            print(f"--- extract_details: Cleaned JSON string is empty. Original: '{final_llm_content_str[:200]}...' ---")
            return {"extracted_details": None, "messages": []}

        print(f"--- extract_details: Cleaned JSON string for parsing: '{cleaned_json_str[:300]}...' ---")
        
        parsed_details: Optional[ExtractedInvoiceDetails] = None
        try:
            # For Pydantic v1/v2 compatibility, model_validate_json is preferred for v2
            if hasattr(ExtractedInvoiceDetails, 'model_validate_json'):
                parsed_details = ExtractedInvoiceDetails.model_validate_json(cleaned_json_str)
            else: # Fallback for Pydantic v1
                parsed_details = ExtractedInvoiceDetails.parse_raw(cleaned_json_str)
            
            print(f"--- extract_details: Successfully parsed: {parsed_details.model_dump_json(indent=2)[:300]}... ---")
            # Add the extracted details as an AIMessage for history if needed, or just update state
            # For now, just returning the details for the state.
            # If you want this interaction to be part of messages history for the main chat_model:
            # ai_summary_of_extraction = f"Extracted invoice details: {parsed_details.invoice_number}, Total: {parsed_details.total_amount}"
            # messages_to_add = [AIMessage(content=ai_summary_of_extraction)]
            # return {"extracted_details": parsed_details, "messages": messages_to_add}

        except Exception as e:
            print(f"!!! Error parsing extracted JSON from LLM: {e} !!!")
            print(f"--- JSON string attempted: {cleaned_json_str} ---")
            traceback.print_exc()
            # Optionally, add an error message to history
            # messages_to_add = [AIMessage(content=f"Error during detail extraction: {e}")]
            # return {"extracted_details": None, "messages": messages_to_add}
            return {"extracted_details": None, "messages": []}

        return {"extracted_details": parsed_details, "messages": []} # No new messages for main chat history from this node

    except Exception as e:
        print(f"!!! Unexpected error in extract_details node: {e} !!!")
        traceback.print_exc()
        return {"extracted_details": None, "messages": []}


# --- Conditional Edges (from user code) ---
def should_execute_sql(state: GraphState) -> str:
    print("--- Condition: should_execute_sql ---")
    if state.get("generated_sql"):
        print("Decision: SQL generated, routing to execute_sql")
        return "execute_sql"
    else:
        print("Decision: No SQL generated, routing to generate_direct_response")
        return "generate_direct_response"

# --- Build the Workflow (from user code) ---
workflow = StateGraph(GraphState)

workflow.add_node("preprocess_input", preprocess_input)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_response_from_sql", generate_response_from_sql)
workflow.add_node("generate_direct_response", generate_direct_response)
workflow.add_node("extract_details", extract_details) # Node that was fixed

workflow.set_entry_point("preprocess_input") # Explicit entry point

workflow.add_edge("preprocess_input", "generate_sql")
workflow.add_conditional_edges(
    "generate_sql",
    should_execute_sql,
    {
        "execute_sql": "execute_sql",
        "generate_direct_response": "generate_direct_response",
    },
)
workflow.add_edge("execute_sql", "generate_response_from_sql")
workflow.add_edge("generate_response_from_sql", "extract_details")
workflow.add_edge("generate_direct_response", "extract_details")
workflow.add_edge("extract_details", END) # End after extraction

# --- Memory and Compilation (from user code) ---
memory = MemorySaver()
# app_runnable = workflow.compile(checkpointer=memory)
# Checkpointer is essential for threads/history
# For testing without full async FastAPI stack, can compile without checkpointer for single runs
app_runnable = workflow.compile()


# --- FastAPI Request Model and Streaming (from user code, slightly adapted for testing) ---
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None # Made optional for easier testing

async def stream_response_generator(graph_stream: AsyncIterator[Dict]) -> AsyncIterator[str]:
    print("--- STREAM GENERATOR (for client) STARTED ---")
    # ... (user's stream_response_generator logic) ...
    # This part is for streaming main LLM responses, not directly related to extract_details fix
    # but important for the overall app.
    async for event_wrapper in graph_stream: # graph_stream now yields {"event": ..., "data": ...}
        event = event_wrapper.get("event")
        data = event_wrapper.get("data")
        if not event or not data:
            continue

        if event == "on_chat_model_stream":
            chunk = data.get("chunk")
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content
    print("--- STREAM GENERATOR (for client) FINISHED ---")


# --- FastAPI Endpoint (from user code, slightly adapted for testing) ---
# app = FastAPI() # Assuming FastAPI app is defined

@app.post("/chat/")
async def run_chat_flow(request: ChatRequest): # Changed from chat_endpoint for direct call test
    user_message_content = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- [Thread: {thread_id}] Received message: '{user_message_content}' ---")
    
    input_message = HumanMessage(content=user_message_content)
    # Initial state for the graph
    initial_graph_state = {"messages": [input_message]}

    # To test extract_details, the flow needs to reach it.
    # If user_message_content is "extract from this invoice", generate_direct_response will output sample text.
    
    # Astream events for observing the flow
    # graph_events_stream = app_runnable.astream_events(initial_graph_state, config, version="v2")
    
    # For non-streaming full run to inspect final state:
    final_state = await app_runnable.ainvoke(initial_graph_state, config)
    print("\n--- Final Graph State ---")
    # print(json.dumps(final_state, indent=2, default=str)) # Default=str for non-serializable like BaseMessage
    
    # More selective printing of final state
    if final_state:
        print(f"  User Query: {final_state.get('user_query')}")
        print(f"  Generated SQL: {final_state.get('generated_sql')}")
        print(f"  SQL Results: {final_state.get('sql_results')}")
        print(f"  LLM Response Content: {final_state.get('llm_response_content')}")
        if final_state.get('extracted_details'):
            print(f"  Extracted Details: {final_state['extracted_details'].model_dump_json(indent=2)}")
        else:
            print(f"  Extracted Details: None")
        # Print message history
        # print("  Message History:")
        # for msg in final_state.get("messages", []):
        #     print(f"    {type(msg).__name__}: {msg.content[:100] if msg.content else '(No content)'}")

    return final_state # Or stream if using astream_events

# --- Example Usage (for testing the flow locally) ---
# async def main_test():
#     print("--- Starting main_test ---")
#     # Test case 1: Simple query that might not generate SQL, leading to direct response
#     # and potentially no specific text for extraction by default.
#     # test_request_1 = ChatRequest(message="Hello there!")
#     # print("\n--- Test Case 1: General Greeting ---")
#     # await run_chat_flow(test_request_1)

#     # Test case 2: Query that should trigger SQL path (mocked)
#     # test_request_2 = ChatRequest(message="find po PO123 details")
#     # print("\n--- Test Case 2: SQL Query ---")
#     # await run_chat_flow(test_request_2) # This path also leads to extract_details

#     # Test case 3: Query specifically to test extraction flow
#     # This relies on `generate_direct_response` providing sample text if "extract" is in query
#     test_request_3 = ChatRequest(message="Please extract from this invoice now")
#     print("\n--- Test Case 3: Extraction Request ---")
#     await run_chat_flow(test_request_3)


if __name__ == "__main__":
    import asyncio
    # OPENAI_API_KEY needs to be valid for this to run.
    if OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("Please replace 'YOUR_OPENAI_API_KEY' with your actual OpenAI API key.")
    else:
        asyncio.run(main_test())