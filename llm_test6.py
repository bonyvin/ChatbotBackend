import datetime
import os
import uuid
import json
import traceback # For detailed error logging
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated, Sequence, AsyncIterator

import operator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse # Import StreamingResponse
from pydantic import AliasChoices, BaseModel, EmailStr, Field, field_validator
from dotenv import load_dotenv
from sqlalchemy import text
from requests import Session # Assuming Session is used within get_db
from fastapi.middleware.cors import CORSMiddleware

# LangChain and LangGraph imports
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    AIMessageChunk,
    FunctionMessage # Import AIMessageChunk for type checking stream
)
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from llm_templates import template_Invoice_without_date,po_database_schema,po_extraction_prompt,invoice_extraction_prompt
# from promoUtils import template_Promotion_without_date

# Assuming database.py and llm_templates.py are in the same directory or accessible
# Mock these if they are not available in the execution environment
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = "YOUR_API_KEY_HERE" # Replace with your actual key if needed
    if OPENAI_API_KEY == "YOUR_API_KEY_HERE":
         print("Warning: OPENAI_API_KEY not found in environment variables or .env. Using placeholder.")

print(f"LangSmith tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2') == 'true'}")
print(f"LangSmith project: {os.getenv('LANGCHAIN_PROJECT')}")
# Using Literal for simplicity as there are a fixed set of strings
InvoiceType = Literal["Merchandise", "Non - Merchandise", "Debit Note", "Credit Note"]
class ExtractedField(BaseModel):
    value: Optional[Any] = None
    is_example: bool = False
class ExtractedInvoiceItem(BaseModel):
    item_id: Optional[ExtractedField] = None
    quantity: Optional[ExtractedField] = None
    invoice_cost: Optional[ExtractedField] = None
class ExtractedInvoiceDetails(BaseModel):
    po_number: Optional[ExtractedField] = None
    invoice_number: Optional[ExtractedField] = None
    invoice_type: Optional[ExtractedField] = None
    date: Optional[ExtractedField] = None
    total_amount: Optional[ExtractedField] = None
    total_tax: Optional[ExtractedField] = None
    items: Optional[List[ExtractedInvoiceItem]] = None
    supplier_id: Optional[ExtractedField] = None
    email: Optional[ExtractedField] = None
class PoIdDetection(BaseModel):
    """
    Structured output for detecting a Purchase Order ID in text.
    """
    po_id_found: bool = Field(False, description="True if a Purchase Order ID was detected in the text.")
    extracted_po_id: Optional[str] = Field(None, description="The extracted Purchase Order ID, if found.")
# Initialize LLMs
chat_model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, streaming=True, temperature=0.7)
# LLM for SQL generation (non-streaming, deterministic)
sql_generator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0)
# LLM for detail extraction (non-streaming)
extractor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0)
extractor_llm_structured = extractor_llm.with_structured_output(
    ExtractedInvoiceDetails,
    method="function_calling", # Or "json_mode"
    include_raw=False)
# LLM for PO ID detection (new)
po_detector_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0).with_structured_output(
    PoIdDetection,
    method="function_calling",
    include_raw=False
)
_table_names_cache = None
today = datetime.datetime.now().strftime("%d/%m/%Y")
system_prompt_content = template_Invoice_without_date.replace("{current_date}", today)
def get_table_names():
    """Fetch all table names from the database, with caching."""
    global _table_names_cache
    if _table_names_cache is not None:
        return _table_names_cache

    db: Optional[Session] = None
    try:
        with next(get_db()) as db: # Use context manager
            # Adjust query for your specific SQL dialect (MySQL example)
            result = db.execute(text("SHOW TABLES")).fetchall()
            # For PostgreSQL:
            # result = db.execute(text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';")).fetchall()
            _table_names_cache = [row[0] for row in result] # Extract table names
            print(f"Fetched and cached table names: {_table_names_cache}")
            return _table_names_cache
    except Exception as e:
        print(f"Error fetching table names: {e}")
        return [] # Return empty list on error
# --- Graph State Definition ---
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str | None
    generated_sql: Optional[str]
    sql_results: Optional[List[Dict] | str]
    llm_response_content: Optional[str]
    extracted_details: Optional[ExtractedInvoiceDetails]
    last_po_id: Optional[str]       # ← we store the last‐used PO here
    candidate_po_id: Optional[str]  # ← this will be set by detect_po
# --- Graph Nodes ---
async def preprocess_input(state: GraphState) -> Dict:
    """Adds the latest user message to the state and history."""
    print("--- Node: preprocess_input ---")
    last_message = state['messages'][-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("Last message is not a HumanMessage")
    user_query = last_message.content
    print(f"User Query: {user_query}")
    return {"user_query": user_query}
async def generate_sql(state: GraphState) -> Dict:
    """
    Node to attempt generating SQL from the user query.
    Uses the dedicated sql_generator_llm.
    Includes post-processing to fix missing closing parenthesis.
    """
    print("--- Node: generate_sql ---")
    user_query = state.get("user_query")
    if not user_query:
        print("--- No user query found. Skipping SQL generation. ---")
        return {"generated_sql": None}
    # Updated prompt for SQL generation
    # Added emphasis on balanced parentheses in rule 7
    sql_generation_prompt = ChatPromptTemplate.from_messages([
        (po_database_schema),
        ("human", "Convert this question to a SQL SELECT query:\n\n```text\n{user_query}\n```")
    ])
    sql_generation_chain = sql_generation_prompt | sql_generator_llm | StrOutputParser()
    generated_sql = None # Default to None
    try:
        max_len = 2000
        truncated_query = user_query[:max_len] + ("..." if len(user_query) > max_len else "")
        print(f"--- Generating SQL for: '{truncated_query}' ---")
        llm_output = await sql_generation_chain.ainvoke({"user_query": truncated_query})
        cleaned_sql = llm_output.strip().strip(';').strip()
        pattern_start_exists = " AND (EXISTS ("
        if pattern_start_exists.lower() in cleaned_sql.lower() and not cleaned_sql.endswith(")"):
             if cleaned_sql.count("(") == cleaned_sql.count(")") + 1:
                 print("--- Post-processing: Adding missing closing parenthesis for EXISTS group. ---")
                 cleaned_sql += ")"
        if not cleaned_sql:
            print("--- Generation failed: LLM returned empty string. ---")
        elif "cannot be answered" in cleaned_sql.lower():
            print(f"--- LLM indicated query cannot be answered: {cleaned_sql} ---")
        elif cleaned_sql.lower().startswith("select"):
            harmful_keywords = ['update ', 'insert ', 'delete ', 'drop ', 'alter ', 'truncate ']
            if any(keyword in cleaned_sql.lower() for keyword in harmful_keywords):
                print(f"--- Generation failed: Potentially harmful non-SELECT keyword detected: {cleaned_sql} ---")
            else:
                print(f"--- SQL Generation Successful (after potential fix): {cleaned_sql} ---")
                generated_sql = cleaned_sql # Assign the potentially fixed SQL
        else:
            print(f"--- Generation failed: Output does not start with SELECT: {cleaned_sql} ---")
    except Exception as e:
        print(f"!!! ERROR during SQL generation: {e} !!!")
        traceback.print_exc()
    return {"generated_sql": generated_sql}

async def execute_sql(state: GraphState) -> Dict:
    """
    Node to execute the generated SQL query using the database helper.
    """
    print("--- Node: execute_sql ---")
    query = state.get("generated_sql")
    sql_results: Optional[List[Dict] | str] = None # Default
    if not query:
        print("--- No SQL query to execute. ---")
        sql_results = "Error: No SQL query was generated."
    if not query.strip().lower().startswith("select"):
        error_msg = "Error: Only SELECT queries are allowed."
        print(f"--- Validation Failed: {error_msg} ---")
        return {"sql_results": error_msg}
    words = query.upper().split()
    valid_tables = get_table_names() # Uses cached names after first call
    if not valid_tables:
        error_msg = "Error: Could not retrieve table names for validation."
        print(f"--- Validation Failed: {error_msg} ---")
        return {"sql_results": error_msg}
    table_found = False
    try:
        from_indices = [i for i, word in enumerate(words) if word == "FROM"]
        join_indices = [i for i, word in enumerate(words) if word == "JOIN"]
        potential_table_indices = [idx + 1 for idx in from_indices + join_indices if idx + 1 < len(words)]
        for idx in potential_table_indices:
            potential_table = words[idx].strip("`;,()")
            if potential_table.lower() in [t.lower() for t in valid_tables]:
                table_found = True
                break # Found at least one valid table reference
    except Exception as parse_err:
         print(f"Warning: Error during simple table name parsing: {parse_err}")
         print(f"Warning: Proceeding with query execution despite table parsing issue: {query}")
         table_found = True # Allow execution if parsing fails, adjust if stricter validation needed
    if not table_found:
        error_msg = f"Error: Could not validate table name in query. Ensure it uses valid tables: {valid_tables}"
        print(f"--- Validation Failed: {error_msg} ---")
        return {"sql_results": error_msg}
    try:
        print(f"--- Executing SQL Query: {query} ---")
        with next(get_db()) as db: # Use context manager
            result = db.execute(text(query)).fetchall()
            print(f"--- Query Execution Successful: Fetched {len(result)} rows ---")
            # Convert rows to list of dicts
            sql_results = [dict(row._mapping) for row in result]
    except Exception as e:
        error_msg = f"Error executing SQL query: {e}"
        print(f"!!! {error_msg} !!!")
        traceback.print_exc()
        sql_results = error_msg # Return the error message string
    return {"sql_results": sql_results}

async def generate_response_from_sql(state: 'GraphState') -> Dict:
    """
    Node to generate a natural language response based on SQL execution results.
    Streams internally using .astream() to aggregate content for state update.
    `astream_events` will handle the client-facing stream.
    """
    print("--- Node: generate_response_from_sql ---")
    user_query = state.get("user_query", "the user's request")
    sql_results = state.get("sql_results")
    generated_sql = state.get("generated_sql")
    messages = state.get("messages", []) # Get history for context
    context = f"The user asked: '{user_query}'\n"
    llm_instruction = "" # Initialize instruction
    if isinstance(sql_results, str): # Error occurred during execution
        context += f"However, there was an error executing the query: {sql_results}\n"
        llm_instruction = "Explain the database query error to the user based on the context."
    elif isinstance(sql_results, list):
        if not sql_results:
            context += "The query executed successfully but returned no results.\n"
            llm_instruction = "Inform the user that the database query found no matching items based on the context."
        else:
            max_results_to_show = 10
            serializable_results = []
            for row in sql_results[:max_results_to_show]:
                 try:
                     serializable_row = dict(row) if not isinstance(row, dict) else row
                     # Further check/convert types within the row if necessary
                     serializable_results.append(serializable_row)
                 except TypeError:
                     print(f"Warning: Could not serialize row for LLM context: {row}")
                     serializable_results.append({"error": "Could not display row data"})
            results_summary = json.dumps(serializable_results, indent=2)
            if len(sql_results) > max_results_to_show:
                results_summary += f"\n... (and {len(sql_results) - max_results_to_show} more rows)"
            context += f"The query returned the following results (showing up to {max_results_to_show}):\n```json\n{results_summary}\n```\n"
            context += f"Total items found: {len(sql_results)}."
            print("Stream Context Results Summary: ",results_summary,"SQL Results",sql_results,"Max Rsults: ",max_results_to_show,"Messages: ",messages)
    else:
         context += "There was an issue processing the SQL results (unexpected type).\n"
         llm_instruction = "Apologize for a technical difficulty processing database results based on the context."
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    system_prompt_content = template_Invoice_without_date.replace("{current_date}", today)
    sql_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        MessagesPlaceholder(variable_name="history"), # Include history
        ("human", "Context:\n{context}") # Provide context and specific instruction
    ])
    summary_chain = sql_summary_prompt | chat_model
    history = list(messages[:-1]) if messages else []
    final_content = ""
    print("--- Node generate_response_from_sql: Starting internal .astream() ---") # DEBUG LOG
    stream_ran = False
    stream_chunk_count = 0
    try:
        async for chunk in summary_chain.astream({
            "history": history,
            "llm_instruction": llm_instruction,
            "context": context
        }):
            stream_ran = True
            stream_chunk_count += 1
            # *** Log internal chunk content ***
            print(f"--- Node generate_response_from_sql: Internal chunk {stream_chunk_count}: '{chunk.content}' ---") # DEBUG LOG
            if chunk.content:
                 final_content += chunk.content
        # Log finish status
        print(f"--- Node generate_response_from_sql: Internal .astream() FINISHED. Ran: {stream_ran}. Chunks: {stream_chunk_count}. Content length: {len(final_content)} ---") # DEBUG LOG
    except Exception as stream_err:
        print(f"!!! ERROR internal .astream() in generate_response_from_sql: {stream_err} !!!")
        traceback.print_exc()
        final_content = f"Sorry, an error occurred while generating the response: {stream_err}" # Fallback content
    response_message = AIMessage(content=final_content)
    return {"llm_response_content": final_content, "messages": [response_message]}

# async def get_po_details(po_id: str) -> List[Dict[str, Any]]:
#     """
#     Fetch all rows from podetails for the given purchase-order ID.
#     Returns a list of dicts, one per row.
#     """
#     # Parameterized SQL to avoid injection and handle quoting
#     sql = text("SELECT * FROM podetails WHERE poId = :po_id")
#     try:
#         # Acquire a session (sync). Adjust if using async.
#         db = next(get_db())
#         print(f"--- Executing SQL Query: {sql} with po_id={po_id} ---")

#         result = db.execute(sql, {"po_id": po_id}).mappings().all()
#         supplier_id= [item['supplierId'] for item in result]
#         item_id=[item['itemId'] for item in result]
#         # `.mappings().all()` returns a list of RowMapping → dict-like
#         print("Supplier Id: ",supplier_id,item_id)
#         print(f"--- Query Execution Successful: Fetched {len(result)} rows ---")
#         return [dict(row) for row in result]

#     except Exception as e:
#         print(f"!!! Error executing SQL query: {e} !!!")
#         traceback.print_exc()
#         return []
#     finally:
#         # Always close the session if that's your pattern
#         try:
#             db.close()
#         except Exception:
#             pass
async def generate_direct_response(state: 'GraphState') -> Dict:
    """
    Node to generate a response directly using the LLM when SQL is not applicable.
    Streams internally using .astream() to aggregate content for state update.
    `astream_events` will handle the client-facing stream.
    """
    print("--- Node: generate_direct_response ---")
    messages = state.get("messages", [])

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}") # Get the last human message
    ])
    last_human_message = ""
    history = []
    if messages:
        history = list(messages[:-1])
        if isinstance(messages[-1], HumanMessage):
            last_human_message = messages[-1].content
        else:
            print("Warning: Last message in state was not HumanMessage.")
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_human_message = msg.content
                    break
            if not last_human_message:
                 last_human_message = "[Could not find last user message]"
    direct_response_chain = prompt_template | chat_model
    final_content = ""
    print("--- Node generate_direct_response: Starting internal .astream() ---") # DEBUG LOG
    stream_ran = False
    stream_chunk_count = 0
    try:
        async for chunk in direct_response_chain.astream({
            "history": history,
            "user_input": last_human_message
        }):
            stream_ran = True
            stream_chunk_count += 1
             # *** Log internal chunk content ***
            print(f"--- Node generate_direct_response: Internal chunk {stream_chunk_count}: '{chunk.content}' ---") # DEBUG LOG
            if chunk.content:
                final_content += chunk.content
        # Log finish status
        print(f"--- Node generate_direct_response: Internal .astream() FINISHED. Ran: {stream_ran}. Chunks: {stream_chunk_count}. Content length: {len(final_content)} ---") # DEBUG LOG
    except Exception as stream_err:
        print(f"!!! ERROR internal .astream() in generate_direct_response: {stream_err} !!!")
        traceback.print_exc()
        final_content = f"Sorry, an error occurred while generating the response: {stream_err}" # Fallback content
    response_message = AIMessage(content=final_content)
    return {"llm_response_content": final_content, "messages": [response_message]}
async def extract_details(state: GraphState) -> Dict:
    """
    Node to extract structured details from the final LLM response content.
    This happens server-side after the response is generated.
    """
    print("--- Node: extract_details ---")
    response_text = state.get("llm_response_content")
    extracted_data: Optional[ExtractedInvoiceDetails] = None # Default
    if not response_text:
        print("--- No LLM response text found for extraction. Skipping. ---")
        return {"extracted_details": None}
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", invoice_extraction_prompt),
        ("human", "Extract purchase order details from this text:\n\n```text\n{text_to_extract}\n```")
    ])
    extraction_chain = extraction_prompt | extractor_llm_structured

    try:
        max_len = 8000
        truncated_text = response_text[:max_len] + ("..." if len(response_text) > max_len else "")

        print(f"--- Extracting details from: '{truncated_text[:200]}...' ---")
        llm_extracted_data = await extraction_chain.ainvoke({"text_to_extract": truncated_text})
        # direct_response=await generate_direct_response(state)
        # print("direct response: ",direct_response)

        if llm_extracted_data and isinstance(llm_extracted_data, ExtractedInvoiceDetails):
            print("--- Extraction Successful ---")
            extracted_data = llm_extracted_data
            print(f"Extracted Data: {extracted_data.model_dump_json(indent=2)}") # Log extracted data
        elif llm_extracted_data:
             print(f"--- Extraction returned unexpected type: {type(llm_extracted_data)} ---")
        else:
            print("--- Extraction returned no data. ---")

    except Exception as e:
        print(f"!!! ERROR during detail extraction: {e} !!!")
        traceback.print_exc()
    return {"extracted_details": extracted_data}
async def get_po_details(po_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all rows from podetails for the given purchase-order ID.
    Returns a list of dicts, one per row.
    """
    # Parameterized SQL to avoid injection and handle quoting
    sql = text("SELECT * FROM podetails WHERE poId = :po_id")
    try:
        # Acquire a session (sync). Adjust if using async.
        db = next(get_db())
        print(f"--- Executing SQL Query: {sql} with po_id={po_id} ---")

        result = db.execute(sql, {"po_id": po_id}).mappings().all()
        supplier_id= [item['supplierId'] for item in result]
        item_id=[item['itemId'] for item in result]
        # `.mappings().all()` returns a list of RowMapping → dict-like
        print("Supplier Id: ",supplier_id,item_id)
        print(f"--- Query Execution Successful: Fetched {len(result)} rows ---")
        return [dict(row) for row in result]

    except Exception as e:
        print(f"!!! Error executing SQL query: {e} !!!")
        traceback.print_exc()
        return []
    finally:
        # Always close the session if that's your pattern
        try:
            db.close()
        except Exception:
            pass

async def generate_response(state: GraphState) -> Dict:
    print("--- Node: generate_response ---")
    # This node already expects state["candidate_po_id"] to exist
    po_id = state["candidate_po_id"]
    print("Po Id:", po_id)

    try:
        po_rows = await get_po_details(po_id)
    except Exception as e:
        print(f"!!! Error calling get_po_details: {e} !!!")
        po_rows = []

    if po_rows:
        supplier_ids = list({row["supplierId"] for row in po_rows})
        item_ids = list({row["itemId"] for row in po_rows})
        print(f"--- PO Details fetched. Suppliers: {supplier_ids}, Items: {item_ids} ---")

        followup_system = (
            f"You retrieved details for PO '{po_id}'. "
            f"Supplier IDs: {supplier_ids}. Items: {item_ids}."
        )
        po_item_ids=(f" Items from PO: {item_ids}.")
        last_human = ""
        if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
            last_human = state["messages"][-1].content

        # followup_prompt = ChatPromptTemplate.from_messages([
        #     ("system", followup_system),
        #     ("human", last_human)
        # ])
        followup_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_content),
            ("human", last_human+po_item_ids)
        ])
        followup_chain = followup_prompt | chat_model

        final_content = ""
        async for chunk in followup_chain.astream({}):
            if chunk.content:
                final_content += chunk.content

        response_message = AIMessage(content=final_content)

        # UPDATE last_po_id so the next time we don’t re-route on the same PO again
        return {
            "llm_response_content": final_content,
            "messages": [response_message],
            "last_po_id": po_id
        }
    else:
        fallback = f"I couldn't find any details for PO '{po_id}'. Please check the number and try again."
        response_message = AIMessage(content=fallback)
        return {
            "llm_response_content": fallback,
            "messages": [response_message],
            # Leave last_po_id unchanged if no rows found
            "last_po_id": state.get("last_po_id")
        }

async def detect_po(state: GraphState) -> Dict:
    """
    Invokes the PO‐detector LLM. If a new PO is found (and differs from last_po_id),
    store it as state['candidate_po_id']; otherwise store None.
    """
    messages = state.get("messages", [])
    last_text = ""
    if messages and isinstance(messages[-1], HumanMessage):
        last_text = messages[-1].content

    print("--- Node: detect_po ---")
    print(f"User text for PO detection: '{last_text}'")

    try:
        po_detection_result: PoIdDetection = await po_detector_llm.ainvoke(last_text)
        # po_detection_result: PoIdDetection = await po_detector_llm.ainvoke({
        #     "text_to_analyze": last_text
        # })
        print(f"LLM PO Detection Result: {po_detection_result.model_dump_json()}")

        if po_detection_result.po_id_found and po_detection_result.extracted_po_id:
            extracted = po_detection_result.extracted_po_id
            last_po = state.get("last_po_id")
            if last_po != extracted:
                print(f"→ Detected new candidate_po_id: {extracted}")
                return {"candidate_po_id": extracted}
            else:
                print("→ PO unchanged; clearing candidate_po_id")
                return {"candidate_po_id": None}
        else:
            print("→ No PO detected in this message.")
            return {"candidate_po_id": None}

    except Exception as e:
        print(f"!!! ERROR in detect_po: {e} !!!")
        traceback.print_exc()
        return {"candidate_po_id": None}
    
async def route_after_detect(state: GraphState) -> str:
    """
    Decides whether to go to generate_response (if a brand-new PO was saved)
    or to generate_sql (if no PO or same PO as before).
    """
    cand = state.get("candidate_po_id")
    last_po = state.get("last_po_id")
    print("--- Condition: route_after_detect ---")
    print(f" candidate_po_id = {cand!r}, last_po_id = {last_po!r}")

    if cand:
        print(f"Decision: New PO='{cand}', route → generate_response")
        return "generate_response"
    else:
        print("Decision: No new PO, route → generate_sql")
        return "generate_sql"   

# # --- Conditional Edges ---
def should_execute_sql(state: GraphState) -> str:
    """Determines the next step based on whether SQL was generated."""
    print("--- Condition: should_execute_sql ---")
    if state.get("generated_sql"):
        print("Decision: SQL generated, routing to execute_sql")
        return "execute_sql"
    else:
        print("Decision: No SQL generated, routing to generate_direct_response")
        return "generate_direct_response"
    
# async def should_route_after_preprocess(state: GraphState) -> str:
#     messages = state.get("messages", [])
#     last_text = messages[-1].content if messages and isinstance(messages[-1], HumanMessage) else ""
#     print(f"--- Condition: should_route_after_preprocess (LLM-based PO detection) ---")
#     try:
#         param={"text_to_analyze": last_text}
#         print("test",param)
#         po_detection_result: PoIdDetection = await po_detector_llm.ainvoke(last_text)
#         print(f"LLM PO Detection Result: {po_detection_result.model_dump_json()}")
#         if po_detection_result.po_id_found and po_detection_result.extracted_po_id:
#             po_id = po_detection_result.extracted_po_id
#             last_po = state.get("last_po_id")
#             if last_po != po_id:
#                 state["candidate_po_id"] = po_id
#                 print(f"Decision: Routing to generate_response_path with PO '{po_id}'")
#                 return "generate_response_path"
#             else:
#                 print("Decision: PO unchanged, routing to generate_sql_path")
#                 return "generate_sql_path"
#         else:
#             print("Decision: No PO detected, routing to generate_sql_path")
#             return "generate_sql_path"

#     except Exception as e:
#         print(f"!!! ERROR during LLM PO detection: {e} !!!")
#         traceback.print_exc()
#         return "generate_sql_path"

# --- Build the Workflow ---
workflow = StateGraph(GraphState)
# Add nodes
workflow.add_node("preprocess_input",      preprocess_input)
workflow.add_node("detect_po",             detect_po)
workflow.add_node("generate_sql",          generate_sql)
workflow.add_node("execute_sql",           execute_sql)
workflow.add_node("generate_response_from_sql", generate_response_from_sql)
workflow.add_node("generate_direct_response",    generate_direct_response)
workflow.add_node("extract_details",       extract_details)
workflow.add_node("generate_response",     generate_response)
# 2) Start → preprocess_input → detect_po
workflow.add_edge(START, "preprocess_input")
workflow.add_edge("preprocess_input", "detect_po")
# 3) Immediately after detect_po, run the conditional route_after_detect
workflow.add_conditional_edges(
    "detect_po",
    route_after_detect,
    {
        "generate_sql":      "generate_sql",
        "generate_response": "generate_response",
    }
)
# 4) SQL branch
workflow.add_conditional_edges(
    "generate_sql",
    should_execute_sql,
    {
        "execute_sql":           "execute_sql",
        "generate_direct_response": "generate_direct_response",
    }
)
workflow.add_edge("execute_sql",            "generate_response_from_sql")
# workflow.add_edge("generate_response_from_sql", "extract_details")
workflow.add_edge("generate_response_from_sql", "generate_direct_response")
# 5) PO-response branch
workflow.add_edge("generate_response",      "extract_details")
# workflow.add_edge("generate_response",      "generate_direct_response")
workflow.add_edge("generate_direct_response",    "extract_details")
# 6) Final step
workflow.add_edge("extract_details",        END)
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
async def stream_response_generator(graph_stream: AsyncIterator[Dict]) -> AsyncIterator[str]:
    """
    Streams LLM response chunks for the client.  
    Fixes “unhashable type: 'dict'” by extracting the actual node name string.
    """
    print("--- STREAM GENERATOR (for client) STARTED ---")
    full_response_for_log = ""
    try:
        async for event in graph_stream:
            if event.get("event") == "on_chat_model_stream":
                metadata = event.get("metadata", {})
                node_info = metadata.get("langgraph_node")
                if isinstance(node_info, dict):
                    node_name = node_info.get("node") or node_info.get("name")
                else:
                    node_name = node_info
                if node_name in {"generate_direct_response", "generate_response_from_sql", "generate_response"}:
                    chunk = event["data"].get("chunk")
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        content = chunk.content
                        full_response_for_log += content
                        yield content

    except Exception as e:
        print(f"!!! ERROR in stream_response_generator: {e} !!!")
        yield f"\n\nStream error: {e}"
    finally:
        print(f"--- STREAM GENERATOR (for client) FINISHED ---")
# --- Memory and Compilation ---
memory = MemorySaver()
app_runnable = workflow.compile(checkpointer=memory)

app = FastAPI(
    title="LangGraph Chatbot API",
    description="API endpoint for a LangChain chatbot using LangGraph, detail extraction, SQL generation, and streaming.",
)            
@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """
    Receives user message, invokes the LangGraph app, and streams the
    appropriate LLM response back to the client. Server-side processing
    (SQL, extraction) happens within the graph nodes.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n--- [Thread: {thread_id}] Received message: '{user_message}' ---")
    input_message = HumanMessage(content=user_message)
    input_state = {"messages": [input_message]} # Pass the new message in the list
    try:
        graph_stream = app_runnable.astream_events(input_state, config, version="v2")
        return StreamingResponse(
            stream_response_generator(graph_stream), # Pass the graph stream to the generator
            media_type="text/event-stream" # Use text/event-stream for Server-Sent Events
        )
    except Exception as e:
        print(f"!!! ERROR invoking graph for thread {thread_id}: {e} !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")



origins = [
    "http://localhost:3000", # Allow your frontend origin
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    # Ensure table names are fetched at least once on startup
    get_table_names()
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)

