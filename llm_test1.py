import datetime
import os
import uuid
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse # Import StreamingResponse
from pydantic import AliasChoices, BaseModel, Field
from dotenv import load_dotenv
from typing import Dict, List, Optional, TypedDict, Annotated, Sequence, AsyncIterator # Typing for state, tools, and streaming
import operator
import traceback # For detailed error logging
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    AIMessageChunk, # Import AIMessageChunk for type checking stream
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END,MessagesState,START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from fastapi.middleware.cors import CORSMiddleware
from requests import Session
from sqlalchemy import text
# Assuming database.py and llm_templates.py are in the same directory or accessible
from database import get_db # Make sure get_db returns a context manager or generator
from llm_templates import template_Promotion_without_date

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# Optional: Configure LangSmith tracing
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Your LangSmith Project Name"
# os.environ["LANGCHAIN_API_KEY"] = "Your LangSmith API Key" # If needed

print(f"LangSmith tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2') == 'true'}")
print(f"LangSmith project: {os.getenv('LANGCHAIN_PROJECT')}")

# --- Tool Definition ---
class ExtractedPromotionDetails(BaseModel):
    # Use validation_alias and AliasChoices for more flexible field naming in LLM output
    promotion_type: str | None = Field(None, validation_alias=AliasChoices('Promotion Type', 'promotion_type'))
    hierarchy_type: str | None = Field(None, validation_alias=AliasChoices('Hierarchy Type', 'hierarchy_type'))
    hierarchy_value: str | None = Field(None, validation_alias=AliasChoices('Hierarchy Value', 'hierarchy_value'))
    brand: str | None = Field(None, validation_alias=AliasChoices('Brand', 'brand'))
    items: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Items', 'items'))
    excluded_items: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Items', 'excluded_items'))
    discount_type: str | None = Field(None, validation_alias=AliasChoices('Discount Type', 'discount_type'))
    discount_value: str | None = Field(None, validation_alias=AliasChoices('Discount Value', 'discount_value')) # Keep as str to handle % vs fixed ambiguity
    start_date: str | None = Field(None, validation_alias=AliasChoices('Start Date', 'start_date'))
    end_date: str | None = Field(None, validation_alias=AliasChoices('End Date', 'end_date'))
    stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Stores', 'stores'))
    excluded_stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Stores', 'excluded_stores'))

    class Config:
        populate_by_name = True # Allows using field names as well as aliases
        extra = 'ignore' # Ignore extra fields the LLM might hallucinate

# --- LangChain/LangGraph Setup ---
# Ensure model is initialized with streaming=True
model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, streaming=True)

# Agent State definition remains the same
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- User's Prompt Template ---
today = datetime.datetime.today().strftime("%d/%m/%Y")

# Ensure template_Promotion_without_date is loaded correctly
# Example placeholder if llm_templates.py is not available:
# template_Promotion_without_date = "System prompt template. Current date: {current_date}"
SYSTEM_PROMPT=template_Promotion_without_date.replace("{current_date}", today)

# 1. Initialize the Chat Model (already done above)

# 2. Define the Prompt Template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 3. Define the LangGraph Workflow
workflow = StateGraph(MessagesState) # Use MessagesState for simplicity if not using custom tools

# Define the function that calls the model
async def call_model(state: MessagesState):
    """Invokes the LLM with the current state and system prompt."""
    messages = state.get("messages", [])
    # Ensure messages is always a list
    if not isinstance(messages, list):
         print(f"Warning: state['messages'] was not a list: {type(messages)}. Resetting.")
         messages = []

    # Ensure all elements in messages are BaseMessage instances
    # (This might be needed if state manipulation introduces other types)
    valid_messages = [msg for msg in messages if isinstance(msg, BaseMessage)]
    if len(valid_messages) != len(messages):
        print(f"Warning: Filtered out non-BaseMessage objects from state['messages']")
        messages = valid_messages

    prompt_input = {"messages": messages}
    try:
        # LangChain automatically traces this call to LangSmith if configured
        # Use stream, not invoke/ainvoke here if the node itself should stream
        # response = await model.ainvoke(prompt) # Use ainvoke for non-streaming full response

        # If using stream_events v2, the graph handles streaming, so ainvoke here is correct
        # to get the final message for the state update.
        prompt = await prompt_template.ainvoke(prompt_input)
        response = await model.ainvoke(prompt)

        # Ensure the response is added correctly to the state
        return {"messages": [response]}
    except Exception as e:
        print(f"Error during model invocation: {e}")
        traceback.print_exc() # Print full traceback for debugging
        # Return an error message within the expected state structure
        error_ai_msg = AIMessage(content=f"Sorry, an error occurred during model invocation: {e}")
        # Append error message to existing messages if possible, or return just the error
        return {"messages": [error_ai_msg]}


# Conditional edge logic remains the same (assuming no tools are defined/bound yet)
# If you add tools later, you'll need a ToolNode and conditional logic
# def should_continue(state: AgentState) -> str:
#     last_message = state['messages'][-1]
#     if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#         print("--- Decision: Route to Tools ---")
#         return "call_tool"
#     else:
#         print("--- Decision: End Turn ---")
#         return END

# Graph construction
workflow.add_node("llm", call_model)
workflow.add_edge(START, "llm")
# If you add tools:
# tool_node = ToolNode(tools) # Define 'tools' list first
# workflow.add_node("call_tool", tool_node)
# workflow.add_conditional_edges(
#     "llm",
#     should_continue,
#     {
#         "call_tool": "call_tool",
#         END: END,
#     },
# )
# workflow.add_edge("call_tool", "llm") # Route back to LLM after tool call
# If no tools, simple edge to END:
workflow.add_edge("llm", END) # Use END constant

# Memory setup remains the same
memory = MemorySaver()

# Compile the graph
app_runnable = workflow.compile(checkpointer=memory)

# --- Detail Extractor Setup ---
extractor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0) # Use low temp for extraction
# Bind the Pydantic model for structured output
extractor_llm_structured = extractor_llm.with_structured_output(
    ExtractedPromotionDetails,
    method="function_calling", # Or "json_mode" if preferred and model supports
    include_raw=False
)

async def extract_details_from_response(response_text: str) -> ExtractedPromotionDetails | None:
    """
    Uses a dedicated LLM call to extract structured promotion details
    from the main AI's response text.
    """
    print("\n--- Attempting Detail Extraction ---")
    if not response_text:
        print("--- No response text provided for extraction. Skipping. ---")
        return None

    # Simple instruction for the extractor LLM
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert extraction system. Analyze the provided text, which is a response from a promotion creation assistant summarizing the current state of a promotion. Extract the promotion details mentioned into the structured format defined by the 'ExtractedPromotionDetails' function/tool.

- Today's date is {today}. Use this ONLY if needed to interpret relative dates mentioned IN THE TEXT (like "starts tomorrow"), standardizing to dd/mm/yyyy.
- Standardize date formats found in the text to dd/mm/yyyy.
- Extract lists for items, excluded items, stores, excluded stores.
- If a field is explicitly mentioned as 'Missing' or not present in the text, leave its value as null or empty list.
- Focus *only* on the details present in the text. Do not infer or add information not explicitly stated.
"""),
        ("human", "Extract promotion details from this text:\n\n```text\n{text_to_extract}\n```")
    ])

    # Create the extraction chain
    extraction_chain = extraction_prompt | extractor_llm_structured

    try:
        # Limit input text length for safety/cost
        max_len = 8000
        truncated_text = response_text[:max_len] + ("..." if len(response_text) > max_len else "")

        extracted_data = await extraction_chain.ainvoke({"text_to_extract": truncated_text})

        if extracted_data:
            print("--- Extraction Successful ---")
            # Ensure it's the correct Pydantic type before returning
            if isinstance(extracted_data, ExtractedPromotionDetails):
                return extracted_data
            else:
                 print(f"--- Extraction returned unexpected type: {type(extracted_data)} ---")
                 return None
        else:
            print("--- Extraction returned no data. ---")
            return None

    except Exception as e:
        print(f"!!! ERROR during detail extraction: {e} !!!")
        traceback.print_exc()
        return None

# --- Database Schema Definition ---
# Define the schema clearly for the LLM
TABLE_SCHEMA = """
Database Schema:

1. `itemmaster` (Alias: im) (Base Table for Item Details):
   - `itemId` (VARCHAR/INT): Unique identifier for each item. (Primary Key)
   - `itemDescription` (VARCHAR): Primary description of the item.
   - `itemSecondaryDescription` (VARCHAR): Additional details about the item.
   - `itemDepartment` (VARCHAR): Broader category (e.g., T-Shirt, Trousers, Jackets). Use LIKE 'value%' for filtering.
   - `itemClass` (VARCHAR): Classification within a department (e.g., Formals, Casuals, Leather). Use LIKE 'value%' for filtering.
   - `itemSubClass` (VARCHAR): Granular classification (e.g., Full Sleeve, Half Sleeve, Zipper, Regular Fit). Use LIKE 'value%' for filtering.
   - `brand` (VARCHAR): Brand associated with the item (e.g., Zara, Adidas, H&M). Use = 'value' for filtering.
   - `diffType1` (INT): Foreign key linking to `itemdiffs.id` (e.g., for color).
   - `diffType2` (INT): Foreign key linking to `itemdiffs.id` (e.g., for size).
   - `diffType3` (INT): Foreign key linking to `itemdiffs.id` (e.g., for material).

2. `itemsupplier` (Alias: isup) (For Cost & Supplier Data):
   - `id` (INT): Unique identifier for this relationship. (Primary Key)
   - `supplierCost` (DECIMAL/FLOAT): Cost of the item from the supplier.
   - `supplierId` (VARCHAR/INT): Identifier for the supplier.
   - `itemId` (VARCHAR/INT): Foreign key linking to `itemmaster.itemId`.

3. `itemdiffs` (Alias: idf) (For Attribute Filtering - Differentiation Types):
   - `id` (INT): Unique identifier for each differentiation attribute. (Primary Key)
   - `diffType` (VARCHAR): The attribute type (e.g., 'color', 'size', 'material'). Often used with diffId.
   - `diffId` (VARCHAR): The actual differentiation value (e.g., 'Red', 'XL', 'Cotton'). Use = 'value' for filtering.

4. `storedetails` (Alias: sd) (For Store Information):
   - `storeId` (INT): Unique identifier for each store. (Primary Key)
   - `storeName` (VARCHAR): Name of the store.
   - `address` (VARCHAR): Street address.
   - `city` (VARCHAR): City.
   - `state` (VARCHAR): State.
   - `zipCode` (VARCHAR): ZIP code.
   - `phone` (VARCHAR): Contact phone number.

Relationships:
- `itemmaster.itemId` links to `itemsupplier.itemId`. JOIN using `ON im.itemId = isup.itemId`.
- `itemmaster.diffType1`, `itemmaster.diffType2`, `itemmaster.diffType3` link to `itemdiffs.id`.
"""

# --- LLM Initialization for SQL Generation ---
sql_generator_llm = ChatOpenAI(
    model="gpt-4o-mini", # Consider more powerful models like gpt-4o if mini struggles
    api_key=OPENAI_API_KEY,
    temperature=0.0 # Use 0 temperature for maximum determinism in SQL generation
)

# --- Database Helper Functions ---
def get_table_names():
    """Fetch all table names from the database."""
    db: Session | None = None
    try:
        db = next(get_db()) # Get DB session
        # Ensure the query is compatible with your specific SQL dialect (MySQL, PostgreSQL, etc.)
        # For MySQL:
        result = db.execute(text("SHOW TABLES")).fetchall()
        # For PostgreSQL:
        # result = db.execute(text("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema';")).fetchall()
        return [row[0] for row in result] # Extract table names (index might vary)
    except Exception as e:
        print(f"Error fetching table names: {e}")
        return [] # Return empty list on error
    finally:
        if db:
            db.close() # Ensure connection is closed

def execute_mysql_query(query: str) -> List[Dict] | str:
    """
    Executes a given SQL SELECT query after basic validation.

    Args:
        query: The SQL SELECT query string.

    Returns:
        A list of dictionaries representing the rows, or an error string.
    """
    db: Session | None = None
    try:
        # Basic validation: Check if it's a SELECT query
        if not query or not query.strip().lower().startswith("select"):
            return "Error: Only SELECT queries are allowed."

        # (Optional but recommended) More robust parsing could be added here
        # to prevent unwanted commands or analyze structure further.

        # --- Table Name Validation (Example - adjust if needed) ---
        # This is a basic check; a more robust parser might be needed for complex queries with aliases/subqueries
        words = query.upper().split()
        valid_tables = get_table_names()
        if not valid_tables: # Handle error from get_table_names
             return "Error: Could not retrieve table names for validation."

        table_found = False
        if "FROM" in words:
            try:
                from_index = words.index("FROM")
                potential_table = words[from_index + 1].strip("`;,()") # Basic extraction
                # Check against known tables (case-insensitive comparison might be better)
                if potential_table.lower() in [t.lower() for t in valid_tables]:
                     table_found = True
                # This logic might need enhancement for JOINs, aliases etc.
                # For now, we assume the first table after FROM is the main one.
            except (IndexError, ValueError):
                pass # Could not parse table name simply

        if not table_found:
            # You might want to be stricter or allow queries if parsing fails
            print(f"Warning: Could not definitively validate table name in query: {query}")
            return f"Error: Could not validate table name in query. Ensure it uses valid tables: {valid_tables}"
        # --- End Table Name Validation ---


        print(f"--- Executing SQL Query: {query} ---")
        db = next(get_db()) # Get DB session
        result = db.execute(text(query)).fetchall()
        print(f"--- Query Execution Successful: Fetched {len(result)} rows ---")
        # Convert rows to list of dicts
        return [dict(row._mapping) for row in result]

    except Exception as e:
        error_msg = f"Error executing SQL query: {e}"
        print(f"!!! {error_msg} !!!")
        traceback.print_exc()
        return error_msg # Return the error message string
    finally:
        if db:
            db.close() # Ensure connection is closed


# --- SQL Generation Function ---
async def generate_sql_from_natural_language(natural_language_query: str) -> str | None:
    """
    Uses an LLM call to convert a natural language query into a SQL SELECT query,
    based on the predefined TABLE_SCHEMA and specific instructions.

    Args:
        natural_language_query: The user's query in plain English.

    Returns:
        The generated SQL SELECT query string, or None if generation fails or is invalid.
    """
    print("\n--- Attempting SQL Query Generation ---")
    if not natural_language_query:
        print("--- No natural language query provided. Skipping SQL generation. ---")
        return None

    # Updated prompt for SQL generation with detailed instructions and examples
    sql_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert SQL generator. Your task is to convert the user's natural language question into a valid SQL SELECT statement based ONLY on the database schema and rules provided below.

Database Schema:
{TABLE_SCHEMA}

Core Task: Generate a SQL SELECT statement to retrieve `itemmaster.itemId` (and potentially other requested columns like `supplierCost`) based on the user's filtering criteria.

Rules & Instructions:
1.  **Focus on Selection Criteria:** Analyze the user's request and extract ONLY the criteria relevant for selecting items (e.g., brand, color, size, department, class, subclass, supplier cost).
2.  **Ignore Irrelevant Information:** Completely IGNORE any information not directly related to filtering or selecting items based on the schema. This includes discounts, promotion details, validity dates, action verbs like "Create", "Offer", "Update", store IDs (unless specifically asked to filter by store details from `storedetails`). Your output MUST be a SELECT query, nothing else.
3.  **SELECT Clause:** Primarily select `im.itemId`. If supplier cost is mentioned or requested, also select `isup.supplierCost`. If other specific columns are requested, include them using the appropriate aliases (im, isup, idf, sd). Use `DISTINCT` if joins might produce duplicate `itemId`s based on the query structure.
4.  **FROM Clause:** Start with `FROM itemmaster im`.
5.  **JOIN Clauses:**
    * If filtering by `supplierCost` or selecting it, `JOIN itemsupplier isup ON im.itemId = isup.itemId`.
    * Filtering by attributes (color, size, material, etc. stored in `itemdiffs`) requires checking `diffType1`, `diffType2`, `diffType3`. Use the `EXISTS` method for this. See Example 1 below.
    * If filtering by store details, `JOIN storedetails sd ON ...` (Note: There's no direct link given between itemmaster/itemsupplier and storedetails in the schema, assume filtering by store details applies elsewhere or cannot be done with this schema unless a link is implied or added).
6.  **WHERE Clause Construction:**
    * **Attributes (`itemdiffs`):** To filter by an attribute like 'Red' or 'Large', use `EXISTS` subqueries checking `itemdiffs` linked via `diffType1`, `diffType2`, or `diffType3`. Example: `WHERE EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Red')`.
    * **Direct `itemmaster` Fields:**
        * Use `im.brand = 'Value'` for exact brand matches.
        * Use `im.itemDepartment LIKE 'Value%'` for department matches.
        * Use `im.itemClass LIKE 'Value%'` for class matches.
        * Use `im.itemSubClass LIKE 'Value%'` for subclass matches.
    * **`itemsupplier` Fields:** Use `isup.supplierCost < Value`, `isup.supplierCost > Value`, etc.
    * **Multiple Values (Same Field):** Use `OR` (e.g., `im.brand = 'Zara' OR im.brand = 'Adidas'`). Consider using `IN` for longer lists (e.g., `im.brand IN ('Zara', 'Adidas')`).
    * **Multiple Conditions (Different Fields):** Use `AND` (e.g., `im.itemDepartment LIKE 'T-Shirt%' AND im.brand = 'Zara'`).
7.  **Output Format:** Generate ONLY the SQL SELECT statement. No explanations, no comments, no markdown backticks (```sql ... ```), no trailing semicolon.
8.  **Invalid Queries:** If the user's query asks for something impossible with the schema (e.g., filtering items by store without a link, asking for non-SELECT operations), respond with "Query cannot be answered with the provided schema."

Examples (Study these carefully):

Example 1: Select all red colored items
User Query: "Select all red colored items"
SQL: SELECT DISTINCT im.itemId FROM itemmaster im WHERE EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Red')

Example 2: Select all red colored items with a supplier cost below $50
User Query: "Select all red colored items with a supplier cost below $50"
SQL: SELECT DISTINCT im.itemId, isup.supplierCost FROM itemmaster im JOIN itemsupplier isup ON im.itemId = isup.itemId WHERE isup.supplierCost < 50 AND (EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Red'))

Example 3: Select all items from FashionX and Zara brands
User Query: "Select all items from FashionX and Zara brands"
SQL: SELECT im.itemId FROM itemmaster im WHERE im.brand = 'FashionX' OR im.brand = 'Zara'

Example 4: Select all items from T-Shirt department and Casuals class
User Query: "Select all items from T-Shirt department and Casuals class"
SQL: SELECT im.itemId FROM itemmaster im WHERE im.itemDepartment LIKE 'T-Shirt%' AND im.itemClass LIKE 'Casuals%'

Example 5: Complex request with irrelevant info
User Query: "Create a simple promotion offering 30% off all yellow items from the FashionX Brand in the T-Shirt Department, valid from 17/04/2025 until the end of May 2025."
SQL: SELECT DISTINCT im.itemId FROM itemmaster im WHERE im.brand = 'FashionX' AND im.itemDepartment LIKE 'T-Shirt%' AND (EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Yellow') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Yellow') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Yellow'))

"""),
        ("human", "Convert this question to a SQL SELECT query:\n\n```text\n{user_query}\n```")
    ])

    # Create the generation chain (Prompt -> LLM -> String Output)
    sql_generation_chain = sql_generation_prompt | sql_generator_llm | StrOutputParser()

    try:
        # Limit input text length (optional)
        max_len = 2000
        truncated_query = natural_language_query[:max_len] + ("..." if len(natural_language_query) > max_len else "")

        print(f"--- Generating SQL for: '{truncated_query}' ---")
        generated_sql = await sql_generation_chain.ainvoke({"user_query": truncated_query})

        # Basic validation and cleaning
        generated_sql = generated_sql.strip().strip(';').strip() # Remove surrounding whitespace and trailing semicolon

        if not generated_sql:
             print("--- Generation failed: LLM returned empty string. ---")
             return None

        if "cannot be answered" in generated_sql.lower():
             print(f"--- LLM indicated query cannot be answered: {generated_sql} ---")
             return None # Explicitly return None if LLM says it's impossible

        # Check if it starts with SELECT (case-insensitive)
        if generated_sql.lower().startswith("select"):
            # Further check for potentially harmful keywords (basic protection)
            harmful_keywords = ['update ', 'insert ', 'delete ', 'drop ', 'alter ', 'truncate ']
            if any(keyword in generated_sql.lower() for keyword in harmful_keywords):
                 print(f"--- Generation failed: Potentially harmful non-SELECT keyword detected: {generated_sql} ---")
                 return None
            print("--- SQL Generation Successful ---")
            return generated_sql
        else:
            print(f"--- Generation failed: Output does not start with SELECT: {generated_sql} ---")
            return None

    except Exception as e:
        print(f"!!! ERROR during SQL generation: {e} !!!")
        traceback.print_exc()
        return None

# --- FastAPI Application ---
app = FastAPI(
    title="LangChain Chatbot API",
    description="API endpoint for a LangChain chatbot using detail extraction, SQL generation, and streaming.",
)
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
# --- API Request Model ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


# --- Streaming Generator (Using astream_events) ---
async def stream_response_generator(app_runnable, input_state: dict, config: dict) -> AsyncIterator[str]:
    """
    Asynchronously streams LLM response chunks for the client using astream_events.
    """
    print("--- STREAM GENERATOR (for client) STARTED ---")
    full_response = ""
    try:
        # Stream events from the graph execution
        async for event in app_runnable.astream_events(input_state, config, version="v2"):
            kind = event["event"]
            # Look for chunks specifically from the chat model stream
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                # Ensure it's an AIMessageChunk and has content
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    content_to_yield = chunk.content
                    full_response += content_to_yield
                    # print(f"Streaming chunk: {content_to_yield}") # Debug print
                    yield content_to_yield # Yield the content string

            # Optional: Log other events for debugging if needed
            # elif kind == "on_chain_start":
            #     print(f"--- Chain Start: {event['name']} ---")
            # elif kind == "on_chain_end":
            #     print(f"--- Chain End: {event['name']} ---")
            # elif kind == "on_llm_end":
            #      print(f"--- LLM End ---")

    except Exception as e:
        print(f"!!! ERROR in stream_response_generator: {e} !!!")
        traceback.print_exc()
        # Yield an error message to the client stream
        yield f"\n\nStream error: {e}"
    finally:
        print(f"--- STREAM GENERATOR (for client) FINISHED ---")
        # Log the full response assembled from chunks (optional)
        # print(f"--- Full response streamed to client ({config['configurable']['thread_id']}):\n{full_response}\n---")


@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """
    Receives user message, gets AI response, performs server-side detail extraction
    and SQL generation/execution, and streams the AI response back to the client.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- [Thread: {thread_id}] Received message: '{user_message}' ---")

    # Prepare input for the graph
    input_message = HumanMessage(content=user_message)

    # Get current state to maintain conversation history
    try:
        current_state = await app_runnable.aget_state(config)
        # Ensure messages exist and are a list, default to empty list if not
        current_messages = current_state.values.get("messages", []) if current_state and current_state.values else []
        if not isinstance(current_messages, list):
             print(f"Warning: current_state messages was not a list: {type(current_messages)}. Resetting.")
             current_messages = []
    except Exception as e:
        # Handle cases where the thread doesn't exist yet or state retrieval fails
        print(f"Warning: Could not retrieve state for thread {thread_id}. Starting new history. Error: {e}")
        current_messages = []

    # Append the new user message to the history for this turn's input
    input_state = {"messages": current_messages + [input_message]}

    # --- Server-Side Extraction & SQL Step ---
    # These happen *after* the main response is generated in this flow
    extracted_details_dict: Dict | None = None
    extracted_sql_query: str | None = None
    sql_result: List[Dict] | str | None = None # Can be list of results, error string, or None
    ai_response_text = ""

    try:
        print(f"--- [Thread: {thread_id}] Invoking main graph to get full response (for extraction/SQL) ---")
        # Use ainvoke to get the complete final state *without* streaming to the client yet.
        # This runs the graph (including the LLM call) once.
        final_state = await app_runnable.ainvoke(input_state, config)

        # Extract the final AI message content from the state
        if final_state and isinstance(final_state.get("messages"), list) and final_state["messages"]:
            # The last message should be the AI's response from the 'call_model' node
            last_message = final_state["messages"][-1]
            if isinstance(last_message, AIMessage):
                ai_response_text = last_message.content or "" # Ensure it's a string
                print(f"--- [Thread: {thread_id}] Full AI response obtained (length: {len(ai_response_text)}) ---")

                # 1. Perform Detail Extraction (Optional, based on AI response)
                extracted_details_obj = await extract_details_from_response(ai_response_text)
                if extracted_details_obj:
                    extracted_details_dict = extracted_details_obj.model_dump(by_alias=False) # Use field names
                    print(f"--- [Thread: {thread_id}] Extracted Details (Server-Side Log) ---")
                    print(json.dumps(extracted_details_dict, indent=2))
                    print("-------------------------------------------------------------")
                else:
                    print(f"--- [Thread: {thread_id}] Detail extraction failed or returned no data. ---")

                # 2. Perform SQL Generation (Based on original user message)
                extracted_sql_query = await generate_sql_from_natural_language(user_message)

                if extracted_sql_query:
                    print(f"--- [Thread: {thread_id}] Generated SQL Query: {extracted_sql_query} ---")
                    # 3. Execute SQL Query
                    sql_result = execute_mysql_query(extracted_sql_query)

                    # Check if execution returned results or an error message
                    if isinstance(sql_result, list):
                        print(f"--- [Thread: {thread_id}] SQL Execution Successful: {len(sql_result)} rows fetched. ---")
                        # print("SQL FINAL RESULT:", sql_result) # Log the actual data if needed (can be large)

                        # ***** NEW STEP: Use the SQL Result *****
                        if sql_result: # Check if the list is not empty
                            print(f"--- [Thread: {thread_id}] Using non-empty SQL result ---")
                            # --- Placeholder for using the SQL result ---
                            # How you use it depends on your application's logic.
                            # Examples:
                            # - Log the findings.
                            # - Store it for context in the *next* user turn.
                            # - Perform further processing based on the data.
                            # - NOTE: As mentioned, this happens *after* the current AI response
                            #   is generated. To influence the *current* response, the workflow
                            #   needs restructuring (SQL gen/exec before main LLM call).
                            print(f"--- [Thread: {thread_id}] SQL Result Data (first 5 rows): {sql_result[:5]}")
                            # Add your logic here, e.g., update_context_for_next_turn(thread_id, sql_result)
                            # -----------------------------------------
                        else:
                             print(f"--- [Thread: {thread_id}] SQL query executed but returned no rows. ---")

                    elif isinstance(sql_result, str): # Indicates an error message from execute_mysql_query
                        print(f"--- [Thread: {thread_id}] SQL Execution Failed: {sql_result} ---")
                        # sql_result already contains the error string
                else:
                    print(f"--- [Thread: {thread_id}] SQL Generation failed or not applicable. ---")

            else:
                 print(f"--- [Thread: {thread_id}] Last message in final_state was not AIMessage ({type(last_message)}). Cannot extract/process. ---")
        else:
            print(f"--- [Thread: {thread_id}] No messages or invalid format in final_state after invoke. Cannot extract/process. State: {final_state} ---")

    except Exception as e:
        print(f"!!! [Thread: {thread_id}] ERROR during server-side invoke/extraction/SQL phase: {e} !!!")
        traceback.print_exc()
        # We might still attempt to stream an error or a default message if ai_response_text wasn't set
        if not ai_response_text:
             ai_response_text = f"An error occurred processing your request: {e}"
        # Consider if you should proceed to streaming or raise HTTP exception here

    # --- Stream the Response Back to Client ---
    # This uses the *original* input_state which triggers the graph run again,
    # but this time `astream_events` is used to capture and yield the streaming chunks.
    try:
        print(f"--- [Thread: {thread_id}] Initiating streaming response for client ---")
        # Pass the original input_state and config to the generator
        return StreamingResponse(
            stream_response_generator(app_runnable, input_state, config),
            media_type="text/event-stream" # Use text/event-stream for Server-Sent Events
        )
    except Exception as e:
        print(f"!!! [Thread: {thread_id}] Error setting up stream: {e} !!!")
        traceback.print_exc()
        # If invoke/extraction succeeded but streaming fails, raise an error
        raise HTTPException(status_code=500, detail=f"Internal Server Error during streaming setup: {e}")

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "Welcome to the LangChain Chatbot API (Detail Extraction, SQL, Streaming Enabled)!"}

# --- Optional: Run with Uvicorn (for local testing) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# import datetime
# import os
# import uuid
# import json
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import StreamingResponse # Import StreamingResponse
# from pydantic import AliasChoices, BaseModel, Field
# from dotenv import load_dotenv
# from typing import Dict, List, Optional, TypedDict, Annotated, Sequence, AsyncIterator # Typing for state, tools, and streaming
# import operator
# import traceback # For detailed error logging
# from langchain_core.output_parsers import StrOutputParser 
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import (
#     HumanMessage,
#     SystemMessage,
#     AIMessage,
#     ToolMessage,
#     BaseMessage,
#     AIMessageChunk, # Import AIMessageChunk for type checking stream
# )
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.tools import tool
# from langgraph.graph import StateGraph, END,MessagesState,START
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.prebuilt import ToolNode
# from fastapi.middleware.cors import CORSMiddleware
# from requests import Session
# from sqlalchemy import text
# from database import get_db
# from llm_templates import template_Promotion_without_date
# # --- Configuration ---
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY environment variable not set.")

# print(f"LangSmith tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2') == 'true'}")
# print(f"LangSmith project: {os.getenv('LANGCHAIN_PROJECT')}")

# # --- Tool Definition ---
# class ExtractedPromotionDetails(BaseModel):
#     # Use validation_alias and AliasChoices for more flexible field naming in LLM output
#     promotion_type: str | None = Field(None, validation_alias=AliasChoices('Promotion Type', 'promotion_type'))
#     hierarchy_type: str | None = Field(None, validation_alias=AliasChoices('Hierarchy Type', 'hierarchy_type'))
#     hierarchy_value: str | None = Field(None, validation_alias=AliasChoices('Hierarchy Value', 'hierarchy_value'))
#     brand: str | None = Field(None, validation_alias=AliasChoices('Brand', 'brand'))
#     items: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Items', 'items'))
#     excluded_items: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Items', 'excluded_items'))
#     discount_type: str | None = Field(None, validation_alias=AliasChoices('Discount Type', 'discount_type'))
#     discount_value: str | None = Field(None, validation_alias=AliasChoices('Discount Value', 'discount_value')) # Keep as str to handle % vs fixed ambiguity
#     start_date: str | None = Field(None, validation_alias=AliasChoices('Start Date', 'start_date'))
#     end_date: str | None = Field(None, validation_alias=AliasChoices('End Date', 'end_date'))
#     stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Stores', 'stores'))
#     excluded_stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Stores', 'excluded_stores'))

#     class Config:
#         populate_by_name = True # Allows using field names as well as aliases
#         extra = 'ignore' # Ignore extra fields the LLM might hallucinate

# # --- LangChain/LangGraph Setup ---
# # Ensure model is initialized with streaming=True
# model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, streaming=True)
# # model_with_tools = model.bind_tools(tools)

# # Agent State definition remains the same
# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], operator.add]

# # --- User's Prompt Template ---
# today = datetime.datetime.today().strftime("%d/%m/%Y")

# SYSTEM_PROMPT=template_Promotion_without_date.replace("{current_date}", today)

# # 1. Initialize the Chat Model
# model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# # 2. Define the Prompt Template
# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", SYSTEM_PROMPT),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

# # 3. Define the LangGraph Workflow
# workflow = StateGraph(MessagesState)

# # Define the function that calls the model
# async def call_model(state: MessagesState):
#     """Invokes the LLM with the current state and system prompt."""
#     messages = state.get("messages", [])
#     if not isinstance(messages, list):
#          print(f"Warning: state['messages'] was not a list: {type(messages)}. Resetting.")
#          messages = []

#     prompt_input = {"messages": messages}
#     try:
#         prompt = await prompt_template.ainvoke(prompt_input)
#         # LangChain automatically traces this call to LangSmith if configured
#         response = await model.ainvoke(prompt)
#         return {"messages": [response]}
#     except Exception as e:
#         print(f"Error during model invocation: {e}")
#         error_message = AIMessage(content=f"Sorry, an error occurred: {e}")
        
# # Conditional edge logic remains the same
# def should_continue(state: AgentState) -> str:
#     last_message = state['messages'][-1]
#     if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#         print("--- Decision: Route to Tools ---")
#         return "call_tool"
#     else:
#         print("--- Decision: End Turn ---")
#         return END

# # Graph construction remains the same
# # workflow = StateGraph(AgentState)
# workflow.add_node("llm", call_model)
# workflow.add_edge(START, "llm")
# workflow.add_edge("llm", "__end__")

# # Memory setup remains the same
# memory = MemorySaver()

# # Compile the graph
# app_runnable = workflow.compile(checkpointer=memory)

# # --- Detail Extractor Setup ---
# extractor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0) # Use low temp for extraction
# # Bind the Pydantic model for structured output
# extractor_llm_structured = extractor_llm.with_structured_output(
#     ExtractedPromotionDetails,
#     method="function_calling", # Or "json_mode" if preferred and model supports
#     include_raw=False
# )

# async def extract_details_from_response(response_text: str) -> ExtractedPromotionDetails | None:
#     """
#     Uses a dedicated LLM call to extract structured promotion details
#     from the main AI's response text.
#     """
#     print("\n--- Attempting Detail Extraction ---")
#     if not response_text:
#         print("--- No response text provided for extraction. Skipping. ---")
#         return None

#     # Simple instruction for the extractor LLM
#     extraction_prompt = ChatPromptTemplate.from_messages([
#         ("system", f"""
# You are an expert extraction system. Analyze the provided text, which is a response from a promotion creation assistant summarizing the current state of a promotion. Extract the promotion details mentioned into the structured format defined by the 'ExtractedPromotionDetails' function/tool.

# - Today's date is {today}. Use this ONLY if needed to interpret relative dates mentioned IN THE TEXT (like "starts tomorrow"), standardizing to dd/mm/yyyy.
# - Standardize date formats found in the text to dd/mm/yyyy.
# - Extract lists for items, excluded items, stores, excluded stores.
# - If a field is explicitly mentioned as 'Missing' or not present in the text, leave its value as null or empty list.
# - Focus *only* on the details present in the text. Do not infer or add information not explicitly stated.
# """),
#         ("human", "Extract promotion details from this text:\n\n```text\n{text_to_extract}\n```")
#     ])

#     # Create the extraction chain
#     extraction_chain = extraction_prompt | extractor_llm_structured

#     try:
#         # Limit input text length for safety/cost
#         max_len = 8000
#         truncated_text = response_text[:max_len] + ("..." if len(response_text) > max_len else "")

#         extracted_data = await extraction_chain.ainvoke({"text_to_extract": truncated_text})

#         if extracted_data:
#             print("--- Extraction Successful ---")
#             return extracted_data
#         else:
#             print("--- Extraction returned no data. ---")
#             return None

#     except Exception as e:
#         print(f"!!! ERROR during detail extraction: {e} !!!")
#         traceback.print_exc()
#         return None
    
# # --- Database Schema Definition ---
# # Define the schema clearly for the LLM
# TABLE_SCHEMA = """
# Database Schema:

# 1. `itemmaster` (Alias: im) (Base Table for Item Details):
#    - `itemId` (VARCHAR/INT): Unique identifier for each item. (Primary Key)
#    - `itemDescription` (VARCHAR): Primary description of the item.
#    - `itemSecondaryDescription` (VARCHAR): Additional details about the item.
#    - `itemDepartment` (VARCHAR): Broader category (e.g., T-Shirt, Trousers, Jackets). Use LIKE 'value%' for filtering.
#    - `itemClass` (VARCHAR): Classification within a department (e.g., Formals, Casuals, Leather). Use LIKE 'value%' for filtering.
#    - `itemSubClass` (VARCHAR): Granular classification (e.g., Full Sleeve, Half Sleeve, Zipper, Regular Fit). Use LIKE 'value%' for filtering.
#    - `brand` (VARCHAR): Brand associated with the item (e.g., Zara, Adidas, H&M). Use = 'value' for filtering.
#    - `diffType1` (INT): Foreign key linking to `itemdiffs.id` (e.g., for color).
#    - `diffType2` (INT): Foreign key linking to `itemdiffs.id` (e.g., for size).
#    - `diffType3` (INT): Foreign key linking to `itemdiffs.id` (e.g., for material).

# 2. `itemsupplier` (Alias: isup) (For Cost & Supplier Data):
#    - `id` (INT): Unique identifier for this relationship. (Primary Key)
#    - `supplierCost` (DECIMAL/FLOAT): Cost of the item from the supplier.
#    - `supplierId` (VARCHAR/INT): Identifier for the supplier.
#    - `itemId` (VARCHAR/INT): Foreign key linking to `itemmaster.itemId`.

# 3. `itemdiffs` (Alias: idf) (For Attribute Filtering - Differentiation Types):
#    - `id` (INT): Unique identifier for each differentiation attribute. (Primary Key)
#    - `diffType` (VARCHAR): The attribute type (e.g., 'color', 'size', 'material'). Often used with diffId.
#    - `diffId` (VARCHAR): The actual differentiation value (e.g., 'Red', 'XL', 'Cotton'). Use = 'value' for filtering.

# 4. `storedetails` (Alias: sd) (For Store Information):
#    - `storeId` (INT): Unique identifier for each store. (Primary Key)
#    - `storeName` (VARCHAR): Name of the store.
#    - `address` (VARCHAR): Street address.
#    - `city` (VARCHAR): City.
#    - `state` (VARCHAR): State.
#    - `zipCode` (VARCHAR): ZIP code.
#    - `phone` (VARCHAR): Contact phone number.

# Relationships:
# - `itemmaster.itemId` links to `itemsupplier.itemId`. JOIN using `ON im.itemId = isup.itemId`.
# - `itemmaster.diffType1`, `itemmaster.diffType2`, `itemmaster.diffType3` link to `itemdiffs.id`.
# """

# # --- LLM Initialization ---
# # Use a lower temperature for more precise SQL generation
# sql_generator_llm = ChatOpenAI(
#     model="gpt-4o-mini", # Consider more powerful models like gpt-4o if mini struggles
#     api_key=OPENAI_API_KEY,
#     temperature=0.0 # Use 0 temperature for maximum determinism in SQL generation
# )

# def execute_mysql_query(query):
#     """Executes a MySQL query after validating the table name."""
#     try:
#         words = query.split()
#         if "FROM" in words:
#             table_index = words.index("FROM") + 1
#             table_name = words[table_index].strip("`;,")  # Extract table name safely
#         else:
#             return "Invalid SQL query structure."

#         # Get existing tables
#         valid_tables = get_table_names()
#         if table_name not in valid_tables:
#             return f"Error: Table `{table_name}` does not exist. Available tables: {valid_tables}"

#         # Execute the validated query
#         db: Session = next(get_db())
#         result = db.execute(text(query)).fetchall()
#         db.close()
#         return [dict(row._mapping) for row in result]

#     except Exception as e:
#         return str(e)

# def get_table_names():
#     """Fetch all table names from the database."""
#     try:
#         db: Session = next(get_db())
#         result = db.execute(text("SHOW TABLES")).fetchall()
#         db.close()
#         return [list(row)[0] for row in result]  # Extract table names
#     except Exception as e:
#         return str(e)
# # --- SQL Generation Function ---
# async def generate_sql_from_natural_language(natural_language_query: str) -> str | None:
#     """
#     Uses an LLM call to convert a natural language query into a SQL SELECT query,
#     based on the predefined TABLE_SCHEMA and specific instructions.

#     Args:
#         natural_language_query: The user's query in plain English.

#     Returns:
#         The generated SQL SELECT query string, or None if generation fails.
#     """
#     print("\n--- Attempting SQL Query Generation (v2) ---")
#     if not natural_language_query:
#         print("--- No natural language query provided. Skipping. ---")
#         return None

#     # Updated prompt for SQL generation with detailed instructions and examples
#     sql_generation_prompt = ChatPromptTemplate.from_messages([
#         ("system", f"""You are an expert SQL generator. Your task is to convert the user's natural language question into a valid SQL SELECT statement based ONLY on the database schema and rules provided below.

# Database Schema:
# {TABLE_SCHEMA}

# Core Task: Generate a SQL SELECT statement to retrieve `itemmaster.itemId` (and potentially other requested columns like `supplierCost`) based on the user's filtering criteria.

# Rules & Instructions:
# 1.  **Focus on Selection Criteria:** Analyze the user's request and extract ONLY the criteria relevant for selecting items (e.g., brand, color, size, department, class, subclass, supplier cost).
# 2.  **Ignore Irrelevant Information:** Completely IGNORE any information not directly related to filtering or selecting items based on the schema. This includes discounts, promotion details, validity dates, action verbs like "Create", "Offer", "Update", store IDs (unless specifically asked to filter by store details from `storedetails`). Your output MUST be a SELECT query, nothing else.
# 3.  **SELECT Clause:** Primarily select `im.itemId`. If supplier cost is mentioned or requested, also select `isup.supplierCost`. If other specific columns are requested, include them using the appropriate aliases (im, isup, idf, sd). Use `DISTINCT` if joins might produce duplicate `itemId`s based on the query structure.
# 4.  **FROM Clause:** Start with `FROM itemmaster im`.
# 5.  **JOIN Clauses:**
#     * If filtering by `supplierCost` or selecting it, `JOIN itemsupplier isup ON im.itemId = isup.itemId`.
#     * Filtering by attributes (color, size, material, etc. stored in `itemdiffs`) requires checking `diffType1`, `diffType2`, `diffType3`. Use the `EXISTS` method for this. See Example 1 below.
#     * If filtering by store details, `JOIN storedetails sd ON ...` (Note: There's no direct link given between itemmaster/itemsupplier and storedetails in the schema, assume filtering by store details applies elsewhere or cannot be done with this schema unless a link is implied or added).
# 6.  **WHERE Clause Construction:**
#     * **Attributes (`itemdiffs`):** To filter by an attribute like 'Red' or 'Large', use `EXISTS` subqueries checking `itemdiffs` linked via `diffType1`, `diffType2`, or `diffType3`. Example: `WHERE EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Red') OR EXISTS (...) OR EXISTS (...)`.
#     * **Direct `itemmaster` Fields:**
#         * Use `im.brand = 'Value'` for exact brand matches.
#         * Use `im.itemDepartment LIKE 'Value%'` for department matches.
#         * Use `im.itemClass LIKE 'Value%'` for class matches.
#         * Use `im.itemSubClass LIKE 'Value%'` for subclass matches.
#     * **`itemsupplier` Fields:** Use `isup.supplierCost < Value`, `isup.supplierCost > Value`, etc.
#     * **Multiple Values (Same Field):** Use `OR` (e.g., `im.brand = 'Zara' OR im.brand = 'Adidas'`). Consider using `IN` for longer lists (e.g., `im.brand IN ('Zara', 'Adidas')`).
#     * **Multiple Conditions (Different Fields):** Use `AND` (e.g., `im.itemDepartment LIKE 'T-Shirt%' AND im.brand = 'Zara'`).
# 7.  **Output Format:** Generate ONLY the SQL SELECT statement. No explanations, no comments, no markdown backticks (```sql ... ```), no trailing semicolon.
# 8.  **Invalid Queries:** If the user's query asks for something impossible with the schema (e.g., filtering items by store without a link, asking for non-SELECT operations), respond with "Query cannot be answered with the provided schema."

# Examples (Study these carefully):

# Example 1: Select all red colored items
# User Query: "Select all red colored items"
# SQL: SELECT DISTINCT im.itemId FROM itemmaster im WHERE EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Red')

# Example 2: Select all red colored items with a supplier cost below $50
# User Query: "Select all red colored items with a supplier cost below $50"
# SQL: SELECT DISTINCT im.itemId, isup.supplierCost FROM itemmaster im JOIN itemsupplier isup ON im.itemId = isup.itemId WHERE isup.supplierCost < 50 AND (EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Red'))

# Example 3: Select all items from FashionX and Zara brands
# User Query: "Select all items from FashionX and Zara brands"
# SQL: SELECT im.itemId FROM itemmaster im WHERE im.brand = 'FashionX' OR im.brand = 'Zara'

# Example 4: Select all items from T-Shirt department and Casuals class
# User Query: "Select all items from T-Shirt department and Casuals class"
# SQL: SELECT im.itemId FROM itemmaster im WHERE im.itemDepartment LIKE 'T-Shirt%' AND im.itemClass LIKE 'Casuals%'

# Example 5: Complex request with irrelevant info
# User Query: "Create a simple promotion offering 30% off all yellow items from the FashionX Brand in the T-Shirt Department, valid from 17/04/2025 until the end of May 2025."
# SQL: SELECT DISTINCT im.itemId FROM itemmaster im WHERE im.brand = 'FashionX' AND im.itemDepartment LIKE 'T-Shirt%' AND (EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Yellow') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Yellow') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Yellow'))

# """),
#         ("human", "Convert this question to a SQL SELECT query:\n\n```text\n{user_query}\n```")
#     ])

#     # Create the generation chain (Prompt -> LLM -> String Output)
#     sql_generation_chain = sql_generation_prompt | sql_generator_llm | StrOutputParser()

#     try:
#         # Limit input text length (optional)
#         max_len = 2000
#         truncated_query = natural_language_query[:max_len] + ("..." if len(natural_language_query) > max_len else "")

#         print(f"--- Generating SQL for: '{truncated_query}' ---")
#         generated_sql = await sql_generation_chain.ainvoke({"user_query": truncated_query})

#         # Basic validation and cleaning
#         generated_sql = generated_sql.strip().strip(';') # Remove surrounding whitespace and trailing semicolon

#         if generated_sql.lower().startswith("select"):
#             print("--- SQL Generation Successful ---")
#             return generated_sql
#         elif "cannot be answered" in generated_sql:
#              print(f"--- LLM indicated query cannot be answered: {generated_sql} ---")
#              return None
#         else:
#             # Check if it looks like SQL but doesn't start with select (e.g., LLM hallucinated UPDATE/INSERT)
#             if any(keyword in generated_sql.lower() for keyword in ['update ', 'insert ', 'delete ', 'drop ']):
#                  print(f"--- Generation failed: Non-SELECT statement generated: {generated_sql} ---")
#                  return None
#             print(f"--- Generation failed or produced non-SQL/invalid output: {generated_sql} ---")
#             return None

#     except Exception as e:
#         print(f"!!! ERROR during SQL generation: {e} !!!")
#         traceback.print_exc()
#         return None
       
# # --- FastAPI Application ---
# app = FastAPI(
#     title="LangChain Chatbot API (Function Calling, Streaming with stream_mode='messages')",
#     description="API endpoint for a LangChain chatbot using tools and streaming final message tokens.",
# )
# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:3000",
#     "http://localhost:3000/*",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # --- API Request Model ---
# class ChatRequest(BaseModel):
#     message: str
#     thread_id: str | None = None

      
# # --- Streaming Generator (Using astream_events) ---
# async def stream_response_generator(app_runnable, input_state: dict, config: dict) -> AsyncIterator[str]:
#     """
#     Asynchronously streams LLM response chunks for the client using astream_events.
#     """
#     print("--- STREAM GENERATOR (for client) STARTED ---")
#     full_response = ""
#     try:
#         async for event in app_runnable.astream_events(input_state, config, version="v2"):
#             kind = event["event"]
#             if kind == "on_chat_model_stream":
#                 chunk = event["data"].get("chunk")
#                 if isinstance(chunk, AIMessageChunk) and chunk.content:
#                     content_to_yield = chunk.content
#                     full_response += content_to_yield
#                     yield content_to_yield

#             # Optional: Log end event
#             elif kind == "on_chain_end":
#                  pass # print(f"Stream Chain Ended for config: {config}")

#     except Exception as e:
#         print(f"!!! ERROR in stream_response_generator: {e} !!!")
#         traceback.print_exc()
#         yield f"\n\nStream error: {e}"
#     finally:
#         print(f"--- STREAM GENERATOR (for client) FINISHED ---")
#         # print(f"--- Full response streamed to client ({config['configurable']['thread_id']}):\n{full_response}\n---")

# @app.post("/chat/")
# async def chat_endpoint(request: ChatRequest):
#     """
#     Receives user message, gets AI response, performs server-side detail extraction,
#     and streams the AI response back to the client.
#     """
#     user_message = request.message
#     thread_id = request.thread_id or str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}

#     # Prepare input for the graph
#     input_message = HumanMessage(content=user_message)
#     # Get current state to maintain conversation history
#     # Use aget_state to fetch the history associated with the thread_id
#     try:
#         current_state = await app_runnable.aget_state(config)
#         current_messages = current_state.values.get("messages", []) if current_state else []
#         if not isinstance(current_messages, list):
#              print(f"Warning: current_state messages was not a list: {type(current_messages)}. Resetting.")
#              current_messages = []
#     except Exception as e:
#         print(f"Warning: Could not retrieve state for thread {thread_id}. Starting new history. Error: {e}")
#         current_messages = []

#     # Append the new user message to the history
#     input_state = {"messages": current_messages + [input_message]}

#     # --- Server-Side Extraction Step ---
#     extracted_details_dict: Dict | None = None
#     ai_response_text = ""
#     try:
#         print(f"\n--- [Thread: {thread_id}] Invoking main graph for full response (for extraction) ---")
#         # Use ainvoke to get the complete final state *without* streaming
#         final_state = await app_runnable.ainvoke(input_state, config)
#         if final_state and final_state.get("messages"):
#             final_messages = final_state["messages"]
#             if isinstance(final_messages, list) and final_messages:
#                 # The last message in the state after invoke should be the AI's response
#                 last_message = final_messages[-1]
#                 if isinstance(last_message, AIMessage):
#                     ai_response_text = last_message.content
#                     print(f"--- [Thread: {thread_id}] Full AI response obtained for extraction (length: {len(ai_response_text)}) ---")

#                     # Perform extraction on the full response text
#                     extracted_details_obj = await extract_details_from_response(ai_response_text)
#                     extracted_sql_query = await generate_sql_from_natural_language(user_message)
                    
#                     print("Extracted Query from user's message: ",extracted_sql_query)

#                     if extracted_details_obj:
#                         # Convert Pydantic model to dict for logging/potential future use
#                         extracted_details_dict = extracted_details_obj.model_dump(by_alias=False) # Use field names for logging
#                         print("--- [Thread: {thread_id}] Extracted Details (Server-Side Log) ---")
#                         print(json.dumps(extracted_details_dict, indent=2))
#                         print("-------------------------------------------------------------")
#                     else:
#                         print(f"--- [Thread: {thread_id}] Detail extraction failed or returned no data. ---")
#                     if extracted_sql_query:
#                         sql_result=execute_mysql_query(extracted_sql_query)
#                         print("SQL FINAL RESULT: ", sql_result)
#                     else:
#                         print("No query extracted") 
#                 else:
#                      print(f"--- [Thread: {thread_id}] Last message in final_state was not AIMessage ({type(last_message)}). Cannot extract. ---")
#             else:
#                  print(f"--- [Thread: {thread_id}] No messages or invalid format in final_state. Cannot extract. ---")
#         else:
#             print(f"--- [Thread: {thread_id}] Failed to get final state from ainvoke. Cannot extract. ---")

#     except Exception as e:
#         print(f"!!! [Thread: {thread_id}] ERROR during invoke/extraction phase: {e} !!!")
#         traceback.print_exc()
#     try:
#         print(f"--- [Thread: {thread_id}] Initiating streaming response for client ---")
#         return StreamingResponse(
#             stream_response_generator(app_runnable, input_state, config),
#             media_type="text/plain"
#         )
#     except Exception as e:
#         print(f"!!! [Thread: {thread_id}] Error setting up stream: {e} !!!")
#         traceback.print_exc()
#         # If invoke/extraction succeeded but streaming fails, we might still want to inform the client
#         raise HTTPException(status_code=500, detail=f"Internal Server Error during streaming setup: {e}")

# # --- Root Endpoint ---
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the LangChain Chatbot API (Function Calling & Streaming Enabled)!"}
