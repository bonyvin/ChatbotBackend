import datetime
import os
import uuid
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, AliasChoices
from dotenv import load_dotenv
# Typing imports
from typing import List, TypedDict, Annotated, Sequence, AsyncIterator, Dict, Optional, Any

import operator
import traceback

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage,
    AIMessageChunk,
    ToolMessage # Import ToolMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool # Import tool decorator
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode # Import ToolNode
from database import engine,SessionLocal,get_db

# FastAPI middleware
from fastapi.middleware.cors import CORSMiddleware

# Database imports (assuming these are set up correctly in your project)
from sqlalchemy.orm import Session
from sqlalchemy import text
# Assume get_db is defined elsewhere and yields a Session
# from .database import get_db # Example import

# --- Mock Database Setup (Replace with your actual setup) ---
# This is a placeholder to make the code runnable without a live DB connection.
# You MUST replace this with your actual SQLAlchemy session management (e.g., using FastAPI dependencies).

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

print(f"LangSmith tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2') == 'true'}")
print(f"LangSmith project: {os.getenv('LANGCHAIN_PROJECT')}")


# --- Database Helper Functions (Adapted for Tool) ---

def get_table_names() -> List[str] | str:
    """Fetch all table names from the database."""
    print("--- Getting table names ---")
    try:
        # Use context manager for session handling if get_db provides one
        db_gen = get_db()
        db = next(db_gen)
        result = db.execute(text("SHOW TABLES")).fetchall()
        # Ensure proper closing/cleanup if needed by your get_db implementation
        # next(db_gen, None) # Consume the rest of the generator if it yields then cleans up
        db.close() # Or call close if it's expected
        print(f"Raw table names result: {result}")
        # Handle potential differences in result format (list of tuples vs list of mappings)
        if result and isinstance(result[0], tuple):
             names = [row[0] for row in result]
        elif result and isinstance(result[0], dict):
             # Assuming the key is the table name or similar
             key = list(result[0].keys())[0]
             names = [row[key] for row in result]
        else:
             names = [] # Handle empty or unexpected format

        print(f"Fetched table names: {names}")
        return names
    except Exception as e:
        print(f"Error fetching table names: {e}")
        traceback.print_exc()
        return f"Error fetching table names: {str(e)}"

# Define a simple schema description (enhance with column details for better results)
# Fetch dynamically or define statically based on your needs
DB_SCHEMA = f"""
Database Schema:
The database contains the following tables:
{json.dumps(get_table_names(), indent=2)}

Relevant Table Columns :
    - `itemmaster` (Base Table):
        - **`itemId`**: Unique identifier for each item.  
        - **`itemDescription`**: Primary description of the item.  
        - **`itemSecondaryDescription`**: Additional details about the item.  
        - **`itemDepartment`**: The broader category an item belongs to (e.g., T-Shirt, Trousers, Jackets).  
        - **`itemClass`**: A classification within a department (e.g., Formals, Casuals, Leather).  
        - **`itemSubClass`**: A more granular classification under the item class (e.g., Full Sleeve, Half Sleeve, Zipper, Regular Fit).  
        - **`brand`**: The brand associated with the item (e.g., Zara, Adidas, H&M).  
        - **`diffType1`, `diffType2`, `diffType3`**: Foreign keys linking to `itemdiffs.id`, representing specific item attributes such as color, size, or material.   
    - `itemsupplier` (For Cost & Supplier Data):
        - **`id`**: Unique identifier for each supplier-item relationship.  
        - **`supplierCost`**: The cost of the item from the supplier.  
        - **`supplierId`**: The identifier for the supplier providing the item.  
        - **`itemId`**: Foreign key linking to `itemmaster.itemId`, establishing the relationship between items and suppliers.   
    - `itemdiffs` (For Attribute Filtering):
        - **`id`**: Unique identifier for each differentiation type.  
        - **`diffType`**: The attribute type used to differentiate items (e.g., color, size, material).  
        - **`diffId`**: The actual differentiation value corresponding to `diffType`.  
    - `storedetails` (For Store Information):
        - **`storeId`**: Unique identifier for each store.
        - **`storeName`**: The name of the store.
        - **`address`**: Street address of the store.
        - **`city`**: City where the store is located.
        - **`state`**: State where the store is located.
        - **`zipCode`**: ZIP code of the store's location.
        - **`phone`**: Contact phone number for the store.

Use standard SQL syntax compatible with MySQL.
Always query primary identifiers like itemId or storeId when asked for items or stores unless specifically asked for descriptions or other fields.
"""

# --- Tool Definitions ---

@tool
async def text_to_sql_tool(natural_language_query: str) -> str:
    """
    Converts a natural language query about promotion items, stores, or related data into a SQL query
    based on the database schema. Returns only the SQL query string or an error message.
    """
    print(f"--- Tool Called: text_to_sql_tool ---")
    print(f"Natural language query: {natural_language_query}")

    # Use a dedicated LLM for SQL generation
    sql_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert SQL generator. Given the database schema below and a natural language query,
generate a *single*, valid, safe (read-only SELECT) SQL query to retrieve the requested information.
Respond ONLY with the SQL query string, without any explanation, comments, or markdown formatting.

{DB_SCHEMA}

Constraints:
- Generate ONLY SELECT statements. Do NOT generate INSERT, UPDATE, DELETE, DROP, or other modifying statements.
- Ensure the query is syntactically correct for MySQL.
- Use aliases for table names if needed (e.g., `itemmaster im`).
- If the query is ambiguous or requires information not present in the schema, respond with "Error: Ambiguous query or missing schema information."
- If the request seems unsafe or malicious, respond with "Error: Unsafe query request."
- Prefer querying specific IDs (itemId, storeId) unless names/descriptions are explicitly requested.
"""),
        ("human", "Generate the SQL query for: {nl_query}")
    ])

    chain = prompt | sql_llm

    try:
        response = await chain.ainvoke({"nl_query": natural_language_query})
        sql_query = response.content.strip().strip('`').strip(';') # Clean up potential LLM artifacts
        print(f"Generated SQL: {sql_query}")

        # Basic safety check (redundant with prompt but good practice)
        if not sql_query.lower().startswith("select"):
             print("Error: Generated query is not a SELECT statement.")
             return "Error: Generated query is not a SELECT statement."

        return sql_query
    except Exception as e:
        print(f"Error generating SQL: {e}")
        traceback.print_exc()
        return f"Error generating SQL: {str(e)}"


@tool
def execute_sql_tool(query: str) -> str:
    """
    Executes a provided SQL SELECT query against the database after basic validation.
    Returns the query results as a JSON string list or an error message.
    Only executes SELECT queries on allowed tables.
    """
    print(f"--- Tool Called: execute_sql_tool ---")
    print(f"Executing SQL: {query}")

    # Validate query type (redundant safety check)
    if not query or not isinstance(query, str) or not query.lower().strip().startswith("select"):
        print("Error: Invalid or non-SELECT query provided.")
        return "Error: Invalid or non-SELECT query provided."

    try:
        # Basic table name extraction and validation
        words = query.split()
        table_name = None
        if "FROM" in words:
            from_index = words.index("FROM")
            if from_index + 1 < len(words):
                # Handle potential alias: table_name alias
                potential_name = words[from_index + 1].strip("`;,()")
                # Check if the *next* word is likely an alias (common patterns)
                if from_index + 2 < len(words) and words[from_index + 2].strip("`;,()").lower() not in ['where', 'join', 'left', 'right', 'inner', 'outer', 'group', 'order', 'limit']:
                     table_name = potential_name # Assume it's the table name
                else:
                     table_name = potential_name # Assume it's the table name even if followed by keyword

        elif "UPDATE" in words: # Add checks for other DML if needed, though SELECT is primary
             update_index = words.index("UPDATE")
             if update_index + 1 < len(words):
                  table_name = words[update_index + 1].strip("`;,()")
        # Add checks for JOINs if necessary, might involve multiple tables

        if not table_name:
             print("Error: Could not determine table name from query.")
             return "Error: Could not determine table name from query."

        valid_tables_result = get_table_names()
        if isinstance(valid_tables_result, str): # Means get_table_names returned an error
             print(f"Error fetching valid tables: {valid_tables_result}")
             return f"Error fetching valid tables: {valid_tables_result}"
        valid_tables = valid_tables_result

        # Check if extracted table name is in the valid list
        # This is a basic check; complex queries with joins need more robust parsing
        if table_name not in valid_tables:
            print(f"Error: Table `{table_name}` is not in the allowed list: {valid_tables}")
            return f"Error: Table `{table_name}` is not allowed or does not exist. Allowed tables: {valid_tables}"

        # Execute the validated query
        db_gen = get_db()
        db = next(db_gen)
        result = db.execute(text(query)).fetchall()
        db.close() # Or however your session needs cleanup

        # Convert result to list of dicts, then to JSON string for LLM
        result_list = [dict(row._mapping) for row in result]
        print(f"Query Result (list of dicts): {result_list}")
        # Limit result size before returning to LLM
        max_results = 50
        if len(result_list) > max_results:
             result_str = json.dumps(result_list[:max_results]) + f"... (truncated to {max_results} results)"
        else:
             result_str = json.dumps(result_list)

        return result_str

    except Exception as e:
        print(f"Error executing SQL: {e}")
        traceback.print_exc()
        # Return specific error if possible, otherwise generic message
        return f"Error executing SQL query: {str(e)}"

# List of tools for the agent
tools = [text_to_sql_tool, execute_sql_tool]

# --- Pydantic Model for Extracted Details ---
# (Keep ExtractedPromotionDetails as defined before)
class ExtractedPromotionDetails(BaseModel):
    promotion_type: Optional[str] = Field(None, validation_alias=AliasChoices('Promotion Type', 'promotion_type'))
    hierarchy_type: Optional[str] = Field(None, validation_alias=AliasChoices('Hierarchy Type', 'hierarchy_type'))
    hierarchy_value: Optional[str] = Field(None, validation_alias=AliasChoices('Hierarchy Value', 'hierarchy_value'))
    brand: Optional[str] = Field(None, validation_alias=AliasChoices('Brand', 'brand'))
    items: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Items', 'items'))
    excluded_items: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Items', 'excluded_items'))
    discount_type: Optional[str] = Field(None, validation_alias=AliasChoices('Discount Type', 'discount_type'))
    discount_value: Optional[str] = Field(None, validation_alias=AliasChoices('Discount Value', 'discount_value'))
    start_date: Optional[str] = Field(None, validation_alias=AliasChoices('Start Date', 'start_date'))
    end_date: Optional[str] = Field(None, validation_alias=AliasChoices('End Date', 'end_date'))
    stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Stores', 'stores'))
    excluded_stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Stores', 'excluded_stores'))

    class Config:
        populate_by_name = True
        extra = 'ignore'


# --- LangChain/LangGraph Setup ---
# Main conversational LLM
main_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
# Bind tools to the main LLM
main_llm_with_tools = main_llm.bind_tools(tools)

# --- User's Prompt Template ---
today = datetime.datetime.now().strftime("%d/%m/%Y")
# --- Updated System Prompt ---
SYSTEM_PROMPT = f"""
Hello and welcome! I'm ExpX, your dedicated assistant for creating promotions. Today is {today}.

*My Goal*: Help you define all the required details for a promotion.

*Required Promotion Details*:
- Promotion Type: (Simple, Buy X/Get Y, Threshold, Gift With Purchase)
- Hierarchy Level (Type & Value): [Department | Class | Sub Class] and its value
- Brand: (e.g., FashionX, H&M)
- Items: Specific Item IDs (e.g., ITEM001, ITEM003)
- Discount (Type & Value): [% Off | Fixed Price | BOGO] and the value
- Dates (Start & End): dd/mm/yyyy format
- Stores: Specific Store IDs (e.g., STORE001, STORE004)

*Optional Details*:
- Excluded Items: Specific Item IDs to exclude
- Excluded Stores: Specific Store IDs to exclude

*How to Interact*:
- Provide details step-by-step or all at once.
- **NEW: Natural Language Queries:** For 'Items' or 'Stores', you can ask me to find them using natural language (e.g., "Items: all t-shirts from FashionX", "Stores: all stores in the northeast").
- I will standardize formats (dates to dd/mm/yyyy).
- I will keep track of details and show a summary.
- I will list missing required fields.
- I will ask for confirmation ('Yes') only when all *required* fields are filled.

*My Tools*:
- `text_to_sql_tool`: When you provide a natural language query for items or stores, I will use this tool FIRST to generate the appropriate SQL query.
- `execute_sql_tool`: AFTER generating the SQL with `text_to_sql_tool`, I will use this tool to run the query and get the results (e.g., list of item IDs or store IDs). I will then populate the relevant field (Items or Stores) with these results.

*Example NL Query Flow*:
User: "Items: all yellow t-shirts from FashionX"
Assistant (Internal Thought): I need to find item IDs. I'll use `text_to_sql_tool` with the query "all yellow t-shirts from FashionX".
Assistant (Calls Tool): `text_to_sql_tool(natural_language_query="all yellow t-shirts from FashionX")`
Tool Returns: "SELECT im.itemId FROM itemmaster im WHERE im.department = 'T-Shirt' AND im.brand = 'FashionX' AND im.itemId IN (SELECT itemId FROM itemsiffs WHERE diffType1 = 'Color' AND diffValue1 = 'Yellow')" (Example SQL)
Assistant (Calls Tool): `execute_sql_tool(query="<SQL from previous step>")`
Tool Returns: "[{{\\"itemId\\": \\"ITEM001\\"}}, {{\\"itemId\\": \\"ITEM007\\"}}] "
Assistant (Updates State & Responds): Okay, I found the following items matching 'all yellow t-shirts from FashionX': ITEM001, ITEM007.
  *Current Promotion Details Summary*:
  ... (summary with Items populated) ...

*Important*:
- I need specific Item IDs and Store IDs in the final promotion details. Natural language queries will be resolved to these IDs using my tools.
- I will simulate database validation for now.

Let's begin! Please provide the first detail for your promotion.

*Current Promotion Details Summary*:
(Summary logic based on conversation history - LLM manages this)

*Missing Required Fields*:
[List missing fields]

{{#if all_details_present}} # Hypothetical
Would you like to submit this promotion? (Yes/No)
{{else}}
Please provide the missing details or your next instruction.
{{/if}}
"""

# Define the Prompt Template (Main agent)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the LangGraph Workflow with Tools
workflow = StateGraph(MessagesState) # Use MessagesState

# Define the function that calls the main model with tools
async def call_main_model_with_tools(state: MessagesState):
    """Invokes the main LLM (with tools bound) with the current state."""
    print("--- Calling Main Model Node (with tools) ---")
    messages = state["messages"]
    # Use the LLM instance with tools bound
    response = await main_llm_with_tools.ainvoke(messages)
    print(f"--- Model Response Type: {type(response)} ---")
    if hasattr(response, 'tool_calls') and response.tool_calls:
         print(f"--- Model requested tool calls: {response.tool_calls} ---")
    else:
         print("--- Model did not request tool calls. ---")
    # The state automatically updates with the AIMessage or AIMessage with tool_calls
    return {"messages": [response]}

# Tool node executes the requested tools
tool_node = ToolNode(tools)

# Conditional edge logic
def should_route_to_tools(state: MessagesState) -> str:
    """Determines whether to route to the tool node or end the turn."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("--- Decision: Route to Tools ---")
        return "call_tool" # Name corresponds to the ToolNode name
    else:
        print("--- Decision: End Turn ---")
        return END

# Build the graph with tools
workflow.add_node("llm", call_main_model_with_tools)
workflow.add_node("call_tool", tool_node) # Add the tool node

workflow.add_edge(START, "llm") # Start goes to LLM

# Add conditional edge from LLM
workflow.add_conditional_edges(
    "llm", # Source node
    should_route_to_tools, # Function to determine route
    {
        "call_tool": "call_tool", # If function returns "call_tool", go to tool_node
        END: END # If function returns END, finish the graph run
    }
)

workflow.add_edge("call_tool", "llm") # After tools execute, route back to LLM

# Memory setup
memory = MemorySaver()

# Compile the graph
app_runnable = workflow.compile(checkpointer=memory)


# --- Detail Extractor Setup ---
# (Keep extractor_llm, extractor_llm_structured, and extract_details_from_response as before)
extractor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
extractor_llm_structured = extractor_llm.with_structured_output(
    ExtractedPromotionDetails,
    method="function_calling",
    include_raw=False
)
async def extract_details_from_response(response_text: str) -> Optional[ExtractedPromotionDetails]:
    """
    Uses a dedicated LLM call to extract structured promotion details
    from the main AI's response text.
    """
    print("\n--- Attempting Detail Extraction ---")
    if not response_text:
        print("--- No response text provided for extraction. Skipping. ---")
        return None
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert extraction system. Analyze the provided text from a promotion assistant and extract the promotion details into the structured format defined by the 'ExtractedPromotionDetails' function/tool.
- Today's date is {today}. Use this ONLY if needed to interpret relative dates mentioned IN THE TEXT, standardizing to dd/mm/yyyy.
- Standardize date formats found in the text to dd/mm/yyyy.
- Extract lists for items, excluded items, stores, excluded stores. Item/Store lists should contain specific IDs if available in the text.
- If a field is explicitly mentioned as 'Missing' or not present in the text, leave its value as null or empty list.
- Focus *only* on the details present in the text.
"""),
        ("human", "Extract promotion details from this text:\n\n```text\n{text_to_extract}\n```")
    ])
    extraction_chain = extraction_prompt | extractor_llm_structured
    try:
        max_len = 8000
        truncated_text = response_text[:max_len] + ("..." if len(response_text) > max_len else "")
        extracted_data = await extraction_chain.ainvoke({"text_to_extract": truncated_text})
        if extracted_data:
            print("--- Extraction Successful ---")
            return extracted_data
        else:
            print("--- Extraction returned no data. ---")
            return None
    except Exception as e:
        print(f"!!! ERROR during detail extraction: {e} !!!")
        traceback.print_exc()
        return None

# --- FastAPI Application ---
app = FastAPI(
    title="LangChain Chatbot API (NL2SQL, Streaming, Extraction)",
    description="API endpoint for LangChain chatbot with NL2SQL tools, streaming, and extraction.",
)
# (Keep CORS middleware setup as before)
origins = [
    "http://localhost:3000",
    "http://localhost",
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
    thread_id: Optional[str] = None

# --- Streaming Generator (Using astream_events) ---
# (Keep stream_response_generator as defined before)
async def stream_response_generator(app_runnable, input_state: dict, config: dict) -> AsyncIterator[str]:
    """
    Asynchronously streams LLM response chunks for the client using astream_events.
    Handles potential tool calls within the stream.
    """
    print("--- STREAM GENERATOR (for client) STARTED ---")
    full_response = ""
    try:
        # Stream events from the graph which now includes tool handling
        async for event in app_runnable.astream_events(input_state, config, version="v2"):
            kind = event["event"]
            # print(f"Stream Event: {kind}, Name: {event.get('name')}, Data: {event.get('data')}") # Detailed log

            # Stream final LLM token chunks to the client
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    content_to_yield = chunk.content
                    full_response += content_to_yield
                    yield content_to_yield
            # Optional: Log tool start/end if needed for frontend status updates
            elif kind == "on_tool_start":
                 print(f"--- Tool Started: {event['name']} Input: {event['data'].get('input')} ---")
                 # yield f"[Tool running: {event['name']}]" # Example: Send status to client
            elif kind == "on_tool_end":
                 print(f"--- Tool Ended: {event['name']} Output: {event['data'].get('output')} ---")
                 # yield f"[Tool finished: {event['name']}]" # Example: Send status to client

    except Exception as e:
        print(f"!!! ERROR in stream_response_generator: {e} !!!")
        traceback.print_exc()
        yield f"\n\nStream error: {e}"
    finally:
        print(f"--- STREAM GENERATOR (for client) FINISHED ---")


# --- API Endpoint ---
# (Keep chat_endpoint using the temporary invoke ID logic for extraction)
@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """
    Receives message, gets AI response (potentially using tools), performs
    server-side detail extraction using a temporary config, and streams the
    AI response (reflecting tool results) back to the client using the original config.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    input_message = HumanMessage(content=user_message)
    try:
        current_state = await app_runnable.aget_state(config)
        current_messages = current_state.values.get("messages", []) if current_state and current_state.values else []
        if not isinstance(current_messages, list):
             print(f"Warning: State messages for {thread_id} was not a list: {type(current_messages)}. Resetting.")
             current_messages = []
    except Exception as e:
        print(f"Warning: Could not retrieve state for thread {thread_id}. Starting new history. Error: {e}")
        current_messages = []

    input_state = {"messages": current_messages + [input_message]}

    # --- Server-Side Extraction Step (using temporary config) ---
    extracted_details_dict: Optional[Dict] = None
    ai_response_text = ""
    try:
        invoke_thread_id = f"invoke-{uuid.uuid4()}"
        invoke_config = {"configurable": {"thread_id": invoke_thread_id}}
        print(f"\n--- [Thread: {thread_id}] Invoking graph for extraction using temp ID: {invoke_thread_id} ---")

        # Use ainvoke with the *temporary* config. This WILL execute tools if needed.
        final_state = await app_runnable.ainvoke(input_state, invoke_config)

        # Process the result from the temporary invoke run
        if final_state and final_state.get("messages"):
            final_messages = final_state["messages"]
            # Check the structure: LangGraph might return state dict or just messages list
            if isinstance(final_messages, dict) and "messages" in final_messages:
                 final_messages = final_messages["messages"] # Adjust if invoke returns full state dict

            if isinstance(final_messages, list) and final_messages:
                # The final message should be the AI's response AFTER potential tool calls
                last_message = final_messages[-1]
                if isinstance(last_message, AIMessage): # It should be AIMessage, not ToolMessage
                    ai_response_text = last_message.content
                    print(f"--- [Thread: {thread_id}] Full AI response obtained post-tools for extraction (length: {len(ai_response_text)}) ---")

                    extracted_details_obj = await extract_details_from_response(ai_response_text)
                    if extracted_details_obj:
                        extracted_details_dict = extracted_details_obj.model_dump(by_alias=False)
                        print("--- [Thread: {thread_id}] Extracted Details (Server-Side Log) ---")
                        print(json.dumps(extracted_details_dict, indent=2))
                        print("-------------------------------------------------------------")
                    else:
                        print(f"--- [Thread: {thread_id}] Detail extraction failed or returned no data. ---")
                else:
                     # If the last message is a ToolMessage, something might be off in the graph flow for invoke
                     print(f"--- [Thread: {thread_id}] Last message in temp invoke was not AIMessage ({type(last_message)}). Cannot extract. ---")
            else:
                 print(f"--- [Thread: {thread_id}] No messages or invalid format in temp invoke result. Cannot extract. ---")
        else:
            print(f"--- [Thread: {thread_id}] Failed to get final state from temp invoke. Cannot extract. ---")

    except Exception as e:
        print(f"!!! [Thread: {thread_id}] ERROR during invoke/extraction phase: {e} !!!")
        traceback.print_exc()

    # --- Streaming Step (using original config) ---
    try:
        print(f"--- [Thread: {thread_id}] Initiating streaming response for client using original config ---")
        # This run will execute the graph again, including tools if needed by the LLM for this run.
        return StreamingResponse(
            stream_response_generator(app_runnable, input_state, config),
            media_type="text/plain"
        )
    except Exception as e:
        print(f"!!! [Thread: {thread_id}] Error setting up stream: {e} !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error during streaming setup: {e}")


# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "Welcome to the LangChain Chatbot API (NL2SQL, Streaming, Extraction Enabled)!"}

# --- To Run ---
# Ensure .env file has OPENAI_API_KEY.
# Ensure requirements installed: uvicorn, fastapi, langchain, langchain-openai, python-dotenv, langgraph, pydantic, sqlalchemy (and mysql driver if needed)
# Replace mock DB setup with your actual database connection logic.
# Run in terminal: uvicorn your_filename:app --reload --port 8000
