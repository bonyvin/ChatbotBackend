import datetime
import os
import uuid
import json
import traceback # For detailed error logging
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Sequence, AsyncIterator

import operator
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse # Import StreamingResponse
from fastapi_mail import FastMail, MessageSchema, MessageType
from pydantic import AliasChoices, BaseModel, EmailStr, Field
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
    AIMessageChunk, # Import AIMessageChunk for type checking stream
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from llm_templates import template_Promotion_without_date
# from promoUtils import template_Promotion_without_date
from langchain_core.tools import tool
from send_email import conf 

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

#All Classes
# --- Tool Definition / Pydantic Models ---
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

class UserIntent(BaseModel):
    intent: str | None = Field(None)

# --- API Request Model Class---
class ChatRequestPromotion(BaseModel):
    message: str
    thread_id: str | None = None

# --- Graph State Definition ---
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add] # Conversation history
    user_query: str | None # Most recent user query
    generated_sql: Optional[str] # SQL generated by the LLM
    sql_results: Optional[List[Dict] | str] # Results from DB execution or error string
    llm_response_content: Optional[str] # Content from the main LLM response node (to be streamed)
    extracted_details: Optional[ExtractedPromotionDetails] # Details extracted from LLM response

#All Models
chat_model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, streaming=True, temperature=0.7)

# LLM for SQL generation (non-streaming, deterministic)
sql_generator_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0)

# LLM for detail extraction (non-streaming)
extractor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0)
extractor_llm_structured = extractor_llm.with_structured_output(
    ExtractedPromotionDetails,
    method="function_calling", # Or "json_mode"
    include_raw=False
)

# New LLM chain definitions for intent extraction in extract details
intent_extractor_llm = ChatOpenAI(
    model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0
)
intent_extractor_llm_structured = intent_extractor_llm.with_structured_output(
    UserIntent, method="function_calling", include_raw=False
)
llm_tool_test = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- Database Schema and Helper Functions ---
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

# Memoized table names to avoid repeated DB calls
_table_names_cache = None

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
    # No finally db.close() needed with context manager


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
    * **Multiple Conditions (Different Fields):** Use `AND` (e.g., `im.itemDepartment LIKE 'T-Shirt%' AND im.brand = 'Zara'`). When combining multiple `EXISTS` clauses with `OR`, group them using parentheses: `AND (EXISTS(...) OR EXISTS(...))`.
7.    **Output Format:** 
    - **Step 1: Draft**: First, think step by step (invisible to user) about how you will build the SELECT, FROM, JOIN, WHERE clauses.
    - **Step 2: Self-Review**: Before emitting the final answer, perform these checks:
         a. Balanced parentheses: count opening “(” vs closing “)”.  
         b. Presence of required clauses: does the query start with SELECT and include FROM?  
         c. No trailing logical operators: ensure no dangling `AND`/`OR` at end of WHERE.  
         d. Balanced quotes around string literals.  
         e. Joins only when needed per rules.  
    - **Step 3: Final Output**: Emit ONLY the SQL SELECT statement, without any commentary, explanations, backticks, or trailing semicolon.
8.  **Error Handling:** If after self-review you find syntax issues or impossibility to answer, output exactly:  
   `Query cannot be answered with the provided schema.`
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

    generated_sql = None # Default to None
    try:
        # Limit input text length (optional)
        max_len = 2000
        truncated_query = user_query[:max_len] + ("..." if len(user_query) > max_len else "")

        print(f"--- Generating SQL for: '{truncated_query}' ---")
        llm_output = await sql_generation_chain.ainvoke({"user_query": truncated_query})

        # Basic validation and cleaning
        cleaned_sql = llm_output.strip().strip(';').strip()

        # *** START FIX: Add missing closing parenthesis for EXISTS groups ***
        # Check if the query uses the specific pattern ' AND (EXISTS ...' and lacks the closing parenthesis
        # Use case-insensitive check for robustness
        pattern_start_exists = " AND (EXISTS ("
        if pattern_start_exists.lower() in cleaned_sql.lower() and not cleaned_sql.endswith(")"):
             # Check if the number of opening parentheses is one more than closing ones
             # This is a slightly more robust check than just checking the end
             if cleaned_sql.count("(") == cleaned_sql.count(")") + 1:
                 print("--- Post-processing: Adding missing closing parenthesis for EXISTS group. ---")
                 cleaned_sql += ")"
        # *** END FIX ***

        if not cleaned_sql:
            print("--- Generation failed: LLM returned empty string. ---")
        elif "cannot be answered" in cleaned_sql.lower():
            print(f"--- LLM indicated query cannot be answered: {cleaned_sql} ---")
        elif cleaned_sql.lower().startswith("select"):
            # Further check for potentially harmful keywords (basic protection)
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
        # Keep generated_sql as None

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
        return {"sql_results": sql_results}

    # --- Validation before execution ---
    # Basic validation: Check if it's a SELECT query
    if not query.strip().lower().startswith("select"):
        error_msg = "Error: Only SELECT queries are allowed."
        print(f"--- Validation Failed: {error_msg} ---")
        return {"sql_results": error_msg}

    # Table Name Validation
    words = query.upper().split()
    valid_tables = get_table_names() # Uses cached names after first call
    if not valid_tables:
        error_msg = "Error: Could not retrieve table names for validation."
        print(f"--- Validation Failed: {error_msg} ---")
        return {"sql_results": error_msg}

    table_found = False
    # Simple check: look for table names directly after FROM or JOIN
    try:
        from_indices = [i for i, word in enumerate(words) if word == "FROM"]
        join_indices = [i for i, word in enumerate(words) if word == "JOIN"]
        potential_table_indices = [idx + 1 for idx in from_indices + join_indices if idx + 1 < len(words)]

        for idx in potential_table_indices:
            potential_table = words[idx].strip("`;,()")
            # Check against known tables (case-insensitive)
            if potential_table.lower() in [t.lower() for t in valid_tables]:
                table_found = True
                break # Found at least one valid table reference
    except Exception as parse_err:
         print(f"Warning: Error during simple table name parsing: {parse_err}")
         # Decide if you want to proceed without validation or fail
         # For now, let's proceed with a warning if parsing fails but allow execution
         print(f"Warning: Proceeding with query execution despite table parsing issue: {query}")
         table_found = True # Allow execution if parsing fails, adjust if stricter validation needed

    if not table_found:
        error_msg = f"Error: Could not validate table name in query. Ensure it uses valid tables: {valid_tables}"
        print(f"--- Validation Failed: {error_msg} ---")
        return {"sql_results": error_msg}
    # --- End Validation ---

    # --- Execution ---
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

    # Prepare context for the LLM
    context = f"The user asked: '{user_query}'\n"
    # context += f"I generated the following SQL query to find relevant data: \n```sql\n{generated_sql}\n```\n"
    llm_instruction = "" # Initialize instruction

    if isinstance(sql_results, str): # Error occurred during execution
        context += f"However, there was an error executing the query: {sql_results}\n"
        # context += "Please inform the user about the error and ask if they want to rephrase their request." # Removed redundant instruction
        llm_instruction = "Explain the database query error to the user based on the context."
    elif isinstance(sql_results, list):
        if not sql_results:
            context += "The query executed successfully but returned no results.\n"
            # context += "Please inform the user that no items matched their criteria." # Removed redundant instruction
            llm_instruction = "Inform the user that the database query found no matching items based on the context."
        else:
            # Limit the number of results shown to the LLM to avoid large context
            max_results_to_show = 10
            # Ensure results are serializable before json.dumps
            serializable_results = []
            for row in sql_results[:max_results_to_show]:
                 try:
                     # Attempt to convert row to dict if needed (adjust based on actual row type)
                     serializable_row = dict(row) if not isinstance(row, dict) else row
                     # Further check/convert types within the row if necessary
                     serializable_results.append(serializable_row)
                 except TypeError:
                     print(f"Warning: Could not serialize row for LLM context: {row}")
                     serializable_results.append({"error": "Could not display row data"})


            results_summary = json.dumps(serializable_results, indent=2)
            if len(sql_results) > max_results_to_show:
                results_summary += f"\n... (and {len(sql_results) - max_results_to_show} more rows)"

            # context += f"The query returned the following results (showing up to {max_results_to_show}):\n```json\n{results_summary}\n```\n"
            # context += f"Please summarize these findings for the user in a clear and concise way. Mention that you looked this up in the database. Total items found: {len(sql_results)}."
            # llm_instruction = "Summarize the database query results for the user based on the context."
            context += f"The query returned the following results (showing up to {max_results_to_show}):\n```json\n{results_summary}\n```\n"
            context += f"Total items found: {len(sql_results)}."
            print("Stream Context Results Summary: ",results_summary,"SQL Results",sql_results,"Max Rsults: ",max_results_to_show,"Messages: ",messages)
    else:
         context += "There was an issue processing the SQL results (unexpected type).\n"
         # context += "Please apologize to the user and mention a technical difficulty." # Removed redundant instruction
         llm_instruction = "Apologize for a technical difficulty processing database results based on the context."


    # Define the prompt for summarizing SQL results
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    system_prompt_content = template_Promotion_without_date.replace("{current_date}", today)
    sql_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        MessagesPlaceholder(variable_name="history"), # Include history
        ("human", "Context:\n{context}") # Provide context and specific instruction
    ])

    # Chain for generating the summary (using the main chat_model for streaming)
    # Ensure chat_model is defined and is a streaming-capable LLM instance
    summary_chain = sql_summary_prompt | chat_model

    # Get history excluding the last human message which triggered this flow
    history = list(messages[:-1]) if messages else []

    # --- Use .astream() to collect content for state ---
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

    # Construct the full AIMessage for history
    response_message = AIMessage(content=final_content)

    # Return the fully accumulated content for the state
    # The client stream is handled by astream_events processing the same LLM call
    return {"llm_response_content": final_content, "messages": [response_message]}

async def generate_direct_response(state: 'GraphState') -> Dict:
    """
    Node to generate a response directly using the LLM when SQL is not applicable.
    Streams internally using .astream() to aggregate content for state update.
    `astream_events` will handle the client-facing stream.
    """
    print("--- Node: generate_direct_response ---")
    messages = state.get("messages", [])

    # Use the original system prompt and conversation history
    today = datetime.datetime.now().strftime("%d/%m/%Y")
    system_prompt_content = template_Promotion_without_date.replace("{current_date}", today)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{user_input}") # Get the last human message
    ])

    # Get the last human message for the current turn's input
    last_human_message = ""
    history = []
    if messages:
        # Separate history from the last message which acts as input
        history = list(messages[:-1])
        if isinstance(messages[-1], HumanMessage):
            last_human_message = messages[-1].content
        else:
            # Fallback if the last message isn't Human
            print("Warning: Last message in state was not HumanMessage.")
            # Attempt to find the last human message in history
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_human_message = msg.content
                    break
            if not last_human_message:
                 last_human_message = "[Could not find last user message]"


    # Chain for direct response (using the main chat_model for streaming)
    # Ensure chat_model is defined and is a streaming-capable LLM instance
    direct_response_chain = prompt_template | chat_model

    # --- Use .astream() to collect content for state ---
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

    # Construct the full AIMessage for history
    response_message = AIMessage(content=final_content)

    # Return the fully accumulated content for the state
    # The client stream is handled by astream_events processing the same LLM call
    return {"llm_response_content": final_content, "messages": [response_message]}

# async def extract_details(state: GraphState) -> Dict:
#     """
#     Node to extract structured details from the final LLM response content.
#     This happens server-side after the response is generated.
#     """
#     print("--- Node: extract_details ---")
#     response_text = state.get("llm_response_content")
#     extracted_data: Optional[ExtractedPromotionDetails] = None # Default

#     if not response_text:
#         print("--- No LLM response text found for extraction. Skipping. ---")
#         return {"extracted_details": None}

#     # Simple instruction for the extractor LLM
#     today = datetime.datetime.now().strftime("%d/%m/%Y")
#     extraction_prompt = ChatPromptTemplate.from_messages([
#         ("system", f"""
# You are an expert extraction system. Analyze the provided text, which is a response from a promotion creation assistant summarizing the current state of a promotion or answering a query. Extract the promotion details mentioned into the structured format defined by the 'ExtractedPromotionDetails' function/tool.

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

#         print(f"--- Extracting details from: '{truncated_text[:200]}...' ---")
#         llm_extracted_data = await extraction_chain.ainvoke({"text_to_extract": truncated_text})

#         if llm_extracted_data and isinstance(llm_extracted_data, ExtractedPromotionDetails):
#             print("--- Extraction Successful ---")
#             extracted_data = llm_extracted_data
#             print(f"Extracted Data: {extracted_data.model_dump_json(indent=2)}") # Log extracted data
#         elif llm_extracted_data:
#              print(f"--- Extraction returned unexpected type: {type(llm_extracted_data)} ---")
#         else:
#             print("--- Extraction returned no data. ---")
        

#     except Exception as e:
#         print(f"!!! ERROR during detail extraction: {e} !!!")
#         traceback.print_exc()
#         # Keep extracted_data as None

#     return {"extracted_details": extracted_data}

async def extract_details(state: GraphState) -> Dict:
    """
    Node to extract structured promotion details and user intent from the conversation so far.
    Returns a dict with:
      - "extracted_details": ExtractedPromotionDetails or None on failure
      - "user_intent": UserIntent or None on failure
    """
    print("--- Node: extract_details ---")
    extracted_data: Optional[ExtractedPromotionDetails] = None
    user_intent: Optional[UserIntent] = None

    # 1. Build chat_history string from state['messages']
    chat_history_lines: List[str] = []
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            chat_history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            chat_history_lines.append(f"Bot: {msg.content}")
        else:
            # Other message types (ToolMessage, SystemMessage) can be included or skipped.
            # For clarity, we skip or treat ToolMessage as Bot.
            if hasattr(msg, "content"):
                chat_history_lines.append(f"Bot: {msg.content}")
    chat_history_str = "\n".join(chat_history_lines)

    # 2. Determine missing fields based on any previously extracted details
    prev: Optional[ExtractedPromotionDetails] = state.get("extracted_details")
    missing_fields: List[str] = []

    # Required fields: promotion_type, hierarchy_type & hierarchy_value, brand, items, discount_type & discount_value, start_date, end_date, stores.
    if prev is None or not prev.promotion_type:
        missing_fields.append("Promotion Type")
    # Hierarchy: requires both type and value
    if prev is None or not prev.hierarchy_type:
        missing_fields.append("Hierarchy Level Type")
    if prev is None or not prev.hierarchy_value:
        missing_fields.append("Hierarchy Level Value")
    if prev is None or not prev.brand:
        missing_fields.append("Brand")
    # Items: require at least one item or a natural-language instruction that can be expanded to items
    if prev is None or not prev.items:
        missing_fields.append("Items")
    # Discount
    if prev is None or not prev.discount_type:
        missing_fields.append("Discount Type")
    if prev is None or not prev.discount_value:
        missing_fields.append("Discount Value")
    # Dates
    if prev is None or not prev.start_date:
        missing_fields.append("Start Date")
    if prev is None or not prev.end_date:
        missing_fields.append("End Date")
    # Stores
    if prev is None or not prev.stores:
        missing_fields.append("Stores")
    # Note: excluded_items and excluded_stores are optional; we do not require them here.

    missing_fields_str = ""
    if missing_fields:
        # Format as bullet list
        missing_fields_str = "\n".join(f"- {fld}" for fld in missing_fields)
    else:
        missing_fields_str = "None"

    # 3. Fill in the prompt template
    # Compute current date in dd/mm/yyyy
    current_date = datetime.date.today().strftime("%d/%m/%Y")

    # The template uses {current_date}, {{chat_history}}, {{missing_fields}}.
    # We do a simple replace. Be careful that the template indeed contains these placeholders exactly.
    try:
        prompt_filled = template_Promotion_without_date \
            .replace("{current_date}", current_date) \
            .replace("{{chat_history}}", chat_history_str) \
            .replace("{{missing_fields}}", missing_fields_str)
    except Exception as e:
        print(f"Error formatting template_Promotion_without_date: {e}")
        prompt_filled = template_Promotion_without_date
        # Fallback: we could still proceed, but chat_history/missing_fields won't be in prompt.

    # 4. Invoke the structured extraction LLM to parse promotion details
    try:
        # Build a ChatPromptTemplate without further variables since prompt is already filled
        detail_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_filled)
        ])
        # Use the structured chain
        llm_result = await (detail_prompt | extractor_llm_structured).ainvoke({})
        # llm_result should be an ExtractedPromotionDetails instance
        if isinstance(llm_result, ExtractedPromotionDetails):
            extracted_data = llm_result
        else:
            print("--- Warning: extract_details: LLM returned unexpected type:", type(llm_result))
    except Exception as e:
        print(f"!!! ERROR during promotion detail extraction: {e} !!!")
        traceback.print_exc()

    # 5. Extract user intent from last bot/user turn
    try:
        last_bot_message = None
        current_user_message = None
        current_bot_message = None

        # Collect only Human and AI messages
        relevant_msgs = [msg for msg in state.get("messages", []) if isinstance(msg, (HumanMessage, AIMessage))]

        # Separate Human and AI messages
        human_msgs = [msg for msg in relevant_msgs if isinstance(msg, HumanMessage)]
        ai_msgs = [msg for msg in relevant_msgs if isinstance(msg, AIMessage)]

        # Last bot message: 2nd-last AI message
        if len(ai_msgs) >= 2:
            last_bot_message = ai_msgs[-2].content

        # Current user message: last Human message
        if human_msgs:
            current_user_message = human_msgs[-1].content

        # Current bot message: last AI message
        if ai_msgs:
            current_bot_message = ai_msgs[-1].content

        # Fallback if not enough data
        last_bot_message = last_bot_message or ""
        current_user_message = current_user_message or ""
        current_bot_message = current_bot_message or ""

        # Build a complete context string
        full_context = f"""Previous Bot: {last_bot_message}
    Current User: {current_user_message}
    Current Bot: {current_bot_message}"""

        print("INTENT CLASSIFIER CONTEXT >>>")
        print(full_context)

        # Updated classification instruction
        # intent_system = (
        #     """
        #     "You are an intelligent intent classification engine.\n"
        #     "You will be given the previous bot message, the current user message, and the current bot response.\n"
        #     "Classify the user's **actual intention** at this point in the conversation.\n"
        #     "Available intent types: [\"Promotion Creation\", \"Detail Filling\", \"Submission\", \"Other\"]\n"
        #     "Return a valid JSON object: {{ \"intent\": string }} where the intent is one of the listed values.\n"
        #     "Use context wisely — for example:\n"
        #     "- If the user initiates a new promotion (e.g., 'I want to create a promotion'), the intent is 'Promotion Creation'.\n"
        #     "- If the user is just providing fields (like brand, type), it's 'Detail Filling'.\n"
        #     "- If the bot asks for submission and the user says 'yes', it's 'Submission'."
        #     Examples:
        #     ---
        #     Previous Bot:
        #     Current User: I want to create a promotion
        #     Current Bot: Got it. Please fill the following:
        #     → "Promotion Creation"
        #     ---
        #     Previous Bot: Would you like to submit this?
        #     Current User: Yes
        #     Current Bot: Promotion submitted successfully.
        #     → "Submission"
        #     """
        # )
        # intent_system = (
        #     "You are a highly reliable intent classification engine for a promotions chatbot.\n"
        #     "You will be given three lines:\n"
        #     "  • Previous Bot: what the bot said before the user's last utterance\n"
        #     "  • Current User: the user's most recent message\n"
        #     "  • Current Bot: the bot's immediate reply to that message\n\n"
        #     "**Rules (in order of precedence):**\n"
        #     "1. **If** the Current User line contains a direct request to start or create a promotion (e.g. “I want to create a promotion”, “Let's make a promo”), **output** intent=\"Promotion Creation\" and ignore everything else.\n"
        #     "2. **Else if** the Previous Bot line is something like 'Would you like to submit this information?  ', and the Current User line is simply 'Yes'/'No'/'Sure'/'Okay'/'Any other positive response confirming submission', and the Current Bot response contains 'Promotion created successfully. Thank you for choosing us.' in response, **output** intent=\"Submission\".\n"
        #     # "2. **Else if** the Current User line is simply “Yes”/“No”/“Sure”/“Okay” in answer to a submission-prompt (“Would you like to submit?”), **output** intent=\"Submission\".\n"
        #     "3. **Else if** the Current User line provides or updates one or more promotion fields (brand, items, discount, dates, etc.), **output** intent=\"Detail Filling\".\n"
        #     "4. **Otherwise**, **output** intent=\"Other\".\n\n"
        #     "**Return exactly** a JSON object: {{\"intent\": \"<one of the four>\"}}\n"
        # )
        intent_system = (
            "You are a highly reliable intent classification engine for a promotions chatbot.\n"
            "**Your job**: Given three lines of context—“Previous Bot,” “Current User,” and “Current Bot”—decide which of these four intents the user is expressing _right now_:\n\n"
            "  1. **Promotion Creation**  \n"
            "     The user is explicitly initiating a brand-new promotion.  \n"
            "     *Trigger phrases* include “I want to create a promotion,” “Let's make a promo,” “Start a new discount,” etc.\n\n"
            "  2. **Submission**  \n"
            # "     The user is confirming final submission _and_ the bot has just acknowledged success.  \n"
            # "     **Strict Rule:** Previous Bot must have asked “Would you like to submit this information?”,\n"
            # "     Current User must reply with a confirmation (“Yes”/“Sure”/“Okay” etc.),\n"
            # "     **and** Current Bot must be a success message containing “Promotion created successfully. Thank you for choosing us.”\n\n"
            "     - The user has just confirmed submission **AND** the bot has immediately replied with a success message.\n"
            "     - **Strict rule**: \n"
            "         • Current User must be a confirmation (e.g. “Yes”, “Sure”, “Okay”).  \n"
            "         • Current Bot must contain exactly “Promotion created successfully. Thank you for choosing us.”  \n\n"
            "  3. **Detail Filling**  \n"
            "     The user is providing or updating one or more promotion fields (e.g. “Promotion Type: Simple,” “Brand: H&M,” “Discount: 20%,” “Start: 01/07/2025”), or answering intermediate questions about stores/items.\n\n"
            "  4. **Other**  \n"
            "     Anything that doesn't fit the above: greetings, small talk, help requests, mistakes, etc.\n\n"
            "**Rules (in order)**:\n"
            "  1. If the **Current User** line matches a _new promotion request_ pattern, return intent=\"Promotion Creation\".  Stop.\n"
            "  2. Else if the Submission strict rule applies (Previous Bot asked for submission, Current User confirms, Current Bot gives a success message), return intent=\"Submission\".\n"
            "  3. Else if **Current User** mentions any of the required fields—Type, Hierarchy, Brand, Items, Discount, Dates, or Stores—return intent=\"Detail Filling\".  Stop.\n"
            "  4. Otherwise, return intent=\"Other\".\n\n"
            "**Example Inputs → Outputs**:\n"
            "```\n"
            "Previous Bot: (nothing yet)\n"
            "Current User: I want to create a promotion\n"
            "Current Bot: To create a promotion, I need these details…\n"
            "→ {{\"intent\": \"Promotion Creation\"}}\n\n"
            "Previous Bot: Would you like to submit this information?\n"
            "Current User: Yes\n"
            "Current Bot: Promotion created successfully. Thank you for choosing us.\n"
            "→ {{\"intent\": \"Submission\"}}\n\n"
            "Previous Bot: Promotion Type: Simple\n"
            "Current User: Brand: Zara and H&M\n"
            "Current Bot: Recorded Brand: Zara, H&M. Missing fields: Items, Discount, Dates, Stores.\n"
            "→ {{\"intent\": \"Detail Filling\"}}\n"
            "```\n\n"
            "**Return exactly** a JSON object with one key: `intent`, whose value is one of:\n"
            "  [\"Promotion Creation\", \"Detail Filling\", \"Submission\", \"Other\"]\n"
        )
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", intent_system),
            ("human", "{context}")
        ])

        llm_intent = await (intent_prompt | intent_extractor_llm_structured).ainvoke({
            "context": full_context
        })

        print("Predicted user intent:", llm_intent)

        if isinstance(llm_intent, UserIntent):
            user_intent = llm_intent

    except Exception as e:
        print(f"!!! ERROR during intent extraction: {e} !!!")
        traceback.print_exc()

        # Return results
        return {
            "extracted_details": extracted_data,
            "user_intent": user_intent
        }

# --- Conditional Edges ---
def should_execute_sql(state: GraphState) -> str:
    """Determines the next step based on whether SQL was generated."""
    print("--- Condition: should_execute_sql ---")
    if state.get("generated_sql"):
        print("Decision: SQL generated, routing to execute_sql")
        return "execute_sql"
    else:
        print("Decision: No SQL generated, routing to generate_direct_response")
        return "generate_direct_response"


# --- Build the Workflow ---
workflow = StateGraph(GraphState)
# Add nodes
workflow.add_node("preprocess_input", preprocess_input)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_response_from_sql", generate_response_from_sql)
workflow.add_node("generate_direct_response", generate_direct_response)
workflow.add_node("extract_details", extract_details)
# Define edges
workflow.add_edge(START, "preprocess_input") # Start with preprocessing
workflow.add_edge("preprocess_input", "generate_sql")
# Conditional routing after trying to generate SQL
workflow.add_conditional_edges(
    "generate_sql",
    should_execute_sql,
    {
        "execute_sql": "execute_sql",
        "generate_direct_response": "generate_direct_response",
    },
)
# Path if SQL was executed
workflow.add_edge("execute_sql", "generate_response_from_sql")
workflow.add_edge("generate_response_from_sql", "extract_details") # Extract after generating response
# Path if SQL was not generated
workflow.add_edge("generate_direct_response", "extract_details") # Extract after generating response
# Final step after extraction
workflow.add_edge("extract_details", END)
# --- Memory and Compilation ---
memory = MemorySaver()
app_runnable_promotion = workflow.compile(checkpointer=memory)

# --- API Request Model ---
async def stream_response_generator_promotion(
    graph_stream: AsyncIterator[Dict]
) -> AsyncIterator[str]:
    """
    Asynchronously streams LLM response chunks from specified nodes
    to the client as soon as they arrive.
    """
    print("--- STREAM GENERATOR (for client) STARTED ---")
    full_response_for_log = ""
    try:
        async for event in graph_stream:
            if event["event"] == "on_chat_model_stream":
                metadata = event.get("metadata", {})
                node_name = metadata.get("langgraph_node")
                if node_name in {"generate_direct_response", "generate_response_from_sql"}:
                    chunk = event["data"].get("chunk")
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        content = chunk.content
                        full_response_for_log += content
                        yield content
    except Exception as e:
        print(f"!!! ERROR in stream_response_generator_promotion: {e} !!!")
        # If nothing has been sent yet, at least send an error
        yield f"\n\nStream error: {e}"
    finally:
        print(f"--- STREAM GENERATOR (for client) FINISHED ---")
        # For debugging, you could log full_response_for_log here

# --- FastAPI Application ---
app = FastAPI(
    title="LangGraph Chatbot API",
    description="API endpoint for a LangChain chatbot using LangGraph, detail extraction, SQL generation, and streaming.",
)
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

tools = [add]

llm_with_tools = llm_tool_test.bind_tools(tools)
# --- FastAPI Endpoint ---
# @app.post("/chat/")
# async def chat_endpoint(request: ChatRequestPromotion):
#     """
#     Receives user message, invokes the LangGraph app, and streams the
#     appropriate LLM response back to the client. Server-side processing
#     (SQL, extraction) happens within the graph nodes.
#     """
#     user_message = request.message
#     thread_id = request.thread_id or str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}

#     print(f"\n--- [Thread: {thread_id}] Received message: '{user_message}' ---")

#     # Prepare input for the graph - the input is the list of messages
#     input_message = HumanMessage(content=user_message)
#     input_state = {"messages": [input_message]} # Pass the new message in the list

#     try:
#         # Use astream_events to get the stream of events from the graph execution
#         graph_stream = app_runnable_promotion.astream_events(input_state, config, version="v2")

#         # Return a StreamingResponse that iterates over the generator
#         # Prepare query
#         query = "What is 3 * 12? Also, what is 11 + 49?"
#         messages = [HumanMessage(content=query)]
#         # Invoke LLM
#         ai_msg = llm_with_tools.invoke(messages)
#         # Print outputs
#         print("=== Tool Calls ===")
#         print(ai_msg.tool_calls)
#         print("\n=== Model Response ===")
#         messages.append(ai_msg)
#         for tool_call in ai_msg.tool_calls:
#             if tool_call["name"].lower() == "add":
#                 tool_msg = add.invoke(tool_call)
#                 messages.append(tool_msg)
#             else:
#                 raise ValueError(f"Unsupported tool: {tool_call['name']}")
                
#         print("Final response: ",messages)
#         return StreamingResponse(
#             stream_response_generator_promotion(graph_stream), # Pass the graph stream to the generator
#             media_type="text/event-stream" # Use text/event-stream for Server-Sent Events
#             # media_type="text/plain" # Or application/jsonl if streaming JSON chunks
#         )

#     except Exception as e:
#         print(f"!!! ERROR invoking graph for thread {thread_id}: {e} !!!")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")


# @app.post("/filenew")
# async def send_file_new(
#     file: UploadFile = File(...),
#     email: EmailStr = Form(...),
#     body: str = Form(...)
# ) -> JSONResponse:
#     try:
#         body_dict: Dict[str, Any] = json.loads(body)
#     except json.JSONDecodeError:
#         return JSONResponse(status_code=400, content={"message": "Invalid JSON in 'body' field."})

#     message = MessageSchema(
#         subject="Fastapi-Mail module",
#         recipients=[email],
#         template_body=body_dict,
#         subtype=MessageType.html,
#         attachments=[file]  # ✅ Pass UploadFile directly
#     )

#     fm = FastMail(conf)
#     await fm.send_message(message, template_name="email-document.html")
#     return JSONResponse(status_code=200, content={"message": "Email has been sent."})


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

