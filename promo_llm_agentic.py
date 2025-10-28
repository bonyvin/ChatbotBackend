import datetime
import os
from platform import node
import re
import uuid
import json
import traceback # For detailed error logging
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Sequence, AsyncIterator,Set
import operator
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse # Import StreamingResponse
from fastapi_mail import FastMail, MessageSchema, MessageType
from pydantic import AliasChoices, BaseModel, EmailStr, Field
from dotenv import load_dotenv
from sqlalchemy import text
from requests import Session # Assuming Session is used within get_db
from fastapi.middleware.cors import CORSMiddleware
from fastapi import WebSocket, WebSocketDisconnect


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
from llm_templates import template_Promotion_without_date,promotion_extraction_prompt
# from promoUtils import template_Promotion_without_date
from langchain_core.tools import tool
from send_email import conf 
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError

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
# # class ExtractedPromotionDetails(BaseModel):
#     # Use validation_alias and AliasChoices for more flexible field naming in LLM output
#     promotion_type: str | None = Field(None, validation_alias=AliasChoices('Promotion Type', 'promotion_type'))
#     # hierarchy_type: str | None = Field(None, validation_alias=AliasChoices('Hierarchy Type', 'hierarchy_type'))
#     hierarchy_type: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Hierarchy Type', 'hierarchy_type'))
#     # hierarchy_value: str | None = Field(None, validation_alias=AliasChoices('Hierarchy Value', 'hierarchy_value'))
#     # brand: str | None = Field(None, validation_alias=AliasChoices('Brand', 'brand'))
#     hierarchy_value: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Hierarchy Value', 'hierarchy_value'))
#     brand: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Brand', 'brand'))
#     items: List[str] | None = Field(default=None, validation_alias=AliasChoices('Items', 'items'))
#     excluded_items: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Items', 'excluded_items'))
#     discount_type: str | None = Field(None, validation_alias=AliasChoices('Discount Type', 'discount_type'))
#     discount_value: str | None = Field(None, validation_alias=AliasChoices('Discount Value', 'discount_value')) # Keep as str to handle % vs fixed ambiguity
#     start_date: str | None = Field(None, validation_alias=AliasChoices('Start Date', 'start_date'))
#     end_date: str | None = Field(None, validation_alias=AliasChoices('End Date', 'end_date'))
#     stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Stores', 'stores'))
#     excluded_stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Stores', 'excluded_stores'))
#     email: str | None = Field(None, validation_alias=AliasChoices('Email', 'email'))

#     class Config:
#         populate_by_name = True # Allows using field names as well as aliases
#         extra = 'ignore' # Ignore extra fields the LLM might hallucinate

class ExtractedPromotionDetails(BaseModel):
    promotion_type: str | None = Field(None, validation_alias=AliasChoices('Promotion Type', 'promotion_type'))
    hierarchy_type: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Hierarchy Type', 'hierarchy_type'))
    hierarchy_value: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Hierarchy Value', 'hierarchy_value'))
    brand: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Brand', 'brand'))
    
    # Changed: Allow None or List
    items: List[str] | None = Field(default=None, validation_alias=AliasChoices('Items', 'items'))
    excluded_items: List[str] | None = Field(default=None, validation_alias=AliasChoices('Excluded Items', 'excluded_items'))
    
    discount_type: str | None = Field(None, validation_alias=AliasChoices('Discount Type', 'discount_type'))
    discount_value: str | None = Field(None, validation_alias=AliasChoices('Discount Value', 'discount_value'))
    start_date: str | None = Field(None, validation_alias=AliasChoices('Start Date', 'start_date'))
    end_date: str | None = Field(None, validation_alias=AliasChoices('End Date', 'end_date'))
    
    # Changed: Allow None or List
    stores: List[str] | None = Field(default=None, validation_alias=AliasChoices('Stores', 'stores'))
    excluded_stores: List[str] | None = Field(default=None, validation_alias=AliasChoices('Excluded Stores', 'excluded_stores'))
    
    email: str | None = Field(None, validation_alias=AliasChoices('Email', 'email'))

    class Config:
        populate_by_name = True
        extra = 'ignore'

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
    user_intent: Optional[UserIntent] # Intent extracted from user input - for persisting state if needed

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


_table_names_cache = None
_table_columns_cache = {}

def is_valid_identifier(name: str) -> bool:
    """Validate that a string is a safe SQL identifier (alphanumeric + underscore only)."""
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def get_table_names() -> List[str]:
    """Fetch all table names from the database, with caching."""
    global _table_names_cache
    if _table_names_cache is not None:
        return _table_names_cache

    db = next(get_db())
    try:
        inspector = inspect(db.get_bind())
        _table_names_cache = inspector.get_table_names()
        print(f"Fetched and cached table names: {_table_names_cache}")
        return _table_names_cache
    except SQLAlchemyError as e:
        print(f"Error fetching table names: {e}")
        return []
    finally:
        db.close()


def get_valid_columns(db: Session, table_name: str) -> Set[str]:
    """Get valid columns for a table with caching."""
    global _table_columns_cache
    
    if table_name in _table_columns_cache:
        return _table_columns_cache[table_name]
    
    inspector = inspect(db.get_bind())
    columns = {col['name'] for col in inspector.get_columns(table_name)}
    _table_columns_cache[table_name] = columns
    return columns


def get_unique_column_values(table_name: str, columns: List[str]) -> Dict[str, List[Any]]:
    """Fetch unique values for each column in the given table."""
    unique_values = {}
    db = next(get_db())
    
    try:
        # SECURITY LAYER 1: Validate identifier format (prevent special characters)
        if not is_valid_identifier(table_name):
            print(f"Invalid table name format: {table_name}")
            return {}
        
        for col in columns:
            if not is_valid_identifier(col):
                print(f"Invalid column name format: {col}")
                unique_values[col] = []
                continue
        
        # SECURITY LAYER 2: Validate table exists in database
        inspector = inspect(db.get_bind())
        valid_tables = inspector.get_table_names()
        
        if table_name not in valid_tables:
            print(f"Table does not exist: {table_name}")
            return {}
        
        # SECURITY LAYER 3: Validate columns exist in the table
        valid_columns = get_valid_columns(db, table_name)
        
        for col in columns:
            if col not in valid_columns:
                print(f"Column '{col}' does not exist in table '{table_name}'")
                unique_values[col] = []
                continue
            
            try:
                # After triple validation, use quoted identifiers
                # Get the dialect-specific identifier quote character
                dialect = db.get_bind().dialect
                quote_char = dialect.identifier_preparer.initial_quote
                
                # Manually quote identifiers (safe after validation)
                quoted_table = f"{quote_char}{table_name}{quote_char}"
                quoted_col = f"{quote_char}{col}{quote_char}"
                
                # Build query with validated and quoted identifiers
                query = text(f"SELECT DISTINCT {quoted_col} FROM {quoted_table}")
                result = db.execute(query).fetchall()
                
                unique_values[col] = [row[0] for row in result if row[0] is not None]
                
            except SQLAlchemyError as col_err:
                print(f"Error fetching column '{col}': {col_err}")
                unique_values[col] = []
                
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
    finally:
        db.close()
    
    return unique_values

# --- Graph Nodes ---
async def preprocess_input(state: GraphState) -> Dict:
    """Adds the latest user message to the state and history."""
    print("--- Node: preprocess_input ---")
    last_message = state['messages'][-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("Last message is not a HumanMessage")
    user_query = last_message.content
    print(f"User Query: {user_query}")
    # print("Unique Item Values :", get_unique_column_values("itemmaster", ["brand", "itemDepartment", "itemClass", "itemSubClass"]))
    return {"user_query": user_query}


async def generate_sql(state: GraphState) -> Dict:
    """
    Node to attempt generating SQL from the user query.
    Uses the dedicated sql_generator_llm with improved unique values integration.
    """
    print("--- Node: generate_sql ---")
    user_query = state.get("user_query")
    if not user_query:
        print("--- No user query found. Skipping SQL generation. ---")
        return {"generated_sql": None}
    
    # Fetch unique values with caching
    unique_item_values = get_unique_column_values(
        "itemmaster", 
        ["brand", "itemDepartment", "itemClass", "itemSubClass"]
    )
    unique_item_diffs = get_unique_column_values(
        "itemdiffs", 
        ["diffId", "diffType"]
    )
    unique_store_values = get_unique_column_values(
        "storedetails", 
        ["storeName", "city", "state", "zipCode"]
    )
    
    # Format unique values more clearly for LLM
    def format_unique_values(data: Dict[str, List[Any]], indent: str = "  ") -> str:
        """Format unique values in a clean, readable format."""
        if not data:
            return f"{indent}(No values available)"
        
        lines = []
        for key, values in data.items():
            if not values:
                lines.append(f"{indent}- {key}: (empty)")
            elif len(values) <= 20:  # Show all if reasonable
                lines.append(f"{indent}- {key}: {', '.join(map(str, values))}")
            else:  # Show sample for large lists
                sample = ', '.join(map(str, values[:15]))
                lines.append(f"{indent}- {key}: {sample}... ({len(values)} total values)")
        return '\n'.join(lines)
    
    formatted_item_values = format_unique_values(unique_item_values)
    formatted_diff_values = format_unique_values(unique_item_diffs)
    formatted_store_values = format_unique_values(unique_store_values)
    
    # Extract specific lists for inline referencing
    available_brands = unique_item_values.get("brand", [])
    available_departments = unique_item_values.get("itemDepartment", [])
    available_classes = unique_item_values.get("itemClass", [])
    available_subclasses = unique_item_values.get("itemSubClass", [])
    available_diff_ids = unique_item_diffs.get("diffId", [])
    available_diff_types = unique_item_diffs.get("diffType", [])
    
    sql_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert SQL generator. Your task is to convert natural language questions into valid SQL SELECT statements based on the database schema and available data values.

DATABASE SCHEMA:
{TABLE_SCHEMA}

AVAILABLE DATA VALUES (Use these for exact matching):

itemmaster table:
{formatted_item_values}

itemdiffs table (for attributes like color, size, material):
{formatted_diff_values}

storedetails table:
{formatted_store_values}

CRITICAL INSTRUCTIONS:

1. **Value Matching & Fuzzy Logic:**
   - When the user mentions a brand, department, class, or attribute, find the CLOSEST MATCH from the available values above
   - Examples of fuzzy matching:
     * User says "zara" → Match to exact value "Zara" from available brands
     * User says "tshirt" or "t shirt" → Match to "T-Shirt" from available departments
     * User says "red" or "Red color" → Match to "Red" from available diffId values
     * User says "nike shoes" → Extract brand="Nike" and potentially itemClass matching "Shoes"
   - ALWAYS use the exact casing and spelling from the available values in your SQL
   - If no close match exists, use the user's value but mention this in your reasoning

2. **Attribute Filtering (itemdiffs):**
   - Attributes like color, size, material are stored in the itemdiffs table
   - Available attribute values: {', '.join(available_diff_ids[:30])}{"..." if len(available_diff_ids) > 30 else ""}
   - Available attribute types: {', '.join(available_diff_types)}
   - To filter by an attribute, use EXISTS with OR across diffType1, diffType2, diffType3
   - Example pattern for filtering by 'Red':
     ```sql
     AND (
       EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Red')
       OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Red')
       OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Red')
     )
     ```

3. **Core Query Structure:**
   - **SELECT:** Start with `SELECT DISTINCT im.itemId` (add other columns if requested)
   - **FROM:** Always `FROM itemmaster im`
   - **JOIN:** Add `JOIN itemsupplier isup ON im.itemId = isup.itemId` ONLY if filtering/selecting supplierCost
   - **WHERE:** Build conditions based on user requirements

4. **WHERE Clause Best Practices:**
   - Brand: `im.brand = 'ExactBrandName'` (use exact match from available values)
   - Department: `im.itemDepartment LIKE 'DepartmentName%'` (use LIKE for partial matches)
   - Class: `im.itemClass LIKE 'ClassName%'`
   - SubClass: `im.itemSubClass LIKE 'SubClassName%'`
   - Multiple values (same field): Use `IN` clause: `im.brand IN ('Zara', 'Nike', 'Adidas')`
   - Multiple attributes: Combine with `AND` between different conditions
   - Supplier cost: `isup.supplierCost < value` or `isup.supplierCost BETWEEN min AND max`

5. **Ignore Non-Selection Information:**
   - Completely ignore: discounts, promotions, dates, validity periods, action verbs (Create, Offer, Update)
   - Focus ONLY on: what items to select and their filtering criteria
   - Output must ONLY be a SELECT query

6. **Syntax Validation (Self-Check before output):**
   - ✓ Balanced parentheses: count opening "(" equals closing ")"
   - ✓ Balanced quotes: all string literals properly quoted
   - ✓ No trailing AND/OR operators
   - ✓ Joins only when needed
   - ✓ No semicolon at the end
   - ✓ Query starts with SELECT

7. **Output Format:**
   - Think through your query construction step-by-step (invisible reasoning)
   - Perform self-validation checklist
   - Output ONLY the final SQL statement with no backticks, comments, or explanations
   - If query is impossible with schema, output exactly: `Query cannot be answered with the provided schema.`

EXAMPLES:

Example 1 - Simple brand filter:
User: "Show me all Nike products"
Reasoning: User wants brand="Nike". Check available brands → "Nike" exists
SQL: SELECT DISTINCT im.itemId FROM itemmaster im WHERE im.brand = 'Nike'

Example 2 - Attribute filter (color):
User: "All red items"
Reasoning: "red" is an attribute. Check diffId values → "Red" exists. Use EXISTS pattern.
SQL: SELECT DISTINCT im.itemId FROM itemmaster im WHERE (EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Red') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Red'))

Example 3 - Multiple brands:
User: "Items from Zara, Nike, or Adidas"
Reasoning: Multiple brands. Check available → all exist. Use IN clause.
SQL: SELECT DISTINCT im.itemId FROM itemmaster im WHERE im.brand IN ('Zara', 'Nike', 'Adidas')

Example 4 - Brand + Department + Cost:
User: "Zara t-shirts under $30"
Reasoning: brand=Zara, department=T-Shirt (fuzzy match "t-shirts"), cost filter. Need JOIN for cost.
SQL: SELECT DISTINCT im.itemId, isup.supplierCost FROM itemmaster im JOIN itemsupplier isup ON im.itemId = isup.itemId WHERE im.brand = 'Zara' AND im.itemDepartment LIKE 'T-Shirt%' AND isup.supplierCost < 30

Example 5 - Complex with attribute + multiple conditions:
User: "Show blue Nike shoes from the sportswear department"
Reasoning: brand=Nike, attribute=Blue, itemClass or department contains shoes/sportswear
SQL: SELECT DISTINCT im.itemId FROM itemmaster im WHERE im.brand = 'Nike' AND im.itemDepartment LIKE 'Sportswear%' AND (EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Blue') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Blue') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Blue'))

Example 6 - Ignore promotional info:
User: "Create a 40% discount on all yellow FashionX t-shirts valid until May 2025"
Reasoning: Ignore "discount", "40%", "valid until May". Extract: brand=FashionX, department=T-Shirt, attribute=Yellow
SQL: SELECT DISTINCT im.itemId FROM itemmaster im WHERE im.brand = 'FashionX' AND im.itemDepartment LIKE 'T-Shirt%' AND (EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType1 AND idf.diffId = 'Yellow') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType2 AND idf.diffId = 'Yellow') OR EXISTS (SELECT 1 FROM itemdiffs idf WHERE idf.id = im.diffType3 AND idf.diffId = 'Yellow'))

Remember: Use exact values from the available data. Perform fuzzy matching mentally but output exact database values.
"""),
        ("human", "Convert this question to a SQL SELECT query:\n\n{user_query}")
    ])

    # Create the generation chain
    sql_generation_chain = sql_generation_prompt | sql_generator_llm | StrOutputParser()

    generated_sql = None
    try:
        # Limit input text length
        max_len = 2000
        truncated_query = user_query[:max_len] + ("..." if len(user_query) > max_len else "")

        print(f"--- Generating SQL for: '{truncated_query}' ---")
        llm_output = await sql_generation_chain.ainvoke({"user_query": truncated_query})

        # Clean output
        cleaned_sql = llm_output.strip().strip(';').strip()
        
        # Remove markdown code blocks if present
        if cleaned_sql.startswith("```"):
            lines = cleaned_sql.split('\n')
            cleaned_sql = '\n'.join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            cleaned_sql = cleaned_sql.strip()

        # Validate parentheses balance
        if cleaned_sql.count("(") != cleaned_sql.count(")"):
            print(f"--- WARNING: Unbalanced parentheses detected. Opening: {cleaned_sql.count('(')}, Closing: {cleaned_sql.count(')')} ---")
            # Attempt to fix if only one closing paren is missing
            if cleaned_sql.count("(") == cleaned_sql.count(")") + 1:
                print("--- Auto-fixing: Adding missing closing parenthesis ---")
                cleaned_sql += ")"

        if not cleaned_sql:
            print("--- Generation failed: LLM returned empty string. ---")
        elif "cannot be answered" in cleaned_sql.lower():
            print(f"--- LLM indicated query cannot be answered: {cleaned_sql} ---")
        elif cleaned_sql.lower().startswith("select"):
            # Security check for harmful keywords
            harmful_keywords = ['update ', 'insert ', 'delete ', 'drop ', 'alter ', 'truncate ', 'exec ', 'execute ']
            if any(keyword in cleaned_sql.lower() for keyword in harmful_keywords):
                print(f"--- Generation failed: Potentially harmful keyword detected: {cleaned_sql} ---")
            else:
                print(f"--- SQL Generation Successful: {cleaned_sql} ---")
                generated_sql = cleaned_sql
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

from fastapi.encoders import jsonable_encoder

def _to_plain(obj):
    """Return a plain-serializable Python object from dict / pydantic v1/v2 / simple values."""
    if obj is None:
        return None
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    # fallback to jsonable_encoder
    try:
        return jsonable_encoder(obj)
    except Exception:
        return str(obj)

intent_system = (
    "You are a highly reliable intent classification engine for a promotions chatbot.\n"
    "**Your job**: Given three lines of context—'Previous Bot,' 'Current User,' and 'Current Bot'—decide which of these intents the user is expressing *right now*:\n\n"
    
    "  1. **Promotion Creation**  \n"
    "     The user is explicitly initiating a brand-new promotion.  \n"
    "     *Trigger phrases* include 'I want to create a promotion,' 'Let's make a promo,' 'Start a new discount,' etc.\n\n"

    "  2. **Submission**  \n"
    "     CRITICAL: Submission occurs when:\n"
    "       - Previous Bot: Contains 'Would you like to submit this information?'\n"
    "       - Current User: User confirms (e.g., 'Yes', 'Sure', 'Confirm')\n"
    "       - Current Bot: Bot replies with success message ('successfully created', 'Thank you')\n\n"
    
    "     IMPORTANT: Even if Current Bot ALSO asks about email in the same message, this is STILL Submission (not Email Fetching).\n"
    "     Email Fetching only occurs AFTER the user responds to the email question.\n\n"
    
    "     ✅ CORRECT Example (IS Submission, even with email question):\n"
    "       Previous Bot: Would you like to submit this information?\n"
    "       Current User: Yes\n"
    "       Current Bot: Promotion created successfully. Would you like the document sent to your email?\n"
    "       → This IS 'Submission' (user just confirmed submission, hasn't answered email yet)\n\n"
    
    "  3. **Detail Filling**  \n"
    "     The user is providing or updating promotion fields, or the bot is asking for missing information.\n"
    "     If the bot is asking 'Would you like to submit?', classify as Detail Filling.\n\n"
    
    "  4. **Email Fetching**  \n"
    "     CRITICAL: Email Fetching ONLY occurs when the user is RESPONDING to an email question:\n"
    "       - Previous Bot: Must ask 'Would you like the document sent to your email?' (AND contain success message)\n"
    "       - Current User: User responds to the email question (e.g., 'Yes', 'user@example.com', 'No thanks')\n"
    "       - Current Bot: Bot processes the email response (e.g., 'Please provide your email', 'Email sent', 'Understood')\n\n"
    
    "     KEY RULE: If Current Bot is the FIRST message asking about email, this is NOT Email Fetching yet.\n"
    "     Email Fetching only happens when the user has ALREADY been asked and is now responding.\n\n"
    
    "     ❌ WRONG Example (NOT Email Fetching):\n"
    "       Previous Bot: Would you like to submit?\n"
    "       Current User: Yes\n"
    "       Current Bot: Promotion created successfully. Would you like the document sent to your email?\n"
    "       → This is 'Submission', NOT 'Email Fetching' (bot is asking for the FIRST time)\n\n"
    
    "     ✅ CORRECT Example (IS Email Fetching):\n"
    "       Previous Bot: Promotion created successfully. Would you like the document sent to your email?\n"
    "       Current User: Yes\n"
    "       Current Bot: Please provide your email address.\n"
    "       → This IS 'Email Fetching' (user is responding to email question)\n\n"
    
    "     ✅ CORRECT Example 2 (IS Email Fetching):\n"
    "       Previous Bot: Promotion created successfully. Would you like the document sent to your email?\n"
    "       Current User: user@example.com\n"
    "       Current Bot: Email sent successfully to user@example.com\n"
    "       → This IS 'Email Fetching' (user provided email in response to question)\n\n"
    
    "  5. **Other**  \n"
    "     Anything that doesn't fit the above.\n\n"
    
    "**Rules (in strict order - FOLLOW EXACTLY):**:\n"
    "  1. If Current Bot asks 'Would you like to submit' → return 'Detail Filling'. STOP.\n"
    "  2. If Previous Bot asked 'Would you like to submit' AND Current User said 'Yes' AND Current Bot has success message → return 'Submission'. STOP.\n"
    "     (Even if Current Bot also asks about email, it's still Submission because user just confirmed submission)\n"
    "  3. If Previous Bot asked about email AND Current User is responding about email AND Current Bot is processing email response → return 'Email Fetching'. STOP.\n"
    "  4. If Current User initiates new promotion → return 'Promotion Creation'. STOP.\n"
    "  5. If Current User mentions promotion fields → return 'Detail Filling'. STOP.\n"
    "  6. Otherwise → return 'Other'.\n\n"
    
    "**CRITICAL: The difference between Submission and Email Fetching:**\n"
    "- Submission = User responding to 'submit?' question\n"
    "- Email Fetching = User responding to 'email?' question\n"
    "- If bot asks BOTH in same message, classify based on what the user is responding to\n"
    "- If user just said 'Yes' to submission, it's Submission (even if bot asks about email next)\n\n"
    
    "**Return exactly** a JSON object with one key: `intent`, whose value is one of:\n"
    "  [\"Promotion Creation\", \"Detail Filling\", \"Submission\", \"Email Fetching\", \"Other\"]\n"
)

# Replace your existing extract_details with this version
async def extract_details(state: GraphState) -> Dict:
    """
    Node to extract structured promotion details and user intent from the conversation so far.
    Returns:
      - extracted_details
      - user_intent
    This version prints user_intent at several stages for debugging.
    """
    print("--- Node: extract_details (ENTER) ---")
    extracted_data: Optional[ExtractedPromotionDetails] = None
    user_intent: Optional[UserIntent] = None

    # Build chat_history string (unchanged)
    chat_history_lines: List[str] = []
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            chat_history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            chat_history_lines.append(f"Bot: {msg.content}")
        else:
            if hasattr(msg, "content"):
                chat_history_lines.append(f"Bot: {msg.content}")
    chat_history_str = "\n".join(chat_history_lines)

    # build missing_fields (unchanged)
    prev: Optional[ExtractedPromotionDetails] = state.get("extracted_details")
    missing_fields: List[str] = []
    if prev is None or not getattr(prev, "promotion_type", None):
        missing_fields.append("Promotion Type")
    if prev is None or not getattr(prev, "hierarchy_type", None):
        missing_fields.append("Hierarchy Level Type")
    if prev is None or not getattr(prev, "hierarchy_value", None):
        missing_fields.append("Hierarchy Level Value")
    if prev is None or not getattr(prev, "brand", None):
        missing_fields.append("Brand")
    if prev is None or not getattr(prev, "items", None):
        missing_fields.append("Items")
    if prev is None or not getattr(prev, "discount_type", None):
        missing_fields.append("Discount Type")
    if prev is None or not getattr(prev, "discount_value", None):
        missing_fields.append("Discount Value")
    if prev is None or not getattr(prev, "start_date", None):
        missing_fields.append("Start Date")
    if prev is None or not getattr(prev, "end_date", None):
        missing_fields.append("End Date")
    if prev is None or not getattr(prev, "stores", None):
        missing_fields.append("Stores")
    if prev is None or not getattr(prev, "email", None):
        missing_fields.append("Email")

    missing_fields_str = "\n".join(f"- {f}" for f in missing_fields) if missing_fields else "None"

    # prepare prompt_filled (unchanged)
    current_date = datetime.date.today().strftime("%d/%m/%Y")
    try:
        prompt_filled = promotion_extraction_prompt \
            .replace("{{extracted_text}}", chat_history_str) 
    except Exception as e:
        print(f"Error formatting template_Promotion_without_date: {e}")
        prompt_filled = promotion_extraction_prompt

    # 4. Invoke structured extractor LLM
    try:
        detail_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_filled)
        ])
        print("--- extract_details: invoking extractor_llm_structured ---")
        llm_result = await (detail_prompt | extractor_llm_structured).ainvoke({})
        print("--- extract_details: extractor_llm_structured returned ---")
        # Debug print raw llm_result
        try:
            print("RAW llm_result (repr):", repr(llm_result))
        except Exception:
            print("RAW llm_result: <unprintable>")

        if isinstance(llm_result, ExtractedPromotionDetails):
            extracted_data = llm_result
            print("--- extract_details: extracted_data populated ---")
            print("extracted_data (plain):", _to_plain(extracted_data))
        else:
            print("--- Warning: extract_details: LLM returned unexpected type:", type(llm_result))
            # Attempt to salvage if the LLM returned a dict-like shape
            try:
                extracted_data = _to_plain(llm_result)
                print("extract_details: salvaged extracted_data:", extracted_data)
            except Exception:
                pass
    except Exception as e:
        print(f"!!! ERROR during promotion detail extraction: {e} !!!")
        traceback.print_exc()

    # 5. Extract user intent (with verbose prints)
    try:
        print("--- extract_details: Starting intent extraction ---")
        last_bot_message = None
        current_user_message = None
        current_bot_message = None

        relevant_msgs = [msg for msg in state.get("messages", []) if isinstance(msg, (HumanMessage, AIMessage))]
        human_msgs = [msg for msg in relevant_msgs if isinstance(msg, HumanMessage)]
        ai_msgs = [msg for msg in relevant_msgs if isinstance(msg, AIMessage)]

        if len(ai_msgs) >= 2:
            last_bot_message = ai_msgs[-2].content
        if human_msgs:
            current_user_message = human_msgs[-1].content
        if ai_msgs:
            current_bot_message = ai_msgs[-1].content

        last_bot_message = last_bot_message or ""
        current_user_message = current_user_message or ""
        current_bot_message = current_bot_message or ""

        full_context = f"""Previous Bot: {last_bot_message}
    Current User: {current_user_message}
    Current Bot: {current_bot_message}"""

        print("INTENT CLASSIFIER CONTEXT >>>")
        print(full_context)

        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", intent_system),
            ("human", "{context}")
        ])

        # BEFORE calling LLM: print context shape
        print("--- extract_details: calling intent LLM with context ---")
        llm_intent = await (intent_prompt | intent_extractor_llm_structured).ainvoke({
            "context": full_context
        })
        print("--- extract_details: intent LLM returned ---")
        print("RAW llm_intent (repr):", repr(llm_intent))
        # if it's a Pydantic model class `UserIntent`, convert and inspect
        if isinstance(llm_intent, UserIntent):
            user_intent = llm_intent
            print("Predicted user intent (UserIntent object):", _to_plain(user_intent))
        else:
            # try to coerce to plain
            try:
                print("llm_intent is not UserIntent instance; trying to plain-encode it.")
                print("llm_intent (jsonable):", _to_plain(llm_intent))
                # If it contains the expected shape, convert to UserIntent-like dict
                # but do not force an actual UserIntent object unless required.
                user_intent = _to_plain(llm_intent)
            except Exception:
                user_intent = None
                print("Could not parse llm_intent into user_intent. Setting None.")

    except Exception as e:
        print(f"!!! ERROR during intent extraction: {e} !!!")
        traceback.print_exc()

    # Final debug print before returning node output
    print("--- Node: extract_details (EXIT) ---")
    print("final extracted_data (plain):", _to_plain(extracted_data))
    print("final user_intent (plain):", _to_plain(user_intent))

    return {
        "extracted_details": extracted_data,
        "user_intent": user_intent
    }

def should_continue(state: GraphState) -> str:
    """
    Determines if the conversation should continue or end.
    The chatbot loops until the user confirms submission.
    Conditions to end:
      - user intent == 'Submission'
    """
    print("--- Condition: should_continue ---")
    user_intent = None

    try:
        user_intent = getattr(state.get("user_intent"), "intent", None)
    except Exception as e:
        print(f"Error reading state in should_continue: {e}")

    if user_intent and user_intent.lower() == "submission":
        print("✅ User confirmed submission. Ending conversation.")
        return END

    print("🔁 Missing details or incomplete flow. Continuing conversation.")
    return "preprocess_input"

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
# workflow.add_edge("extract_details", END)
workflow.add_conditional_edges(
    "extract_details",
    should_continue,
    {
        "preprocess_input": "preprocess_input",  # Continue the loop
        END: END,  # Stop when complete or on submission
    },
)
# --- Memory and Compilation ---
memory = MemorySaver()
app_runnable_promotion_agentic = workflow.compile(checkpointer=memory)

# # --- API Request Model ---
# async def stream_response_generator_promotion(
#     graph_stream: AsyncIterator[Dict]
# ) -> AsyncIterator[str]:
#     """
#     Asynchronously streams LLM response chunks from specified nodes
#     to the client as soon as they arrive.
#     """
#     print("--- STREAM GENERATOR (for client) STARTED ---")
#     full_response_for_log = ""
#     try:
#         async for event in graph_stream:
#             if event["event"] == "on_chat_model_stream":
#                 metadata = event.get("metadata", {})
#                 node_name = metadata.get("langgraph_node")
#                 if node_name in {"generate_direct_response", "generate_response_from_sql"}:
#                     chunk = event["data"].get("chunk")
#                     if isinstance(chunk, AIMessageChunk) and chunk.content:
#                         content = chunk.content
#                         full_response_for_log += content
#                         yield content
#             elif event["event"] == "on_node_end" and node == "extract_details":
#                 output = event["data"]["output"] or {}
#                 # Safely serialize any dates or Pydantic models
#                 payload = {
#                     "extracted_details": getattr(output, "extracted_details", None),
#                     "user_intent": getattr(output, "user_intent", None),
#                 }
#                 # Convert Pydantic objects to dicts if needed
#                 if hasattr(payload["extracted_details"], "dict"):
#                     payload["extracted_details"] = payload["extracted_details"].dict()
#                 if hasattr(payload["user_intent"], "dict"):
#                     payload["user_intent"] = payload["user_intent"].dict()
#                 # SSE custom event
#                 yield (
#                     "event: extracted_details\n"
#                     f"data: {json.dumps(payload, default=str)}\n\n"
#                 )
#     except Exception as e:
#         print(f"!!! ERROR in stream_response_generator_promotion: {e} !!!")
#         # If nothing has been sent yet, at least send an error
#         yield f"\n\nStream error: {e}"
#     finally:
#         print(f"--- STREAM GENERATOR (for client) FINISHED ---")
#         # For debugging, you could log full_response_for_log here

# Connection manager to handle multiple WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, thread_id: str):
        await websocket.accept()
        self.active_connections[thread_id] = websocket
        print(f"WebSocket connected for thread: {thread_id}")

    def disconnect(self, thread_id: str):
        if thread_id in self.active_connections:
            del self.active_connections[thread_id]
            print(f"WebSocket disconnected for thread: {thread_id}")

    async def send_message(self, message: dict, thread_id: str):
        if thread_id in self.active_connections:
            await self.active_connections[thread_id].send_json(message)

    async def send_text(self, text: str, thread_id: str):
        if thread_id in self.active_connections:
            await self.active_connections[thread_id].send_text(text)

manager = ConnectionManager()

# Modified stream generator for WebSocket
async def stream_response_generator_websocket(
    graph_stream: AsyncIterator[Dict],
    websocket: WebSocket,
    thread_id: str
) -> None:
    """
    Streams LLM response chunks to WebSocket client with structured events.
    """
    print(f"--- WEBSOCKET STREAM GENERATOR STARTED for thread: {thread_id} ---")
    full_response_for_log = ""
    
    try:
        async for event in graph_stream:
            # Stream text chunks
            if event["event"] == "on_chat_model_stream":
                metadata = event.get("metadata", {})
                node_name = metadata.get("langgraph_node")
                
                if node_name in {"generate_direct_response", "generate_response_from_sql"}:
                    chunk = event["data"].get("chunk")
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        content = chunk.content
                        full_response_for_log += content
                        
                        # Send as structured message
                        await manager.send_message({
                            "type": "stream_chunk",
                            "content": content,
                            "node": node_name
                        }, thread_id)
            
            # Send extracted details when available
            elif event["event"] == "on_node_end" and event.get("name") == "extract_details":
                output = event["data"].get("output", {})
                
                # Safely serialize extracted details
                extracted_details = output.get("extracted_details")
                user_intent = output.get("user_intent")
                
                payload = {
                    "type": "extracted_details",
                    "extracted_details": _to_plain(extracted_details) if extracted_details else None,
                    "user_intent": _to_plain(user_intent) if user_intent else None,
                }
                
                await manager.send_message(payload, thread_id)
                print(f"Sent extracted details: {payload}")
        
        # Send completion signal
        await manager.send_message({
            "type": "stream_complete",
            "full_response": full_response_for_log
        }, thread_id)
        
    except Exception as e:
        print(f"!!! ERROR in websocket stream generator: {e} !!!")
        traceback.print_exc()
        
        await manager.send_message({
            "type": "error",
            "message": str(e)
        }, thread_id)
    
    finally:
        print(f"--- WEBSOCKET STREAM GENERATOR FINISHED for thread: {thread_id} ---")

app = FastAPI(
    title="LangGraph Chatbot API",
    description="API endpoint for a LangChain chatbot using LangGraph, detail extraction, SQL generation, and streaming.",
)
# WebSocket endpoint (replace your POST endpoint)
@app.websocket("/ws/promotion_chat/{thread_id}")
async def websocket_promotion_chat(websocket: WebSocket, thread_id: str):
    """
    WebSocket endpoint for promotion chat with agentic capabilities.
    
    Expected message format from client:
    {
        "type": "message",
        "content": "user message text",
        "thread_id": "optional-override"
    }
    
    Response format:
    {
        "type": "stream_chunk" | "extracted_details" | "stream_complete" | "error",
        "content": "...",  // for stream_chunk
        "extracted_details": {...},  // for extracted_details
        "user_intent": {...},  // for extracted_details
        "message": "..."  // for error
    }
    """
    await manager.connect(websocket, thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            print(f"\n--- [Thread: {thread_id}] Received WebSocket message: {data} ---")
            
            message_type = data.get("type")
            
            if message_type == "message":
                user_message = data.get("content", "")
                
                if not user_message.strip():
                    await manager.send_message({
                        "type": "error",
                        "message": "Empty message received"
                    }, thread_id)
                    continue
                
                # Send acknowledgment
                await manager.send_message({
                    "type": "ack",
                    "message": "Processing your request..."
                }, thread_id)
                
                # Prepare input for graph
                input_message = HumanMessage(content=user_message)
                input_state = {"messages": [input_message]}
                
                try:
                    # Stream graph execution
                    graph_stream = app_runnable_promotion_agentic.astream_events(
                        input_state, 
                        config,
                        version="v1"
                    )
                    
                    # Process and stream results
                    await stream_response_generator_websocket(
                        graph_stream,
                        websocket,
                        thread_id
                    )
                    
                except Exception as graph_error:
                    print(f"!!! ERROR in graph execution: {graph_error} !!!")
                    traceback.print_exc()
                    
                    await manager.send_message({
                        "type": "error",
                        "message": f"Error processing request: {str(graph_error)}"
                    }, thread_id)
            
            elif message_type == "ping":
                # Heartbeat mechanism
                await manager.send_message({
                    "type": "pong"
                }, thread_id)
            
            else:
                await manager.send_message({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }, thread_id)
    
    except WebSocketDisconnect:
        manager.disconnect(thread_id)
        print(f"Client disconnected: {thread_id}")
    
    except Exception as e:
        print(f"!!! WebSocket error for thread {thread_id}: {e} !!!")
        traceback.print_exc()
        manager.disconnect(thread_id)

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

