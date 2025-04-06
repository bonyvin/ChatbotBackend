
#24-MArch-2025- Before improving promotion department handling
from fastapi import FastAPI,Depends,HTTPException,status
from utils import gpt_response,extract_information,conversation,checkValidation,run_conversation,collect_invoice_data,bot_action,openaifunction,testModel,test_submission,extract_invoice_details,extract_text_from_pdf,extract_text_from_image,collect_invoice_data_from_file,extract_text_with_openai,categorize_invoice_details,client,categorize_invoice_details_new,previous_invoice_details,template_5,extract_details_gpt_vision,client_new,llm_gpt3,llm_gpt4,async_client 
from poUtils import template_PO,DEFAULT_PO_STRUCTURE,categorize_po_details,previous_po_details
from pydantic import BaseModel;
from models import Base, StoreDetails,User,PoDetails,PoHeader,InvHeader,InvDetails,Supplier,ShipmentHeader,ItemDiffs,ItemSupplier,ItemMaster,PromotionDetails,PromotionHeader
from schemas import StoreDetailsSchema, UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach,ChatRequest,SupplierCreate,ShipmentHeader,ShipmentDetails,ItemDiffsSchema,ItemSupplierSchema,ItemMasterSchema,PromotionDetailsSchema,PromotionHeaderSchema
from schemas import ShipmentDetails as ShipmentDetailsSchema
from database import engine,SessionLocal,get_db
from promoUtils import extract_and_classify_promo_details, template_Promotion,categorize_promo_details,previous_promo_details,categorize_promo_details_fun_call
from sqlalchemy.orm import Session
import models
from typing import List,Any,Optional
from fastapi.middleware.cors import CORSMiddleware
from utils_extractor import run_conversation
from typing import Dict,Tuple, Optional
import easyocr
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import openai
from datetime import datetime  # Add this import at the top of your file
import copy
# from exampleextrator import run_conversation
from fastapi.responses import JSONResponse
import mysql.connector
Base.metadata.create_all(bind=engine)
from sqlalchemy.sql import text
import re;
import json;
import os
import signal
from collections import defaultdict
import shutil
from collections import defaultdict
# Updated LangChain imports (community versions and new locations)
# Import updated LangChain classes from the recommended modules
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback 
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import hashlib
import asyncio
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from promoTest import handle_promotion_chat_promo_test
from langchain_community.chat_models import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import difflib

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#PO Chatbot Functions
chat_histories = {}
user_po_details = {}


#Store
# @app.post("/storeCreation/", status_code=status.HTTP_201_CREATED)
# def create_store_details(storeDetails:StoreDetailsSchema, db: Session = Depends(get_db)):
#     db_storeDetails = models.StoreDetails(
#         storeId=storeDetails.storeId,
#         storeName=storeDetails.storeName,
#         address=storeDetails.address,
#         city=storeDetails.city,
#         state=storeDetails.state,
#         zipCode=storeDetails.zipCode,
#         phone=storeDetails.phone
#     )
#     db.add(db_storeDetails)
#     db.commit()
#     db.refresh(db_storeDetails)
#     return db_storeDetails

@app.post("/storeCreation/", status_code=status.HTTP_201_CREATED)
def create_store_details(storeDetails:StoreDetailsSchema, db: Session = Depends(get_db)):
    db_storeDetails = models.StoreDetails(**storeDetails.dict())
    db.add(db_storeDetails)
    db.commit()
    db.refresh(db_storeDetails)
    return db_storeDetails

@app.get("/storeList/")
def get_stores(db: Session = Depends(get_db)):
    return db.query(StoreDetails).all()

# Get supplier by ID
@app.get("/storeList/{storeId}")
def get_stores_by_id(storeId : str, db: Session = Depends(get_db)):
    store = db.query(StoreDetails).filter(StoreDetails.storeId  == storeId ).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store Id not found")
    return store

#Promo
@app.get("/promotionHeader/{promotionId}")
def get_promotion_header(promotionId: str, db: Session = Depends(get_db)):
    promoHeader = db.query(models.PromotionHeader).filter(models.PromotionHeader.promotionId == promotionId).first()
    if not promoHeader:
        raise HTTPException(status_code=404, detail="Promotion not found")
    promoDetails = db.query(models.PromotionDetails).filter(models.PromotionDetails.promotionId == promotionId).all()
    return {"promotion_header": promoHeader, "promotion_details": promoDetails}

@app.get("/promotionDetails/{promotionId}", response_model=List[PromotionDetailsSchema])
def get_promotion_details(promotionId: str, db: Session = Depends(get_db)):
    promoDetails = db.query(models.PromotionDetails).filter(models.PromotionDetails.promotionId == promotionId).all()
    if not promoDetails:
        raise HTTPException(status_code=404, detail="No details found for this promotion")
    return promoDetails

@app.post("/promotionHeader/", status_code=status.HTTP_201_CREATED)
def create_promotion_header(promoHeader:PromotionHeaderSchema, db: Session = Depends(get_db)):
    db_promoHeader = models.PromotionHeader(**promoHeader.dict())
    db.add(db_promoHeader)
    db.commit()
    db.refresh(db_promoHeader)
    return db_promoHeader

# @app.post("/promotionHeader/", status_code=status.HTTP_201_CREATED)
# def create_promotion_header(promoHeader:PromotionHeaderSchema, db: Session = Depends(get_db)):
#     db_promoHeader = models.PromotionHeader(
#         promotionId=promoHeader.promotionId,
#         componentId=promoHeader.componentId,
#         startDate=promoHeader.startDate,
#         endDate=promoHeader.endDate,
#         promotionType=promoHeader.promotionType,
#         storeId=promoHeader.storeId

#     )
#     db.add(db_promoHeader)
#     db.commit()
#     db.refresh(db_promoHeader)
#     return db_promoHeader

@app.post("/promotionDetails/", status_code=status.HTTP_201_CREATED)
def create_promotion_details(promoDetails: List[PromotionDetailsSchema], db: Session = Depends(get_db)):
    for details in promoDetails:
        db_promoDetails = models.PromotionDetails(
            promotionId=details.promotionId,
            componentId=details.componentId,
            itemId=details.itemId,
            discountType=details.discountType,
            discountValue=details.discountValue
        )
        db.add(db_promoDetails)
        db.commit()
        db.refresh(db_promoDetails)
    return {"message": "Promotion details added successfully!"}


# Database Functions
# def get_valid_hierarchy_values(hierarchy_level: str) -> list:
#     """Fetch valid values for a hierarchy level from itemsMaster"""
#     column_map = {
#         "department": "itemDepartment",
#         "class": "itemClass",
#         "subclass": "itemSubClass"
#     }
    
#     if hierarchy_level.lower() not in column_map:
#         return []
    
#     column = column_map[hierarchy_level.lower()]
#     query = f"SELECT DISTINCT `{column}` FROM itemmaster"
#     result = execute_mysql_query(query)
#     return [str(row[column]) for row in result if row[column]]

# def validate_items(item_ids: list) -> dict:
#     """Validate item IDs against database"""
#     if not item_ids:
#         return {"valid": [], "invalid": []}
    
#     ids_str = ",".join([f"'{id.strip()}'" for id in item_ids])
#     query = f"SELECT itemId FROM itemmaster WHERE itemId IN ({ids_str})"
#     valid_ids = [row["itemId"] for row in execute_mysql_query(query)]
#     invalid_ids = list(set(item_ids) - set(valid_ids))
    
#     return {"valid": valid_ids, "invalid": invalid_ids}

# def validate_items_query(query: str) -> bool:
#     diff_attributes = ['color', 'size', 'material']
#     query_lower = query.lower()
#     for attr in diff_attributes:
#         if f'itemmaster.`{attr}`' in query_lower or f'where {attr} =' in query_lower:
#             return False
#     return True
def query_database_function_promo(question: str, db: Session) -> str:
    # Use the optimized function to extract attribute details (e.g., color, size, or material)
    attr, value, attr_id = find_attributes(question)
    # Now including 'storedetails' table
    tables = ["itemmaster", "itemsupplier", "itemdiffs", "storedetails"]
    tables_str = ", ".join(tables)
    
    # If an attribute filter is detected, prepare an optimization hint to use direct equality.
    optimization_hint = ""
    if attr and value and attr_id:
        optimization_hint = (
            f"\n### **Optimization for Attribute Filtering:**\n"
            f"- Detected filter for {attr} with value '{value}'. Use direct equality checks with id {attr_id} "
            f"instead of subqueries, i.e., (im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id}).\n"
        )
    
    promotion_sql_prompt = f"""
    You are a SQL assistant for fashion retail data. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 4 tables in my database, namely:
    - `itemmaster` (aliased as `im`)  
    - `itemsupplier` (aliased as `isup`)  
    - `itemdiffs` (aliased as `idf`)
    - `storedetails` (for store information with columns: storeId, storeName, address, city, state, zipCode, phone)

    ### **Critical Rules:**  

    #### **1. Color, Size, and Material Filters**  
    - If the user searches for an **attribute** like **color, size, or material**, follow this strict process:  
      1. **Find the `id` in `itemdiffs`** where `diffType` matches the attribute (e.g., 'color') and `diffId` is the user-provided value (e.g., 'Red').  
      2. **Filter `itemmaster` (`im`)** by checking if the retrieved `idf.id` exists in **`im.diffType1`**, **`im.diffType2`**, or **`im.diffType3`**.  
      3. **NEVER** search for colors, sizes, or materials inside `itemmaster` directly.  

    #### **2. Allowed Tables Only**  
    - You may only use the following tables:  
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

    #### **3. Query Format & Execution**  
    - **Start queries from `itemmaster`** as the primary table.
    - Use **explicit JOINs** when needed (e.g., joining `itemsupplier` for cost-related queries).
    - **Return only valid SQL queries** without additional explanations or markdown formatting.
    - The generated SQL query should be fully structured and dynamically adapt to the user query.
    {optimization_hint}
    #### **4. Department Name Flexibility**
    - When filtering by `itemDepartment`, use the `LIKE` operator with a trailing wildcard `%` to match plural/singular forms. 
      For example, if the user specifies 'T-Shirts', use `im.itemDepartment LIKE 'T-Shirt%'`.
    ---
    ### **SQL Examples:**

    #### **Example 1: Select All Red Colored Items**
    User: "Select all red colored items"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id}
    ```
    #### **Example 2: Select all red colored items with a supplier cost below $50**
    User: "Select all red colored items with a supplier cost below $50"
    ```sql
    SELECT im.itemId, isup.supplierCost
    FROM itemmaster im
    JOIN itemsupplier isup ON im.itemId = isup.itemId
    WHERE isup.supplierCost < 50
    AND (im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id})
    ```
    #### **Example 3: Select All Items of Size "Large"**
    User: "Select All Items of Size 'Large'"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id})
    ```
    #### **Example 4: Select All Items from T-Shirts Department**
    User: "Select All Items from T-Shirts Department"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.itemDepartment LIKE 'T-Shirt%'
    ```
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": promotion_sql_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    print("Query response:", response)
    mysql_query = response.choices[0].message.content.strip()
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate the generated SQL query
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result:", result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error:", {str(e)})
        return f"Error executing query: {str(e)}"

def query_database_function_promo(question: str, db: Session) -> str:
    # Use the optimized function to extract attribute details (e.g., color, size, or material)
    attr, value, attr_id = find_attributes(question)
    tables = ["itemmaster", "itemsupplier", "itemdiffs"]
    tables_str = ", ".join(tables)
    # If an attribute filter is detected, prepare an optimization hint to use direct equality.
    optimization_hint = ""
    if attr and value and attr_id:
        optimization_hint = (
            f"\n### **Optimization for Attribute Filtering:**\n"
            f"- Detected filter for {attr} with value '{value}'. Use direct equality checks with id {attr_id} "
            f"instead of subqueries, i.e., (im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id}).\n"
        )
    
    promotion_sql_prompt = f"""
    You are a SQL assistant for fashion retail data. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 3 tables in my database, namely:
    - `itemmaster` (aliased as `im`)  
    - `itemsupplier` (aliased as `isup`)  
    - `itemdiffs` (aliased as `idf`)  

    ### **Critical Rules:**  

    #### **1. Color, Size, and Material Filters**  
    - If the user searches for an **attribute** like **color, size, or material**, follow this strict process:  
      1. **Find the `id` in `itemdiffs`** where `diffType` matches the attribute (e.g., 'color') and `diffId` is the user-provided value (e.g., 'Red').  
      2. **Filter `itemmaster` (`im`)** by checking if the retrieved `idf.id` exists in **`im.diffType1`**, **`im.diffType2`**, or **`im.diffType3`**.  
      3. **NEVER** search for colors, sizes, or materials inside `itemmaster` directly.  

    #### **2. Allowed Tables Only**  
    - You may only use the following tables:  
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
        - **`itemId`**: Foreign key linking to `itemsMaster.itemId`, establishing the relationship between items and suppliers.   
    - `itemdiffs` (For Attribute Filtering):
        - **`id`**: Unique identifier for each differentiation type.  
        - **`diffType`**: The attribute type used to differentiate items (e.g., color, size, material).  
          - If the user wants to filter items by a specific attribute (e.g., color or size), the query should check `diffType` and retrieve the corresponding `diffId`.  
        - **`diffId`**: The actual differentiation value corresponding to `diffType`.  
          - Example: If `diffType = 'color'`, then `diffId` could be "Red"; if `diffType = 'size'`, then `diffId` could be "M".

    #### **3. Query Format & Execution**  
    - **Start queries from `itemmaster`** as the primary table.
    - Use **explicit JOINs** when needed (e.g., joining `itemsupplier` for cost-related queries).
    - **Return only valid SQL queries** without additional explanations or markdown formatting.
    - The generated SQL query should be fully structured and dynamically adapt to the user query.
    {optimization_hint}
    ---
    ### **SQL Examples:**

    #### **Example 1: Select All Red Colored Items**
    User: "Select all red colored items"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id}
    ```
    #### **Example 2: Select all red colored items with a supplier cost below $50**
    User: "Select all red colored items with a supplier cost below $50"
    ```sql
    SELECT im.itemId, isup.supplierCost
    FROM itemmaster im
    JOIN itemsupplier isup ON im.itemId = isup.itemId
    WHERE isup.supplierCost < 50
    AND (im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id})
    ```
    #### **Example 3: Select All Items of Size "Large"**
    User: "Select All Items of Size 'Large'"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id})
    ```
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": promotion_sql_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    print("Query response:", response)
    mysql_query = response.choices[0].message.content.strip()
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate the generated SQL query
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result:", result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error:", {str(e)})
        return f"Error executing query: {str(e)}"


# Helper functions

def db_query(query: str, params: dict = None) -> List:
    """Generic database query executor with error handling."""
    try:
        db = next(get_db())
        result = db.execute(text(query), params or {}).fetchall()
        db.close()
        return [row for row in result]
    except Exception as e:
        print(f"Database error: {e}")
        return []

# Optimized core functions
def find_attributes(question: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Find attributes, values, and their ID in a single optimized flow."""
    print(f"\nProcessing query: '{question}'")
    
    # Get attribute metadata in single query
    attr_data = db_query(
        "SELECT diffType, diffId, id FROM itemdiffs GROUP BY diffType, diffId"
    )
    
    if not attr_data:
        print("No attribute data found in database")
        return (None, None, None)

    # Build optimized attribute structure {diffType: {diffId: id}}
    attribute_map = {}
    for diff_type, diff_id, id_val in attr_data:
        attribute_map.setdefault(diff_type, {})[diff_id] = id_val

    print(f"Attribute map: {attribute_map.keys()}")
    
    # Detect attributes using optimized function
    attr, value = detect_attribute(question, attribute_map)
    if not attr or not value:
        return (None, None, None)

    # Get ID from pre-loaded map
    item_id = attribute_map.get(attr, {}).get(value)
    return (attr, value, item_id)

def detect_attribute(question: str, attribute_map: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Optimized attribute detection with caching."""
    if not attribute_map:
        return (None, None)

    try:
        # Build prompt from pre-loaded data
        attr_desc = "\n".join(
            f"- {attr}: {', '.join(values)}" 
            for attr, values in attribute_map.items()
        )

        prompt = f"""Analyze query: "{question}"
        Identify attributes from:
        {attr_desc}
        Respond in JSON format: {{"attribute": "...", "value": "..."}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a retail data analyst. Identify product attributes."
            }, {
                "role": "user", 
                "content": prompt
            }],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)
        attr = result.get('attribute')
        value = result.get('value')

        # Validate against known data
        if not attribute_map.get(attr, {}).get(value):
            print(f"Invalid detection: {attr}/{value}")
            return (None, None)

        return (attr, value)

    except Exception as e:
        print(f"Detection error: {e}")
        return (None, None)


# Instantiate the LLM using the latest ChatOpenAI interface
llm_gpt4 = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)
response_cache = {}  # In-memory cache keyed by conversation hash
user_promo_details={}
conversation_memory = {}
promo_details_memory = {}
conversation_states = defaultdict(list)
previous_promo_details = defaultdict(dict)

#working old code    
promo_states = defaultdict(dict)
DEFAULT_PROMO_STRUCTURE = {
    "type": "",
    "hierarchy": {"level": "", "value": ""},
    "items": [],
    "excluded_items": [],
    "discount": {"type": "", "amount": 0},
    "dates": {"start": "", "end": ""},
    "locations": [],
    "excluded_locations": [],
    "status": "draft"
}
user_promo_details={}

def classify_query(user_message: str, conversation_context: str) -> str:
    """
    Classifies user queries as "General" or "Not General" based on promotion-related intent.
    Returns "Not General" if the user is providing promotion details, even implicitly.
    """
    # Enhanced prompt with decision rules and examples
    prompt = f"""
**Role**: You are a query classifier for a retail promotion system. Classify the user's intent as either "General" or "Not General".

**Conversation Context (User's Previous Messages)**:
{conversation_context}

**Current User Query**:
{user_message}

**Classification Criteria**:
1. **Key Promotion Fields**: 
   - Explicit mentions of: Promotion Type, Hierarchy Type/Value, Brand, Items/SKUs, Discounts (%, Fixed, BOGO), Dates, Stores.
   - Implicit cues: "discount", "off", "promotion", "brand", "items", "start/end", "exclude", "department/class/subclass".

2. **Intent Signals**:
   - "Not General" if the user is:
     - Building/updating a promotion (e.g., "Create a promotion...", "Change discount to 20%").
     - Providing values for promotion fields (e.g., "Brand: H&M", "30% off").
     - Selecting items/locations (e.g., "Include all red shirts", "Exclude STORE003").

3. **General Queries**:
   - Data lookups (e.g., "List suppliers for ITEM001", "What's the description of X?").
   - Non-promotion questions about products/brands.

**Examples**:
- "List items by H&M" → General
- "30% off FashionX items in T-Shirts Dept" → Not General
- "Select all L-sized SKUs" → Not General
- "What's the hierarchy of ITEM002?" → General
- "Simple, Department, T-Shirts, 10% Off" → Not General

**Output**: JSON with "query_type" as "General" or "Not General".
    """

    FUNCTION_SCHEMA_CLASSIFY = {
        "name": "classify_query",
        "description": "Classify if the user is providing promotion details (Not General) or asking general questions (General).",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["General", "Not General"],
                    "description": "Classification result. 'Not General' if promotion fields, discounts, or setup intent are detected."
                }
            },
            "required": ["query_type"]
        }
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You classify queries for a promotion system."},
                {"role": "user", "content": prompt}
            ],
            functions=[FUNCTION_SCHEMA_CLASSIFY],
            function_call={"name": "classify_query"},
            temperature=0.0,  # Minimize randomness
            max_tokens=100
        )
        
        # Parse response (handles both function call and JSON modes)
        if response.choices[0].message.function_call:
            arguments = json.loads(response.choices[0].message.function_call.arguments)
        else:
            arguments = json.loads(response.choices[0].message.content)
        
        return arguments.get("query_type", "General")
    
    except Exception as e:
        logging.error(f"Classification error: {e}")
        return "General"

# LCEL integration placeholder – extend this as needed for your LCEL configuration
def apply_lcel(chain, lcel_expression):
    print("Applying LCEL configuration:", lcel_expression)
    # For now, return the chain unchanged
    return chain

# --- Modified handle_promotion_chat Endpoint ---
# @app.post("/promo-chat")
# async def handle_promotion_chat(request: dict):
#     """
#     Updated promo-chat endpoint that integrates:
#       • Advanced prompt engineering, memory management, LCEL, caching, and async execution.
#       • Original function calling for database queries.
#       • Query classification via the classify_query function.
#     """
#     user_id = request.get("user_id")
#     user_message = request.get("message")
#     if not user_id or not user_message:
#         raise HTTPException(status_code=400, detail="Missing user_id or message")
    
#     # Initialize user session state if needed
#     if user_id not in promo_states:
#         promo_states[user_id] = []
#         user_promo_details[user_id] = DEFAULT_PROMO_STRUCTURE.copy()
#         conversation_memory[user_id] = ConversationBufferWindowMemory(
#             k=10,
#             return_messages=True,
#             input_key="input",
#             output_key="output"
#         )
    
#     # Append user message and update conversation text
#     promo_states[user_id].append(f"User: {user_message}")
#     conversation_text = "\n".join(promo_states[user_id])
#     conversation_memory[user_id].save_context({"input": user_message}, {"output": ""})
    
#     # Create a cache key based on the conversation text
#     conversation_hash = hashlib.md5(conversation_text.encode("utf-8")).hexdigest()
#     if conversation_hash in response_cache:
#         return response_cache[conversation_hash]
    
#     # --- New Advanced Prompt Template & Memory ---
#     system_prompt_template = PromptTemplate(
#         input_variables=["current_date", "chat_history"],
#         template=(
#             "You are ExpX, a smart promotion assistant.\n"
#             "Today is {current_date}. Use the following conversation history to generate an efficient and cost-effective response.\n"
#             "Conversation History:\n{chat_history}\n\n"
#             "Leverage the following advanced LLM techniques:\n"
#             "• Prompt Templates & Prompt Engineering\n"
#             "• Chains & Modular Workflow Composition\n"
#             "• Memory Management\n"
#             "• Retrieval-Augmented Generation (RAG)\n"
#             "• Output Parsing & Structured Output\n"
#             "• Tool & Function Calling\n"
#             "• Agents & Dynamic Decision-Making\n"
#             "• Callbacks & Streaming\n"
#             "• Text Splitting Techniques\n"
#             "• Few-Shot Prompting & Example Selection\n"
#             "• Parallel Execution & Asynchronous Processing\n"
#             "• Caching & Token Management\n"
#             "• LangChain Expression Language (LCEL)\n\n"
#             "Respond concisely and in a structured format."
#         )
#     )
#     current_date = datetime.today().strftime("%d/%m/%Y")
#     system_prompt = system_prompt_template.format(current_date=current_date, chat_history=conversation_text)
    
#     # Optionally split conversation if needed
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     _ = text_splitter.split_text(conversation_text)  # Not used directly here
    
#     # --- First, run the advanced LLM chain with caching & LCEL ---
#     with get_openai_callback() as callback:
#         # Define an LLMChain with memory management
#         # llm_chain = LLMChain(
#         #     llm=llm_gpt4,
#         #     prompt=PromptTemplate(
#         #         input_variables=["chat_history"],
#         #         template="Given the following conversation history, generate an accurate promotion response:\n{chat_history}\nRespond concisely."
#         #     ),
#         #     memory=conversation_memory[user_id]
#         # )
#         prompt = PromptTemplate(
#             input_variables=["input", "chat_history"],
#             template="Given the following conversation history, generate an accurate promotion response:\n{chat_history}\nNew Input: {input}\nRespond concisely."
#         )
#         llm_chain = (
#             RunnablePassthrough.assign(
#                 chat_history=lambda x: conversation_memory[user_id].load_memory_variables(x)["history"]
#             )
#             | prompt
#             | llm_gpt4
#             | StrOutputParser()
#         )
#         # Apply LCEL configuration (placeholder)
#         lcel_expression = "parallelize(chain, async=True, retry=2)"
#         llm_chain = apply_lcel(llm_chain, lcel_expression)
#         # chain_response = await llm_chain.ainvoke(input={"chat_history": conversation_text})
#         chain_response = await llm_chain.ainvoke({"input": user_message})
#         bot_reply = chain_response 
    
#     bot_reply = chain_response
#     conversation_memory[user_id].save_context({"input": ""}, {"output": bot_reply})
#     promo_states[user_id].append(f"Bot: {bot_reply}")
    
#     # --- Append Original Function Calling Logic ---
#     # Build messages using the original template_Promotion
#     messages_fc = [
#         {"role": "system", "content": template_Promotion},
#         {"role": "user", "content": conversation_text}
#     ]
#     functions = [{
#         "name": "query_database",
#         "description": "Execute database queries for validation/data retrieval",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "question": {
#                     "type": "string",
#                     "description": "Natural language question requiring database data"
#                 }
#             },
#             "required": ["question"]
#         }
#     }]
#     # Invoke function calling via the LLM (if applicable)
#     try:
#         fc_response = await llm_gpt4.agenerate(
#             messages=messages_fc,
#             functions=functions,
#             function_call="auto",
#             temperature=0.7,
#             max_tokens=500
#         )
#         fc_message = fc_response["choices"][0]["message"]
#         query_called = False
#         if "function_call" in fc_message and fc_message["function_call"]:
#             args = json.loads(fc_message["function_call"]["arguments"])
#             db_session = next(get_db())
#             query_result = query_database_function_promo(args["question"], db_session)
#             query_called = True
#             messages_fc.append({
#                 "role": "function",
#                 "name": "query_database",
#                 "content": query_result
#             })
#             second_response = await llm_gpt4.agenerate(
#                 messages=messages_fc,
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             bot_reply = second_response["choices"][0]["message"]["content"]
#     except Exception as e:
#         logging.error("Error during function calling: %s", e)
#         query_called = False

#     # --- Classify the Query using the classify_query function ---
#     classification_result = await classify_query(user_message, conversation_text)
    
#     # --- Extract structured promotion details (simulate async extraction) ---
#     try:
#         promo_json = await categorize_promo_details_fun_call(bot_reply, user_id)
#     except Exception as e:
#         promo_json = user_promo_details[user_id]
    
#     # Determine submission status from bot reply content
#     if "Would you like to submit" in bot_reply:
#         submissionStatus = "pending"
#     elif "Promotion created successfully" in bot_reply:
#         submissionStatus = "submitted"
#     elif "I want to change something" in bot_reply:
#         submissionStatus = "cancelled"
#     else:
#         submissionStatus = "in_progress"
    
#     response_data = {
#         "user_id": user_id,
#         "bot_reply": bot_reply,
#         "chat_history": promo_states[user_id],
#         "promo_json": promo_json,
#         "submissionStatus": submissionStatus,
#         "query_called": query_called,
#         "classification_result": classification_result,
#         "callback_details": str(callback)
#     }
    
#     # Cache the response for future identical conversation contexts
#     response_cache[conversation_hash] = response_data
#     return response_data

@app.post("/promo-chat")
async def handle_promotion_chat(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in promo_states:
        promo_states[user_id] = []
        user_promo_details[user_id] = DEFAULT_PROMO_STRUCTURE.copy()
    
    promo_states[user_id].append(f"User: {user_message}")
    conversation = "\n".join(promo_states[user_id])
    try:
        messages = [
            {"role": "system", "content": template_Promotion},
            {"role": "user", "content": conversation}
        ]
        # print("Messages first: ",messages)
        functions = [{
            "name": "query_database",
            "description": "Execute database queries for validation/data retrieval",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"}
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )
        # print("Response: ",response)
        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False
        db_session = get_db() 

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function_promo(args["question"],db_session)
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })
            print("Messages: ",messages)
            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            # print("Second Response: ",second_response)
            bot_reply = second_response.choices[0].message.content

        # Retain promo_json from previous interaction if query_called is True
        # if not query_called:
        user_promo_details[user_id] = await categorize_promo_details(bot_reply, user_id)
        # user_promo_details[user_id] = await categorize_promo_details_fun_call(bot_reply, user_id)
        # user_promo_details[user_id] = await extract_and_classify_promo_details(bot_reply, user_id)
        promo_json = user_promo_details[user_id]  # Assign retained promo_json
        classification_result = classify_query(user_message, conversation)
        promo_states[user_id].append(f"Bot: {bot_reply}")
        # print("Promo JSON:", promo_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Promotion created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("Promotion submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": promo_states,
            "promo_json": promo_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus,
            "query_called":query_called,
            "classification_result":classification_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#DIFF
# GET all diffs
@app.get("/diffs", response_model=list[ItemDiffsSchema])
def get_diffs(db: Session = Depends(get_db)):
    diffs = db.query(models.ItemDiffs).all()
    return diffs

# POST a new diff
@app.post("/diffs", response_model=ItemDiffsSchema)
def create_diff(diff: ItemDiffsSchema, db: Session = Depends(get_db)):
    db_diff = models.ItemDiffs(**diff.dict())

    # Check if diff already exists
    existing_diff = db.query(models.ItemDiffs).filter(models.ItemDiffs.id == diff.id).first()
    if existing_diff:
        raise HTTPException(status_code=400, detail="Diff with this ID already exists")

    db.add(db_diff)
    db.commit()
    db.refresh(db_diff)
    return db_diff
#ITEM
@app.get("/items", response_model=List[ItemMasterSchema])
def get_items(db: Session = Depends(get_db)):
    items = db.query(models.ItemMaster).all()
    return items

@app.post("/items", response_model=List[ItemMasterSchema])
def create_items(items: List[ItemMasterSchema], db: Session = Depends(get_db)):
    new_items = [models.ItemMaster(**item.dict()) for item in items]
    db.add_all(new_items)
    db.commit()
    return new_items

#ITEM SUPPLIER
@app.post("/itemSuppliers/", response_model=ItemSupplierSchema)
def create_item_supplier(item_supplier: ItemSupplierSchema, db: Session = Depends(get_db)):
    # Optional: Check if the itemSupplier record already exists by supplierId and itemId
    existing_item_supplier = db.query(ItemSupplier).filter(
        ItemSupplier.supplierId == item_supplier.supplierId,
        ItemSupplier.itemId == item_supplier.itemId
    ).first()
    if existing_item_supplier:
        raise HTTPException(status_code=400, detail="ItemSupplier entry already exists")
    
    new_item_supplier = ItemSupplier(
        supplierCost=item_supplier.supplierCost,
        supplierId=item_supplier.supplierId,
        itemId=item_supplier.itemId
    )
    db.add(new_item_supplier)
    db.commit()
    db.refresh(new_item_supplier)
    return new_item_supplier

@app.get("/itemSuppliers/")
def get_item_suppliers(db: Session = Depends(get_db)):
    return db.query(ItemSupplier).all()

@app.get("/itemSuppliers/{id}", response_model=ItemSupplierSchema)
def get_item_supplier_by_id(id: int, db: Session = Depends(get_db)):
    item_supplier = db.query(ItemSupplier).filter(ItemSupplier.id == id).first()
    if item_supplier is None:
        raise HTTPException(status_code=404, detail="ItemSupplier not found")
    return item_supplier


#SHIPMENT 
@app.post("/shipments/", response_model=ShipmentHeader)
def create_shipment(shipment: ShipmentHeader, db: Session = Depends(get_db)):
    db_shipment = models.ShipmentHeader(**shipment.dict())
    db.add(db_shipment)
    db.commit()
    db.refresh(db_shipment)
    return db_shipment

@app.post("/shipments/{receipt_id}/details", response_model=List[ShipmentDetailsSchema])
def add_shipment_details(receipt_id: str, details: List[ShipmentDetailsSchema], db: Session = Depends(get_db)):
    shipment = db.query(models.ShipmentHeader).filter(models.ShipmentHeader.receiptId == receipt_id).first()
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    po_items = {po_detail.itemId for po_detail in db.query(models.PoDetails).filter(models.PoDetails.poId == shipment.poId).all()}
    
    for detail in details:
        if detail.itemId not in po_items:
            raise HTTPException(status_code=400, detail=f"Item {detail.itemId} is not in the PO {shipment.poId}")
    
    db_details = [models.ShipmentDetails(**{**detail.dict(), "receiptId": receipt_id}) for detail in details]
    db.add_all(db_details)
    db.commit()
    for db_detail in db_details:
        db.refresh(db_detail)
        
    return db_details

@app.get("/shipments/{receipt_id}", response_model=Dict[str, Any])
def get_shipment_with_details(receipt_id: str, db: Session = Depends(get_db)):
    shipment = db.query(ShipmentHeader).filter(ShipmentHeader.receiptId == receipt_id).first()
    if shipment is None:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    details = db.query(ShipmentDetails).filter(ShipmentDetails.receiptId == receipt_id).all()
    return {"shipment": shipment, "details": details}


#SUPPPLIER
@app.post("/suppliers/", response_model=SupplierCreate)
def create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    # Check if supplier already exists
    existing_supplier = db.query(Supplier).filter(Supplier.email == supplier.email).first()
    if existing_supplier:
        raise HTTPException(status_code=400, detail="Supplier with this email already exists")

    new_supplier = Supplier(
        supplierId=supplier.supplierId,
        name=supplier.name,
        email=supplier.email,
        phone=supplier.phone,
        address=supplier.address,
        lead_time=supplier.lead_time
    )
    db.add(new_supplier)
    db.commit()
    db.refresh(new_supplier)
    return new_supplier

# Get all suppliers
@app.get("/suppliers/")
def get_suppliers(db: Session = Depends(get_db)):
    return db.query(Supplier).all()

# Get supplier by ID
@app.get("/suppliers/{supplierId}")
def get_supplier(supplierId : str, db: Session = Depends(get_db)):
    supplier = db.query(Supplier).filter(Supplier.supplierId  == supplierId ).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return supplier

#po chat
def execute_mysql_query(query):
    """Executes a MySQL query after validating the table name."""
    try:
        words = query.split()
        if "FROM" in words:
            table_index = words.index("FROM") + 1
            table_name = words[table_index].strip("`;,")  # Extract table name safely
        else:
            return "Invalid SQL query structure."

        # Get existing tables
        valid_tables = get_table_names()
        if table_name not in valid_tables:
            return f"Error: Table `{table_name}` does not exist. Available tables: {valid_tables}"

        # Execute the validated query
        db: Session = next(get_db())
        result = db.execute(text(query)).fetchall()
        db.close()
        return [dict(row._mapping) for row in result]

    except Exception as e:
        return str(e)

def get_table_names():
    """Fetch all table names from the database."""
    try:
        db: Session = next(get_db())
        result = db.execute(text("SHOW TABLES")).fetchall()
        db.close()
        return [list(row)[0] for row in result]  # Extract table names
    except Exception as e:
        return str(e)

def query_database_function(question: str) -> str:
    """Generates and executes an SQL query based on a natural language question."""
    # Fetch available tables
    available_tables = get_table_names()
    tables_str = ", ".join(available_tables)

    # SQL generation prompt (same as original)
    sql_query_prompt = f"""
    The user wants to query the MySQL database. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 7 tables in my database, namely: invoicedetails, invoiceheader, podetails, poheader, suppliers, shipmentHeader, and shipmentDetails.
    - **invoicedetails**: id, itemId, itemQuantity, itemDescription, itemCost, totalItemCost, poId, invoiceNumber.
    - **invoiceheader**: invoiceId, invoicedate, invoiceType, currency, payment_term, invoice_status, total_qty, total_cost, total_tax, total_amount, userInvNo.
    - **podetails**: id, itemId, itemQuantity, supplierId, itemDescription, itemCost, totalItemCost, poId.
    - **poheader**: poNumber, shipByDate, leadTime, estimatedDeliveryDate, totalQuantity, totalCost, totalTax, currency, payment_term.
    - **suppliers**: supplierId, name, email, phone, address, lead_time.
    - **shipmentheader**: receiptId, asnId, expectedDate, receivedDate, receivedBy, status, sourceLocation, destinationLocation, totalQuantity, totalCost, poId.
    - **shipmentdetails**: id, itemId, itemDescription, itemCost, expectedItemQuantity, receivedItemQuantity, damagedItemQuantity, containerId, receiptId, invoiced.

    ### **Field Mapping Rules** 
    - "purchase order", "purchaseorder", "PURCHASE ORDER", "PURCHASEORDER", "Purchase Order", and "PurchaseOrder" refer to the **poheader** table.
    - "PO number", "po_number", "po no", "PONumber", "order number" refer to **poNumber**.
    - "po_id" and "poId" are equivalent and refer to **poId**.
    - "cost" and "total cost" refer to **totalCost**.
    - "invoiceNumber", "invoice Number", "invoice No", "invoiceNo", "invoiceId", and "invoice Id" refer to **userInvNo**.
    - "amount" refers to **totalAmount**.
    - "invoice number" and "invoiceNumber" refer to **invoiceNumber**.
    - "payment terms" and "payment_term" are the same.
    - "lead time" and "lead_time" are the same.
    - "supplier" refers to the **suppliers** table.
    - "supplierId" and "supplier id" refer to **supplierId**.
    - "shipment", "shipment header", "receipt" refer to **shipmentHeader**.
    - "shipment details", "shipment detail", "shipment items" refer to **shipmentDetails**.
    - "receipt id" and "receiptId" refer to **receiptId** in **shipmentHeader** or **shipmentDetails**.
    - "ASN" refers to **asnId**.
    - "expected date" refers to **expectedDate**.
    - "received date" refers to **receivedDate**.
    - "received by" refers to **receivedBy**.
    - "status" refers to **status**.
    - "source location" refers to **sourceLocation**.
    - "destination location" refers to **destinationLocation**.
    - "total quantity" refers to **totalQuantity**.
    - "total cost" refers to **totalCost**.
    - "item id" refers to **itemId**.
    - "item description" refers to **itemDescription**.
    - "item cost" refers to **itemCost**.
    - "expected item quantity" refers to **expectedItemQuantity**.
    - "received item quantity" refers to **receivedItemQuantity**.
    - "damaged item quantity" refers to **damagedItemQuantity**.
    - "container id" refers to **containerId**.
    - "invoiced" refers to **invoiced**.
    -  "quality" and "quality issues" refer to the **percentage of damagedItemQuantity** calculated as:
        **(damagedItemQuantity / receivedItemQuantity) * 100**.
    ### **Scenario for Invoice Listing**
    If the user asks for all invoices of a PO number, generate a query that retrieves all the userInvNo values available for that PO number. To do this, join the invoiceheader table with the invoicedetails table (using invoiceheader.invoiceId = invoicedetails.invoiceNumber) and filter by the given poId.

    ### **Case-Insensitive Handling**
    ### **Handling Special Characters & Spaces**
    """
    sql_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sql_query_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    
    # Extract and clean query
    mysql_query = sql_response.choices[0].message.content.strip()
    print("query_database_function result: ",sql_response)
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate and execute
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result: ",result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error: ",{str(e)})
        return f"Error executing query: {str(e)}"

@app.post("/chat")
async def chat_with_po_assistant(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()
    
    chat_histories[user_id].append(f"User: {user_message}")
    conversation = "\n".join(chat_histories[user_id])

    try:
        # First API call - with function definition
        messages = [
            {"role": "system", "content": template_PO},
            {"role": "user", "content": conversation}
        ]

        functions = [{
            "name": "query_database",
            "description": "Retrieve data from the database using SQL queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"
                    }
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function(args["question"])
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })

            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Retain po_json from previous interaction if query_called is True
        if not query_called:
            user_po_details[user_id] = await categorize_po_details(bot_reply, user_id)

        po_json = user_po_details[user_id]  # Assign retained po_json

        chat_histories[user_id].append(f"Bot: {bot_reply}")
        print("PO JSON:", po_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Purchase Order created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("PO submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": chat_histories,
            "po_json": po_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adduser")
async def add_user(request:UserSchema,db:Session=Depends(get_db)):
    user = User(name=request.name,email=request.email,nickname=request.nickname)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/poCreation/", status_code=status.HTTP_201_CREATED)
def create_po(poHeader: poHeaderCreate, db: Session = Depends(get_db)):
    db_poHeader = models.PoHeader(**poHeader.dict())
    db.add(db_poHeader)
    db.commit()
    db.refresh(db_poHeader)
    return db_poHeader

@app.post("/poDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_poDetail(poDetailData: List[poDetailsCreate], db: Session = Depends(get_db)):
    for details in poDetailData:
        db_poDetails = models.PoDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            supplierId=details.supplierId
        )
        db.add(db_poDetails)
        db.commit()
        db.refresh(db_poDetails)
    return {
        "message":"Items added Sucessfully!"
    }
    # db_poDetails = models.PoDetails(**poDetailData.dict())
    # db.add(db_poDetails)
    # db.commit()
    # db.refresh(db_poDetails)
    # return [db_poDetails]

@app.post("/invCreation/", status_code=status.HTTP_201_CREATED)
def create_invoice(invHeader: invHeaderCreate, db: Session = Depends(get_db)):
    db_invHeader = models.InvHeader(**invHeader.dict())
    db.add(db_invHeader)
    db.commit()
    db.refresh(db_invHeader)
    return db_invHeader

@app.post("/invDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_invDetail(invDetailData: List[invDetailsCreate], db: Session = Depends(get_db)):
    for details in invDetailData:
        db_invDetails = models.InvDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            invoiceNumber= details.invoiceNumber
        )
        db.add(db_invDetails)
        db.commit()
        db.refresh(db_invDetails)
    return {
        "message":"Items added Sucessfully!"
    }

@app.get("/poDetails/{po_id}")
def read_poDeatils(po_id: str, db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        detail = "Po Number is not found in our database! Please add a valid PO number!"
        conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        
    po_info = db.query(models.PoDetails).filter(models.PoDetails.poId == po_id).all()
    return { "po_header":po,"po_details":po_info}

@app.post("/invoiceValidation")
def po_data_validations(po_id:str,detail:Dict[str,int],db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        raise HTTPException(status_code=404, detail="PO is not found!")
    for item,quantity in detail:
        po_details = db.query(models.PoDetails).filter(models.PoDetails.itemId==item).first()
        if po_details is None:
            detail = "Item which you added is not present in this PO"
            conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        if(po_details.itemQuantity>quantity):
            detail = po_details.itemId + "quantity is excced according to PO quantity is" + po_details.itemQuantity
            conversation.append('Bot: ' + detail)
            raise HTTPException(status_code=404, detail=conversation)
    return {"details":conversation}

@app.get("/invoiceDetails/{inv_id}")
def read_invDeatils(inv_id: str, db: Session = Depends(get_db)):
    inv = db.query(models.InvHeader).filter(models.InvHeader.invoiceId == inv_id).first()
    if inv is None:
        raise HTTPException(status_code=404, detail="Invoice not found!")
    inv_info = db.query(models.InvDetails).filter(models.InvDetails.invoiceNumber == inv_id).all()
    return { "inv_header":inv,"inv_details":inv_info}

@app.post("/ok")  
async def ok_endpoint(query: str, db: Session = Depends(get_db)):
    # Pass the db session to query_database_function_promo
    returned_attributes = query_database_function_promo(query, db)
    return {"message": "ok", "attributes": returned_attributes}

@app.post("/testing")
async def testing_endpoint(request: ChatRequest):
    promo_response = await handle_promotion_chat_promo_test(request.model_dump())  # Call function from promo.py
    return {"promo": promo_response}

@app.post("/uploadPromo")  
async def upload_promo(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Call the extraction function
        data = await extract_details_gpt_vision(temp_file_path)
        result=await categorize_promo_details(data,"admin")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove the temporary file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return {"structured_data": result}

@app.get
async def findPoDetails(po:str):
        db: Session = Depends(get_db)
        po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po).first()
        if po is None:
            return {"Po not found!"}
        else:
            return {"Po Found!"}


@app.post("/clearData")
async def clearConversation(submitted:str):
    conversation.clear()
    chat_histories.clear()
    previous_invoice_details.clear()
    previous_po_details.clear()
    previous_promo_details.clear()
    submissionStatus = "not submitted"
    return {"conversation":conversation,"submissionStatus":"not submitted","chat_history":chat_histories}

@app.post("/testSubmission")
async def submiission(query:str):
    result=test_submission(query)
    return {"result":result}

@app.post("/uploadGpt/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_invoice_details(extracted_text)

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

@app.post("/uploadPo/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_po_details(extracted_text,"admin")

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

    
@app.post("/uploadOpenAi/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    extracted_text = extract_text_with_openai(file)
    return JSONResponse(content={"extracted_text": extracted_text})


@app.post("/upload/")
async def upload_invoice(file: UploadFile = File(...)):
    """API to upload an invoice file and extract details."""
    if file.content_type not in ["image/png", "image/jpeg", "application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, PDF, and TXT are supported.")

    # Read file bytes
    file_bytes = await file.read()

    # Extract text based on file type
    if file.content_type in ["image/png", "image/jpeg"]:
        image = Image.open(BytesIO(file_bytes))
        extracted_text = extract_text_from_image(image)
    elif file.content_type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
    else:  # Text file
        extracted_text = file_bytes.decode("utf-8")

    # Process extracted text
    invoice_details = extract_invoice_details(extracted_text)
    invoice_data_from_conversation = {
        "quantities": extract_invoice_details(extracted_text).get("quantities", []),
        "items": extract_invoice_details(extracted_text).get("items", [])
    }
    # invoice_json=json.dumps(invoice_details)
    # await generate_response(invoice_details)

    return {"file_name": file.filename, "invoice_details": invoice_details,"invoice_data_from_conversation":invoice_data_from_conversation,"extracted_text":extracted_text}


def fetch_po_items(po_number: str) -> list:
    """Fetch PO items with error handling"""
    try:
        query = f"""
        SELECT itemId, itemQuantity, itemDescription, itemCost 
        FROM podetails 
        WHERE poId = '{po_number}'
        """
        result = execute_mysql_query(query)
        return [dict(item) for item in result] if result else []
    except Exception as e:
        print(f"PO fetch error: {str(e)}")
        return []
    
invoice_chat_histories = {}
user_invoice_details = {}

@app.post("/creation/response") 
async def generate_response(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session for invoice creation
    if user_id not in invoice_chat_histories:
        invoice_chat_histories[user_id] = []
        user_invoice_details[user_id] = {}
    
    invoice_chat_histories[user_id].append(f"User: {user_message}")
    conversation = "\n".join(invoice_chat_histories[user_id])

    try:
        # Prepare messages with system template
        messages = [
            {"role": "system", "content": template_5},
            {"role": "user", "content": conversation}
        ]

        # Define available functions
        functions = [{
            "name": "query_database",
            "description": "Retrieve data from the database using SQL queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"
                    }
                },
                "required": ["question"]
            }
        }]

        # First API call with function definition
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function(args["question"])
            query_called = True

            # Append function response and make second call
            messages.append({
                "role": "function",
                "name": "query_database",
                "content": query_result
            })

            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Update conversation history
        invoice_chat_histories[user_id].append(f"Bot: {bot_reply}")

        # Extract invoice details if no function was called
        if not query_called:
            user_invoice_details[user_id] = await categorize_invoice_details_new(bot_reply, user_id)

        inv_details = user_invoice_details[user_id]
        print("invDetails",inv_details)
        po_items=fetch_po_items(inv_details["PO Number"])

        # Determine submission status
        submissionStatus = "not submitted"
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Invoice created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"

        # Additional processing from original implementation
        test_model_reply = testModel(user_message, bot_reply)
        form_submission = test_submission(bot_reply)
        
        # Determine action type
        action = 'action'
        past_pattern = re.compile(r"(past\s+invoice|invoice\s+past|last\s+invoice)", re.IGNORECASE)
        create_pattern = re.compile(r"(create\s+invoice|invoice\s+create)", re.IGNORECASE)
        
        for line in invoice_chat_histories[user_id]:
            if line.startswith("User:"):
                user_input = line.split(":")[1].strip()
                if past_pattern.search(user_input):
                    action = "last invoice created"
                elif create_pattern.search(user_input):
                    action = "create invoice"
                elif "create po" in user_input.lower():
                    action = "create PO"
        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": invoice_chat_histories[user_id],
            "conversation": invoice_chat_histories[user_id],
            "invoice_json": inv_details,
            "action": action,
            "submissionStatus":form_submission,
            "po_items":po_items,
            # "submissionStatus": submissionStatus,
            "test_model_reply": test_model_reply,
            "invoiceDatafromConversation":inv_details,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
# @app.post("/creation/response")
# async def generate_response(query:str):
#     conversation.append('User: ' + query)
#     output = gpt_response(query)
#     conversation.append('Bot: ' + output)
#     test_model_reply=testModel(query,output)
#     form_submission=test_submission(output)
#     action='action'
#     submissionStatus="not submitted"
#     past_invoice_regex = r"(past\s+invoice|invoice\s+past|last\s+invoice|invoice\s+last|previous\s+invoice|invoice\s+previous)\s*"
#     invoice_regex = r"(create\s+invoice|invoice\s+create|create\san\sinvoice)\s*"
#     patternInvoice = re.compile(invoice_regex, re.IGNORECASE)
#     pastPatternInvoice = re.compile(past_invoice_regex, re.IGNORECASE)
#     for line in conversation:
#         if line.startswith("User:"):
#             user_input = line.split(":")[1].strip().lower()
#             if re.search(pastPatternInvoice, user_input):
#                 action = "last invoice created"
#                 # break
#             elif re.search(patternInvoice, user_input):
#                 action = "create invoice"
#             elif "create po" in user_input:
#                 action = "create PO"

#     invoiceDatafromConversation=collect_invoice_data(conversation)
#     invDetails=await categorize_invoice_details_new(conversation,"admin")
#     print("invDetails",invDetails)
#     return {"conversation":conversation,"invoice_json":invDetails,"action":action,
#     "submissionStatus":form_submission,"invoiceDatafromConversation":invDetails,
#     "test_model_reply":test_model_reply }
   


# @app.post("/creation/response") 
# async def generate_response(request: ChatRequest):
#     user_id = request.user_id
#     user_message = request.message
#     po_validated = False
    
#     # Maintain user session for invoice creation
#     if user_id not in invoice_chat_histories:
#         invoice_chat_histories[user_id] = []
#         user_invoice_details[user_id] = {'po_validated': False, 'po_data': {}}
    
#     invoice_chat_histories[user_id].append(f"User: {user_message}")
#     conversation = "\n".join(invoice_chat_histories[user_id])

#     try:
#         # Enhanced system prompt with PO validation emphasis
#         messages = [
#             {"role": "system", "content": f"{template_5}\nFIRST PRIORITY: Always check for and validate PO numbers if mentioned."},
#             {"role": "user", "content": conversation}
#         ]

#         # Define prioritized functions
#         functions = [
#             {
#                 "name": "validate_po",
#                 "description": "MUST CALL THIS FIRST if user mentions a PO number. Validates PO and retrieves items.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "po_number": {
#                             "type": "string",
#                             "description": "Exact PO number from user message (e.g. PO123, purchase order 456)"
#                         }
#                     },
#                     "required": ["po_number"]
#                 }
#             },
#             {
#                 "name": "query_database",
#                 "description": "For general database queries not related to PO validation",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "question": {
#                             "type": "string", 
#                             "description": "Natural language question requiring database data"
#                         }
#                     },
#                     "required": ["question"]
#                 }
#             }
#         ]

#         # First try to force PO validation
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             functions=functions,
#             function_call={"name": "validate_po"},
#             temperature=0.7,
#             max_tokens=500
#         )

#         response_message = response.choices[0].message
#         function_call = response_message.function_call

#         # Fallback to auto if no PO detected
#         if not function_call:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 functions=functions,
#                 function_call="auto",
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             response_message = response.choices[0].message

#         bot_reply = response_message.content
#         function_call = response_message.function_call
#         query_called = False

#         # Handle PO validation first
#         if function_call and function_call.name == "validate_po":
#             args = json.loads(function_call.arguments)
#             po_number = args["po_number"]
            
#             # Fetch and store PO items
#             items = fetch_po_items(po_number)
#             user_invoice_details[user_id]['po_data'] = {
#                 'po_number': po_number,
#                 'items': items,
#                 'validated_at': datetime.now().isoformat()
#             }
#             user_invoice_details[user_id]['po_validated'] = True
#             po_validated = True

#             # Build validation response
#             content = json.dumps({
#                 "status": "validated",
#                 "po_number": po_number,
#                 "item_count": len(items),
#                 "items_sample": items[:3] if len(items) > 3 else items
#             }, default=str)

#             messages.append({
#                 "role": "function",
#                 "name": "validate_po",
#                 "content": content
#             })

#             # Generate final response with PO context
#             second_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             bot_reply = second_response.choices[0].message.content

#         elif function_call and function_call.name == "query_database":
#             args = json.loads(function_call.arguments)
#             query_result = query_database_function(args["question"])
#             query_called = True

#             messages.append({
#                 "role": "function",
#                 "name": "query_database",
#                 "content": query_result
#             })

#             second_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             bot_reply = second_response.choices[0].message.content

#         # Update conversation history
#         invoice_chat_histories[user_id].append(f"Bot: {bot_reply}")

#         # Only extract details if no PO validation occurred
#         if not po_validated and not query_called:
#             user_invoice_details[user_id] = await categorize_invoice_details_new(bot_reply, user_id)

#         # Merge PO data with invoice details
#         if po_validated:
#             user_invoice_details[user_id].setdefault('invoice_data', {})
#             user_invoice_details[user_id]['invoice_data'].update({
#                 'po_number': user_invoice_details[user_id]['po_data']['po_number'],
#                 'po_items': user_invoice_details[user_id]['po_data']['items']
#             })

#         # Ensure final validated status
#         po_validated = user_invoice_details[user_id].get('po_validated', False)

#         # Determine submission status
#         submission_status = "in_progress"
#         if "Would you like to submit" in bot_reply:
#             submission_status = "pending"
#         elif "Invoice created successfully" in bot_reply:
#             submission_status = "submitted"
#         elif "I want to change something" in bot_reply:
#             submission_status = "cancelled"

#         # Action detection
#         action = 'create invoice' if 'create invoice' in user_message.lower() else 'other'
        
#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "chat_history": invoice_chat_histories[user_id],
#             "invoice_json": user_invoice_details[user_id].get('invoice_data', {}),
#             "action": action,
#             "submissionStatus": submission_status,
#             "test_model_reply": testModel(user_message, bot_reply),
#             "invoiceDatafromConversation": user_invoice_details[user_id],
#             "po_validated": po_validated,
#             "validated_po": user_invoice_details[user_id].get('po_data', {})
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#24-MArch-2025- Before improving promotion department handling

#16 MArch 2025- Testing dfferent variations of LLM enhancements and classification
from fastapi import FastAPI,Depends,HTTPException,status
from utils import gpt_response,extract_information,conversation,checkValidation,run_conversation,collect_invoice_data,bot_action,openaifunction,testModel,test_submission,extract_invoice_details,extract_text_from_pdf,extract_text_from_image,collect_invoice_data_from_file,extract_text_with_openai,categorize_invoice_details,client,categorize_invoice_details_new,previous_invoice_details,template_5,extract_details_gpt_vision,client_new,llm_gpt3,llm_gpt4,async_client 
from poUtils import template_PO,DEFAULT_PO_STRUCTURE,categorize_po_details,previous_po_details
from pydantic import BaseModel;
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails,Supplier,ShipmentHeader,ItemDiffs,ItemSupplier,ItemMaster,PromotionDetails,PromotionHeader
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach,ChatRequest,SupplierCreate,ShipmentHeader,ShipmentDetails,ItemDiffsSchema,ItemSupplierSchema,ItemMasterSchema,PromotionDetailsSchema,PromotionHeaderSchema
from schemas import ShipmentDetails as ShipmentDetailsSchema
from database import engine,SessionLocal,get_db
from promoUtils import template_Promotion,categorize_promo_details,previous_promo_details,categorize_promo_details_fun_call
from sqlalchemy.orm import Session
import models
from typing import List,Any,Optional
from fastapi.middleware.cors import CORSMiddleware
from utils_extractor import run_conversation
from typing import Dict,Tuple, Optional
import easyocr
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import openai
from datetime import datetime  # Add this import at the top of your file
import copy
# from exampleextrator import run_conversation
from fastapi.responses import JSONResponse
import mysql.connector
Base.metadata.create_all(bind=engine)
from sqlalchemy.sql import text
import re;
import json;
import os
import signal
from collections import defaultdict
import shutil
from collections import defaultdict
# Updated LangChain imports (community versions and new locations)
# Import updated LangChain classes from the recommended modules
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import logging
Base.metadata.create_all(bind=engine)

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#PO Chatbot Functions
chat_histories = {}
user_po_details = {}


#Promo
@app.get("/promotionHeader/{promotionId}")
def get_promotion_header(promotionId: str, db: Session = Depends(get_db)):
    promoHeader = db.query(models.PromotionHeader).filter(models.PromotionHeader.promotionId == promotionId).first()
    if not promoHeader:
        raise HTTPException(status_code=404, detail="Promotion not found")
    promoDetails = db.query(models.PromotionDetails).filter(models.PromotionDetails.promotionId == promotionId).all()
    return {"promotion_header": promoHeader, "promotion_details": promoDetails}

@app.get("/promotionDetails/{promotionId}", response_model=List[PromotionDetailsSchema])
def get_promotion_details(promotionId: str, db: Session = Depends(get_db)):
    promoDetails = db.query(models.PromotionDetails).filter(models.PromotionDetails.promotionId == promotionId).all()
    if not promoDetails:
        raise HTTPException(status_code=404, detail="No details found for this promotion")
    return promoDetails
@app.post("/promotionHeader/", status_code=status.HTTP_201_CREATED)
def create_promotion_header(promoHeader:PromotionHeaderSchema, db: Session = Depends(get_db)):
    db_promoHeader = models.PromotionHeader(
        promotionId=promoHeader.promotionId,
        componentId=promoHeader.componentId,
        startDate=promoHeader.startDate,
        endDate=promoHeader.endDate,
        promotionType=promoHeader.promotionType
    )
    db.add(db_promoHeader)
    db.commit()
    db.refresh(db_promoHeader)
    return db_promoHeader
@app.post("/promotionDetails/", status_code=status.HTTP_201_CREATED)
def create_promotion_details(promoDetails: List[PromotionDetailsSchema], db: Session = Depends(get_db)):
    for details in promoDetails:
        db_promoDetails = models.PromotionDetails(
            promotionId=details.promotionId,
            componentId=details.componentId,
            itemId=details.itemId,
            discountType=details.discountType,
            discountValue=details.discountValue
        )
        db.add(db_promoDetails)
        db.commit()
        db.refresh(db_promoDetails)
    return {"message": "Promotion details added successfully!"}


# Database Functions
# def get_valid_hierarchy_values(hierarchy_level: str) -> list:
#     """Fetch valid values for a hierarchy level from itemsMaster"""
#     column_map = {
#         "department": "itemDepartment",
#         "class": "itemClass",
#         "subclass": "itemSubClass"
#     }
    
#     if hierarchy_level.lower() not in column_map:
#         return []
    
#     column = column_map[hierarchy_level.lower()]
#     query = f"SELECT DISTINCT `{column}` FROM itemmaster"
#     result = execute_mysql_query(query)
#     return [str(row[column]) for row in result if row[column]]

# def validate_items(item_ids: list) -> dict:
#     """Validate item IDs against database"""
#     if not item_ids:
#         return {"valid": [], "invalid": []}
    
#     ids_str = ",".join([f"'{id.strip()}'" for id in item_ids])
#     query = f"SELECT itemId FROM itemmaster WHERE itemId IN ({ids_str})"
#     valid_ids = [row["itemId"] for row in execute_mysql_query(query)]
#     invalid_ids = list(set(item_ids) - set(valid_ids))
    
#     return {"valid": valid_ids, "invalid": invalid_ids}

# def validate_items_query(query: str) -> bool:
#     diff_attributes = ['color', 'size', 'material']
#     query_lower = query.lower()
#     for attr in diff_attributes:
#         if f'itemmaster.`{attr}`' in query_lower or f'where {attr} =' in query_lower:
#             return False
#     return True
def query_database_function_promo(question: str, db: Session) -> str:
    # Use the optimized function to extract attribute details (e.g., color, size, or material)
    attr, value, attr_id = find_attributes(question)
    tables = ["itemmaster", "itemsupplier", "itemdiffs"]
    tables_str = ", ".join(tables)
    # If an attribute filter is detected, prepare an optimization hint to use direct equality.
    optimization_hint = ""
    if attr and value and attr_id:
        optimization_hint = (
            f"\n### **Optimization for Attribute Filtering:**\n"
            f"- Detected filter for {attr} with value '{value}'. Use direct equality checks with id {attr_id} "
            f"instead of subqueries, i.e., (im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id}).\n"
        )
    
    promotion_sql_prompt = f"""
    You are a SQL assistant for fashion retail data. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 3 tables in my database, namely:
    - `itemmaster` (aliased as `im`)  
    - `itemsupplier` (aliased as `isup`)  
    - `itemdiffs` (aliased as `idf`)  

    ### **Critical Rules:**  

    #### **1. Color, Size, and Material Filters**  
    - If the user searches for an **attribute** like **color, size, or material**, follow this strict process:  
      1. **Find the `id` in `itemdiffs`** where `diffType` matches the attribute (e.g., 'color') and `diffId` is the user-provided value (e.g., 'Red').  
      2. **Filter `itemmaster` (`im`)** by checking if the retrieved `idf.id` exists in **`im.diffType1`**, **`im.diffType2`**, or **`im.diffType3`**.  
      3. **NEVER** search for colors, sizes, or materials inside `itemmaster` directly.  

    #### **2. Allowed Tables Only**  
    - You may only use the following tables:  
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
        - **`itemId`**: Foreign key linking to `itemsMaster.itemId`, establishing the relationship between items and suppliers.   
    - `itemdiffs` (For Attribute Filtering):
        - **`id`**: Unique identifier for each differentiation type.  
        - **`diffType`**: The attribute type used to differentiate items (e.g., color, size, material).  
          - If the user wants to filter items by a specific attribute (e.g., color or size), the query should check `diffType` and retrieve the corresponding `diffId`.  
        - **`diffId`**: The actual differentiation value corresponding to `diffType`.  
          - Example: If `diffType = 'color'`, then `diffId` could be "Red"; if `diffType = 'size'`, then `diffId` could be "M".

    #### **3. Query Format & Execution**  
    - **Start queries from `itemmaster`** as the primary table.
    - Use **explicit JOINs** when needed (e.g., joining `itemsupplier` for cost-related queries).
    - **Return only valid SQL queries** without additional explanations or markdown formatting.
    - The generated SQL query should be fully structured and dynamically adapt to the user query.
    {optimization_hint}
    ---
    ### **SQL Examples:**

    #### **Example 1: Select All Red Colored Items**
    User: "Select all red colored items"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id}
    ```
    #### **Example 2: Select all red colored items with a supplier cost below $50**
    User: "Select all red colored items with a supplier cost below $50"
    ```sql
    SELECT im.itemId, isup.supplierCost
    FROM itemmaster im
    JOIN itemsupplier isup ON im.itemId = isup.itemId
    WHERE isup.supplierCost < 50
    AND (im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id})
    ```
    #### **Example 3: Select All Items of Size "Large"**
    User: "Select All Items of Size 'Large'"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id})
    ```
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": promotion_sql_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    print("Query response:", response)
    mysql_query = response.choices[0].message.content.strip()
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate the generated SQL query
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result:", result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error:", {str(e)})
        return f"Error executing query: {str(e)}"


# Helper functions
def clean_sql(raw: str) -> str:
    """Remove markdown and validate structure"""
    return raw.replace("```sql", "").replace("```", "").strip()

def format_results(results: list) -> str:
    """Convert results to JSON string"""
    return json.dumps([dict(row) for row in results])

def db_query(query: str, params: dict = None) -> List:
    """Generic database query executor with error handling."""
    try:
        db = next(get_db())
        result = db.execute(text(query), params or {}).fetchall()
        db.close()
        return [row for row in result]
    except Exception as e:
        print(f"Database error: {e}")
        return []

# Optimized core functions
def find_attributes(question: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Find attributes, values, and their ID in a single optimized flow."""
    print(f"\nProcessing query: '{question}'")
    
    # Get attribute metadata in single query
    attr_data = db_query(
        "SELECT diffType, diffId, id FROM itemdiffs GROUP BY diffType, diffId"
    )
    
    if not attr_data:
        print("No attribute data found in database")
        return (None, None, None)

    # Build optimized attribute structure {diffType: {diffId: id}}
    attribute_map = {}
    for diff_type, diff_id, id_val in attr_data:
        attribute_map.setdefault(diff_type, {})[diff_id] = id_val

    print(f"Attribute map: {attribute_map.keys()}")
    
    # Detect attributes using optimized function
    attr, value = detect_attribute(question, attribute_map)
    if not attr or not value:
        return (None, None, None)

    # Get ID from pre-loaded map
    item_id = attribute_map.get(attr, {}).get(value)
    return (attr, value, item_id)

def detect_attribute(question: str, attribute_map: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Optimized attribute detection with caching."""
    if not attribute_map:
        return (None, None)

    try:
        # Build prompt from pre-loaded data
        attr_desc = "\n".join(
            f"- {attr}: {', '.join(values)}" 
            for attr, values in attribute_map.items()
        )

        prompt = f"""Analyze query: "{question}"
        Identify attributes from:
        {attr_desc}
        Respond in JSON format: {{"attribute": "...", "value": "..."}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a retail data analyst. Identify product attributes."
            }, {
                "role": "user", 
                "content": prompt
            }],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)
        attr = result.get('attribute')
        value = result.get('value')

        # Validate against known data
        if not attribute_map.get(attr, {}).get(value):
            print(f"Invalid detection: {attr}/{value}")
            return (None, None)

        return (attr, value)

    except Exception as e:
        print(f"Detection error: {e}")
        return (None, None)



user_promo_details={}
conversation_memory = {}
promo_details_memory = {}

 
class PromotionDetailsChat(BaseModel):
    PromotionType: Optional[str] = None
    HierarchyType: Optional[str] = None
    HierarchyValue: Optional[str] = None
    Brand: Optional[str] = None
    Items: List[str] = []
    ExcludedItems: List[str] = []
    DiscountType: Optional[str] = None
    DiscountValue: Optional[float] = None
    StartDate: Optional[str] = None
    EndDate: Optional[str] = None
    Stores: List[str] = []
    ExcludedLocations: List[str] = []
# -------------------------------
# Database Integration
# -------------------------------
class DatabaseHandler:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    async def execute(self, query: str, context: str) -> Any:
        """Unified database query executor"""
        if context == "validation":
            return self._validate_promotion_data(query)
        return self._handle_general_query(query)
    
    def _validate_promotion_data(self, query: str) -> dict:
        """Structured validation queries (Approach 1 style)"""
        # Implementation of query_database_function_promo
        return {"status": "valid", "details": query}
    
    def _handle_general_query(self, query: str) -> str:
        """Natural language queries (Approach 2 style)"""
        # Implementation of general query handling
        return f"Results for: {query}"

# -------------------------------
# State Management
# -------------------------------
conversation_states = defaultdict(list)
promotion_states = defaultdict(PromotionDetailsChat)
previous_promo_details = defaultdict(dict)

# -------------------------------
# Core Utilities
# -------------------------------
async def classify_query(user_message: str) -> str:
    """Cost-effective query classification"""
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "Classify as 'promotion' or 'general'. Respond with one word."
        }, {
            "role": "user",
            "content": user_message
        }],
        temperature=0.0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

async def extract_promo_details(text: str) -> Dict:
    """Optimized structured extraction (Approach 1)"""
    try:
        response = await async_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[{
                "role": "system",
                "content": "Extract promotion details to JSON using exact keys: "
                           "PromotionType, HierarchyType, HierarchyValue, Brand, "
                           "Items, ExcludedItems, DiscountType, DiscountValue, "
                           "StartDate, EndDate, Stores, ExcludedLocations"
            }, {
                "role": "user",
                "content": text
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Extraction error: {e}")
        return {}

# -------------------------------
# Agent Setup (Approach 2)
# -------------------------------
def init_agent(db: Session):
    tools = [
        Tool(
            name="query_database",
            func=lambda q: DatabaseHandler(db).execute(q, "general"),
            description="Access product database for general queries"
        )
    ]
    return initialize_agent(
        tools,
        llm_gpt4,
        agent="zero-shot-react-description",
        verbose=True
    )

# -------------------------------
# Promotion Template (Common Code)
# -------------------------------
# -------------------------------
# Main Endpoint
# -------------------------------
# @app.post("/promo-chat")
# async def handle_chat(request: ChatRequest, db: Session = Depends(get_db)):
#     user_id = request.user_id
#     user_message = request.message
#     response = {"bot_reply": "", "promo_details": {}, "status": "in_progress"}
    
#     try:
#         # 1. Classify query type
#         query_type = await classify_query(user_message)
        
#         # 2. Handle general queries
#         if query_type != "promotion":
#             agent = init_agent(db)
#             result = agent.invoke({"input": user_message})
#             return {
#                 "bot_reply": result["output"],
#                 "promo_details": {},
#                 "status": "general_response"
#             }
        
#         # 3. Promotion handling flow
#         conversation_states[user_id].append(f"User: {user_message}")
        
#         # 4. Process with GPT-4
#         messages = [{
#             "role": "system",
#             "content": template_Promotion
#         }, {
#             "role": "user",
#             "content": "\n".join(conversation_states[user_id][-5:])
#         }]
        
#         completion = await async_client.chat.completions.create(            
#             model="gpt-4-1106-preview",
#             messages=messages,
#             temperature=0.7,
#             max_tokens=500
#         )
#         bot_reply = completion.choices[0].message.content
        
#         # 5. Handle function calls
#         if completion.choices[0].message.function_call:
#             db_handler = DatabaseHandler(db)
#             args = json.loads(completion.choices[0].message.function_call.arguments)
#             result = await db_handler.execute(args["question"], "validation")
#             # ... additional function handling logic ...
        
#         # 6. Extract and merge promotion details
#         extracted = await categorize_promo_details_fun_call(user_message,user_id)
#         # extracted = await categorize_promo_details(user_message,user_id)
#         # extracted = await extract_promo_details(user_message)
#         # current = promotion_states[user_id].dict()
#         # cleaned_extracted = {
#         # k: v for k, v in extracted.items() 
#         # if v not in [None, "", []] and not (isinstance(v, float) and math.isnan(v))
#         # }
#         # merged = PromotionDetailsChat(**{
#         #     **current,
#         #     **cleaned_extracted
#         # })
#         # merged_dict = merged.dict()
#         # for field in merged_dict:
#         #     if merged_dict[field] is None:
#         #         merged_dict[field] = "" if isinstance(merged_dict[field], str) else None

#         # promotion_states[user_id] = merged
#         # promotion_states[user_id] = PromotionDetailsChat(**merged)
        
#         # 7. Update conversation history
#         conversation_states[user_id].append(f"Bot: {bot_reply}")
        
#         return {
#             "bot_reply": bot_reply,
#             "promo_details": extracted,
#             "status": "submission_pending" if "submit" in bot_reply.lower() else "in_progress",
#             "query type":query_type
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
#Langchain solution
# Default promotion structure
# DEFAULT_PROMO_STRUCTURE = {
#     "Promotion Type": None,
#     "Hierarchy Type": None,
#     "Hierarchy Value": None,
#     "Brand": None,
#     "Items": [],
#     "Excluded Item List": [],
#     "Discount Type": None,
#     "Discount Value": None,
#     "Start Date": None,
#     "End Date": None,
#     "Stores": None,
#     "Excluded Location List": None,
# }

# # Simple caching placeholder for demonstration
# cache: Dict[str, Any] = {}

# # -------------------------------
# # Helper: Text Splitting Function
# # -------------------------------
# def split_long_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
#     """Splits long text into chunks with overlap."""
#     from langchain.text_splitter import RecursiveCharacterTextSplitter
#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
#     return splitter.split_text(text)

# # -------------------------------
# # Promotion Extraction Chain
# # -------------------------------
# promo_extraction_template = PromptTemplate(
#     input_variables=["promotion_text"],
#     template="""
# Extract and structure the following promotion details from the text.
# Return a JSON object that exactly follows these keys:
# - Promotion Type
# - Hierarchy Type
# - Hierarchy Value
# - Brand
# - Items (as an array)
# - Excluded Item List (as an array)
# - Discount Type
# - Discount Value
# - Start Date (dd/mm/yyyy)
# - End Date (dd/mm/yyyy)
# - Stores
# - Excluded Location List

# If a field is missing, return null (or an empty array for Items and Excluded Item List).
# Promotion Text: {promotion_text}
# """,
# )
# promo_extraction_chain = LLMChain(
#     llm=client_new,
#     prompt=promo_extraction_template,
# )

# # -------------------------------
# # Router Chain (Query Classification)
# # -------------------------------
# classification_template = PromptTemplate(
#     input_variables=["user_query"],
#     template="""
# You are a query classifier for a promotion chatbot.
# For example, if the user says "20% off on T-Shirts from H&M", respond with "promotion".
# If the user asks "What are the current promotions?", respond with "general".
# User Query: {user_query}
# """,
# )
# classification_chain = LLMChain(
#     llm=client_new,
#     prompt=classification_template,
# )

# def router_chain(user_query: str):
#     # Pass input as a dictionary and use invoke
#     result = classification_chain.invoke({"user_query": user_query})
#     query_type = result["user_query"].strip().lower()
#     return query_type

# # -------------------------------
# # Agent Initialization for General Queries
# # -------------------------------
# def query_database_tool(query: str):
#     db = next(get_db())
#     return query_database_function_promo(query, db)

# tools = [
#     Tool(
#         name="query_database",
#         func=query_database_tool,
#         description="Execute database queries for general queries"
#     ),
# ]
# agent = initialize_agent(tools, client_new, agent="zero-shot-react-description", verbose=True)

# ############################################
# # (Optional) Parallel Chain Execution Example
# ############################################
# # This function demonstrates how you might run two chains (e.g. extraction and general query) in parallel.
# # Uncomment and modify as needed.
# # def run_parallel_chains(user_query: str, promo_text: str):
# #     parallel_runner = RunnableParallel({
# #         "extraction": promo_extraction_chain, 
# #         "general": general_query_chain
# #     })
# #     # This will run both chains concurrently and return a dict with both outputs.
# #     results = parallel_runner.invoke({"promotion_text": promo_text, "user_query": user_query})
# #     return results

# # -------------------------------
# # FastAPI Endpoint for Promotion Chat
# # -------------------------------
# # -------------------------------
# # FastAPI Endpoint for Promotion Chat
# # -------------------------------
# @app.post("/promo-chat")
# async def handle_promotion_chat(request: ChatRequest, db: Session = Depends(get_db)):
#     user_id = request.user_id
#     user_message = request.message

#     # Initialize user memory if not exists
#     if user_id not in conversation_memory:
#         conversation_memory[user_id] = ConversationBufferWindowMemory(k=5)
#         promo_details_memory[user_id] = DEFAULT_PROMO_STRUCTURE.copy()

#     try:
#         # Check cache for repeated queries
#         if user_message in cache:
#             bot_reply = cache[user_message]
#         else:
#             # Classify query type: promotion update vs general query
#             query_type = router_chain(user_message)

#             if query_type == "promotion":
#                 # Concatenate conversation history
#                 conversation_text = " ".join(conversation_memory[user_id].buffer)
#                 if len(conversation_text) > 2000:
#                     conversation_chunks = split_long_text(conversation_text)
#                     combined_promo_text = f"Current Promotion Details:\n{' '.join(conversation_chunks)}"
#                 else:
#                     combined_promo_text = f"Current Promotion Details:\n{conversation_text}"

#                 # Extract promotion details using asynchronous invoke (ainvoke)
#                 extracted_details_json = await promo_extraction_chain.ainvoke({"promotion_text": combined_promo_text})
#                 print("Extracted Details JSON:", extracted_details_json)
#                 try:
#                     promo_details = json.loads(extracted_details_json)
#                     promo_details_memory[user_id] = promo_details.copy()
#                 except Exception as e:
#                     print(f"JSON Parsing Error: {e}")
#                     promo_details = promo_details_memory[user_id]  # Fallback to stored details

#                 bot_reply = "Promotion details updated successfully."

#             else:
#                 # For general queries, use the agent and convert output to string if needed
#                 with get_openai_callback() as cb:
#                     response = agent.invoke({"input": user_message})
#                     # If the response is an AIMessage object, extract its content.
#                     bot_reply = response.content if hasattr(response, "content") else str(response)
#                     print(f"Total Tokens: {cb.total_tokens}")
#                     print(f"Prompt Tokens: {cb.prompt_tokens}")
#                     print(f"Completion Tokens: {cb.completion_tokens}")
#                     print(f"Total Cost (USD): ${cb.total_cost}")

#             # Cache the reply
#             cache[user_message] = bot_reply

#         # Save conversation history
#         conversation_memory[user_id].save_context({"input": user_message}, {"output": bot_reply})

#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "chat_history": conversation_memory[user_id].buffer,
#             "promo_json": promo_details_memory[user_id],
#             "submissionStatus": "in_progress" if query_type == "promotion" else "n/a",
#             "query_called": True if query_type == "general" else False
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#working old code    
promo_states = defaultdict(dict)
DEFAULT_PROMO_STRUCTURE = {
    "type": "",
    "hierarchy": {"level": "", "value": ""},
    "items": [],
    "excluded_items": [],
    "discount": {"type": "", "amount": 0},
    "dates": {"start": "", "end": ""},
    "locations": [],
    "excluded_locations": [],
    "status": "draft"
}
user_promo_details={}

# @app.post("/promo-chat")
# async def handle_promotion_chat(request: ChatRequest):
#     user_id = request.user_id
#     user_message = request.message

#     # Maintain user session
#     if user_id not in promo_states:
#         promo_states[user_id] = []
#         user_promo_details[user_id] = DEFAULT_PROMO_STRUCTURE.copy()
    
#     promo_states[user_id].append(f"User: {user_message}")
#     conversation = "\n".join(promo_states[user_id])
#     try:
#         messages = [
#             {"role": "system", "content": template_Promotion},
#             {"role": "user", "content": conversation}
#         ]
#         print("Messages first: ",messages)
#         functions = [{
#             "name": "query_database",
#             "description": "Execute database queries for validation/data retrieval",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "question": {
#                         "type": "string", 
#                         "description": "Natural language question requiring database data"}
#                 },
#                 "required": ["question"]
#             }
#         }]

#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             functions=functions,
#             function_call="auto",
#             temperature=0.7,
#             max_tokens=500
#         )
#         print("Response: ",response)
#         response_message = response.choices[0].message
#         bot_reply = response_message.content
#         function_call = response_message.function_call
#         query_called = False
#         db_session = get_db() 

#         # Handle function call
#         if function_call and function_call.name == "query_database":
#             args = json.loads(function_call.arguments)
#             query_result = query_database_function_promo(args["question"],db_session)
#             query_called = True

#             # Append function response to messages
#             messages.append({
#                 "role": "function", 
#                 "name": "query_database",
#                 "content": query_result
#             })
#             print("Messages: ",messages)
#             # Second API call with function result
#             second_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             print("Second Response: ",second_response)
#             bot_reply = second_response.choices[0].message.content

#         # Retain promo_json from previous interaction if query_called is True
#         # if not query_called:
#         #     user_promo_details[user_id] = await categorize_promo_details(bot_reply, user_id)
#         user_promo_details[user_id] = await categorize_promo_details_fun_call(bot_reply, user_id)
#         promo_json = user_promo_details[user_id]  # Assign retained promo_json

#         promo_states[user_id].append(f"Bot: {bot_reply}")
#         # print("Promo JSON:", promo_json, "User ID:", user_id)

#         # Determine submission status
#         if "Would you like to submit" in bot_reply:
#             submissionStatus = "pending"
#         elif "Promotion created successfully" in bot_reply:
#             submissionStatus = "submitted"
#         elif "I want to change something" in bot_reply:
#             submissionStatus = "cancelled"
#         else:
#             submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
#         print("Promotion submission status:", submissionStatus)

#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "chat_history": promo_states,
#             "promo_json": promo_json,  # Retains values if query_called is True
#             "submissionStatus": submissionStatus,
#             "query_called":query_called
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


def classify_query(user_message: str, conversation_context: str) -> str:
    """
    Classify whether the user query is a general query or a promotion details query.
    Returns "General" or "Not General".
    
    The function now uses function calling with GPT-4 to ensure context awareness and to verify if the query
    is intended to fill promotion detail fields:
    "Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", "Discount Value",
    "Start Date", "End Date", "Promotion Type", "Excluded Item List", "Items", "Excluded Location List", "Stores".
    
    If the query focuses on these fields (e.g., "Select all yellow colored items" in a promotion data entry context),
    it is classified as "Not General". Otherwise, queries like "List all the items in this department" are classified as "General".
    """
    prompt = f"""
You are a query classifier. Based on the provided conversation context, decide whether the following query is a general query or a promotion details query.

Conversation Context:
{conversation_context}

User Query:
{user_message}

If the query is intended to fill promotion detail fields (i.e., one or more of:
"Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", "Discount Value", "Start Date", "End Date", "Promotion Type", "Excluded Item List", "Items", "Excluded Location List", "Stores"),
then classify the query as "Not General". Otherwise, if the query is for general inquiry or to retrieve data based on previous context, classify it as "General".

Provide your answer in JSON format as:
{{"query_type": "General"}} or {{"query_type": "Not General"}}
    """
    
    # Define a function schema for robust classification via function calling.
    FUNCTION_SCHEMA_CLASSIFY = {
        "name": "classify_query",
        "description": "Classify the user query as General or Not General based on context and promotion detail fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["General", "Not General"],
                    "description": "The classification of the query."
                }
            },
            "required": ["query_type"]
        }
    }
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-0613",
            messages=[
                {"role": "system", "content": "You are a query classifier."},
                {"role": "user", "content": prompt}
            ],
            functions=[FUNCTION_SCHEMA_CLASSIFY],
            function_call={"name": "classify_query"},
            temperature=0.1,
            max_tokens=50
        )
        
        # Check if a function call response is present.
        message = response.choices[0].message
        if message.get("function_call"):
            arguments = message.function_call.arguments
            result = json.loads(arguments)
            return result.get("query_type", "General")
        else:
            # Fallback: parse the content if function_call is not used.
            result = json.loads(message.content)
            return result.get("query_type", "General")
    except Exception as e:
        logging.error(f"Error in classify_query with function calling: {e}")
        return "General"

# --- Modified handle_promotion_chat Endpoint ---
@app.post("/promo-chat")
async def handle_promotion_chat(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in promo_states:
        promo_states[user_id] = []
        user_promo_details[user_id] = DEFAULT_PROMO_STRUCTURE.copy()
    
    promo_states[user_id].append(f"User: {user_message}")
    conversation = "\n".join(promo_states[user_id])
    try:
        messages = [
            {"role": "system", "content": template_Promotion},
            {"role": "user", "content": conversation}
        ]
        print("Messages first: ",messages)
        functions = [{
            "name": "query_database",
            "description": "Execute database queries for validation/data retrieval",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"}
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )
        print("Response: ",response)
        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False
        db_session = get_db() 

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function_promo(args["question"],db_session)
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })
            print("Messages: ",messages)
            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            print("Second Response: ",second_response)
            bot_reply = second_response.choices[0].message.content

        # Retain promo_json from previous interaction if query_called is True
        # if not query_called:
        #     user_promo_details[user_id] = await categorize_promo_details(bot_reply, user_id)
        user_promo_details[user_id] = await categorize_promo_details_fun_call(bot_reply, user_id)
        promo_json = user_promo_details[user_id]  # Assign retained promo_json

        promo_states[user_id].append(f"Bot: {bot_reply}")
        # print("Promo JSON:", promo_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Promotion created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("Promotion submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": promo_states,
            "promo_json": promo_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus,
            "query_called":query_called
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/promo-chat")
# async def handle_promotion_chat(request: ChatRequest):
#     user_id = request.user_id
#     user_message = request.message

#     # Maintain user session
#     if user_id not in promo_states:
#         promo_states[user_id] = []
#         user_promo_details[user_id] = DEFAULT_PROMO_STRUCTURE.copy()
    
#     promo_states[user_id].append(f"User: {user_message}")
#     conversation = "\n".join(promo_states[user_id])
    
#     # --- Classify the Query ---
#     query_type = classify_query(user_message, conversation)
#     logging.info(f"Query classified as: {query_type}")

#     if query_type == "General":
#         # Process as a general query (e.g. SQL generation for general query)
#         try:
#             db_session = get_db() 
#             query_result = query_database_function_promo(user_message, db_session)
#             promo_states[user_id].append(f"Bot: {query_result}")
#             return {
#                 "user_id": user_id,
#                 "bot_reply": query_result,
#                 "chat_history": promo_states,
#                 "promo_json": user_promo_details[user_id],
#                 "submissionStatus": "in_progress",
#                 "query_type": query_type
#             }
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))
#     else:
#         # Process as a promotion details query using your existing chain with function calling
#         try:
#             messages = [
#                 {"role": "system", "content": template_Promotion},
#                 {"role": "user", "content": conversation}
#             ]
#             functions = [{
#                 "name": "query_database",
#                 "description": "Execute database queries for validation/data retrieval",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "question": {
#                             "type": "string", 
#                             "description": "Natural language question requiring database data"
#                         }
#                     },
#                     "required": ["question"]
#                 }
#             }]

#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 functions=functions,
#                 function_call="auto",
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             response_message = response.choices[0].message
#             bot_reply = response_message.content
#             function_call = response_message.function_call
#             query_called = False
#             db_session = get_db() 

#             if function_call and function_call.name == "query_database":
#                 args = json.loads(function_call.arguments)
#                 query_result = query_database_function_promo(args["question"], db_session)
#                 query_called = True

#                 messages.append({
#                     "role": "function", 
#                     "name": "query_database",
#                     "content": query_result
#                 })

#                 second_response = client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=messages,
#                     temperature=0.7,
#                     max_tokens=500
#                 )
#                 bot_reply = second_response.choices[0].message.content

#             user_promo_details[user_id] = await categorize_promo_details_fun_call(bot_reply, user_id)
#             promo_states[user_id].append(f"Bot: {bot_reply}")

#             # Determine submission status based on bot reply
#             if "Would you like to submit" in bot_reply:
#                 submissionStatus = "pending"
#             elif "Promotion created successfully" in bot_reply:
#                 submissionStatus = "submitted"
#             elif "I want to change something" in bot_reply:
#                 submissionStatus = "cancelled"
#             else:
#                 submissionStatus = "in_progress"

#             return {
#                 "user_id": user_id,
#                 "bot_reply": bot_reply,
#                 "chat_history": promo_states,
#                 "promo_json": user_promo_details[user_id],
#                 "submissionStatus": submissionStatus,
#                 "query_called": query_called,
#                 "query_type": query_type
#             }

#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

#DIFF
# GET all diffs
@app.get("/diffs", response_model=list[ItemDiffsSchema])
def get_diffs(db: Session = Depends(get_db)):
    diffs = db.query(models.ItemDiffs).all()
    return diffs

# POST a new diff
@app.post("/diffs", response_model=ItemDiffsSchema)
def create_diff(diff: ItemDiffsSchema, db: Session = Depends(get_db)):
    db_diff = models.ItemDiffs(**diff.dict())

    # Check if diff already exists
    existing_diff = db.query(models.ItemDiffs).filter(models.ItemDiffs.id == diff.id).first()
    if existing_diff:
        raise HTTPException(status_code=400, detail="Diff with this ID already exists")

    db.add(db_diff)
    db.commit()
    db.refresh(db_diff)
    return db_diff
#ITEM
@app.get("/items", response_model=List[ItemMasterSchema])
def get_items(db: Session = Depends(get_db)):
    items = db.query(models.ItemMaster).all()
    return items

@app.post("/items", response_model=List[ItemMasterSchema])
def create_items(items: List[ItemMasterSchema], db: Session = Depends(get_db)):
    new_items = [models.ItemMaster(**item.dict()) for item in items]
    db.add_all(new_items)
    db.commit()
    return new_items

#ITEM SUPPLIER
@app.post("/itemSuppliers/", response_model=ItemSupplierSchema)
def create_item_supplier(item_supplier: ItemSupplierSchema, db: Session = Depends(get_db)):
    # Optional: Check if the itemSupplier record already exists by supplierId and itemId
    existing_item_supplier = db.query(ItemSupplier).filter(
        ItemSupplier.supplierId == item_supplier.supplierId,
        ItemSupplier.itemId == item_supplier.itemId
    ).first()
    if existing_item_supplier:
        raise HTTPException(status_code=400, detail="ItemSupplier entry already exists")
    
    new_item_supplier = ItemSupplier(
        supplierCost=item_supplier.supplierCost,
        supplierId=item_supplier.supplierId,
        itemId=item_supplier.itemId
    )
    db.add(new_item_supplier)
    db.commit()
    db.refresh(new_item_supplier)
    return new_item_supplier

@app.get("/itemSuppliers/")
def get_item_suppliers(db: Session = Depends(get_db)):
    return db.query(ItemSupplier).all()

@app.get("/itemSuppliers/{id}", response_model=ItemSupplierSchema)
def get_item_supplier_by_id(id: int, db: Session = Depends(get_db)):
    item_supplier = db.query(ItemSupplier).filter(ItemSupplier.id == id).first()
    if item_supplier is None:
        raise HTTPException(status_code=404, detail="ItemSupplier not found")
    return item_supplier


#SHIPMENT 
@app.post("/shipments/", response_model=ShipmentHeader)
def create_shipment(shipment: ShipmentHeader, db: Session = Depends(get_db)):
    db_shipment = models.ShipmentHeader(**shipment.dict())
    db.add(db_shipment)
    db.commit()
    db.refresh(db_shipment)
    return db_shipment

@app.post("/shipments/{receipt_id}/details", response_model=List[ShipmentDetailsSchema])
def add_shipment_details(receipt_id: str, details: List[ShipmentDetailsSchema], db: Session = Depends(get_db)):
    shipment = db.query(models.ShipmentHeader).filter(models.ShipmentHeader.receiptId == receipt_id).first()
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    po_items = {po_detail.itemId for po_detail in db.query(models.PoDetails).filter(models.PoDetails.poId == shipment.poId).all()}
    
    for detail in details:
        if detail.itemId not in po_items:
            raise HTTPException(status_code=400, detail=f"Item {detail.itemId} is not in the PO {shipment.poId}")
    
    db_details = [models.ShipmentDetails(**{**detail.dict(), "receiptId": receipt_id}) for detail in details]
    db.add_all(db_details)
    db.commit()
    for db_detail in db_details:
        db.refresh(db_detail)
        
    return db_details

@app.get("/shipments/{receipt_id}", response_model=Dict[str, Any])
def get_shipment_with_details(receipt_id: str, db: Session = Depends(get_db)):
    shipment = db.query(ShipmentHeader).filter(ShipmentHeader.receiptId == receipt_id).first()
    if shipment is None:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    details = db.query(ShipmentDetails).filter(ShipmentDetails.receiptId == receipt_id).all()
    return {"shipment": shipment, "details": details}


#SUPPPLIER
@app.post("/suppliers/", response_model=SupplierCreate)
def create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    # Check if supplier already exists
    existing_supplier = db.query(Supplier).filter(Supplier.email == supplier.email).first()
    if existing_supplier:
        raise HTTPException(status_code=400, detail="Supplier with this email already exists")

    new_supplier = Supplier(
        supplierId=supplier.supplierId,
        name=supplier.name,
        email=supplier.email,
        phone=supplier.phone,
        address=supplier.address,
        lead_time=supplier.lead_time
    )
    db.add(new_supplier)
    db.commit()
    db.refresh(new_supplier)
    return new_supplier

# Get all suppliers
@app.get("/suppliers/")
def get_suppliers(db: Session = Depends(get_db)):
    return db.query(Supplier).all()

# Get supplier by ID
@app.get("/suppliers/{supplierId}")
def get_supplier(supplierId : str, db: Session = Depends(get_db)):
    supplier = db.query(Supplier).filter(Supplier.supplierId  == supplierId ).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return supplier

#po chat
def execute_mysql_query(query):
    """Executes a MySQL query after validating the table name."""
    try:
        words = query.split()
        if "FROM" in words:
            table_index = words.index("FROM") + 1
            table_name = words[table_index].strip("`;,")  # Extract table name safely
        else:
            return "Invalid SQL query structure."

        # Get existing tables
        valid_tables = get_table_names()
        if table_name not in valid_tables:
            return f"Error: Table `{table_name}` does not exist. Available tables: {valid_tables}"

        # Execute the validated query
        db: Session = next(get_db())
        result = db.execute(text(query)).fetchall()
        db.close()
        return [dict(row._mapping) for row in result]

    except Exception as e:
        return str(e)
def get_table_names():
    """Fetch all table names from the database."""
    try:
        db: Session = next(get_db())
        result = db.execute(text("SHOW TABLES")).fetchall()
        db.close()
        return [list(row)[0] for row in result]  # Extract table names
    except Exception as e:
        return str(e)


def query_database_function(question: str) -> str:
    """Generates and executes an SQL query based on a natural language question."""
    # Fetch available tables
    available_tables = get_table_names()
    tables_str = ", ".join(available_tables)

    # SQL generation prompt (same as original)
    sql_query_prompt = f"""
    The user wants to query the MySQL database. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 7 tables in my database, namely: invoicedetails, invoiceheader, podetails, poheader, suppliers, shipmentHeader, and shipmentDetails.
    - **invoicedetails**: id, itemId, itemQuantity, itemDescription, itemCost, totalItemCost, poId, invoiceNumber.
    - **invoiceheader**: invoiceId, invoicedate, invoiceType, currency, payment_term, invoice_status, total_qty, total_cost, total_tax, total_amount, userInvNo.
    - **podetails**: id, itemId, itemQuantity, supplierId, itemDescription, itemCost, totalItemCost, poId.
    - **poheader**: poNumber, shipByDate, leadTime, estimatedDeliveryDate, totalQuantity, totalCost, totalTax, currency, payment_term.
    - **suppliers**: supplierId, name, email, phone, address, lead_time.
    - **shipmentheader**: receiptId, asnId, expectedDate, receivedDate, receivedBy, status, sourceLocation, destinationLocation, totalQuantity, totalCost, poId.
    - **shipmentdetails**: id, itemId, itemDescription, itemCost, expectedItemQuantity, receivedItemQuantity, damagedItemQuantity, containerId, receiptId, invoiced.

    ### **Field Mapping Rules** 
    - "purchase order", "purchaseorder", "PURCHASE ORDER", "PURCHASEORDER", "Purchase Order", and "PurchaseOrder" refer to the **poheader** table.
    - "PO number", "po_number", "po no", "PONumber", "order number" refer to **poNumber**.
    - "po_id" and "poId" are equivalent and refer to **poId**.
    - "cost" and "total cost" refer to **totalCost**.
    - "invoiceNumber", "invoice Number", "invoice No", "invoiceNo", "invoiceId", and "invoice Id" refer to **userInvNo**.
    - "amount" refers to **totalAmount**.
    - "invoice number" and "invoiceNumber" refer to **invoiceNumber**.
    - "payment terms" and "payment_term" are the same.
    - "lead time" and "lead_time" are the same.
    - "supplier" refers to the **suppliers** table.
    - "supplierId" and "supplier id" refer to **supplierId**.
    - "shipment", "shipment header", "receipt" refer to **shipmentHeader**.
    - "shipment details", "shipment detail", "shipment items" refer to **shipmentDetails**.
    - "receipt id" and "receiptId" refer to **receiptId** in **shipmentHeader** or **shipmentDetails**.
    - "ASN" refers to **asnId**.
    - "expected date" refers to **expectedDate**.
    - "received date" refers to **receivedDate**.
    - "received by" refers to **receivedBy**.
    - "status" refers to **status**.
    - "source location" refers to **sourceLocation**.
    - "destination location" refers to **destinationLocation**.
    - "total quantity" refers to **totalQuantity**.
    - "total cost" refers to **totalCost**.
    - "item id" refers to **itemId**.
    - "item description" refers to **itemDescription**.
    - "item cost" refers to **itemCost**.
    - "expected item quantity" refers to **expectedItemQuantity**.
    - "received item quantity" refers to **receivedItemQuantity**.
    - "damaged item quantity" refers to **damagedItemQuantity**.
    - "container id" refers to **containerId**.
    - "invoiced" refers to **invoiced**.
    -  "quality" and "quality issues" refer to the **percentage of damagedItemQuantity** calculated as:
        **(damagedItemQuantity / receivedItemQuantity) * 100**.
    ### **Scenario for Invoice Listing**
    If the user asks for all invoices of a PO number, generate a query that retrieves all the userInvNo values available for that PO number. To do this, join the invoiceheader table with the invoicedetails table (using invoiceheader.invoiceId = invoicedetails.invoiceNumber) and filter by the given poId.

    ### **Case-Insensitive Handling**
    ### **Handling Special Characters & Spaces**
    """
    sql_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sql_query_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    
    # Extract and clean query
    mysql_query = sql_response.choices[0].message.content.strip()
    print("query_database_function result: ",sql_response)
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate and execute
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result: ",result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error: ",{str(e)})
        return f"Error executing query: {str(e)}"



@app.post("/chat")
async def chat_with_po_assistant(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()
    
    chat_histories[user_id].append(f"User: {user_message}")
    conversation = "\n".join(chat_histories[user_id])

    try:
        # First API call - with function definition
        messages = [
            {"role": "system", "content": template_PO},
            {"role": "user", "content": conversation}
        ]

        functions = [{
            "name": "query_database",
            "description": "Retrieve data from the database using SQL queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"
                    }
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function(args["question"])
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })

            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Retain po_json from previous interaction if query_called is True
        if not query_called:
            user_po_details[user_id] = await categorize_po_details(bot_reply, user_id)

        po_json = user_po_details[user_id]  # Assign retained po_json

        chat_histories[user_id].append(f"Bot: {bot_reply}")
        print("PO JSON:", po_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Purchase Order created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("PO submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": chat_histories,
            "po_json": po_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adduser")
async def add_user(request:UserSchema,db:Session=Depends(get_db)):
    user = User(name=request.name,email=request.email,nickname=request.nickname)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/poCreation/", status_code=status.HTTP_201_CREATED)
def create_po(poHeader: poHeaderCreate, db: Session = Depends(get_db)):
    db_poHeader = models.PoHeader(**poHeader.dict())
    db.add(db_poHeader)
    db.commit()
    db.refresh(db_poHeader)
    return db_poHeader

@app.post("/poDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_poDetail(poDetailData: List[poDetailsCreate], db: Session = Depends(get_db)):
    for details in poDetailData:
        db_poDetails = models.PoDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            supplierId=details.supplierId
        )
        db.add(db_poDetails)
        db.commit()
        db.refresh(db_poDetails)
    return {
        "message":"Items added Sucessfully!"
    }
    # db_poDetails = models.PoDetails(**poDetailData.dict())
    # db.add(db_poDetails)
    # db.commit()
    # db.refresh(db_poDetails)
    # return [db_poDetails]

@app.post("/invCreation/", status_code=status.HTTP_201_CREATED)
def create_invoice(invHeader: invHeaderCreate, db: Session = Depends(get_db)):
    db_invHeader = models.InvHeader(**invHeader.dict())
    db.add(db_invHeader)
    db.commit()
    db.refresh(db_invHeader)
    return db_invHeader

@app.post("/invDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_invDetail(invDetailData: List[invDetailsCreate], db: Session = Depends(get_db)):
    for details in invDetailData:
        db_invDetails = models.InvDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            invoiceNumber= details.invoiceNumber
        )
        db.add(db_invDetails)
        db.commit()
        db.refresh(db_invDetails)
    return {
        "message":"Items added Sucessfully!"
    }

@app.get("/poDetails/{po_id}")
def read_poDeatils(po_id: str, db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        detail = "Po Number is not found in our database! Please add a valid PO number!"
        conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        
    po_info = db.query(models.PoDetails).filter(models.PoDetails.poId == po_id).all()
    return { "po_header":po,"po_details":po_info}

@app.post("/invoiceValidation")
def po_data_validations(po_id:str,detail:Dict[str,int],db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        raise HTTPException(status_code=404, detail="PO is not found!")
    for item,quantity in detail:
        po_details = db.query(models.PoDetails).filter(models.PoDetails.itemId==item).first()
        if po_details is None:
            detail = "Item which you added is not present in this PO"
            conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        if(po_details.itemQuantity>quantity):
            detail = po_details.itemId + "quantity is excced according to PO quantity is" + po_details.itemQuantity
            conversation.append('Bot: ' + detail)
            raise HTTPException(status_code=404, detail=conversation)
    return {"details":conversation}
        
    


@app.get("/invoiceDetails/{inv_id}")
def read_invDeatils(inv_id: str, db: Session = Depends(get_db)):
    inv = db.query(models.InvHeader).filter(models.InvHeader.invoiceId == inv_id).first()
    if inv is None:
        raise HTTPException(status_code=404, detail="Invoice not found!")
    inv_info = db.query(models.InvDetails).filter(models.InvDetails.invoiceNumber == inv_id).all()
    return { "inv_header":inv,"inv_details":inv_info}



@app.post("/ok")  
async def ok_endpoint(query: str, db: Session = Depends(get_db)):
    # Pass the db session to query_database_function_promo
    returned_attributes = query_database_function_promo(query, db)
    return {"message": "ok", "attributes": returned_attributes}

@app.post("/uploadPromo")  
async def upload_promo(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Call the extraction function
        data = await extract_details_gpt_vision(temp_file_path)
        result=await categorize_promo_details(data,"admin")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Remove the temporary file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    return {"structured_data": result}

@app.get
async def findPoDetails(po:str):
        db: Session = Depends(get_db)
        po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po).first()
        if po is None:
            return {"Po not found!"}
        else:
            return {"Po Found!"}


@app.post("/clearData")
async def clearConversation(submitted:str):
    conversation.clear()
    chat_histories.clear()
    previous_invoice_details.clear()
    previous_po_details.clear()
    previous_promo_details.clear()
    submissionStatus = "not submitted"
    return {"conversation":conversation,"submissionStatus":"not submitted","chat_history":chat_histories}

@app.post("/testSubmission")
async def submiission(query:str):
    result=test_submission(query)
    return {"result":result}

@app.post("/uploadGpt/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_invoice_details(extracted_text)

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

@app.post("/uploadPo/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_po_details(extracted_text,"admin")

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

    
@app.post("/uploadOpenAi/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    extracted_text = extract_text_with_openai(file)
    return JSONResponse(content={"extracted_text": extracted_text})


@app.post("/upload/")
async def upload_invoice(file: UploadFile = File(...)):
    """API to upload an invoice file and extract details."""
    if file.content_type not in ["image/png", "image/jpeg", "application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, PDF, and TXT are supported.")

    # Read file bytes
    file_bytes = await file.read()

    # Extract text based on file type
    if file.content_type in ["image/png", "image/jpeg"]:
        image = Image.open(BytesIO(file_bytes))
        extracted_text = extract_text_from_image(image)
    elif file.content_type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
    else:  # Text file
        extracted_text = file_bytes.decode("utf-8")

    # Process extracted text
    invoice_details = extract_invoice_details(extracted_text)
    invoice_data_from_conversation = {
        "quantities": extract_invoice_details(extracted_text).get("quantities", []),
        "items": extract_invoice_details(extracted_text).get("items", [])
    }
    # invoice_json=json.dumps(invoice_details)
    # await generate_response(invoice_details)

    return {"file_name": file.filename, "invoice_details": invoice_details,"invoice_data_from_conversation":invoice_data_from_conversation,"extracted_text":extracted_text}


# @app.post("/creation/response")
# async def generate_response(query:str):
#     conversation.append('User: ' + query)
#     output = gpt_response(query)
#     conversation.append('Bot: ' + output)
#     test_model_reply=testModel(query,output)
#     form_submission=test_submission(output)
#     action='action'
#     submissionStatus="not submitted"
#     past_invoice_regex = r"(past\s+invoice|invoice\s+past|last\s+invoice|invoice\s+last|previous\s+invoice|invoice\s+previous)\s*"
#     invoice_regex = r"(create\s+invoice|invoice\s+create|create\san\sinvoice)\s*"
#     patternInvoice = re.compile(invoice_regex, re.IGNORECASE)
#     pastPatternInvoice = re.compile(past_invoice_regex, re.IGNORECASE)
#     for line in conversation:
#         if line.startswith("User:"):
#             user_input = line.split(":")[1].strip().lower()
#             if re.search(pastPatternInvoice, user_input):
#                 action = "last invoice created"
#                 # break
#             elif re.search(patternInvoice, user_input):
#                 action = "create invoice"
#             elif "create po" in user_input:
#                 action = "create PO"

#     invoiceDatafromConversation=collect_invoice_data(conversation)
#     invDetails=await categorize_invoice_details_new(conversation,"admin")
#     print("invDetails",invDetails)
#     return {"conversation":conversation,"invoice_json":invDetails,"action":action,
#     "submissionStatus":form_submission,"invoiceDatafromConversation":invDetails,
#     "test_model_reply":test_model_reply }
   
invoice_chat_histories = {}
user_invoice_details = {}


# @app.post("/creation/response") 
# async def generate_response(request: ChatRequest):
#     user_id = request.user_id
#     user_message = request.message
#     po_validated = False
    
#     # Maintain user session for invoice creation
#     if user_id not in invoice_chat_histories:
#         invoice_chat_histories[user_id] = []
#         user_invoice_details[user_id] = {'po_validated': False, 'po_data': {}}
    
#     invoice_chat_histories[user_id].append(f"User: {user_message}")
#     conversation = "\n".join(invoice_chat_histories[user_id])

#     try:
#         # Enhanced system prompt with PO validation emphasis
#         messages = [
#             {"role": "system", "content": f"{template_5}\nFIRST PRIORITY: Always check for and validate PO numbers if mentioned."},
#             {"role": "user", "content": conversation}
#         ]

#         # Define prioritized functions
#         functions = [
#             {
#                 "name": "validate_po",
#                 "description": "MUST CALL THIS FIRST if user mentions a PO number. Validates PO and retrieves items.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "po_number": {
#                             "type": "string",
#                             "description": "Exact PO number from user message (e.g. PO123, purchase order 456)"
#                         }
#                     },
#                     "required": ["po_number"]
#                 }
#             },
#             {
#                 "name": "query_database",
#                 "description": "For general database queries not related to PO validation",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "question": {
#                             "type": "string", 
#                             "description": "Natural language question requiring database data"
#                         }
#                     },
#                     "required": ["question"]
#                 }
#             }
#         ]

#         # First try to force PO validation
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             functions=functions,
#             function_call={"name": "validate_po"},
#             temperature=0.7,
#             max_tokens=500
#         )

#         response_message = response.choices[0].message
#         function_call = response_message.function_call

#         # Fallback to auto if no PO detected
#         if not function_call:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 functions=functions,
#                 function_call="auto",
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             response_message = response.choices[0].message

#         bot_reply = response_message.content
#         function_call = response_message.function_call
#         query_called = False

#         # Handle PO validation first
#         if function_call and function_call.name == "validate_po":
#             args = json.loads(function_call.arguments)
#             po_number = args["po_number"]
            
#             # Fetch and store PO items
#             items = fetch_po_items(po_number)
#             user_invoice_details[user_id]['po_data'] = {
#                 'po_number': po_number,
#                 'items': items,
#                 'validated_at': datetime.now().isoformat()
#             }
#             user_invoice_details[user_id]['po_validated'] = True
#             po_validated = True

#             # Build validation response
#             content = json.dumps({
#                 "status": "validated",
#                 "po_number": po_number,
#                 "item_count": len(items),
#                 "items_sample": items[:3] if len(items) > 3 else items
#             }, default=str)

#             messages.append({
#                 "role": "function",
#                 "name": "validate_po",
#                 "content": content
#             })

#             # Generate final response with PO context
#             second_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             bot_reply = second_response.choices[0].message.content

#         elif function_call and function_call.name == "query_database":
#             args = json.loads(function_call.arguments)
#             query_result = query_database_function(args["question"])
#             query_called = True

#             messages.append({
#                 "role": "function",
#                 "name": "query_database",
#                 "content": query_result
#             })

#             second_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             bot_reply = second_response.choices[0].message.content

#         # Update conversation history
#         invoice_chat_histories[user_id].append(f"Bot: {bot_reply}")

#         # Only extract details if no PO validation occurred
#         if not po_validated and not query_called:
#             user_invoice_details[user_id] = await categorize_invoice_details_new(bot_reply, user_id)

#         # Merge PO data with invoice details
#         if po_validated:
#             user_invoice_details[user_id].setdefault('invoice_data', {})
#             user_invoice_details[user_id]['invoice_data'].update({
#                 'po_number': user_invoice_details[user_id]['po_data']['po_number'],
#                 'po_items': user_invoice_details[user_id]['po_data']['items']
#             })

#         # Ensure final validated status
#         po_validated = user_invoice_details[user_id].get('po_validated', False)

#         # Determine submission status
#         submission_status = "in_progress"
#         if "Would you like to submit" in bot_reply:
#             submission_status = "pending"
#         elif "Invoice created successfully" in bot_reply:
#             submission_status = "submitted"
#         elif "I want to change something" in bot_reply:
#             submission_status = "cancelled"

#         # Action detection
#         action = 'create invoice' if 'create invoice' in user_message.lower() else 'other'
        
#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "chat_history": invoice_chat_histories[user_id],
#             "invoice_json": user_invoice_details[user_id].get('invoice_data', {}),
#             "action": action,
#             "submissionStatus": submission_status,
#             "test_model_reply": testModel(user_message, bot_reply),
#             "invoiceDatafromConversation": user_invoice_details[user_id],
#             "po_validated": po_validated,
#             "validated_po": user_invoice_details[user_id].get('po_data', {})
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

def fetch_po_items(po_number: str) -> list:
    """Fetch PO items with error handling"""
    try:
        query = f"""
        SELECT itemId, itemQuantity, itemDescription, itemCost 
        FROM podetails 
        WHERE poId = '{po_number}'
        """
        result = execute_mysql_query(query)
        return [dict(item) for item in result] if result else []
    except Exception as e:
        print(f"PO fetch error: {str(e)}")
        return []

@app.post("/creation/response") 
async def generate_response(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session for invoice creation
    if user_id not in invoice_chat_histories:
        invoice_chat_histories[user_id] = []
        user_invoice_details[user_id] = {}
    
    invoice_chat_histories[user_id].append(f"User: {user_message}")
    conversation = "\n".join(invoice_chat_histories[user_id])

    try:
        # Prepare messages with system template
        messages = [
            {"role": "system", "content": template_5},
            {"role": "user", "content": conversation}
        ]

        # Define available functions
        functions = [{
            "name": "query_database",
            "description": "Retrieve data from the database using SQL queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"
                    }
                },
                "required": ["question"]
            }
        }]

        # First API call with function definition
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function(args["question"])
            query_called = True

            # Append function response and make second call
            messages.append({
                "role": "function",
                "name": "query_database",
                "content": query_result
            })

            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Update conversation history
        invoice_chat_histories[user_id].append(f"Bot: {bot_reply}")

        # Extract invoice details if no function was called
        if not query_called:
            user_invoice_details[user_id] = await categorize_invoice_details_new(bot_reply, user_id)

        inv_details = user_invoice_details[user_id]
        print("invDetails",inv_details)
        po_items=fetch_po_items(inv_details["PO Number"])

        # Determine submission status
        submissionStatus = "not submitted"
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Invoice created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"

        # Additional processing from original implementation
        test_model_reply = testModel(user_message, bot_reply)
        form_submission = test_submission(bot_reply)
        
        # Determine action type
        action = 'action'
        past_pattern = re.compile(r"(past\s+invoice|invoice\s+past|last\s+invoice)", re.IGNORECASE)
        create_pattern = re.compile(r"(create\s+invoice|invoice\s+create)", re.IGNORECASE)
        
        for line in invoice_chat_histories[user_id]:
            if line.startswith("User:"):
                user_input = line.split(":")[1].strip()
                if past_pattern.search(user_input):
                    action = "last invoice created"
                elif create_pattern.search(user_input):
                    action = "create invoice"
                elif "create po" in user_input.lower():
                    action = "create PO"
        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": invoice_chat_histories[user_id],
            "conversation": invoice_chat_histories[user_id],
            "invoice_json": inv_details,
            "action": action,
            "submissionStatus":form_submission,
            "po_items":po_items,
            # "submissionStatus": submissionStatus,
            "test_model_reply": test_model_reply,
            "invoiceDatafromConversation":inv_details,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
#16 MArch 2025- Testing dfferent variations of LLM enhancements and classification
#10 March 2025 -Soln where PO id validated and items are fetched using GPT 4o logic
@app.post("/creation/response")
async def generate_response(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session for invoice creation
    if user_id not in invoice_chat_histories:
        invoice_chat_histories[user_id] = []
        user_invoice_details[user_id] = {}
    
    invoice_chat_histories[user_id].append(f"User: {user_message}")
    conversation = "\n".join(invoice_chat_histories[user_id])

    try:
        # Prepare messages with system template
        messages = [
            {"role": "system", "content": template_5},
            {"role": "user", "content": conversation}
        ]

        # Define available functions
        functions = [{
            "name": "query_database",
            "description": "Retrieve data from the database using SQL queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"
                    }
                },
                "required": ["question"]
            }
        }]

        # First API call with function definition
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            result = query_database_function(args["question"])
            print("generate query result: ", result["query"])
            query_result = result["result"]
            result_dict = json.loads(query_result)
            query_called = True

            # Extract generated SQL
            generated_sql =result["query"]

            # GPT-4o validation check for PO number filter
            is_po_validation = False
            if generated_sql:
                validation_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Analyze if the SQL query contains a WHERE clause that filters by a single purchase order identifier "
                            "using either the 'ponumber' or 'poid' column. Consider case-insensitive matches and various SQL formats."
                        )
                    },
                    {
                        "role": "user",
                        "content": generated_sql
                    }
                ]

                po_validation_function = [{
                    "name": "check_po_validation",
                    "description": (
                        "Determine if the SQL query validates by filtering on a single purchase order identifier using either "
                        "'ponumber' or 'poid'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_po_validation": {
                                "type": "boolean",
                                "description": (
                                    "True if the query filters using a single condition on either 'ponumber' or 'poid', "
                                    "False otherwise."
                                )
                            }
                        },
                        "required": ["is_po_validation"]
                    }
                }]

                try:
                    validation_response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=validation_messages,
                        functions=po_validation_function,
                        function_call={"name": "check_po_validation"},
                        temperature=0,
                        max_tokens=50
                    )
                    print("PO validation fn")
                    func_call = validation_response.choices[0].message.function_call
                    if func_call and func_call.name == "check_po_validation":
                        args = json.loads(func_call.arguments)
                        is_po_validation = args.get("is_po_validation", False)
                        print("ispoval:",is_po_validation)

                except Exception as e:
                    print(f"PO validation check error: {str(e)}")

            # Bypass function call flow for PO validation
            if is_po_validation:
                query_called = False

            # Append function response and make second call
            messages.append({
                "role": "function",
                "name": "query_database",
                "content": query_result
            })

            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Update conversation history
        invoice_chat_histories[user_id].append(f"Bot: {bot_reply}")

        # Execute invoice detail categorization if needed
        if not query_called:
            user_invoice_details[user_id] = await categorize_invoice_details_new(bot_reply, user_id)

        inv_details = user_invoice_details[user_id]
        print("invDetails", inv_details)

        # Determine submission status
        submissionStatus = "not submitted"
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Invoice created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"

        # Additional processing
        test_model_reply = testModel(user_message, bot_reply)
        form_submission = test_submission(bot_reply)
        
        # Determine action type
        action = 'action'
        past_pattern = re.compile(r"(past\s+invoice|invoice\s+past|last\s+invoice)", re.IGNORECASE)
        create_pattern = re.compile(r"(create\s+invoice|invoice\s+create)", re.IGNORECASE)
        
        for line in invoice_chat_histories[user_id]:
            if line.startswith("User:"):
                user_input = line.split(":", 1)[1].strip()
                if past_pattern.search(user_input):
                    action = "last invoice created"
                elif create_pattern.search(user_input):
                    action = "create invoice"
                elif "create po" in user_input.lower():
                    action = "create PO"
                    
        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": invoice_chat_histories[user_id],
            "conversation": invoice_chat_histories[user_id],
            "invoice_json": inv_details,
            "action": action,
            "submissionStatus": form_submission,
            "test_model_reply": test_model_reply,
            "invoiceDatafromConversation": inv_details,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def query_database_function(question: str) -> str:
    """Generates and executes an SQL query based on a natural language question."""
    # Fetch available tables
    available_tables = get_table_names()
    tables_str = ", ".join(available_tables)

    # SQL generation prompt (same as original)
    sql_query_prompt = f"""
    The user wants to query the MySQL database. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 7 tables in my database, namely: invoicedetails, invoiceheader, podetails, poheader, suppliers, shipmentHeader, and shipmentDetails.
    - **invoicedetails**: id, itemId, itemQuantity, itemDescription, itemCost, totalItemCost, poId, invoiceNumber.
    - **invoiceheader**: invoiceId, invoicedate, invoiceType, currency, payment_term, invoice_status, total_qty, total_cost, total_tax, total_amount, userInvNo.
    - **podetails**: id, itemId, itemQuantity, supplierId, itemDescription, itemCost, totalItemCost, poId.
    - **poheader**: poNumber, shipByDate, leadTime, estimatedDeliveryDate, totalQuantity, totalCost, totalTax, currency, payment_term.
    - **suppliers**: supplierId, name, email, phone, address, lead_time.
    - **shipmentheader**: receiptId, asnId, expectedDate, receivedDate, receivedBy, status, sourceLocation, destinationLocation, totalQuantity, totalCost, poId.
    - **shipmentdetails**: id, itemId, itemDescription, itemCost, expectedItemQuantity, receivedItemQuantity, damagedItemQuantity, containerId, receiptId, invoiced.

    ### **Field Mapping Rules** 
    - "purchase order", "purchaseorder", "PURCHASE ORDER", "PURCHASEORDER", "Purchase Order", and "PurchaseOrder" refer to the **poheader** table.
    - "PO number", "po_number", "po no", "PONumber", "order number" refer to **poNumber**.
    - "po_id" and "poId" are equivalent and refer to **poId**.
    - "cost" and "total cost" refer to **totalCost**.
    - "invoiceNumber", "invoice Number", "invoice No", "invoiceNo", "invoiceId", and "invoice Id" refer to **userInvNo**.
    - "amount" refers to **totalAmount**.
    - "invoice number" and "invoiceNumber" refer to **invoiceNumber**.
    - "payment terms" and "payment_term" are the same.
    - "lead time" and "lead_time" are the same.
    - "supplier" refers to the **suppliers** table.
    - "supplierId" and "supplier id" refer to **supplierId**.
    - "shipment", "shipment header", "receipt" refer to **shipmentHeader**.
    - "shipment details", "shipment detail", "shipment items" refer to **shipmentDetails**.
    - "receipt id" and "receiptId" refer to **receiptId** in **shipmentHeader** or **shipmentDetails**.
    - "ASN" refers to **asnId**.
    - "expected date" refers to **expectedDate**.
    - "received date" refers to **receivedDate**.
    - "received by" refers to **receivedBy**.
    - "status" refers to **status**.
    - "source location" refers to **sourceLocation**.
    - "destination location" refers to **destinationLocation**.
    - "total quantity" refers to **totalQuantity**.
    - "total cost" refers to **totalCost**.
    - "item id" refers to **itemId**.
    - "item description" refers to **itemDescription**.
    - "item cost" refers to **itemCost**.
    - "expected item quantity" refers to **expectedItemQuantity**.
    - "received item quantity" refers to **receivedItemQuantity**.
    - "damaged item quantity" refers to **damagedItemQuantity**.
    - "container id" refers to **containerId**.
    - "invoiced" refers to **invoiced**.
    -  "quality" and "quality issues" refer to the **percentage of damagedItemQuantity** calculated as:
        **(damagedItemQuantity / receivedItemQuantity) * 100**.
    ### **Scenario for Invoice Listing**
    If the user asks for all invoices of a PO number, generate a query that retrieves all the userInvNo values available for that PO number. To do this, join the invoiceheader table with the invoicedetails table (using invoiceheader.invoiceId = invoicedetails.invoiceNumber) and filter by the given poId.
    
    ## **Scenario for PO Number Validation**
    If the user provides a PO number (e.g., "PO123"), generate a query that validates the PO number by checking its existence in the poheader table and returns all itemId values from the podetails table for that PO number.

    ### **Case-Insensitive Handling**
    ### **Handling Special Characters & Spaces**
    """
    # Generate SQL query
    sql_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sql_query_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    
    # Extract and clean query
    mysql_query = sql_response.choices[0].message.content.strip()
    print("query_database_function result: ",sql_response)
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate and execute
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result: ",result)
        query=mysql_query
        result=json.dumps(result, default=str)
        print("query and result",query,result)
        return {"query":query,"result":result}
    except Exception as e:
        print("Query error: ",{str(e)})
        return f"Error executing query: {str(e)}"
#10 March 2025
# 09 March 2025
from fastapi import FastAPI,Depends,HTTPException,status
from utils import gpt_response,extract_information,conversation,checkValidation,run_conversation,collect_invoice_data,bot_action,openaifunction,testModel,test_submission,extract_invoice_details,extract_text_from_pdf,extract_text_from_image,collect_invoice_data_from_file,extract_text_with_openai,categorize_invoice_details,client,categorize_invoice_details_new,previous_invoice_details
from poUtils import template_PO,DEFAULT_PO_STRUCTURE,categorize_po_details,previous_po_details
from pydantic import BaseModel;
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails,Supplier,ShipmentHeader,ItemDiffs,ItemSupplier,ItemMaster,PromotionDetails,PromotionHeader
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach,ChatRequest,SupplierCreate,ShipmentHeader,ShipmentDetails,ItemDiffsSchema,ItemSupplierSchema,ItemMasterSchema,PromotionDetailsSchema,PromotionHeaderSchema
from schemas import ShipmentDetails as ShipmentDetailsSchema
from database import engine,SessionLocal,get_db
from promoUtils import template_Promotion,categorize_promo_details,previous_promo_details
from sqlalchemy.orm import Session
import models
from typing import List,Any
from fastapi.middleware.cors import CORSMiddleware
from utils_extractor import run_conversation
from typing import Dict,Tuple, Optional
import easyocr
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import openai
# from exampleextrator import run_conversation
from fastapi.responses import JSONResponse
import mysql.connector
Base.metadata.create_all(bind=engine)
from sqlalchemy.sql import text
import re;
import json;
import os
import signal
from collections import defaultdict

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#PO Chatbot Functions
chat_histories = {}
user_po_details = {}


#Promo
@app.get("/promotionHeader/{promotionId}", response_model=PromotionHeaderSchema)
def get_promotion_header(promotionId: str, db: Session = Depends(get_db)):
    promoHeader = db.query(models.PromotionHeader).filter(models.PromotionHeader.promotionId == promotionId).first()
    if not promoHeader:
        raise HTTPException(status_code=404, detail="Promotion not found")
    return promoHeader
@app.get("/promotionDetails/{promotionId}", response_model=List[PromotionDetailsSchema])
def get_promotion_details(promotionId: str, db: Session = Depends(get_db)):
    promoDetails = db.query(models.PromotionDetails).filter(models.PromotionDetails.promotionId == promotionId).all()
    if not promoDetails:
        raise HTTPException(status_code=404, detail="No details found for this promotion")
    return promoDetails
@app.post("/promotionHeader/", status_code=status.HTTP_201_CREATED)
def create_promotion_header(promoHeader:PromotionHeaderSchema, db: Session = Depends(get_db)):
    db_promoHeader = models.PromotionHeader(
        promotionId=promoHeader.promotionId,
        componentId=promoHeader.componentId,
        startDate=promoHeader.startDate,
        endDate=promoHeader.endDate,
        promotionType=promoHeader.promotionType
    )
    db.add(db_promoHeader)
    db.commit()
    db.refresh(db_promoHeader)
    return db_promoHeader
@app.post("/promotionDetails/", status_code=status.HTTP_201_CREATED)
def create_promotion_details(promoDetails: List[PromotionDetailsSchema], db: Session = Depends(get_db)):
    for details in promoDetails:
        db_promoDetails = models.PromotionDetails(
            promotionId=details.promotionId,
            componentId=details.componentId,
            itemId=details.itemId,
            discountType=details.discountType,
            discountValue=details.discountValue
        )
        db.add(db_promoDetails)
        db.commit()
        db.refresh(db_promoDetails)
    return {"message": "Promotion details added successfully!"}


# Database Functions
# def get_valid_hierarchy_values(hierarchy_level: str) -> list:
#     """Fetch valid values for a hierarchy level from itemsMaster"""
#     column_map = {
#         "department": "itemDepartment",
#         "class": "itemClass",
#         "subclass": "itemSubClass"
#     }
    
#     if hierarchy_level.lower() not in column_map:
#         return []
    
#     column = column_map[hierarchy_level.lower()]
#     query = f"SELECT DISTINCT `{column}` FROM itemmaster"
#     result = execute_mysql_query(query)
#     return [str(row[column]) for row in result if row[column]]

# def validate_items(item_ids: list) -> dict:
#     """Validate item IDs against database"""
#     if not item_ids:
#         return {"valid": [], "invalid": []}
    
#     ids_str = ",".join([f"'{id.strip()}'" for id in item_ids])
#     query = f"SELECT itemId FROM itemmaster WHERE itemId IN ({ids_str})"
#     valid_ids = [row["itemId"] for row in execute_mysql_query(query)]
#     invalid_ids = list(set(item_ids) - set(valid_ids))
    
#     return {"valid": valid_ids, "invalid": invalid_ids}

# def validate_items_query(query: str) -> bool:
#     diff_attributes = ['color', 'size', 'material']
#     query_lower = query.lower()
#     for attr in diff_attributes:
#         if f'itemmaster.`{attr}`' in query_lower or f'where {attr} =' in query_lower:
#             return False
#     return True
def query_database_function_promo(question: str, db: Session) -> str:
    # Use the optimized function to extract attribute details (e.g., color, size, or material)
    attr, value, attr_id = find_attributes(question)
    tables = ["itemmaster", "itemsupplier", "itemdiffs"]
    tables_str = ", ".join(tables)
    # If an attribute filter is detected, prepare an optimization hint to use direct equality.
    optimization_hint = ""
    if attr and value and attr_id:
        optimization_hint = (
            f"\n### **Optimization for Attribute Filtering:**\n"
            f"- Detected filter for {attr} with value '{value}'. Use direct equality checks with id {attr_id} "
            f"instead of subqueries, i.e., (im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id}).\n"
        )
    
    promotion_sql_prompt = f"""
    You are a SQL assistant for fashion retail data. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 3 tables in my database, namely:
    - `itemmaster` (aliased as `im`)  
    - `itemsupplier` (aliased as `isup`)  
    - `itemdiffs` (aliased as `idf`)  

    ### **Critical Rules:**  

    #### **1. Color, Size, and Material Filters**  
    - If the user searches for an **attribute** like **color, size, or material**, follow this strict process:  
      1. **Find the `id` in `itemdiffs`** where `diffType` matches the attribute (e.g., 'color') and `diffId` is the user-provided value (e.g., 'Red').  
      2. **Filter `itemmaster` (`im`)** by checking if the retrieved `idf.id` exists in **`im.diffType1`**, **`im.diffType2`**, or **`im.diffType3`**.  
      3. **NEVER** search for colors, sizes, or materials inside `itemmaster` directly.  

    #### **2. Allowed Tables Only**  
    - You may only use the following tables:  
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
        - **`itemId`**: Foreign key linking to `itemsMaster.itemId`, establishing the relationship between items and suppliers.   
    - `itemdiffs` (For Attribute Filtering):
        - **`id`**: Unique identifier for each differentiation type.  
        - **`diffType`**: The attribute type used to differentiate items (e.g., color, size, material).  
          - If the user wants to filter items by a specific attribute (e.g., color or size), the query should check `diffType` and retrieve the corresponding `diffId`.  
        - **`diffId`**: The actual differentiation value corresponding to `diffType`.  
          - Example: If `diffType = 'color'`, then `diffId` could be "Red"; if `diffType = 'size'`, then `diffId` could be "M".

    #### **3. Query Format & Execution**  
    - **Start queries from `itemmaster`** as the primary table.
    - Use **explicit JOINs** when needed (e.g., joining `itemsupplier` for cost-related queries).
    - **Return only valid SQL queries** without additional explanations or markdown formatting.
    - The generated SQL query should be fully structured and dynamically adapt to the user query.
    {optimization_hint}
    ---
    ### **SQL Examples:**

    #### **Example 1: Select All Red Colored Items**
    User: "Select all red colored items"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id}
    ```
    #### **Example 2: Select all red colored items with a supplier cost below $50**
    User: "Select all red colored items with a supplier cost below $50"
    ```sql
    SELECT im.itemId, isup.supplierCost
    FROM itemmaster im
    JOIN itemsupplier isup ON im.itemId = isup.itemId
    WHERE isup.supplierCost < 50
    AND (im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id})
    ```
    #### **Example 3: Select All Items of Size "Large"**
    User: "Select All Items of Size 'Large'"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 = {attr_id} OR im.diffType2 = {attr_id} OR im.diffType3 = {attr_id})
    ```
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": promotion_sql_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    print("Query response:", response)
    mysql_query = response.choices[0].message.content.strip()
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate the generated SQL query
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result:", result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error:", {str(e)})
        return f"Error executing query: {str(e)}"


# Helper functions
def clean_sql(raw: str) -> str:
    """Remove markdown and validate structure"""
    return raw.replace("```sql", "").replace("```", "").strip()

def format_results(results: list) -> str:
    """Convert results to JSON string"""
    return json.dumps([dict(row) for row in results])

def db_query(query: str, params: dict = None) -> List:
    """Generic database query executor with error handling."""
    try:
        db = next(get_db())
        result = db.execute(text(query), params or {}).fetchall()
        db.close()
        return [row for row in result]
    except Exception as e:
        print(f"Database error: {e}")
        return []

# Optimized core functions
def find_attributes(question: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Find attributes, values, and their ID in a single optimized flow."""
    print(f"\nProcessing query: '{question}'")
    
    # Get attribute metadata in single query
    attr_data = db_query(
        "SELECT diffType, diffId, id FROM itemdiffs GROUP BY diffType, diffId"
    )
    
    if not attr_data:
        print("No attribute data found in database")
        return (None, None, None)

    # Build optimized attribute structure {diffType: {diffId: id}}
    attribute_map = {}
    for diff_type, diff_id, id_val in attr_data:
        attribute_map.setdefault(diff_type, {})[diff_id] = id_val

    print(f"Attribute map: {attribute_map.keys()}")
    
    # Detect attributes using optimized function
    attr, value = detect_attribute(question, attribute_map)
    if not attr or not value:
        return (None, None, None)

    # Get ID from pre-loaded map
    item_id = attribute_map.get(attr, {}).get(value)
    return (attr, value, item_id)

def detect_attribute(question: str, attribute_map: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Optimized attribute detection with caching."""
    if not attribute_map:
        return (None, None)

    try:
        # Build prompt from pre-loaded data
        attr_desc = "\n".join(
            f"- {attr}: {', '.join(values)}" 
            for attr, values in attribute_map.items()
        )

        prompt = f"""Analyze query: "{question}"
        Identify attributes from:
        {attr_desc}
        Respond in JSON format: {{"attribute": "...", "value": "..."}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a retail data analyst. Identify product attributes."
            }, {
                "role": "user", 
                "content": prompt
            }],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)
        attr = result.get('attribute')
        value = result.get('value')

        # Validate against known data
        if not attribute_map.get(attr, {}).get(value):
            print(f"Invalid detection: {attr}/{value}")
            return (None, None)

        return (attr, value)

    except Exception as e:
        print(f"Detection error: {e}")
        return (None, None)


promo_states = defaultdict(dict)
DEFAULT_PROMO_STRUCTURE = {
    "type": "",
    "hierarchy": {"level": "", "value": ""},
    "items": [],
    "excluded_items": [],
    "discount": {"type": "", "amount": 0},
    "dates": {"start": "", "end": ""},
    "locations": [],
    "excluded_locations": [],
    "status": "draft"
}

# Modified API endpoint
user_promo_details={}
@app.post("/promo-chat")
async def handle_promotion_chat(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in promo_states:
        promo_states[user_id] = []
        user_promo_details[user_id] = DEFAULT_PROMO_STRUCTURE.copy()
    
    promo_states[user_id].append(f"User: {user_message}")
    conversation = "\n".join(promo_states[user_id])
    try:
        messages = [
            {"role": "system", "content": template_Promotion},
            {"role": "user", "content": conversation}
        ]

        functions = [{
            "name": "query_database",
            "description": "Execute database queries for validation/data retrieval",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"}
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False
        db_session = get_db() 

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function_promo(args["question"],db_session)
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })

            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Retain promo_json from previous interaction if query_called is True
        # if not query_called:
        #     user_promo_details[user_id] = await categorize_promo_details(bot_reply, user_id)
        user_promo_details[user_id] = await categorize_promo_details(bot_reply, user_id)
        promo_json = user_promo_details[user_id]  # Assign retained promo_json

        promo_states[user_id].append(f"Bot: {bot_reply}")
        # print("Promo JSON:", promo_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Promotion created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("Promotion submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": promo_states,
            "promo_json": promo_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus,
            "query_called":query_called
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#DIFF
# GET all diffs
@app.get("/diffs", response_model=list[ItemDiffsSchema])
def get_diffs(db: Session = Depends(get_db)):
    diffs = db.query(models.ItemDiffs).all()
    return diffs

# POST a new diff
@app.post("/diffs", response_model=ItemDiffsSchema)
def create_diff(diff: ItemDiffsSchema, db: Session = Depends(get_db)):
    db_diff = models.ItemDiffs(**diff.dict())

    # Check if diff already exists
    existing_diff = db.query(models.ItemDiffs).filter(models.ItemDiffs.id == diff.id).first()
    if existing_diff:
        raise HTTPException(status_code=400, detail="Diff with this ID already exists")

    db.add(db_diff)
    db.commit()
    db.refresh(db_diff)
    return db_diff
#ITEM
@app.get("/items", response_model=List[ItemMasterSchema])
def get_items(db: Session = Depends(get_db)):
    items = db.query(models.ItemMaster).all()
    return items

@app.post("/items", response_model=List[ItemMasterSchema])
def create_items(items: List[ItemMasterSchema], db: Session = Depends(get_db)):
    new_items = [models.ItemMaster(**item.dict()) for item in items]
    db.add_all(new_items)
    db.commit()
    return new_items

#ITEM SUPPLIER
@app.post("/itemSuppliers/", response_model=ItemSupplierSchema)
def create_item_supplier(item_supplier: ItemSupplierSchema, db: Session = Depends(get_db)):
    # Optional: Check if the itemSupplier record already exists by supplierId and itemId
    existing_item_supplier = db.query(ItemSupplier).filter(
        ItemSupplier.supplierId == item_supplier.supplierId,
        ItemSupplier.itemId == item_supplier.itemId
    ).first()
    if existing_item_supplier:
        raise HTTPException(status_code=400, detail="ItemSupplier entry already exists")
    
    new_item_supplier = ItemSupplier(
        supplierCost=item_supplier.supplierCost,
        supplierId=item_supplier.supplierId,
        itemId=item_supplier.itemId
    )
    db.add(new_item_supplier)
    db.commit()
    db.refresh(new_item_supplier)
    return new_item_supplier

@app.get("/itemSuppliers/")
def get_item_suppliers(db: Session = Depends(get_db)):
    return db.query(ItemSupplier).all()

@app.get("/itemSuppliers/{id}", response_model=ItemSupplierSchema)
def get_item_supplier_by_id(id: int, db: Session = Depends(get_db)):
    item_supplier = db.query(ItemSupplier).filter(ItemSupplier.id == id).first()
    if item_supplier is None:
        raise HTTPException(status_code=404, detail="ItemSupplier not found")
    return item_supplier


#SHIPMENT 
@app.post("/shipments/", response_model=ShipmentHeader)
def create_shipment(shipment: ShipmentHeader, db: Session = Depends(get_db)):
    db_shipment = models.ShipmentHeader(**shipment.dict())
    db.add(db_shipment)
    db.commit()
    db.refresh(db_shipment)
    return db_shipment

@app.post("/shipments/{receipt_id}/details", response_model=List[ShipmentDetailsSchema])
def add_shipment_details(receipt_id: str, details: List[ShipmentDetailsSchema], db: Session = Depends(get_db)):
    shipment = db.query(models.ShipmentHeader).filter(models.ShipmentHeader.receiptId == receipt_id).first()
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    po_items = {po_detail.itemId for po_detail in db.query(models.PoDetails).filter(models.PoDetails.poId == shipment.poId).all()}
    
    for detail in details:
        if detail.itemId not in po_items:
            raise HTTPException(status_code=400, detail=f"Item {detail.itemId} is not in the PO {shipment.poId}")
    
    db_details = [models.ShipmentDetails(**{**detail.dict(), "receiptId": receipt_id}) for detail in details]
    db.add_all(db_details)
    db.commit()
    for db_detail in db_details:
        db.refresh(db_detail)
        
    return db_details

@app.get("/shipments/{receipt_id}", response_model=Dict[str, Any])
def get_shipment_with_details(receipt_id: str, db: Session = Depends(get_db)):
    shipment = db.query(ShipmentHeader).filter(ShipmentHeader.receiptId == receipt_id).first()
    if shipment is None:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    details = db.query(ShipmentDetails).filter(ShipmentDetails.receiptId == receipt_id).all()
    return {"shipment": shipment, "details": details}


#SUPPPLIER
@app.post("/suppliers/", response_model=SupplierCreate)
def create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    # Check if supplier already exists
    existing_supplier = db.query(Supplier).filter(Supplier.email == supplier.email).first()
    if existing_supplier:
        raise HTTPException(status_code=400, detail="Supplier with this email already exists")

    new_supplier = Supplier(
        supplierId=supplier.supplierId,
        name=supplier.name,
        email=supplier.email,
        phone=supplier.phone,
        address=supplier.address,
        lead_time=supplier.lead_time
    )
    db.add(new_supplier)
    db.commit()
    db.refresh(new_supplier)
    return new_supplier

# Get all suppliers
@app.get("/suppliers/")
def get_suppliers(db: Session = Depends(get_db)):
    return db.query(Supplier).all()

# Get supplier by ID
@app.get("/suppliers/{supplierId}")
def get_supplier(supplierId : str, db: Session = Depends(get_db)):
    supplier = db.query(Supplier).filter(Supplier.supplierId  == supplierId ).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return supplier

#po chat
def execute_mysql_query(query):
    """Executes a MySQL query after validating the table name."""
    try:
        words = query.split()
        if "FROM" in words:
            table_index = words.index("FROM") + 1
            table_name = words[table_index].strip("`;,")  # Extract table name safely
        else:
            return "Invalid SQL query structure."

        # Get existing tables
        valid_tables = get_table_names()
        if table_name not in valid_tables:
            return f"Error: Table `{table_name}` does not exist. Available tables: {valid_tables}"

        # Execute the validated query
        db: Session = next(get_db())
        result = db.execute(text(query)).fetchall()
        db.close()
        return [dict(row._mapping) for row in result]

    except Exception as e:
        return str(e)
def get_table_names():
    """Fetch all table names from the database."""
    try:
        db: Session = next(get_db())
        result = db.execute(text("SHOW TABLES")).fetchall()
        db.close()
        return [list(row)[0] for row in result]  # Extract table names
    except Exception as e:
        return str(e)

def query_database_function(question: str) -> str:
    """Generates and executes an SQL query based on a natural language question."""
    # Fetch available tables
    available_tables = get_table_names()
    tables_str = ", ".join(available_tables)

    # SQL generation prompt (same as original)
    sql_query_prompt = f"""
    The user wants to query the MySQL database. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 7 tables in my database, namely: invoicedetails, invoiceheader, podetails, poheader, suppliers, shipmentHeader, and shipmentDetails.    - **invoicedetails** has the following fields: id, itemId, itemQuantity, itemDescription, itemCost, totalItemCost, poId, invoiceNumber.
    - **invoicedetails**: id, itemId, itemQuantity, itemDescription, itemCost, totalItemCost, poId, invoiceNumber.
    - **invoiceheader**: invoiceId, invoicedate, invoiceType, currency, payment_term, invoice_status, total_qty, total_cost, total_tax, total_amount, userInvNo.
    - **podetails**: id, itemId, itemQuantity, supplierId, itemDescription, itemCost, totalItemCost, poId.
    - **poheader**: poNumber, shipByDate, leadTime, estimatedDeliveryDate, totalQuantity, totalCost, totalTax, currency, payment_term.
    - **suppliers**: supplierId, name, email, phone, address, lead_time.
    - **shipmentheader**: receiptId, asnId, expectedDate, receivedDate, receivedBy, status, sourceLocation, destinationLocation, totalQuantity, totalCost, poId.
    - **shipmentdetails**: id, itemId, itemDescription, itemCost, expectedItemQuantity, receivedItemQuantity, damagedItemQuantity, containerId, receiptId, invoiced.

    ### **Field Mapping Rules** 
    - "purchase order", "purchaseorder", "PURCHASE ORDER", "PURCHASEORDER", "Purchase Order", and "PurchaseOrder" refer to the **poheader** table.
    - "PO number", "po_number", "po no", "PONumber", "order number" refer to **poNumber**.
    - "po_id" and "poId" are equivalent and refer to **poId**.
    - "cost" and "total cost" refer to **totalCost**.
    - "amount" refers to **totalAmount**.
    - "invoice number" and "invoiceNumber" refer to **invoiceNumber**.
    - "payment terms" and "payment_term" are the same.
    - "lead time" and "lead_time" are the same.
    - "supplier" refers to the **suppliers** table.
    - "supplierId" and "supplier id" refer to **supplierId**.
    - "shipment", "shipment header", "receipt" refer to **shipmentHeader**.
    - "shipment details", "shipment detail", "shipment items" refer to **shipmentDetails**.
    - "receipt id" and "receiptId" refer to **receiptId** in **shipmentHeader** or **shipmentDetails**.
    - "ASN" refers to **asnId**.
    - "expected date" refers to **expectedDate**.
    - "received date" refers to **receivedDate**.
    - "received by" refers to **receivedBy**.
    - "status" refers to **status**.
    - "source location" refers to **sourceLocation**.
    - "destination location" refers to **destinationLocation**.
    - "total quantity" refers to **totalQuantity**.
    - "total cost" refers to **totalCost**.
    - "item id" refers to **itemId**.
    - "item description" refers to **itemDescription**.
    - "item cost" refers to **itemCost**.
    - "expected item quantity" refers to **expectedItemQuantity**.
    - "received item quantity" refers to **receivedItemQuantity**.
    - "damaged item quantity" refers to **damagedItemQuantity**.
    - "container id" refers to **containerId**.
    - "invoiced" refers to **invoiced**.
    -  "quality" and "quality issues" refer to the **percentage of damagedItemQuantity** calculated as:
        **(damagedItemQuantity / receivedItemQuantity) * 100**.

    ### **Case-Insensitive Handling**
    ### **Handling Special Characters & Spaces**
    """

    # Generate SQL query
    sql_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sql_query_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    
    # Extract and clean query
    mysql_query = sql_response.choices[0].message.content.strip()
    print("Query result: ",sql_response)
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate and execute
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result: ",result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error: ",{str(e)})
        return f"Error executing query: {str(e)}"


@app.post("/chat")
async def chat_with_po_assistant(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()
    
    chat_histories[user_id].append(f"User: {user_message}")
    conversation = "\n".join(chat_histories[user_id])

    try:
        # First API call - with function definition
        messages = [
            {"role": "system", "content": template_PO},
            {"role": "user", "content": conversation}
        ]

        functions = [{
            "name": "query_database",
            "description": "Retrieve data from the database using SQL queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"
                    }
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function(args["question"])
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })

            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Retain po_json from previous interaction if query_called is True
        if not query_called:
            user_po_details[user_id] = await categorize_po_details(bot_reply, user_id)

        po_json = user_po_details[user_id]  # Assign retained po_json

        chat_histories[user_id].append(f"Bot: {bot_reply}")
        print("PO JSON:", po_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Purchase Order created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("PO submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": chat_histories,
            "po_json": po_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adduser")
async def add_user(request:UserSchema,db:Session=Depends(get_db)):
    user = User(name=request.name,email=request.email,nickname=request.nickname)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/poCreation/", status_code=status.HTTP_201_CREATED)
def create_po(poHeader: poHeaderCreate, db: Session = Depends(get_db)):
    db_poHeader = models.PoHeader(**poHeader.dict())
    db.add(db_poHeader)
    db.commit()
    db.refresh(db_poHeader)
    return db_poHeader

@app.post("/poDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_poDetail(poDetailData: List[poDetailsCreate], db: Session = Depends(get_db)):
    for details in poDetailData:
        db_poDetails = models.PoDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            supplierId=details.supplierId
        )
        db.add(db_poDetails)
        db.commit()
        db.refresh(db_poDetails)
    return {
        "message":"Items added Sucessfully!"
    }
    # db_poDetails = models.PoDetails(**poDetailData.dict())
    # db.add(db_poDetails)
    # db.commit()
    # db.refresh(db_poDetails)
    # return [db_poDetails]

@app.post("/invCreation/", status_code=status.HTTP_201_CREATED)
def create_invoice(invHeader: invHeaderCreate, db: Session = Depends(get_db)):
    db_invHeader = models.InvHeader(**invHeader.dict())
    db.add(db_invHeader)
    db.commit()
    db.refresh(db_invHeader)
    return db_invHeader

@app.post("/invDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_invDetail(invDetailData: List[invDetailsCreate], db: Session = Depends(get_db)):
    for details in invDetailData:
        db_invDetails = models.InvDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            invoiceNumber= details.invoiceNumber
        )
        db.add(db_invDetails)
        db.commit()
        db.refresh(db_invDetails)
    return {
        "message":"Items added Sucessfully!"
    }

@app.get("/poDetails/{po_id}")
def read_poDeatils(po_id: str, db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        detail = "Po Number is not found in our database! Please add a valid PO number!"
        conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        
    po_info = db.query(models.PoDetails).filter(models.PoDetails.poId == po_id).all()
    return { "po_header":po,"po_details":po_info}

@app.post("/invoiceValidation")
def po_data_validations(po_id:str,detail:Dict[str,int],db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        raise HTTPException(status_code=404, detail="PO is not found!")
    for item,quantity in detail:
        po_details = db.query(models.PoDetails).filter(models.PoDetails.itemId==item).first()
        if po_details is None:
            detail = "Item which you added is not present in this PO"
            conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        if(po_details.itemQuantity>quantity):
            detail = po_details.itemId + "quantity is excced according to PO quantity is" + po_details.itemQuantity
            conversation.append('Bot: ' + detail)
            raise HTTPException(status_code=404, detail=conversation)
    return {"details":conversation}
        
    


@app.get("/invoiceDetails/{inv_id}")
def read_invDeatils(inv_id: str, db: Session = Depends(get_db)):
    inv = db.query(models.InvHeader).filter(models.InvHeader.invoiceId == inv_id).first()
    if inv is None:
        raise HTTPException(status_code=404, detail="Invoice not found!")
    inv_info = db.query(models.InvDetails).filter(models.InvDetails.invoiceNumber == inv_id).all()
    return { "inv_header":inv,"inv_details":inv_info}



@app.post("/ok")  
async def ok_endpoint(query: str, db: Session = Depends(get_db)):
    # Pass the db session to query_database_function_promo
    returned_attributes = query_database_function_promo(query, db)
    return {"message": "ok", "attributes": returned_attributes}
# @app.post("/ok")
# async def ok_endpoint(query:str,db: Session = Depends(get_db)):
#     returned_attributes=query_database_function_promo(query)
#     # returned_attributes=find_attributes(query)
#     # testModel()
#     # openaifunction()
#     # extractor = run_conversation("Invoice type: Debit Note PO number: PO123 Date: 26/06/2024 Items: ID123, ID124 Supplier Id: SUP1123  Total tax: 342  Quantity: 2, 4",db)

#     return {"message":"ok","attributes":returned_attributes}

@app.get
async def findPoDetails(po:str):
        db: Session = Depends(get_db)
        po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po).first()
        if po is None:
            return {"Po not found!"}
        else:
            return {"Po Found!"}


@app.post("/clearData")
async def clearConversation(submitted:str):
    conversation.clear()
    chat_histories.clear()
    previous_invoice_details.clear()
    previous_po_details.clear()
    previous_promo_details.clear()
    submissionStatus = "not submitted"
    return {"conversation":conversation,"submissionStatus":"not submitted","chat_history":chat_histories}

@app.post("/testSubmission")
async def submiission(query:str):
    result=test_submission(query)
    return {"result":result}

@app.post("/uploadGpt/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_invoice_details(extracted_text)

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

@app.post("/uploadPo/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_po_details(extracted_text,"admin")

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

    
@app.post("/uploadOpenAi/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    extracted_text = extract_text_with_openai(file)
    return JSONResponse(content={"extracted_text": extracted_text})


@app.post("/upload/")
async def upload_invoice(file: UploadFile = File(...)):
    """API to upload an invoice file and extract details."""
    if file.content_type not in ["image/png", "image/jpeg", "application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, PDF, and TXT are supported.")

    # Read file bytes
    file_bytes = await file.read()

    # Extract text based on file type
    if file.content_type in ["image/png", "image/jpeg"]:
        image = Image.open(BytesIO(file_bytes))
        extracted_text = extract_text_from_image(image)
    elif file.content_type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
    else:  # Text file
        extracted_text = file_bytes.decode("utf-8")

    # Process extracted text
    invoice_details = extract_invoice_details(extracted_text)
    invoice_data_from_conversation = {
        "quantities": extract_invoice_details(extracted_text).get("quantities", []),
        "items": extract_invoice_details(extracted_text).get("items", [])
    }
    # invoice_json=json.dumps(invoice_details)
    # await generate_response(invoice_details)

    return {"file_name": file.filename, "invoice_details": invoice_details,"invoice_data_from_conversation":invoice_data_from_conversation,"extracted_text":extracted_text}


@app.post("/creation/response")
async def generate_response(query:str):
    conversation.append('User: ' + query)
    output = gpt_response(query)
    conversation.append('Bot: ' + output)
    extractor = ''
    # extractor = run_conversation(output)
    test_model_reply=testModel(query,output)
    form_submission=test_submission(output)
    action='action'
    submissionStatus="not submitted"
    bot_response_pattern = re.compile(r"Invoice\screated\ssuccessfully", re.IGNORECASE)
    past_invoice_regex = r"(past\s+invoice|invoice\s+past|last\s+invoice|invoice\s+last|previous\s+invoice|invoice\s+previous)\s*"
    invoice_regex = r"(create\s+invoice|invoice\s+create|create\san\sinvoice)\s*"
    patternInvoice = re.compile(invoice_regex, re.IGNORECASE)
    pastPatternInvoice = re.compile(past_invoice_regex, re.IGNORECASE)
    bot_response_pattern = re.compile(r"Invoice\screated\ssuccessfully", re.IGNORECASE)
    for line in conversation:
        if line.startswith("User:"):
            user_input = line.split(":")[1].strip().lower()
            if re.search(pastPatternInvoice, user_input):
                action = "last invoice created"
                # break
            elif re.search(patternInvoice, user_input):
                action = "create invoice"
            elif "create po" in user_input:
                action = "create PO"
        # elif line.startswith("Bot:"):
        #     if re.search(bot_response_pattern, line):
        #          submissionStatus = "submitted"
        #     else:
        #          submissionStatus="not submitted"
 
    # pattern = r"invoice type:(.*?), date:(.*?), po number:(.*?), supplier id:(.*?), total amount:(.*?), total tax:(.*?), items:(.*?), quantity:(.*?)$"
    # invoice_details = {
    #     "invoice type": None,
    #     "date": None,
    #     "po number": None,
    #     "supplier id":None,
    #     "total amount": None,
    #     "total tax": None,
    #     "items": [],
    #     "quantity": []
    # }
    # invoiceDatafromConversation=collect_invoice_data(conversation)    
    # regex_patterns = {
    #     "Invoice type": r"invoice\s+type\s*:\s*([^:]+)",
    #     "Date": r"date\s*:\s*(\d{2}/\d{2}/\d{4})",
    #     "PO number": r"po\s+number\s*:\s*(\w+)",
    #     "Supplier Id": r"supplier\s+id\s*:\s*(\w+)",
    #     "Total amount": r"total\s+amount\s*:\s*(\d+)",
    #     "Total tax": r"total\s+tax\s*:\s*(\d+)",
    #     "Items": r"items\s*:\s*(\w+)",
    #     "Quantity": r"quantity\s*:\s*(\d+)",
    # }
        
               
    # for line in conversation:
        
    #     if "User: " in line:
    #         match_invoice_type = re.match(r".*invoice\s*type\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
    #     elif "Bot: " in line:
    #         for detail, pattern in regex_patterns.items():
    #                 match = re.search(pattern, line, re.IGNORECASE)
    #                 if match:
    #                     invoice_details[detail.lower()] = match.group(1)
    # invoice_json = json.dumps(invoice_details, indent=4)
    invoiceDatafromConversation=collect_invoice_data(conversation)
    invoice_details = {
        "invoice type": None,
        "date": None,       
        "user invoice number":None,
        "po number": None,
        "total amount": None,
        "total tax": None,
        "items": [],
        "quantities": []
    }
    regex_patterns = {
    "invoice type": r"\*\*Invoice Type:\*\*\s*(.+?)\n",
    "date": r"\*\*Date:\*\*\s*(\d{2}/\d{2}/\d{4})",
    "po number": r"\*\*PO Number:\*\*\s*(\w+)",
    "user invoice number": r"\*\*Invoice Number:\*\*\s*(\w+)",
    "total amount": r"\*\*Total Amount:\*\*\s*([\d,]+)",  # Handles commas
    "total tax": r"\*\*Total Tax:\*\*\s*([\d,]+)",  # Handles commas
    "items": r"\*\*Items:\*\*\s*([\w,\s]+)",
    "quantities": r"\*\*Quantities:\*\*\s*([\d,\s]+)"
}

# Extract values using regex
    for line in conversation:
        for key, pattern in regex_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key in ["items", "quantities"]:  # Convert comma-separated values to lists
                    invoice_details[key] = [item.strip() for item in value.split(",")]
                else:
                    invoice_details[key] = value

    # Convert to JSON and print the result
    invoice_json = json.dumps(invoice_details, indent=4)
    print(invoice_json)
    # invDetails=await categorize_invoice_details(conversation)
    invDetails=await categorize_invoice_details_new(conversation,"admin")
    print("invDetails",invDetails)
    return {"conversation":conversation,"invoice_json":invDetails,"action":action,
    "submissionStatus":form_submission,"invoiceDatafromConversation":invDetails,
    "test_model_reply":test_model_reply,"extractor_info":extractor }
    # return {"conversation":conversation,"invoice_json":invoice_json,"action":action,
    # "submissionStatus":form_submission,"invoiceDatafromConversation":invoiceDatafromConversation,
    # "test_model_reply":test_model_reply,"extractor_info":extractor }

#old po
# @app.post("/chat")
# async def chat_with_po_assistant(request: ChatRequest):
#     """
#     Handles chat requests and processes PO creation using OpenAI API.
#     Returns both the bot's reply as a message and structured JSON data (or null if parsing fails).
#     """
#     user_id = request.user_id
#     user_message = request.message

#     # Initialize chat history & PO details if not exists
#     if user_id not in chat_histories:
#         chat_histories[user_id] = []
#         user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()

#     # Append user message to history
#     chat_histories[user_id].append(f"User: {user_message}")

#     # Create conversation history
#     conversation = "\n".join(chat_histories[user_id])

#     try:
#         # Generate bot response
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": template_PO},
#                 {"role": "user", "content": conversation}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )

#         bot_reply = response.choices[0].message.content
#         po_json = await categorize_po_details(bot_reply,user_id)

#         # Attempt to parse the response as JSON
#         print("Bot reply: ",bot_reply)
#         try:
#             structured_data = json.loads(bot_reply)
#         except json.JSONDecodeError:
#             structured_data = None  # If parsing fails, return null

#         # Update user PO details only if structured_data is valid
#         if structured_data:
#             user_po_details[user_id] = structured_data

#         # Append bot response to history
#         chat_histories[user_id].append(f"Bot: {bot_reply}")
#         print("PO_son:",po_json,"user_id: ",user_id)
#         # Determine submission status
#         if "Would you like to submit" in bot_reply:
#             submissionStatus = "pending"
#         elif "Purchase Order created successfully" in bot_reply:
#             submissionStatus = "submitted"
#         elif "I want to change something" in bot_reply:
#             submissionStatus = "cancelled"
#         else:
#             submissionStatus = "in_progress"  # Default state if no clear intent is detected
#         print("PO sumission status: ",submissionStatus)
#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "po_details": structured_data,
#             "chat_history": chat_histories,
#             "po_json": po_json,  # Will be null if parsing fails
#             "submissionStatus": submissionStatus
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat")
# async def chat_with_po_assistant(request: ChatRequest):
#     """
#     Handles chat requests and processes PO creation using OpenAI API.
#     Returns both the bot's reply as a message and structured JSON data (or null if parsing fails).
#     """
#     user_id = request.user_id
#     user_message = request.message

#     # Initialize chat history & PO details if not exists
#     if user_id not in chat_histories:
#         chat_histories[user_id] = []
#         user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()

#     # Append user message to history
#     chat_histories[user_id].append(f"User: {user_message}")
#     print(chat_histories)
#     # Create conversation history
#     conversation = "\n".join(chat_histories[user_id])

#     try:
#         # Use new OpenAI API format
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": template_PO},
#                 {"role": "user", "content": conversation}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )

#         bot_reply = response.choices[0].message.content  # New way to access the message
#         print(bot_reply)
#         po_json=await categorize_po_details(bot_reply)
#         # Attempt to parse the response as JSON
#         try:
#             structured_data = json.loads(bot_reply)
#         except json.JSONDecodeError:
#             structured_data = None  # If parsing fails, return null

#         # Update user PO details only if structured_data is valid
#         if structured_data:
#             user_po_details[user_id] = structured_data

#         # Append bot response to history
#         chat_histories[user_id].append(f"Bot: {bot_reply}")

#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "po_details": structured_data,
#             "chat_history":chat_histories,
#             "po_json":po_json # Will be null if parsing fails
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# 09 March 2025

#02 March 2025#-2
from fastapi import FastAPI,Depends,HTTPException,status
from utils import gpt_response,extract_information,conversation,checkValidation,run_conversation,collect_invoice_data,bot_action,openaifunction,testModel,test_submission,extract_invoice_details,extract_text_from_pdf,extract_text_from_image,collect_invoice_data_from_file,extract_text_with_openai,categorize_invoice_details,client,categorize_invoice_details_new
from poUtils import template_PO,DEFAULT_PO_STRUCTURE,categorize_po_details
from pydantic import BaseModel;
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails,Supplier,ShipmentHeader,ItemDiffs,ItemSupplier,ItemMaster,PromotionDetails,PromotionHeader
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach,ChatRequest,SupplierCreate,ShipmentHeader,ShipmentDetails,ItemDiffsSchema,ItemSupplierSchema,ItemMasterSchema,PromotionDetailsSchema,PromotionHeaderSchema
from schemas import ShipmentDetails as ShipmentDetailsSchema
from database import engine,SessionLocal,get_db
from promoUtils import template_Promotion
from sqlalchemy.orm import Session
import models
from typing import List,Any
from fastapi.middleware.cors import CORSMiddleware
from utils_extractor import run_conversation
from typing import Dict,Tuple, Optional
import easyocr
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import openai
# from exampleextrator import run_conversation
from fastapi.responses import JSONResponse
import mysql.connector
Base.metadata.create_all(bind=engine)
from sqlalchemy.sql import text
import re;
import json;
import os
import signal
from collections import defaultdict

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#PO Chatbot Functions
chat_histories = {}
user_po_details = {}


#Promo
@app.get("/promotionHeader/{promotionId}", response_model=PromotionHeaderSchema)
def get_promotion_header(promotionId: str, db: Session = Depends(get_db)):
    promoHeader = db.query(models.PromotionHeader).filter(models.PromotionHeader.promotionId == promotionId).first()
    if not promoHeader:
        raise HTTPException(status_code=404, detail="Promotion not found")
    return promoHeader
@app.get("/promotionDetails/{promotionId}", response_model=List[PromotionDetailsSchema])
def get_promotion_details(promotionId: str, db: Session = Depends(get_db)):
    promoDetails = db.query(models.PromotionDetails).filter(models.PromotionDetails.promotionId == promotionId).all()
    if not promoDetails:
        raise HTTPException(status_code=404, detail="No details found for this promotion")
    return promoDetails
@app.post("/promotionHeader/", status_code=status.HTTP_201_CREATED)
def create_promotion_header(promoHeader:PromotionHeaderSchema, db: Session = Depends(get_db)):
    db_promoHeader = models.PromotionHeader(
        promotionId=promoHeader.promotionId,
        componentId=promoHeader.componentId,
        startDate=promoHeader.startDate,
        endDate=promoHeader.endDate,
        promotionType=promoHeader.promotionType
    )
    db.add(db_promoHeader)
    db.commit()
    db.refresh(db_promoHeader)
    return db_promoHeader
@app.post("/promotionDetails/", status_code=status.HTTP_201_CREATED)
def create_promotion_details(promoDetails: List[PromotionDetailsSchema], db: Session = Depends(get_db)):
    for details in promoDetails:
        db_promoDetails = models.PromotionDetails(
            promotionId=details.promotionId,
            componentId=details.componentId,
            itemId=details.itemId,
            discountType=details.discountType,
            discountValue=details.discountValue
        )
        db.add(db_promoDetails)
        db.commit()
        db.refresh(db_promoDetails)
    return {"message": "Promotion details added successfully!"}


# Database Functions
# def get_valid_hierarchy_values(hierarchy_level: str) -> list:
#     """Fetch valid values for a hierarchy level from itemsMaster"""
#     column_map = {
#         "department": "itemDepartment",
#         "class": "itemClass",
#         "subclass": "itemSubClass"
#     }
    
#     if hierarchy_level.lower() not in column_map:
#         return []
    
#     column = column_map[hierarchy_level.lower()]
#     query = f"SELECT DISTINCT `{column}` FROM itemmaster"
#     result = execute_mysql_query(query)
#     return [str(row[column]) for row in result if row[column]]

# def validate_items(item_ids: list) -> dict:
#     """Validate item IDs against database"""
#     if not item_ids:
#         return {"valid": [], "invalid": []}
    
#     ids_str = ",".join([f"'{id.strip()}'" for id in item_ids])
#     query = f"SELECT itemId FROM itemmaster WHERE itemId IN ({ids_str})"
#     valid_ids = [row["itemId"] for row in execute_mysql_query(query)]
#     invalid_ids = list(set(item_ids) - set(valid_ids))
    
#     return {"valid": valid_ids, "invalid": invalid_ids}

# def validate_items_query(query: str) -> bool:
#     diff_attributes = ['color', 'size', 'material']
#     query_lower = query.lower()
#     for attr in diff_attributes:
#         if f'itemmaster.`{attr}`' in query_lower or f'where {attr} =' in query_lower:
#             return False
#     return True

# def query_database_function(question: str, db: Session) -> str:
#     # Modified SQL Generation Prompt
#     included_tables = ["itemmaster", "itemsupplier", "itemdiffs"]
#     itemdiffsCols = get_column_names(included_tables[2])
#     attributes = get_unique_values("itemdiffs", itemdiffsCols[1])  # ['colour', 'size']
    
#     # Format tables and attributes for prompt
#     tables_str = ", ".join([f"`{t}`" for t in included_tables])
#     attributes_str = ", ".join([f"'{a}'" for a in attributes])

#     promotion_sql_prompt = """
#     You are a SQL assistant for fashion retail data. The following tables exist: {tables_str}.
#     Generate a **pure SQL query** without explanations, comments, or descriptions.


#     # If the user's query contains any of the {attributes_str}, follow the process mentioned below

#     ### **Critical Rules:**  
#     #### 1. Attribute Handling ({attributes_str})
#     - Process for attribute-based queries:
#       1. Find `id` in `itemdiffs` WHERE `diffType` = [mapped_attribute] AND `diffId` = [user_value]
#       2. Filter `itemmaster` WHERE this `id` appears in `diffType1/2/3`
#       3. **NEVER** search for colors, sizes, or materials inside `itemmaster` directly.  
#       4. **NEVER** use `LIKE` on `itemDescription` (e.g., avoid `WHERE itemDescription LIKE 'red'`). 

#     #### 2. Valid Tables/Columns
#     - Use only: itemmaster (im), itemsupplier (isup), itemdiffs (idf)
#     - Never search itemmaster.itemDescription for attributes
#     - You may only use the following tables:  
#     - `itemmaster` (Base Table):
#         - **`itemId`**: Unique identifier for each item.  
#         - **`itemDescription`**: Primary description of the item.  
#         - **`itemSecondaryDescription`**: Additional details about the item.  
#         - **`itemDepartment`**: The broader category an item belongs to (e.g., T-Shirt, Trousers, Jackets).  
#         - **`itemClass`**: A classification within a department (e.g., Formals, Casuals, Leather).  
#         - **`itemSubClass`**: A more granular classification under the item class (e.g., Full Sleeve, Half Sleeve, Zipper, Regular Fit).  
#         - **`brand`**: The brand associated with the item (e.g., Zara, Adidas, H&M).  
#         - **`diffType1`, `diffType2`, `diffType3`**: Foreign keys linking to `itemdiffs.id`, representing specific item attributes such as color, size, or material.   
#     - `itemsupplier` (For Cost & Supplier Data):
#         - **`id`**: Unique identifier for each supplier-item relationship.  
#         - **`supplierCost`**: The cost of the item from the supplier.  
#         - **`supplierId`**: The identifier for the supplier providing the item.  
#         - **`itemId`**: Foreign key linking to `itemsMaster.itemId`, establishing the relationship between items and suppliers.   
#     - `itemdiffs` (For Attribute Filtering):
#         - **`id`**: Unique identifier for each differentiation type.  
#         - **`diffType`**: The attribute type used to differentiate items (e.g., color, size, material).  
#           - If the user wants to filter items by a specific attribute (e.g., color or size), the query should check `diffType` and retrieve the corresponding `diffId`.  
#         - **`diffId`**: The actual differentiation value corresponding to `diffType`.  
#           - Example: If `diffType = 'color'`, then `diffId` could be "Red"; if `diffType = 'size'`, then `diffId` could be "M".    

#     #### **3. Query Format & Execution**  
#     - **Start queries from `itemmaster`** as the primary table.  
#     - Use **explicit JOINs** when needed (e.g., joining `itemsupplier` for cost-related queries).  
#     # - **DO NOT** use `UNION`.  
#     - **Return only valid SQL queries**. Do not include explanations or markdown formatting.  
#     - The generated SQL query should be a **fully structured SQL query** without additional explanations, comments, or descriptions.  
#     - It should dynamically adapt to user queries, ensuring proper field mappings and query optimizations based on the rules outlined above.  

#     ---
#     ### **SQL Examples:**

#     #### **Example 1: Select All Red Colored Items**
#     User: *"Select all red colored items"*
#     ```sql
#     SELECT im.itemId
#     FROM itemmaster im
#     WHERE im.diffType1 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
#     ) OR im.diffType2 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
#     ) OR im.diffType3 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
#     )
#     ```
#     #### **Example 2: Select all red colored items with a supplier cost below $50**
#     User: *"Select all red colored items with a supplier cost below $50"*
#      ```sql
#     SELECT im.itemId, isup.supplierCost
#     FROM itemmaster im
#     JOIN itemsupplier isup ON im.itemId = isup.itemId
#     WHERE isup.supplierCost < 50
#     AND (im.diffType1 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
#     ) OR im.diffType2 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
#     ) OR im.diffType3 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
#     ))
#      ```
#     #### **Example 3: Select All Items of Size "Large"**
#     User: *"Select All Items of Size "Large""*
#     ```sql
#     SELECT im.itemId
#     FROM itemmaster im
#     WHERE im.diffType1 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'size' AND diffId = 'Large'
#     ) OR im.diffType2 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'size' AND diffId = 'Large'
#     ) OR im.diffType3 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'size' AND diffId = 'Large'
#     )
#     ```

#     """
#     promotion_sql_prompt_new = f"""
#     You are a **strict** SQL assistant for fashion retail data. The following tables exist: {tables_str}.
#     Generate a **pure SQL query** without explanations, comments, or descriptions.
    
#     🚨 **STRICT RULES: NEVER VIOLATE** 🚨  
#     - If the user's query involves any of the attributes {attributes_str}, you must:
#       1. First, retrieve `id` from `itemdiffs` where `diffType` equals '[ATTRIBUTE]' and `diffId` equals '[USER VALUE]'.
#       2. Then, filter `itemmaster` to return those records where the retrieved `id` appears in one of `diffType1`, `diffType2`, or `diffType3`.
#       3. **NEVER** filter attributes using `itemmaster.itemDescription` (for example, do not use `LIKE '%red%'` on itemDescription for attribute filtering).
#       4. If the query attempts to filter attributes by using `itemmaster.itemDescription`, **FAIL THE QUERY IMMEDIATELY**.
    
#     - For general text searches on item descriptions (when not filtering by an attribute), it is acceptable to use `itemmaster.itemDescription`.
    
#     ✅ **Correct Query Example for an Attribute-Based Filter**  
#     (This example uses one of the allowed attributes from {attributes_str}.)
#     ```sql
#     SELECT im.itemId
#     FROM itemmaster im
#     WHERE im.diffType1 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
#     ) OR im.diffType2 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
#     ) OR im.diffType3 IN (
#         SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
#     );
#     ```
    
#     ✅ **Correct Query Example for a General Text Search**  
#     (This is allowed only when not filtering by an attribute from {attributes_str}.)
#     ```sql
#     SELECT im.itemId, im.itemDescription
#     FROM itemmaster im
#     WHERE im.itemDescription LIKE '%casual%';
#     ```
#     """
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": promotion_sql_prompt_new},
#             {"role": "user", "content": question}
#         ],
#         temperature=0.1,
#         max_tokens=300
#     )
    
#     query = response.choices[0].message.content.strip()
    
#     # --- Post-Processing Validation ---
#     # Convert to lower case for robust matching.
#     lower_query = query.lower()
#     lower_question = question.lower()
    
#     # If the question mentions any of the allowed attribute types, enforce proper filtering.
#     if any(attr.lower() in lower_question for attr in attributes_list):
#         if "itemdiffs" not in lower_query:
#             return "❌ Invalid Query: The query must filter attributes using the itemdiffs subquery."
#         # Check if itemDescription is (improperly) used to filter attributes.
#         if "itemdescription" in lower_query and ("like" in lower_query or "=" in lower_query):
#             return "❌ Invalid Query: Filtering attributes by itemDescription is forbidden."
    
#     try:
#         result = db.execute(query).fetchall()
#         return json.dumps([dict(row) for row in result])
#     except Exception as e:
#         return f"Query failed: {str(e)}"

# def query_database_function(question: str, db: Session) -> str:
#     # 1. Fetch Metadata
#     itemdiffs_data = get_all_itemdiffs(db)
#     attribute_map = create_attribute_value_map(itemdiffs_data)
#     attributes_list = list(attribute_map.keys())
    
#     # 2. Enhanced Attribute Detection
#     detected_attrs = detect_attributes(question.lower(), attribute_map)
    
#     # 3. Route Query Generation
#     if detected_attrs:
#         sql_query = generate_attribute_query(question, attribute_map)
#     else:
#         sql_query = generate_general_query(question)
    
#     # 4. Strict Validation
#     if error := validate_query(sql_query, detected_attrs, attribute_map):
#         return error
    
#     # 5. Execute Query
#     try:
#         result = db.execute(text(sql_query)).fetchall()
#         return json.dumps([dict(row) for row in result])
#     except Exception as e:
#         return f"Query failed: {str(e)}"

# def create_attribute_value_map(itemdiffs_data: list) -> dict:
#     """Create {attribute: {values}} mapping"""
#     attribute_map = defaultdict(set)
#     for row in itemdiffs_data:
#         attribute_map[row['diffType']].add(row['diffId'].lower())
#     return attribute_map

# def detect_attributes(question: str, attribute_map: dict) -> set:
#     """Find relevant attributes using both keys and values"""
#     detected = set()
    
#     # Check for attribute names
#     for attr in attribute_map:
#         if attr.lower() in question:
#             detected.add(attr)
    
#     # Check for attribute values
#     for attr, values in attribute_map.items():
#         if any(value in question for value in values):
#             detected.add(attr)
    
#     return detected

# def generate_attribute_query(question: str, attribute_map: dict) -> str:
#     """Generate SQL using explicit attribute-value mapping"""
#     attribute_list = "\n".join(
#         f"- {attr}: {', '.join(values)}" 
#         for attr, values in attribute_map.items()
#     )
    
#     prompt = f"""You MUST follow this structure for attribute queries:

# 1. Identify ONE attribute type and value from:
# {attribute_list}

# 2. Generate this exact pattern:
# SELECT im.itemId 
# FROM itemmaster im
# WHERE im.diffType1 IN (
#     SELECT id FROM itemdiffs 
#     WHERE diffType = '[ATTRIBUTE]' AND LOWER(diffId) = '[VALUE]'
# )
# OR im.diffType2 IN (...)
# OR im.diffType3 IN (...)

# Example for "red items":
# SELECT im.itemId 
# FROM itemmaster im
# WHERE im.diffType1 IN (
#     SELECT id FROM itemdiffs 
#     WHERE diffType = 'color' AND LOWER(diffId) = 'red'
# )
# OR im.diffType2 IN (
#     SELECT id FROM itemdiffs 
#     WHERE diffType = 'color' AND LOWER(diffId) = 'red'
# )
# OR im.diffType3 IN (
#     SELECT id FROM itemdiffs 
#     WHERE diffType = 'color' AND LOWER(diffId) = 'red'
# )

# Generate ONLY SQL following this exact pattern."""
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": question}
#         ],
#         temperature=0
#     )
#     return response.choices[0].message.content.strip()

# def validate_query(sql: str, detected_attrs: set, attribute_map: dict) -> str:
#     """Enforce strict structural validation"""
#     sql_lower = sql.lower()
    
#     # 1. Basic structural checks
#     required = [
#         "select im.itemid",
#         "from itemmaster im",
#         "where",
#         "select id from itemdiffs",
#         "difftype ="
#     ]
    
#     for check in required:
#         if check not in sql_lower:
#             return f"❌ Missing required element: {check}"
    
#     # 2. Attribute-specific validation
#     for attr in detected_attrs:
#         # Check attribute exists in SQL
#         if f"difftype = '{attr.lower()}'" not in sql_lower:
#             return f"❌ Missing attribute '{attr}' in subquery"
        
#         # Check value exists in attribute map
#         value_match = re.search(r"lower\(diffid\)\s*=\s*'(\w+)'", sql_lower)
#         if value_match and value_match[1] not in attribute_map[attr]:
#             return f"❌ Invalid value '{value_match[1]}' for attribute '{attr}'"
    
#     # 3. Block invalid columns
#     if re.search(r"\b(itemcolor|itemsize|color|size)\b", sql_lower):
#         return "❌ Invalid column reference"
    
#     return ""
# def query_database_function_promo(question: str, db: Session) -> str:
#     # First check for attribute-based queries
#     attr_type, attr_value, attr_id = find_attributes(question)
#     print("find attributes: ", attr_type, attr_value, attr_id)
    
#     # Build base SQL prompt
#     promotion_sql_prompt = f"""
#     You are a SQL assistant for fashion retail data. Generate pure SQL without explanations.
#     Tables: itemMaster (im), itemSupplier (isup), itemDiffs (idf)
    
#     {"-- ATTRIBUTE FILTER ACTIVE --" if attr_id else ""}
#     {f"/* Use idf.id = {attr_id} for attribute filtering */" if attr_id else ""}
    
#     ### Critical Rules:
#     1. {"ALWAYS filter using im.diffType1/2/3 = " + str(attr_id) if attr_id else "For attributes, first find id in itemDiffs"}
#     2. Never search itemDescription directly
#     3. Start queries from itemMaster
    
#     ### Valid Columns:
#     - itemMaster: itemId, diffType1, diffType2, diffType3, ...
#     - itemSupplier: supplierCost, itemId
#     """

#     # Generate SQL using OpenAI
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[{
#                 "role": "system", 
#                 "content": promotion_sql_prompt
#             }, {
#                 "role": "user",
#                 "content": question
#             }],
#             temperature=0.3
#         )
        
#         raw_sql = response.choices[0].message.content
#         query = clean_sql(raw_sql)
        
#         # Validate and execute
#         if attr_id and str(attr_id) not in query:
#             return "Error: Attribute filter not properly applied"
            
#         result = db.execute(text(query)).fetchall()
#         return format_results(result)
        
#     except Exception as e:
#         return f"Query failed: {str(e)}"
def query_database_function_promo(question: str, db: Session) -> str:
    # Modified SQL Generation Prompt
    available_tables = get_table_names()
    # Define the tables you want to include
    included_tables = ["itemmaster", "itemsupplier", "itemdiffs"]
    # Filter available tables to only include the ones in included_tables
    filtered_tables = [table for table in available_tables if table in included_tables]
    # Create the string of tables for the query
    tables_str = ", ".join(filtered_tables)
    promotion_sql_prompt = """
    You are a SQL assistant for fashion retail data. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 3 tables in my database, namely:
    - `itemmaster` (aliased as `im`)  
    - `itemsupplier` (aliased as `isup`)  
    - `itemdiffs` (aliased as `idf`)  

    ### **Critical Rules:**  

    #### **1. Color, Size, and Material Filters**  
    - If the user searches for an **attribute** like **color, size, or material**, follow this strict process:  
    1. **Find the `id` in `itemdiffs`** where `diffType` matches the attribute (e.g., `'color'`) and `diffId` is the user-provided value (e.g., `'Red'`).  
    2. **Filter `itemmaster` (`im`)** by checking if the retrieved `idf.id` exists in **`im.diffType1`**, **`im.diffType2`**, or **`im.diffType3`**.  
    3. **NEVER** search for colors, sizes, or materials inside `itemmaster` directly.  
    4. **NEVER** use `LIKE` on `itemDescription` (e.g., avoid `WHERE itemDescription LIKE 'red'`).  

    #### **2. Allowed Tables Only**  
    - You may only use the following tables:  
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
        - **`itemId`**: Foreign key linking to `itemsMaster.itemId`, establishing the relationship between items and suppliers.   
    - `itemdiffs` (For Attribute Filtering):
        - **`id`**: Unique identifier for each differentiation type.  
        - **`diffType`**: The attribute type used to differentiate items (e.g., color, size, material).  
          - If the user wants to filter items by a specific attribute (e.g., color or size), the query should check `diffType` and retrieve the corresponding `diffId`.  
        - **`diffId`**: The actual differentiation value corresponding to `diffType`.  
          - Example: If `diffType = 'color'`, then `diffId` could be "Red"; if `diffType = 'size'`, then `diffId` could be "M".    

    #### **3. Query Format & Execution**  
    - **Start queries from `itemmaster`** as the primary table.  
    - Use **explicit JOINs** when needed (e.g., joining `itemsupplier` for cost-related queries).  
    # - **DO NOT** use `UNION`.  
    - **Return only valid SQL queries**. Do not include explanations or markdown formatting.  
    - The generated SQL query should be a **fully structured SQL query** without additional explanations, comments, or descriptions.  
    - It should dynamically adapt to user queries, ensuring proper field mappings and query optimizations based on the rules outlined above.  

    ---
    ### **SQL Examples:**

    #### **Example 1: Select All Red Colored Items**
    User: *"Select all red colored items"*
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 IN (
        SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
    ) OR im.diffType2 IN (
        SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
    ) OR im.diffType3 IN (
        SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
    )
    ```
    #### **Example 2: Select all red colored items with a supplier cost below $50**
    User: *"Select all red colored items with a supplier cost below $50"*
     ```sql
    SELECT im.itemId, isup.supplierCost
    FROM itemmaster im
    JOIN itemsupplier isup ON im.itemId = isup.itemId
    WHERE isup.supplierCost < 50
    AND (im.diffType1 IN (
        SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
    ) OR im.diffType2 IN (
        SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
    ) OR im.diffType3 IN (
        SELECT id FROM itemdiffs WHERE diffType = 'color' AND diffId = 'Red'
    ))
     ```
    #### **Example 3: Select All Items of Size "Large"**
    User: *"Select All Items of Size "Large""*
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.diffType1 IN (
        SELECT id FROM itemdiffs WHERE diffType = 'size' AND diffId = 'Large'
    ) OR im.diffType2 IN (
        SELECT id FROM itemdiffs WHERE diffType = 'size' AND diffId = 'Large'
    ) OR im.diffType3 IN (
        SELECT id FROM itemdiffs WHERE diffType = 'size' AND diffId = 'Large'
    )
    ```

    """
    response =  client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": promotion_sql_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    print("uery response: ",response)
    mysql_query = response.choices[0].message.content.strip()
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate and execute
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result: ",result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error: ",{str(e)})
        return f"Error executing query: {str(e)}"


# Helper functions
def clean_sql(raw: str) -> str:
    """Remove markdown and validate structure"""
    return raw.replace("```sql", "").replace("```", "").strip()

def format_results(results: list) -> str:
    """Convert results to JSON string"""
    return json.dumps([dict(row) for row in results])

def db_query(query: str, params: dict = None) -> List:
    """Generic database query executor with error handling."""
    try:
        db = next(get_db())
        result = db.execute(text(query), params or {}).fetchall()
        db.close()
        return [row for row in result]
    except Exception as e:
        print(f"Database error: {e}")
        return []

# Optimized core functions
def find_attributes(question: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Find attributes, values, and their ID in a single optimized flow."""
    print(f"\nProcessing query: '{question}'")
    
    # Get attribute metadata in single query
    attr_data = db_query(
        "SELECT diffType, diffId, id FROM itemdiffs GROUP BY diffType, diffId"
    )
    
    if not attr_data:
        print("No attribute data found in database")
        return (None, None, None)

    # Build optimized attribute structure {diffType: {diffId: id}}
    attribute_map = {}
    for diff_type, diff_id, id_val in attr_data:
        attribute_map.setdefault(diff_type, {})[diff_id] = id_val

    print(f"Attribute map: {attribute_map.keys()}")
    
    # Detect attributes using optimized function
    attr, value = detect_attribute(question, attribute_map)
    if not attr or not value:
        return (None, None, None)

    # Get ID from pre-loaded map
    item_id = attribute_map.get(attr, {}).get(value)
    return (attr, value, item_id)

def detect_attribute(question: str, attribute_map: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Optimized attribute detection with caching."""
    if not attribute_map:
        return (None, None)

    try:
        # Build prompt from pre-loaded data
        attr_desc = "\n".join(
            f"- {attr}: {', '.join(values)}" 
            for attr, values in attribute_map.items()
        )

        prompt = f"""Analyze query: "{question}"
        Identify attributes from:
        {attr_desc}
        Respond in JSON format: {{"attribute": "...", "value": "..."}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "You are a retail data analyst. Identify product attributes."
            }, {
                "role": "user", 
                "content": prompt
            }],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        result = json.loads(response.choices[0].message.content)
        attr = result.get('attribute')
        value = result.get('value')

        # Validate against known data
        if not attribute_map.get(attr, {}).get(value):
            print(f"Invalid detection: {attr}/{value}")
            return (None, None)

        return (attr, value)

    except Exception as e:
        print(f"Detection error: {e}")
        return (None, None)


promo_states = defaultdict(dict)
DEFAULT_PROMO_STRUCTURE = {
    "type": "",
    "hierarchy": {"level": "", "value": ""},
    "items": [],
    "excluded_items": [],
    "discount": {"type": "", "amount": 0},
    "dates": {"start": "", "end": ""},
    "locations": [],
    "excluded_locations": [],
    "status": "draft"
}

# Modified API endpoint
user_promo_details={}
@app.post("/promo-chat")
async def handle_promotion_chat(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in promo_states:
        promo_states[user_id] = []
        user_promo_details[user_id] = DEFAULT_PROMO_STRUCTURE.copy()
    
    promo_states[user_id].append(f"User: {user_message}")
    conversation = "\n".join(promo_states[user_id])
    included_tables = ["itemMaster", "itemSupplier", "itemDiffs"]
    itemDiffsCols = get_column_names(included_tables[2])
    attributes = get_unique_values("itemDiffs", itemDiffsCols[1])  # ['colour', 'size']
    
    # Format tables and attributes for prompt
    tables_str = ", ".join([f"`{t}`" for t in included_tables])
    attributes_str = ", ".join([f"'{a}'" for a in attributes])

    print("item diff cols: ",itemDiffsCols,"attributes: ",attributes_str)

    try:
        # First API call - with function definition
        messages = [
            {"role": "system", "content": template_Promotion},
            {"role": "user", "content": conversation}
        ]

        functions = [{
            "name": "query_database",
            "description": "Execute database queries for validation/data retrieval",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function_promo(args["question"])
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })

            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Retain promo_json from previous interaction if query_called is True
        # if not query_called:
        #     user_promo_details[user_id] = await categorize_promo_details(bot_reply, user_id)

        # promo_json = user_promo_details[user_id]  # Assign retained promo_json

        promo_states[user_id].append(f"Bot: {bot_reply}")
        # print("Promo JSON:", promo_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Promotion created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("Promotion submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": promo_states,
            # "promo_json": promo_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#DIFF
# GET all diffs
@app.get("/diffs", response_model=list[ItemDiffsSchema])
def get_diffs(db: Session = Depends(get_db)):
    diffs = db.query(models.ItemDiffs).all()
    return diffs

# POST a new diff
@app.post("/diffs", response_model=ItemDiffsSchema)
def create_diff(diff: ItemDiffsSchema, db: Session = Depends(get_db)):
    db_diff = models.ItemDiffs(**diff.dict())

    # Check if diff already exists
    existing_diff = db.query(models.ItemDiffs).filter(models.ItemDiffs.id == diff.id).first()
    if existing_diff:
        raise HTTPException(status_code=400, detail="Diff with this ID already exists")

    db.add(db_diff)
    db.commit()
    db.refresh(db_diff)
    return db_diff
#ITEM
@app.get("/items", response_model=List[ItemMasterSchema])
def get_items(db: Session = Depends(get_db)):
    items = db.query(models.ItemMaster).all()
    return items

@app.post("/items", response_model=List[ItemMasterSchema])
def create_items(items: List[ItemMasterSchema], db: Session = Depends(get_db)):
    new_items = [models.ItemMaster(**item.dict()) for item in items]
    db.add_all(new_items)
    db.commit()
    return new_items


#SHIPMENT 
@app.post("/shipments/", response_model=ShipmentHeader)
def create_shipment(shipment: ShipmentHeader, db: Session = Depends(get_db)):
    db_shipment = models.ShipmentHeader(**shipment.dict())
    db.add(db_shipment)
    db.commit()
    db.refresh(db_shipment)
    return db_shipment

@app.post("/shipments/{receipt_id}/details", response_model=List[ShipmentDetailsSchema])
def add_shipment_details(receipt_id: str, details: List[ShipmentDetailsSchema], db: Session = Depends(get_db)):
    shipment = db.query(models.ShipmentHeader).filter(models.ShipmentHeader.receiptId == receipt_id).first()
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    po_items = {po_detail.itemId for po_detail in db.query(models.PoDetails).filter(models.PoDetails.poId == shipment.poId).all()}
    
    for detail in details:
        if detail.itemId not in po_items:
            raise HTTPException(status_code=400, detail=f"Item {detail.itemId} is not in the PO {shipment.poId}")
    
    db_details = [models.ShipmentDetails(**{**detail.dict(), "receiptId": receipt_id}) for detail in details]
    db.add_all(db_details)
    db.commit()
    for db_detail in db_details:
        db.refresh(db_detail)
        
    return db_details

@app.get("/shipments/{receipt_id}", response_model=Dict[str, Any])
def get_shipment_with_details(receipt_id: str, db: Session = Depends(get_db)):
    shipment = db.query(ShipmentHeader).filter(ShipmentHeader.receiptId == receipt_id).first()
    if shipment is None:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    details = db.query(ShipmentDetails).filter(ShipmentDetails.receiptId == receipt_id).all()
    return {"shipment": shipment, "details": details}


#SUPPPLIER
@app.post("/suppliers/", response_model=SupplierCreate)
def create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    # Check if supplier already exists
    existing_supplier = db.query(Supplier).filter(Supplier.email == supplier.email).first()
    if existing_supplier:
        raise HTTPException(status_code=400, detail="Supplier with this email already exists")

    new_supplier = Supplier(
        supplierId=supplier.supplierId,
        name=supplier.name,
        email=supplier.email,
        phone=supplier.phone,
        address=supplier.address,
        lead_time=supplier.lead_time
    )
    db.add(new_supplier)
    db.commit()
    db.refresh(new_supplier)
    return new_supplier

# Get all suppliers
@app.get("/suppliers/")
def get_suppliers(db: Session = Depends(get_db)):
    return db.query(Supplier).all()

# Get supplier by ID
@app.get("/suppliers/{supplierId}")
def get_supplier(supplierId : str, db: Session = Depends(get_db)):
    supplier = db.query(Supplier).filter(Supplier.supplierId  == supplierId ).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return supplier

#po chat
def execute_mysql_query(query):
    """Executes a MySQL query after validating the table name."""
    try:
        words = query.split()
        if "FROM" in words:
            table_index = words.index("FROM") + 1
            table_name = words[table_index].strip("`;,")  # Extract table name safely
        else:
            return "Invalid SQL query structure."

        # Get existing tables
        valid_tables = get_table_names()
        if table_name not in valid_tables:
            return f"Error: Table `{table_name}` does not exist. Available tables: {valid_tables}"

        # Execute the validated query
        db: Session = next(get_db())
        result = db.execute(text(query)).fetchall()
        db.close()
        return [dict(row._mapping) for row in result]

    except Exception as e:
        return str(e)
def get_table_names():
    """Fetch all table names from the database."""
    try:
        db: Session = next(get_db())
        result = db.execute(text("SHOW TABLES")).fetchall()
        db.close()
        return [list(row)[0] for row in result]  # Extract table names
    except Exception as e:
        return str(e)

def query_database_function(question: str) -> str:
    """Generates and executes an SQL query based on a natural language question."""
    # Fetch available tables
    available_tables = get_table_names()
    tables_str = ", ".join(available_tables)

    # SQL generation prompt (same as original)
    sql_query_prompt = f"""
    The user wants to query the MySQL database. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 7 tables in my database, namely: invoicedetails, invoiceheader, podetails, poheader, suppliers, shipmentHeader, and shipmentDetails.    - **invoicedetails** has the following fields: id, itemId, itemQuantity, itemDescription, itemCost, totalItemCost, poId, invoiceNumber.
    - **invoicedetails**: id, itemId, itemQuantity, itemDescription, itemCost, totalItemCost, poId, invoiceNumber.
    - **invoiceheader**: invoiceId, invoicedate, invoiceType, currency, payment_term, invoice_status, total_qty, total_cost, total_tax, total_amount, userInvNo.
    - **podetails**: id, itemId, itemQuantity, supplierId, itemDescription, itemCost, totalItemCost, poId.
    - **poheader**: poNumber, shipByDate, leadTime, estimatedDeliveryDate, totalQuantity, totalCost, totalTax, currency, payment_term.
    - **suppliers**: supplierId, name, email, phone, address, lead_time.
    - **shipmentheader**: receiptId, asnId, expectedDate, receivedDate, receivedBy, status, sourceLocation, destinationLocation, totalQuantity, totalCost, poId.
    - **shipmentdetails**: id, itemId, itemDescription, itemCost, expectedItemQuantity, receivedItemQuantity, damagedItemQuantity, containerId, receiptId, invoiced.

    ### **Field Mapping Rules** 
    - "purchase order", "purchaseorder", "PURCHASE ORDER", "PURCHASEORDER", "Purchase Order", and "PurchaseOrder" refer to the **poheader** table.
    - "PO number", "po_number", "po no", "PONumber", "order number" refer to **poNumber**.
    - "po_id" and "poId" are equivalent and refer to **poId**.
    - "cost" and "total cost" refer to **totalCost**.
    - "amount" refers to **totalAmount**.
    - "invoice number" and "invoiceNumber" refer to **invoiceNumber**.
    - "payment terms" and "payment_term" are the same.
    - "lead time" and "lead_time" are the same.
    - "supplier" refers to the **suppliers** table.
    - "supplierId" and "supplier id" refer to **supplierId**.
    - "shipment", "shipment header", "receipt" refer to **shipmentHeader**.
    - "shipment details", "shipment detail", "shipment items" refer to **shipmentDetails**.
    - "receipt id" and "receiptId" refer to **receiptId** in **shipmentHeader** or **shipmentDetails**.
    - "ASN" refers to **asnId**.
    - "expected date" refers to **expectedDate**.
    - "received date" refers to **receivedDate**.
    - "received by" refers to **receivedBy**.
    - "status" refers to **status**.
    - "source location" refers to **sourceLocation**.
    - "destination location" refers to **destinationLocation**.
    - "total quantity" refers to **totalQuantity**.
    - "total cost" refers to **totalCost**.
    - "item id" refers to **itemId**.
    - "item description" refers to **itemDescription**.
    - "item cost" refers to **itemCost**.
    - "expected item quantity" refers to **expectedItemQuantity**.
    - "received item quantity" refers to **receivedItemQuantity**.
    - "damaged item quantity" refers to **damagedItemQuantity**.
    - "container id" refers to **containerId**.
    - "invoiced" refers to **invoiced**.
    -  "quality" and "quality issues" refer to the **percentage of damagedItemQuantity** calculated as:
        **(damagedItemQuantity / receivedItemQuantity) * 100**.

    ### **Case-Insensitive Handling**
    ### **Handling Special Characters & Spaces**
    """

    # Generate SQL query
    sql_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sql_query_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    
    # Extract and clean query
    mysql_query = sql_response.choices[0].message.content.strip()
    print("Query result: ",sql_response)
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate and execute
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result: ",result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error: ",{str(e)})
        return f"Error executing query: {str(e)}"
@app.post("/chat")
async def chat_with_po_assistant(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()
    
    chat_histories[user_id].append(f"User: {user_message}")
    conversation = "\n".join(chat_histories[user_id])

    try:
        # First API call - with function definition
        messages = [
            {"role": "system", "content": template_PO},
            {"role": "user", "content": conversation}
        ]

        functions = [{
            "name": "query_database",
            "description": "Retrieve data from the database using SQL queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"
                    }
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function(args["question"])
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })

            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Retain po_json from previous interaction if query_called is True
        if not query_called:
            user_po_details[user_id] = await categorize_po_details(bot_reply, user_id)

        po_json = user_po_details[user_id]  # Assign retained po_json

        chat_histories[user_id].append(f"Bot: {bot_reply}")
        print("PO JSON:", po_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Purchase Order created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("PO submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": chat_histories,
            "po_json": po_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adduser")
async def add_user(request:UserSchema,db:Session=Depends(get_db)):
    user = User(name=request.name,email=request.email,nickname=request.nickname)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/poCreation/", status_code=status.HTTP_201_CREATED)
def create_po(poHeader: poHeaderCreate, db: Session = Depends(get_db)):
    db_poHeader = models.PoHeader(**poHeader.dict())
    db.add(db_poHeader)
    db.commit()
    db.refresh(db_poHeader)
    return db_poHeader

@app.post("/poDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_poDetail(poDetailData: List[poDetailsCreate], db: Session = Depends(get_db)):
    for details in poDetailData:
        db_poDetails = models.PoDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            supplierId=details.supplierId
        )
        db.add(db_poDetails)
        db.commit()
        db.refresh(db_poDetails)
    return {
        "message":"Items added Sucessfully!"
    }
    # db_poDetails = models.PoDetails(**poDetailData.dict())
    # db.add(db_poDetails)
    # db.commit()
    # db.refresh(db_poDetails)
    # return [db_poDetails]

@app.post("/invCreation/", status_code=status.HTTP_201_CREATED)
def create_invoice(invHeader: invHeaderCreate, db: Session = Depends(get_db)):
    db_invHeader = models.InvHeader(**invHeader.dict())
    db.add(db_invHeader)
    db.commit()
    db.refresh(db_invHeader)
    return db_invHeader

@app.post("/invDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_invDetail(invDetailData: List[invDetailsCreate], db: Session = Depends(get_db)):
    for details in invDetailData:
        db_invDetails = models.InvDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            invoiceNumber= details.invoiceNumber
        )
        db.add(db_invDetails)
        db.commit()
        db.refresh(db_invDetails)
    return {
        "message":"Items added Sucessfully!"
    }

@app.get("/poDetails/{po_id}")
def read_poDeatils(po_id: str, db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        detail = "Po Number is not found in our database! Please add a valid PO number!"
        conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        
    po_info = db.query(models.PoDetails).filter(models.PoDetails.poId == po_id).all()
    return { "po_header":po,"po_details":po_info}

@app.post("/invoiceValidation")
def po_data_validations(po_id:str,detail:Dict[str,int],db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        raise HTTPException(status_code=404, detail="PO is not found!")
    for item,quantity in detail:
        po_details = db.query(models.PoDetails).filter(models.PoDetails.itemId==item).first()
        if po_details is None:
            detail = "Item which you added is not present in this PO"
            conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        if(po_details.itemQuantity>quantity):
            detail = po_details.itemId + "quantity is excced according to PO quantity is" + po_details.itemQuantity
            conversation.append('Bot: ' + detail)
            raise HTTPException(status_code=404, detail=conversation)
    return {"details":conversation}
        
    


@app.get("/invoiceDetails/{inv_id}")
def read_invDeatils(inv_id: str, db: Session = Depends(get_db)):
    inv = db.query(models.InvHeader).filter(models.InvHeader.invoiceId == inv_id).first()
    if inv is None:
        raise HTTPException(status_code=404, detail="Invoice not found!")
    inv_info = db.query(models.InvDetails).filter(models.InvDetails.invoiceNumber == inv_id).all()
    return { "inv_header":inv,"inv_details":inv_info}



@app.post("/ok")  
async def ok_endpoint(query: str, db: Session = Depends(get_db)):
    # Pass the db session to query_database_function_promo
    returned_attributes = query_database_function_promo(query, db)
    return {"message": "ok", "attributes": returned_attributes}
# @app.post("/ok")
# async def ok_endpoint(query:str,db: Session = Depends(get_db)):
#     returned_attributes=query_database_function_promo(query)
#     # returned_attributes=find_attributes(query)
#     # testModel()
#     # openaifunction()
#     # extractor = run_conversation("Invoice type: Debit Note PO number: PO123 Date: 26/06/2024 Items: ID123, ID124 Supplier Id: SUP1123  Total tax: 342  Quantity: 2, 4",db)

#     return {"message":"ok","attributes":returned_attributes}

@app.get
async def findPoDetails(po:str):
        db: Session = Depends(get_db)
        po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po).first()
        if po is None:
            return {"Po not found!"}
        else:
            return {"Po Found!"}


@app.post("/clearData")
async def clearConversation(submitted:str):
    conversation.clear()
    chat_histories.clear()
    submissionStatus = "not submitted"
    return {"conversation":conversation,"submissionStatus":"not submitted","chat_history":chat_histories}

@app.post("/testSubmission")
async def submiission(query:str):
    result=test_submission(query)
    return {"result":result}

@app.post("/uploadGpt/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_invoice_details(extracted_text)

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

@app.post("/uploadPo/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_po_details(extracted_text,"admin")

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

    
@app.post("/uploadOpenAi/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    extracted_text = extract_text_with_openai(file)
    return JSONResponse(content={"extracted_text": extracted_text})


@app.post("/upload/")
async def upload_invoice(file: UploadFile = File(...)):
    """API to upload an invoice file and extract details."""
    if file.content_type not in ["image/png", "image/jpeg", "application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, PDF, and TXT are supported.")

    # Read file bytes
    file_bytes = await file.read()

    # Extract text based on file type
    if file.content_type in ["image/png", "image/jpeg"]:
        image = Image.open(BytesIO(file_bytes))
        extracted_text = extract_text_from_image(image)
    elif file.content_type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
    else:  # Text file
        extracted_text = file_bytes.decode("utf-8")

    # Process extracted text
    invoice_details = extract_invoice_details(extracted_text)
    invoice_data_from_conversation = {
        "quantities": extract_invoice_details(extracted_text).get("quantities", []),
        "items": extract_invoice_details(extracted_text).get("items", [])
    }
    # invoice_json=json.dumps(invoice_details)
    # await generate_response(invoice_details)

    return {"file_name": file.filename, "invoice_details": invoice_details,"invoice_data_from_conversation":invoice_data_from_conversation,"extracted_text":extracted_text}


@app.post("/creation/response")
async def generate_response(query:str):
    conversation.append('User: ' + query)
    output = gpt_response(query)
    conversation.append('Bot: ' + output)
    extractor = ''
    # extractor = run_conversation(output)
    test_model_reply=testModel(query,output)
    form_submission=test_submission(output)
    action='action'
    submissionStatus="not submitted"
    bot_response_pattern = re.compile(r"Invoice\screated\ssuccessfully", re.IGNORECASE)
    past_invoice_regex = r"(past\s+invoice|invoice\s+past|last\s+invoice|invoice\s+last|previous\s+invoice|invoice\s+previous)\s*"
    invoice_regex = r"(create\s+invoice|invoice\s+create|create\san\sinvoice)\s*"
    patternInvoice = re.compile(invoice_regex, re.IGNORECASE)
    pastPatternInvoice = re.compile(past_invoice_regex, re.IGNORECASE)
    bot_response_pattern = re.compile(r"Invoice\screated\ssuccessfully", re.IGNORECASE)
    for line in conversation:
        if line.startswith("User:"):
            user_input = line.split(":")[1].strip().lower()
            if re.search(pastPatternInvoice, user_input):
                action = "last invoice created"
                # break
            elif re.search(patternInvoice, user_input):
                action = "create invoice"
            elif "create po" in user_input:
                action = "create PO"
        # elif line.startswith("Bot:"):
        #     if re.search(bot_response_pattern, line):
        #          submissionStatus = "submitted"
        #     else:
        #          submissionStatus="not submitted"
 
    # pattern = r"invoice type:(.*?), date:(.*?), po number:(.*?), supplier id:(.*?), total amount:(.*?), total tax:(.*?), items:(.*?), quantity:(.*?)$"
    # invoice_details = {
    #     "invoice type": None,
    #     "date": None,
    #     "po number": None,
    #     "supplier id":None,
    #     "total amount": None,
    #     "total tax": None,
    #     "items": [],
    #     "quantity": []
    # }
    # invoiceDatafromConversation=collect_invoice_data(conversation)    
    # regex_patterns = {
    #     "Invoice type": r"invoice\s+type\s*:\s*([^:]+)",
    #     "Date": r"date\s*:\s*(\d{2}/\d{2}/\d{4})",
    #     "PO number": r"po\s+number\s*:\s*(\w+)",
    #     "Supplier Id": r"supplier\s+id\s*:\s*(\w+)",
    #     "Total amount": r"total\s+amount\s*:\s*(\d+)",
    #     "Total tax": r"total\s+tax\s*:\s*(\d+)",
    #     "Items": r"items\s*:\s*(\w+)",
    #     "Quantity": r"quantity\s*:\s*(\d+)",
    # }
        
               
    # for line in conversation:
        
    #     if "User: " in line:
    #         match_invoice_type = re.match(r".*invoice\s*type\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
    #     elif "Bot: " in line:
    #         for detail, pattern in regex_patterns.items():
    #                 match = re.search(pattern, line, re.IGNORECASE)
    #                 if match:
    #                     invoice_details[detail.lower()] = match.group(1)
    # invoice_json = json.dumps(invoice_details, indent=4)
    invoiceDatafromConversation=collect_invoice_data(conversation)
    invoice_details = {
        "invoice type": None,
        "date": None,       
        "user invoice number":None,
        "po number": None,
        "total amount": None,
        "total tax": None,
        "items": [],
        "quantities": []
    }
    regex_patterns = {
    "invoice type": r"\*\*Invoice Type:\*\*\s*(.+?)\n",
    "date": r"\*\*Date:\*\*\s*(\d{2}/\d{2}/\d{4})",
    "po number": r"\*\*PO Number:\*\*\s*(\w+)",
    "user invoice number": r"\*\*Invoice Number:\*\*\s*(\w+)",
    "total amount": r"\*\*Total Amount:\*\*\s*([\d,]+)",  # Handles commas
    "total tax": r"\*\*Total Tax:\*\*\s*([\d,]+)",  # Handles commas
    "items": r"\*\*Items:\*\*\s*([\w,\s]+)",
    "quantities": r"\*\*Quantities:\*\*\s*([\d,\s]+)"
}

# Extract values using regex
    for line in conversation:
        for key, pattern in regex_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key in ["items", "quantities"]:  # Convert comma-separated values to lists
                    invoice_details[key] = [item.strip() for item in value.split(",")]
                else:
                    invoice_details[key] = value

    # Convert to JSON and print the result
    invoice_json = json.dumps(invoice_details, indent=4)
    print(invoice_json)
    # invDetails=await categorize_invoice_details(conversation)
    invDetails=await categorize_invoice_details_new(conversation,"admin")
    print("invDetails",invDetails)
    return {"conversation":conversation,"invoice_json":invDetails,"action":action,
    "submissionStatus":form_submission,"invoiceDatafromConversation":invDetails,
    "test_model_reply":test_model_reply,"extractor_info":extractor }
    # return {"conversation":conversation,"invoice_json":invoice_json,"action":action,
    # "submissionStatus":form_submission,"invoiceDatafromConversation":invoiceDatafromConversation,
    # "test_model_reply":test_model_reply,"extractor_info":extractor }

#old po
# @app.post("/chat")
# async def chat_with_po_assistant(request: ChatRequest):
#     """
#     Handles chat requests and processes PO creation using OpenAI API.
#     Returns both the bot's reply as a message and structured JSON data (or null if parsing fails).
#     """
#     user_id = request.user_id
#     user_message = request.message

#     # Initialize chat history & PO details if not exists
#     if user_id not in chat_histories:
#         chat_histories[user_id] = []
#         user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()

#     # Append user message to history
#     chat_histories[user_id].append(f"User: {user_message}")

#     # Create conversation history
#     conversation = "\n".join(chat_histories[user_id])

#     try:
#         # Generate bot response
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": template_PO},
#                 {"role": "user", "content": conversation}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )

#         bot_reply = response.choices[0].message.content
#         po_json = await categorize_po_details(bot_reply,user_id)

#         # Attempt to parse the response as JSON
#         print("Bot reply: ",bot_reply)
#         try:
#             structured_data = json.loads(bot_reply)
#         except json.JSONDecodeError:
#             structured_data = None  # If parsing fails, return null

#         # Update user PO details only if structured_data is valid
#         if structured_data:
#             user_po_details[user_id] = structured_data

#         # Append bot response to history
#         chat_histories[user_id].append(f"Bot: {bot_reply}")
#         print("PO_son:",po_json,"user_id: ",user_id)
#         # Determine submission status
#         if "Would you like to submit" in bot_reply:
#             submissionStatus = "pending"
#         elif "Purchase Order created successfully" in bot_reply:
#             submissionStatus = "submitted"
#         elif "I want to change something" in bot_reply:
#             submissionStatus = "cancelled"
#         else:
#             submissionStatus = "in_progress"  # Default state if no clear intent is detected
#         print("PO sumission status: ",submissionStatus)
#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "po_details": structured_data,
#             "chat_history": chat_histories,
#             "po_json": po_json,  # Will be null if parsing fails
#             "submissionStatus": submissionStatus
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat")
# async def chat_with_po_assistant(request: ChatRequest):
#     """
#     Handles chat requests and processes PO creation using OpenAI API.
#     Returns both the bot's reply as a message and structured JSON data (or null if parsing fails).
#     """
#     user_id = request.user_id
#     user_message = request.message

#     # Initialize chat history & PO details if not exists
#     if user_id not in chat_histories:
#         chat_histories[user_id] = []
#         user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()

#     # Append user message to history
#     chat_histories[user_id].append(f"User: {user_message}")
#     print(chat_histories)
#     # Create conversation history
#     conversation = "\n".join(chat_histories[user_id])

#     try:
#         # Use new OpenAI API format
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": template_PO},
#                 {"role": "user", "content": conversation}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )

#         bot_reply = response.choices[0].message.content  # New way to access the message
#         print(bot_reply)
#         po_json=await categorize_po_details(bot_reply)
#         # Attempt to parse the response as JSON
#         try:
#             structured_data = json.loads(bot_reply)
#         except json.JSONDecodeError:
#             structured_data = None  # If parsing fails, return null

#         # Update user PO details only if structured_data is valid
#         if structured_data:
#             user_po_details[user_id] = structured_data

#         # Append bot response to history
#         chat_histories[user_id].append(f"Bot: {bot_reply}")

#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "po_details": structured_data,
#             "chat_history":chat_histories,
#             "po_json":po_json # Will be null if parsing fails
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#02 March 2025#-2
#02 March 2025#
from fastapi import FastAPI,Depends,HTTPException,status
from utils import gpt_response,extract_information,conversation,checkValidation,run_conversation,collect_invoice_data,bot_action,openaifunction,testModel,test_submission,extract_invoice_details,extract_text_from_pdf,extract_text_from_image,collect_invoice_data_from_file,extract_text_with_openai,categorize_invoice_details,client,categorize_invoice_details_new
from poUtils import template_PO,DEFAULT_PO_STRUCTURE,categorize_po_details
from pydantic import BaseModel;
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails,Supplier,ShipmentHeader,ItemDiffs,ItemSupplier,ItemMaster
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach,ChatRequest,SupplierCreate,ShipmentHeader,ShipmentDetails,ItemDiffsSchema,ItemSupplierSchema,ItemMasterSchema
from schemas import ShipmentDetails as ShipmentDetailsSchema
from database import engine,SessionLocal,get_db
from promoUtils import template_Promotion
from sqlalchemy.orm import Session
import models
from typing import List,Any
from fastapi.middleware.cors import CORSMiddleware
from utils_extractor import run_conversation
from typing import Dict
import easyocr
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import openai
# from exampleextrator import run_conversation
from fastapi.responses import JSONResponse
import mysql.connector
Base.metadata.create_all(bind=engine)
from sqlalchemy.sql import text
import re;
import json;
import os
import signal
from collections import defaultdict

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#PO Chatbot Functions
chat_histories = {}
user_po_details = {}


#Promo
# Database Functions
def get_valid_hierarchy_values(hierarchy_level: str) -> list:
    """Fetch valid values for a hierarchy level from itemsMaster"""
    column_map = {
        "department": "itemDepartment",
        "class": "itemClass",
        "subclass": "itemSubClass"
    }
    
    if hierarchy_level.lower() not in column_map:
        return []
    
    column = column_map[hierarchy_level.lower()]
    query = f"SELECT DISTINCT `{column}` FROM itemmaster"
    result = execute_mysql_query(query)
    return [str(row[column]) for row in result if row[column]]

def validate_items(item_ids: list) -> dict:
    """Validate item IDs against database"""
    if not item_ids:
        return {"valid": [], "invalid": []}
    
    ids_str = ",".join([f"'{id.strip()}'" for id in item_ids])
    query = f"SELECT itemId FROM itemmaster WHERE itemId IN ({ids_str})"
    valid_ids = [row["itemId"] for row in execute_mysql_query(query)]
    invalid_ids = list(set(item_ids) - set(valid_ids))
    
    return {"valid": valid_ids, "invalid": invalid_ids}


# promotion_sql_prompt =  """
# I have 3 tables in my database: **itemsMaster, itemSupplier, and itemDiffs**. These tables store item-related data, including attributes, classifications, suppliers, and differentiation details.

# ### **Table Structures & Field Descriptions**  

# #### **1. itemsMaster** (Stores core item details)  
# - **`itemId`**: Unique identifier for each item.  
# - **`itemDescription`**: Primary description of the item.  
# - **`itemSecondaryDescription`**: Additional details about the item.  
# - **`itemDepartment`**: The broader category an item belongs to (e.g., T-Shirt, Trousers, Jackets).  
# - **`itemClass`**: A classification within a department (e.g., Formals, Casuals, Leather).  
# - **`itemSubClass`**: A more granular classification under the item class (e.g., Full Sleeve, Half Sleeve, Zipper, Regular Fit).  
# - **`brand`**: The brand associated with the item (e.g., Zara, Adidas, H&M).  
# - **`diffType1`, `diffType2`, `diffType3`**: Foreign keys linking to `itemDiffs.id`, representing specific item attributes such as color, size, or material.  

# #### **3. itemSupplier** (Stores supplier and pricing details)  
# - **`id`**: Unique identifier for each supplier-item relationship.  
# - **`supplierCost`**: The cost of the item from the supplier.  
# - **`supplierId`**: The identifier for the supplier providing the item.  
# - **`itemId`**: Foreign key linking to `itemsMaster.itemId`, establishing the relationship between items and suppliers. 

# #### **2. itemDiffs** (Defines item differentiations)  
# - **`id`**: Unique identifier for each differentiation type.  
# - **`diffType`**: The attribute type used to differentiate items (e.g., color, size, material).  
#   - If the user wants to filter items by a specific attribute (e.g., color or size), the query should check `diffType` and retrieve the corresponding `diffId`.  
# - **`diffId`**: The actual differentiation value corresponding to `diffType`.  
#   - Example: If `diffType = 'color'`, then `diffId` could be "Red"; if `diffType = 'size'`, then `diffId` could be "M".  

# ### **MANDATORY QUERY RULES**
# If a user query includes a filter based on a differentiation attribute (diffType) such as color, size, or material:  
# 1. NEVER reference color/size/material directly in itemsMaster
# 2. ALWAYS use this pattern for attribute filters:
#     -**Identify the `diffId`**:  
#     - Query `itemDiffs` to find the `id` where `diffType` matches the requested attribute (e.g., "color") and `diffId` matches the requested value (e.g., "Red").  
#     - Example query:  
#         ```sql
#         SELECT id FROM itemDiffs WHERE diffType = 'color' AND diffId = 'Red';
#         ```
#     - **Use the retrieved `id` to filter `itemsMaster` on `diffType1`, `diffType2`, or `diffType3`**  
#     - Example Query:
#         ```sql
#         SELECT itemId, itemDescription  
#         FROM itemsMaster  
#         WHERE diffType1 = <retrieved_id> OR diffType2 = <retrieved_id> OR diffType3 = <retrieved_id>;
#         ```
#     - **Final Query Output**:  
#     - The final SQL should dynamically adapt, ensuring correct joins and conditions based on the input query.
#     ### **EXAMPLE QUERIES for diffs**
#         User: "Red items"
#         SQL:
#         SELECT i.itemId 
#         FROM itemsMaster i
#         WHERE i.diffType1 IN (SELECT id FROM itemDiffs WHERE diffType='color' AND diffId='red')
#         OR i.diffType2 IN (SELECT id FROM itemDiffs WHERE diffType='color' AND diffId='red')
#         OR i.diffType3 IN (SELECT id FROM itemDiffs WHERE diffType='color' AND diffId='red')

#         User: "Medium sized dresses"
#         SQL:
#         SELECT i.itemId 
#         FROM itemsMaster i
#         WHERE (i.diffType1 IN (SELECT id FROM itemDiffs WHERE diffType='size' AND diffId='medium')
#         OR i.diffType2 IN (SELECT id FROM itemDiffs WHERE diffType='size' AND diffId='medium')
#         OR i.diffType3 IN (SELECT id FROM itemDiffs WHERE diffType='size' AND diffId='medium'))
#         AND i.itemClass = 'Dresses'

#     ### **ERROR PREVENTION**
#         - If unsure about diffType mapping, ASK FOR CLARIFICATION
#         - On validation failure, RETURN ERROR CODE: INVALID_DIFF_REFERENCE 


# ### **General Query Generation Rules**  

# 1. The SQL query should always return `itemId` as the primary identifier.  
# 2. Use `JOIN` statements when querying for differentiation attributes (`diffType`).  
# 3. Apply `WHERE` clauses for filtering based on user-defined conditions.  
# 4. Handle lists of values (e.g., multiple brands or departments) using `IN()` clauses.  
# 5. Ensure the referenced tables and columns exist before generating queries.  
# 6. Use backticks (\`) around column names to prevent SQL syntax errors. 
# 7. If a differentiation attribute (like **color** or **size**) is provided, first fetch its ID from `itemDiffs`, then use it to filter `itemsMaster`.  
 

# ### **Expected SQL Output**  

# - The generated SQL query should be a **fully structured SQL query** without additional explanations, comments, or descriptions.  
# - It should dynamically adapt to user queries, ensuring proper field mappings and query optimizations based on the rules outlined above.  
# """
def validate_items_query(query: str) -> bool:
    diff_attributes = ['color', 'size', 'material']
    query_lower = query.lower()
    for attr in diff_attributes:
        if f'itemmaster.`{attr}`' in query_lower or f'where {attr} =' in query_lower:
            return False
    return True

def query_database_function(question: str, db: Session) -> str:
    # Modified SQL Generation Prompt
    available_tables = get_table_names()
    # Define the tables you want to include
    included_tables = ["itemMaster", "itemSupplier", "itemDiffs"]
    # Filter available tables to only include the ones in included_tables
    filtered_tables = [table for table in available_tables if table in included_tables]
    # Create the string of tables for the query
    tables_str = ", ".join(filtered_tables)
    promotion_sql_prompt = """
    You are a SQL assistant for fashion retail data. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 3 tables in my database, namely:
    - `itemMaster` (aliased as `im`)  
    - `itemSupplier` (aliased as `isup`)  
    - `itemDiffs` (aliased as `idf`)  

    ### **Critical Rules:**  

    #### **1. Color, Size, and Material Filters**  
    - If the user searches for an **attribute** like **color, size, or material**, follow this strict process:  
    1. **Find the `id` in `itemDiffs`** where `diffType` matches the attribute (e.g., `'color'`) and `diffId` is the user-provided value (e.g., `'Red'`).  
    2. **Filter `itemMaster` (`im`)** by checking if the retrieved `idf.id` exists in **`im.diffType1`**, **`im.diffType2`**, or **`im.diffType3`**.  
    3. **NEVER** search for colors, sizes, or materials inside `itemMaster` directly.  
    4. **NEVER** use `LIKE` on `itemDescription` (e.g., avoid `WHERE itemDescription LIKE 'red'`).  

    #### **2. Allowed Tables Only**  
    - You may only use the following tables:  
    - `itemMaster` (Base Table):
        - **`itemId`**: Unique identifier for each item.  
        - **`itemDescription`**: Primary description of the item.  
        - **`itemSecondaryDescription`**: Additional details about the item.  
        - **`itemDepartment`**: The broader category an item belongs to (e.g., T-Shirt, Trousers, Jackets).  
        - **`itemClass`**: A classification within a department (e.g., Formals, Casuals, Leather).  
        - **`itemSubClass`**: A more granular classification under the item class (e.g., Full Sleeve, Half Sleeve, Zipper, Regular Fit).  
        - **`brand`**: The brand associated with the item (e.g., Zara, Adidas, H&M).  
        - **`diffType1`, `diffType2`, `diffType3`**: Foreign keys linking to `itemDiffs.id`, representing specific item attributes such as color, size, or material.   
    - `itemSupplier` (For Cost & Supplier Data):
        - **`id`**: Unique identifier for each supplier-item relationship.  
        - **`supplierCost`**: The cost of the item from the supplier.  
        - **`supplierId`**: The identifier for the supplier providing the item.  
        - **`itemId`**: Foreign key linking to `itemsMaster.itemId`, establishing the relationship between items and suppliers.   
    - `itemDiffs` (For Attribute Filtering):
        - **`id`**: Unique identifier for each differentiation type.  
        - **`diffType`**: The attribute type used to differentiate items (e.g., color, size, material).  
          - If the user wants to filter items by a specific attribute (e.g., color or size), the query should check `diffType` and retrieve the corresponding `diffId`.  
        - **`diffId`**: The actual differentiation value corresponding to `diffType`.  
          - Example: If `diffType = 'color'`, then `diffId` could be "Red"; if `diffType = 'size'`, then `diffId` could be "M".    

    #### **3. Query Format & Execution**  
    - **Start queries from `itemMaster`** as the primary table.  
    - Use **explicit JOINs** when needed (e.g., joining `itemSupplier` for cost-related queries).  
    # - **DO NOT** use `UNION`.  
    - **Return only valid SQL queries**. Do not include explanations or markdown formatting.  
    - The generated SQL query should be a **fully structured SQL query** without additional explanations, comments, or descriptions.  
    - It should dynamically adapt to user queries, ensuring proper field mappings and query optimizations based on the rules outlined above.  

    ---
    ### **SQL Examples:**

    #### **Example 1: Select All Red Colored Items**
    User: *"Select all red colored items"*
    ```sql
    SELECT im.itemId
    FROM itemMaster im
    WHERE im.diffType1 IN (
        SELECT id FROM itemDiffs WHERE diffType = 'color' AND diffId = 'Red'
    ) OR im.diffType2 IN (
        SELECT id FROM itemDiffs WHERE diffType = 'color' AND diffId = 'Red'
    ) OR im.diffType3 IN (
        SELECT id FROM itemDiffs WHERE diffType = 'color' AND diffId = 'Red'
    )
    ```
    #### **Example 2: Select all red colored items with a supplier cost below $50**
    User: *"Select all red colored items with a supplier cost below $50"*
     ```sql
    SELECT im.itemId, isup.supplierCost
    FROM itemMaster im
    JOIN itemSupplier isup ON im.itemId = isup.itemId
    WHERE isup.supplierCost < 50
    AND (im.diffType1 IN (
        SELECT id FROM itemDiffs WHERE diffType = 'color' AND diffId = 'Red'
    ) OR im.diffType2 IN (
        SELECT id FROM itemDiffs WHERE diffType = 'color' AND diffId = 'Red'
    ) OR im.diffType3 IN (
        SELECT id FROM itemDiffs WHERE diffType = 'color' AND diffId = 'Red'
    ))
     ```
    #### **Example 3: Select All Items of Size "Large"**
    User: *"Select All Items of Size "Large""*
    ```sql
    SELECT im.itemId
    FROM itemMaster im
    WHERE im.diffType1 IN (
        SELECT id FROM itemDiffs WHERE diffType = 'size' AND diffId = 'Large'
    ) OR im.diffType2 IN (
        SELECT id FROM itemDiffs WHERE diffType = 'size' AND diffId = 'Large'
    ) OR im.diffType3 IN (
        SELECT id FROM itemDiffs WHERE diffType = 'size' AND diffId = 'Large'
    )
    ```

    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": promotion_sql_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_tokens=300
    )
    
    query = response.choices[0].message.content
    query = query.replace("```sql", "").replace("```", "").strip()
    
    if not validate_items_query(query):
        return "Invalid attribute reference detected in query"
    
    try:
        result = db.execute(query).fetchall()
        return json.dumps([dict(row) for row in result])
    except Exception as e:
        return f"Query failed: {str(e)}"


promo_states = defaultdict(dict)
DEFAULT_PROMO_STRUCTURE = {
    "type": "",
    "hierarchy": {"level": "", "value": ""},
    "items": [],
    "excluded_items": [],
    "discount": {"type": "", "amount": 0},
    "dates": {"start": "", "end": ""},
    "locations": [],
    "excluded_locations": [],
    "status": "draft"
}

# Modified API endpoint
user_promo_details={}
@app.post("/promo-chat")
async def handle_promotion_chat(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in promo_states:
        promo_states[user_id] = []
        user_promo_details[user_id] = DEFAULT_PROMO_STRUCTURE.copy()
    
    promo_states[user_id].append(f"User: {user_message}")
    conversation = "\n".join(promo_states[user_id])

    try:
        # First API call - with function definition
        messages = [
            {"role": "system", "content": template_Promotion},
            {"role": "user", "content": conversation}
        ]

        functions = [{
            "name": "query_database",
            "description": "Execute database queries for validation/data retrieval",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function(args["question"])
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })

            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Retain promo_json from previous interaction if query_called is True
        # if not query_called:
        #     user_promo_details[user_id] = await categorize_promo_details(bot_reply, user_id)

        # promo_json = user_promo_details[user_id]  # Assign retained promo_json

        promo_states[user_id].append(f"Bot: {bot_reply}")
        # print("Promo JSON:", promo_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Promotion created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("Promotion submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": promo_states,
            # "promo_json": promo_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/promo-chat")
# async def handle_promotion_chat(request: ChatRequest):
#     user_id = request.user_id
#     message = request.message
    
#     # Initialize user state if new
#     # if user_id not in promo_states:
#     #     promo_states[user_id] = {
#     #         "details": DEFAULT_PROMO_STRUCTURE.copy(),
#     #         "chat_history": []
#     #     }
    
#     # # Get current state
#     # current_state = promo_states[user_id]

#     if user_id not in promo_states:
#         promo_states[user_id] = []
#         user_promo_details[user_id] = DEFAULT_PO_STRUCTURE.copy()
    
#     promo_states[user_id].append(f"User: {message}")
#     conversation = "\n".join(promo_states[user_id])

    
#     # Update chat history
#     # current_state["chat_history"].append(f"User: {message}")
    
#     # Process message
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": template_Promotion},
#             {"role": "user", "content": message}
#         ],
#         functions=[{
#             "name": "query_database",
#             "description": "Execute database queries for validation/data retrieval",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "question": {"type": "string"}
#                 },
#                 "required": ["question"]
#             }
#         }]
#     )
    
#     # Handle function calls
#     if response.choices[0].message.function_call:
#         function_call = response.choices[0].message.function_call
#         if function_call.name == "query_database":
#             args = json.loads(function_call.arguments)
#             query_result = query_database_function(args["question"])            
#             print("Query result:",query_result)

#             # Process result and update state
            
#     return {
#         "response": response.choices[0].message.content,
#         # "status": current_state["details"]["status"],
#         "chat_hstory":promo_states, # Corrected access
#         "conversation":conversation
#     }
#DIFF
# GET all diffs
@app.get("/diffs", response_model=list[ItemDiffsSchema])
def get_diffs(db: Session = Depends(get_db)):
    diffs = db.query(models.ItemDiffs).all()
    return diffs

# POST a new diff
@app.post("/diffs", response_model=ItemDiffsSchema)
def create_diff(diff: ItemDiffsSchema, db: Session = Depends(get_db)):
    db_diff = models.ItemDiffs(**diff.dict())

    # Check if diff already exists
    existing_diff = db.query(models.ItemDiffs).filter(models.ItemDiffs.id == diff.id).first()
    if existing_diff:
        raise HTTPException(status_code=400, detail="Diff with this ID already exists")

    db.add(db_diff)
    db.commit()
    db.refresh(db_diff)
    return db_diff
#ITEM
@app.get("/items", response_model=List[ItemMasterSchema])
def get_items(db: Session = Depends(get_db)):
    items = db.query(models.ItemMaster).all()
    return items

@app.post("/items", response_model=List[ItemMasterSchema])
def create_items(items: List[ItemMasterSchema], db: Session = Depends(get_db)):
    new_items = [models.ItemMaster(**item.dict()) for item in items]
    db.add_all(new_items)
    db.commit()
    return new_items


#SHIPMENT 
@app.post("/shipments/", response_model=ShipmentHeader)
def create_shipment(shipment: ShipmentHeader, db: Session = Depends(get_db)):
    db_shipment = models.ShipmentHeader(**shipment.dict())
    db.add(db_shipment)
    db.commit()
    db.refresh(db_shipment)
    return db_shipment

@app.post("/shipments/{receipt_id}/details", response_model=List[ShipmentDetailsSchema])
def add_shipment_details(receipt_id: str, details: List[ShipmentDetailsSchema], db: Session = Depends(get_db)):
    shipment = db.query(models.ShipmentHeader).filter(models.ShipmentHeader.receiptId == receipt_id).first()
    if not shipment:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    po_items = {po_detail.itemId for po_detail in db.query(models.PoDetails).filter(models.PoDetails.poId == shipment.poId).all()}
    
    for detail in details:
        if detail.itemId not in po_items:
            raise HTTPException(status_code=400, detail=f"Item {detail.itemId} is not in the PO {shipment.poId}")
    
    db_details = [models.ShipmentDetails(**{**detail.dict(), "receiptId": receipt_id}) for detail in details]
    db.add_all(db_details)
    db.commit()
    for db_detail in db_details:
        db.refresh(db_detail)
        
    return db_details

@app.get("/shipments/{receipt_id}", response_model=Dict[str, Any])
def get_shipment_with_details(receipt_id: str, db: Session = Depends(get_db)):
    shipment = db.query(ShipmentHeader).filter(ShipmentHeader.receiptId == receipt_id).first()
    if shipment is None:
        raise HTTPException(status_code=404, detail="Shipment not found")
    
    details = db.query(ShipmentDetails).filter(ShipmentDetails.receiptId == receipt_id).all()
    return {"shipment": shipment, "details": details}


#SUPPPLIER
@app.post("/suppliers/", response_model=SupplierCreate)
def create_supplier(supplier: SupplierCreate, db: Session = Depends(get_db)):
    # Check if supplier already exists
    existing_supplier = db.query(Supplier).filter(Supplier.email == supplier.email).first()
    if existing_supplier:
        raise HTTPException(status_code=400, detail="Supplier with this email already exists")

    new_supplier = Supplier(
        supplierId=supplier.supplierId,
        name=supplier.name,
        email=supplier.email,
        phone=supplier.phone,
        address=supplier.address,
        lead_time=supplier.lead_time
    )
    db.add(new_supplier)
    db.commit()
    db.refresh(new_supplier)
    return new_supplier

# Get all suppliers
@app.get("/suppliers/")
def get_suppliers(db: Session = Depends(get_db)):
    return db.query(Supplier).all()

# Get supplier by ID
@app.get("/suppliers/{supplierId}")
def get_supplier(supplierId : str, db: Session = Depends(get_db)):
    supplier = db.query(Supplier).filter(Supplier.supplierId  == supplierId ).first()
    if not supplier:
        raise HTTPException(status_code=404, detail="Supplier not found")
    return supplier

#po chat
def execute_mysql_query(query):
    """Executes a MySQL query after validating the table name."""
    try:
        words = query.split()
        if "FROM" in words:
            table_index = words.index("FROM") + 1
            table_name = words[table_index].strip("`;,")  # Extract table name safely
        else:
            return "Invalid SQL query structure."

        # Get existing tables
        valid_tables = get_table_names()
        if table_name not in valid_tables:
            return f"Error: Table `{table_name}` does not exist. Available tables: {valid_tables}"

        # Execute the validated query
        db: Session = next(get_db())
        result = db.execute(text(query)).fetchall()
        db.close()
        return [dict(row._mapping) for row in result]

    except Exception as e:
        return str(e)
def get_table_names():
    """Fetch all table names from the database."""
    try:
        db: Session = next(get_db())
        result = db.execute(text("SHOW TABLES")).fetchall()
        db.close()
        return [list(row)[0] for row in result]  # Extract table names
    except Exception as e:
        return str(e)

def query_database_function(question: str) -> str:
    """Generates and executes an SQL query based on a natural language question."""
    # Fetch available tables
    available_tables = get_table_names()
    tables_str = ", ".join(available_tables)

    # SQL generation prompt (same as original)
    sql_query_prompt = f"""
    The user wants to query the MySQL database. The following tables exist: {tables_str}.
    Generate a **pure SQL query** without explanations, comments, or descriptions.
    I have 7 tables in my database, namely: invoicedetails, invoiceheader, podetails, poheader, suppliers, shipmentHeader, and shipmentDetails.    - **invoicedetails** has the following fields: id, itemId, itemQuantity, itemDescription, itemCost, totalItemCost, poId, invoiceNumber.
    - **invoicedetails**: id, itemId, itemQuantity, itemDescription, itemCost, totalItemCost, poId, invoiceNumber.
    - **invoiceheader**: invoiceId, invoicedate, invoiceType, currency, payment_term, invoice_status, total_qty, total_cost, total_tax, total_amount, userInvNo.
    - **podetails**: id, itemId, itemQuantity, supplierId, itemDescription, itemCost, totalItemCost, poId.
    - **poheader**: poNumber, shipByDate, leadTime, estimatedDeliveryDate, totalQuantity, totalCost, totalTax, currency, payment_term.
    - **suppliers**: supplierId, name, email, phone, address, lead_time.
    - **shipmentheader**: receiptId, asnId, expectedDate, receivedDate, receivedBy, status, sourceLocation, destinationLocation, totalQuantity, totalCost, poId.
    - **shipmentdetails**: id, itemId, itemDescription, itemCost, expectedItemQuantity, receivedItemQuantity, damagedItemQuantity, containerId, receiptId, invoiced.

    ### **Field Mapping Rules** 
    - "purchase order", "purchaseorder", "PURCHASE ORDER", "PURCHASEORDER", "Purchase Order", and "PurchaseOrder" refer to the **poheader** table.
    - "PO number", "po_number", "po no", "PONumber", "order number" refer to **poNumber**.
    - "po_id" and "poId" are equivalent and refer to **poId**.
    - "cost" and "total cost" refer to **totalCost**.
    - "amount" refers to **totalAmount**.
    - "invoice number" and "invoiceNumber" refer to **invoiceNumber**.
    - "payment terms" and "payment_term" are the same.
    - "lead time" and "lead_time" are the same.
    - "supplier" refers to the **suppliers** table.
    - "supplierId" and "supplier id" refer to **supplierId**.
    - "shipment", "shipment header", "receipt" refer to **shipmentHeader**.
    - "shipment details", "shipment detail", "shipment items" refer to **shipmentDetails**.
    - "receipt id" and "receiptId" refer to **receiptId** in **shipmentHeader** or **shipmentDetails**.
    - "ASN" refers to **asnId**.
    - "expected date" refers to **expectedDate**.
    - "received date" refers to **receivedDate**.
    - "received by" refers to **receivedBy**.
    - "status" refers to **status**.
    - "source location" refers to **sourceLocation**.
    - "destination location" refers to **destinationLocation**.
    - "total quantity" refers to **totalQuantity**.
    - "total cost" refers to **totalCost**.
    - "item id" refers to **itemId**.
    - "item description" refers to **itemDescription**.
    - "item cost" refers to **itemCost**.
    - "expected item quantity" refers to **expectedItemQuantity**.
    - "received item quantity" refers to **receivedItemQuantity**.
    - "damaged item quantity" refers to **damagedItemQuantity**.
    - "container id" refers to **containerId**.
    - "invoiced" refers to **invoiced**.
    -  "quality" and "quality issues" refer to the **percentage of damagedItemQuantity** calculated as:
        **(damagedItemQuantity / receivedItemQuantity) * 100**.

    ### **Case-Insensitive Handling**
    ### **Handling Special Characters & Spaces**
    """

    # Generate SQL query
    sql_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sql_query_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    
    # Extract and clean query
    mysql_query = sql_response.choices[0].message.content.strip()
    print("Query result: ",sql_response)
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate and execute
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result: ",result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error: ",{str(e)})
        return f"Error executing query: {str(e)}"
@app.post("/chat")
async def chat_with_po_assistant(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()
    
    chat_histories[user_id].append(f"User: {user_message}")
    conversation = "\n".join(chat_histories[user_id])

    try:
        # First API call - with function definition
        messages = [
            {"role": "system", "content": template_PO},
            {"role": "user", "content": conversation}
        ]

        functions = [{
            "name": "query_database",
            "description": "Retrieve data from the database using SQL queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string", 
                        "description": "Natural language question requiring database data"
                    }
                },
                "required": ["question"]
            }
        }]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )

        response_message = response.choices[0].message
        bot_reply = response_message.content
        function_call = response_message.function_call
        query_called = False

        # Handle function call
        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            query_result = query_database_function(args["question"])
            query_called = True

            # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })

            # Second API call with function result
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        # Retain po_json from previous interaction if query_called is True
        if not query_called:
            user_po_details[user_id] = await categorize_po_details(bot_reply, user_id)

        po_json = user_po_details[user_id]  # Assign retained po_json

        chat_histories[user_id].append(f"Bot: {bot_reply}")
        print("PO JSON:", po_json, "User ID:", user_id)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Purchase Order created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        print("PO submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": chat_histories,
            "po_json": po_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat")
# async def chat_with_po_assistant(request: ChatRequest):
#     user_id = request.user_id
#     user_message = request.message

#     # ... (Keep existing user session setup code)
#     if user_id not in chat_histories:
#         chat_histories[user_id] = []
#         user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()
#     chat_histories[user_id].append(f"User: {user_message}")
#     conversation = "\n".join(chat_histories[user_id])

#     try:
#         # First API call - with function definition
#         messages = [
#             {"role": "system", "content": template_PO},
#             {"role": "user", "content": conversation}
#         ]

#         functions = [{
#             "name": "query_database",
#             "description": "Retrieve data from the database using SQL queries",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "question": {
#                         "type": "string", 
#                         "description": "Natural language question requiring database data"
#                     }
#                 },
#                 "required": ["question"]
#             }
#         }]

#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=messages,
#             functions=functions,
#             function_call="auto",
#             temperature=0.7,
#             max_tokens=500
#         )

#         response_message = response.choices[0].message
#         bot_reply = response_message.content
#         print("Response bot reply: ",response_message)
#         function_call = response_message.function_call
#         query_called = False
#         # Handle function call
#         if function_call and function_call.name == "query_database":
#             args = json.loads(function_call.arguments)
#             query_result = query_database_function(args["question"])
#             query_called = True
#             # Append function response to messages
#             messages.append({
#                 "role": "function", 
#                 "name": "query_database",
#                 "content": query_result
#             })

#             # Second API call with function result
#             second_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             bot_reply = second_response.choices[0].message.content
#         # po_json={'Supplier ID': None, 'Estimated Delivery Date': None, 'Total Quantity': None, 'Total Cost': None, 'Total Tax': None, 'Items': None}
#         po_json={}
#         if not query_called:
#             po_json = await categorize_po_details(bot_reply, user_id)
#         # po_json = await categorize_po_details(bot_reply,user_id)
#         chat_histories[user_id].append(f"Bot: {bot_reply}")
#         print("PO_son:",po_json,"user_id: ",user_id)
#         # Determine submission status
#         if "Would you like to submit" in bot_reply:
#             submissionStatus = "pending"
#         elif "Purchase Order created successfully" in bot_reply:
#             submissionStatus = "submitted"
#         elif "I want to change something" in bot_reply:
#             submissionStatus = "cancelled"
#         else:
#             submissionStatus = "in_progress"  # Default state if no clear intent is detected
#         print("PO sumission status: ",submissionStatus)
#         # ... (Keep existing PO processing and response handling code)

#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "chat_history": chat_histories,
#             "po_json": po_json,  # Will be null if parsing fails
#             "submissionStatus": submissionStatus
#             # ... (other response fields)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/adduser")
async def add_user(request:UserSchema,db:Session=Depends(get_db)):
    user = User(name=request.name,email=request.email,nickname=request.nickname)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/poCreation/", status_code=status.HTTP_201_CREATED)
def create_po(poHeader: poHeaderCreate, db: Session = Depends(get_db)):
    db_poHeader = models.PoHeader(**poHeader.dict())
    db.add(db_poHeader)
    db.commit()
    db.refresh(db_poHeader)
    return db_poHeader

@app.post("/poDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_poDetail(poDetailData: List[poDetailsCreate], db: Session = Depends(get_db)):
    for details in poDetailData:
        db_poDetails = models.PoDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            supplierId=details.supplierId
        )
        db.add(db_poDetails)
        db.commit()
        db.refresh(db_poDetails)
    return {
        "message":"Items added Sucessfully!"
    }
    # db_poDetails = models.PoDetails(**poDetailData.dict())
    # db.add(db_poDetails)
    # db.commit()
    # db.refresh(db_poDetails)
    # return [db_poDetails]

@app.post("/invCreation/", status_code=status.HTTP_201_CREATED)
def create_invoice(invHeader: invHeaderCreate, db: Session = Depends(get_db)):
    db_invHeader = models.InvHeader(**invHeader.dict())
    db.add(db_invHeader)
    db.commit()
    db.refresh(db_invHeader)
    return db_invHeader

@app.post("/invDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_invDetail(invDetailData: List[invDetailsCreate], db: Session = Depends(get_db)):
    for details in invDetailData:
        db_invDetails = models.InvDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            invoiceNumber= details.invoiceNumber
        )
        db.add(db_invDetails)
        db.commit()
        db.refresh(db_invDetails)
    return {
        "message":"Items added Sucessfully!"
    }

@app.get("/poDetails/{po_id}")
def read_poDeatils(po_id: str, db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        detail = "Po Number is not found in our database! Please add a valid PO number!"
        conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        
    po_info = db.query(models.PoDetails).filter(models.PoDetails.poId == po_id).all()
    return { "po_header":po,"po_details":po_info}

@app.post("/invoiceValidation")
def po_data_validations(po_id:str,detail:Dict[str,int],db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        raise HTTPException(status_code=404, detail="PO is not found!")
    for item,quantity in detail:
        po_details = db.query(models.PoDetails).filter(models.PoDetails.itemId==item).first()
        if po_details is None:
            detail = "Item which you added is not present in this PO"
            conversation.append('Bot: ' + detail)
        raise HTTPException(status_code=404, detail=conversation)
        if(po_details.itemQuantity>quantity):
            detail = po_details.itemId + "quantity is excced according to PO quantity is" + po_details.itemQuantity
            conversation.append('Bot: ' + detail)
            raise HTTPException(status_code=404, detail=conversation)
    return {"details":conversation}
        
    


@app.get("/invoiceDetails/{inv_id}")
def read_invDeatils(inv_id: str, db: Session = Depends(get_db)):
    inv = db.query(models.InvHeader).filter(models.InvHeader.invoiceId == inv_id).first()
    if inv is None:
        raise HTTPException(status_code=404, detail="Invoice not found!")
    inv_info = db.query(models.InvDetails).filter(models.InvDetails.invoiceNumber == inv_id).all()
    return { "inv_header":inv,"inv_details":inv_info}



    

@app.get("/ok")
async def ok_endpoint(db: Session = Depends(get_db)):
    # testModel()
    # openaifunction()
    # extractor = run_conversation("Invoice type: Debit Note PO number: PO123 Date: 26/06/2024 Items: ID123, ID124 Supplier Id: SUP1123  Total tax: 342  Quantity: 2, 4",db)

    return {"message":"ok"}

@app.get
async def findPoDetails(po:str):
        db: Session = Depends(get_db)
        po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po).first()
        if po is None:
            return {"Po not found!"}
        else:
            return {"Po Found!"}


@app.post("/clearData")
async def clearConversation(submitted:str):
    conversation.clear()
    chat_histories.clear()
    submissionStatus = "not submitted"
    return {"conversation":conversation,"submissionStatus":"not submitted","chat_history":chat_histories}

@app.post("/testSubmission")
async def submiission(query:str):
    result=test_submission(query)
    return {"result":result}

@app.post("/uploadGpt/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_invoice_details(extracted_text)

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

@app.post("/uploadPo/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    extracted_text = await extract_text_with_openai(file)
    structured_data = await categorize_po_details(extracted_text,"admin")

    return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

    
@app.post("/uploadOpenAi/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    extracted_text = extract_text_with_openai(file)
    return JSONResponse(content={"extracted_text": extracted_text})


@app.post("/upload/")
async def upload_invoice(file: UploadFile = File(...)):
    """API to upload an invoice file and extract details."""
    if file.content_type not in ["image/png", "image/jpeg", "application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, PDF, and TXT are supported.")

    # Read file bytes
    file_bytes = await file.read()

    # Extract text based on file type
    if file.content_type in ["image/png", "image/jpeg"]:
        image = Image.open(BytesIO(file_bytes))
        extracted_text = extract_text_from_image(image)
    elif file.content_type == "application/pdf":
        extracted_text = extract_text_from_pdf(file_bytes)
    else:  # Text file
        extracted_text = file_bytes.decode("utf-8")

    # Process extracted text
    invoice_details = extract_invoice_details(extracted_text)
    invoice_data_from_conversation = {
        "quantities": extract_invoice_details(extracted_text).get("quantities", []),
        "items": extract_invoice_details(extracted_text).get("items", [])
    }
    # invoice_json=json.dumps(invoice_details)
    # await generate_response(invoice_details)

    return {"file_name": file.filename, "invoice_details": invoice_details,"invoice_data_from_conversation":invoice_data_from_conversation,"extracted_text":extracted_text}


@app.post("/creation/response")
async def generate_response(query:str):
    conversation.append('User: ' + query)
    output = gpt_response(query)
    conversation.append('Bot: ' + output)
    extractor = ''
    # extractor = run_conversation(output)
    test_model_reply=testModel(query,output)
    form_submission=test_submission(output)
    action='action'
    submissionStatus="not submitted"
    bot_response_pattern = re.compile(r"Invoice\screated\ssuccessfully", re.IGNORECASE)
    past_invoice_regex = r"(past\s+invoice|invoice\s+past|last\s+invoice|invoice\s+last|previous\s+invoice|invoice\s+previous)\s*"
    invoice_regex = r"(create\s+invoice|invoice\s+create|create\san\sinvoice)\s*"
    patternInvoice = re.compile(invoice_regex, re.IGNORECASE)
    pastPatternInvoice = re.compile(past_invoice_regex, re.IGNORECASE)
    bot_response_pattern = re.compile(r"Invoice\screated\ssuccessfully", re.IGNORECASE)
    for line in conversation:
        if line.startswith("User:"):
            user_input = line.split(":")[1].strip().lower()
            if re.search(pastPatternInvoice, user_input):
                action = "last invoice created"
                # break
            elif re.search(patternInvoice, user_input):
                action = "create invoice"
            elif "create po" in user_input:
                action = "create PO"
        # elif line.startswith("Bot:"):
        #     if re.search(bot_response_pattern, line):
        #          submissionStatus = "submitted"
        #     else:
        #          submissionStatus="not submitted"
 
    # pattern = r"invoice type:(.*?), date:(.*?), po number:(.*?), supplier id:(.*?), total amount:(.*?), total tax:(.*?), items:(.*?), quantity:(.*?)$"
    # invoice_details = {
    #     "invoice type": None,
    #     "date": None,
    #     "po number": None,
    #     "supplier id":None,
    #     "total amount": None,
    #     "total tax": None,
    #     "items": [],
    #     "quantity": []
    # }
    # invoiceDatafromConversation=collect_invoice_data(conversation)    
    # regex_patterns = {
    #     "Invoice type": r"invoice\s+type\s*:\s*([^:]+)",
    #     "Date": r"date\s*:\s*(\d{2}/\d{2}/\d{4})",
    #     "PO number": r"po\s+number\s*:\s*(\w+)",
    #     "Supplier Id": r"supplier\s+id\s*:\s*(\w+)",
    #     "Total amount": r"total\s+amount\s*:\s*(\d+)",
    #     "Total tax": r"total\s+tax\s*:\s*(\d+)",
    #     "Items": r"items\s*:\s*(\w+)",
    #     "Quantity": r"quantity\s*:\s*(\d+)",
    # }
        
               
    # for line in conversation:
        
    #     if "User: " in line:
    #         match_invoice_type = re.match(r".*invoice\s*type\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
    #     elif "Bot: " in line:
    #         for detail, pattern in regex_patterns.items():
    #                 match = re.search(pattern, line, re.IGNORECASE)
    #                 if match:
    #                     invoice_details[detail.lower()] = match.group(1)
    # invoice_json = json.dumps(invoice_details, indent=4)
    invoiceDatafromConversation=collect_invoice_data(conversation)
    invoice_details = {
        "invoice type": None,
        "date": None,       
        "user invoice number":None,
        "po number": None,
        "total amount": None,
        "total tax": None,
        "items": [],
        "quantities": []
    }
    regex_patterns = {
    "invoice type": r"\*\*Invoice Type:\*\*\s*(.+?)\n",
    "date": r"\*\*Date:\*\*\s*(\d{2}/\d{2}/\d{4})",
    "po number": r"\*\*PO Number:\*\*\s*(\w+)",
    "user invoice number": r"\*\*Invoice Number:\*\*\s*(\w+)",
    "total amount": r"\*\*Total Amount:\*\*\s*([\d,]+)",  # Handles commas
    "total tax": r"\*\*Total Tax:\*\*\s*([\d,]+)",  # Handles commas
    "items": r"\*\*Items:\*\*\s*([\w,\s]+)",
    "quantities": r"\*\*Quantities:\*\*\s*([\d,\s]+)"
}

# Extract values using regex
    for line in conversation:
        for key, pattern in regex_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if key in ["items", "quantities"]:  # Convert comma-separated values to lists
                    invoice_details[key] = [item.strip() for item in value.split(",")]
                else:
                    invoice_details[key] = value

    # Convert to JSON and print the result
    invoice_json = json.dumps(invoice_details, indent=4)
    print(invoice_json)
    # invDetails=await categorize_invoice_details(conversation)
    invDetails=await categorize_invoice_details_new(conversation,"admin")
    print("invDetails",invDetails)
    return {"conversation":conversation,"invoice_json":invDetails,"action":action,
    "submissionStatus":form_submission,"invoiceDatafromConversation":invDetails,
    "test_model_reply":test_model_reply,"extractor_info":extractor }
    # return {"conversation":conversation,"invoice_json":invoice_json,"action":action,
    # "submissionStatus":form_submission,"invoiceDatafromConversation":invoiceDatafromConversation,
    # "test_model_reply":test_model_reply,"extractor_info":extractor }

#old po
# @app.post("/chat")
# async def chat_with_po_assistant(request: ChatRequest):
#     """
#     Handles chat requests and processes PO creation using OpenAI API.
#     Returns both the bot's reply as a message and structured JSON data (or null if parsing fails).
#     """
#     user_id = request.user_id
#     user_message = request.message

#     # Initialize chat history & PO details if not exists
#     if user_id not in chat_histories:
#         chat_histories[user_id] = []
#         user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()

#     # Append user message to history
#     chat_histories[user_id].append(f"User: {user_message}")

#     # Create conversation history
#     conversation = "\n".join(chat_histories[user_id])

#     try:
#         # Generate bot response
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": template_PO},
#                 {"role": "user", "content": conversation}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )

#         bot_reply = response.choices[0].message.content
#         po_json = await categorize_po_details(bot_reply,user_id)

#         # Attempt to parse the response as JSON
#         print("Bot reply: ",bot_reply)
#         try:
#             structured_data = json.loads(bot_reply)
#         except json.JSONDecodeError:
#             structured_data = None  # If parsing fails, return null

#         # Update user PO details only if structured_data is valid
#         if structured_data:
#             user_po_details[user_id] = structured_data

#         # Append bot response to history
#         chat_histories[user_id].append(f"Bot: {bot_reply}")
#         print("PO_son:",po_json,"user_id: ",user_id)
#         # Determine submission status
#         if "Would you like to submit" in bot_reply:
#             submissionStatus = "pending"
#         elif "Purchase Order created successfully" in bot_reply:
#             submissionStatus = "submitted"
#         elif "I want to change something" in bot_reply:
#             submissionStatus = "cancelled"
#         else:
#             submissionStatus = "in_progress"  # Default state if no clear intent is detected
#         print("PO sumission status: ",submissionStatus)
#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "po_details": structured_data,
#             "chat_history": chat_histories,
#             "po_json": po_json,  # Will be null if parsing fails
#             "submissionStatus": submissionStatus
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat")
# async def chat_with_po_assistant(request: ChatRequest):
#     """
#     Handles chat requests and processes PO creation using OpenAI API.
#     Returns both the bot's reply as a message and structured JSON data (or null if parsing fails).
#     """
#     user_id = request.user_id
#     user_message = request.message

#     # Initialize chat history & PO details if not exists
#     if user_id not in chat_histories:
#         chat_histories[user_id] = []
#         user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()

#     # Append user message to history
#     chat_histories[user_id].append(f"User: {user_message}")
#     print(chat_histories)
#     # Create conversation history
#     conversation = "\n".join(chat_histories[user_id])

#     try:
#         # Use new OpenAI API format
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": template_PO},
#                 {"role": "user", "content": conversation}
#             ],
#             temperature=0.7,
#             max_tokens=500
#         )

#         bot_reply = response.choices[0].message.content  # New way to access the message
#         print(bot_reply)
#         po_json=await categorize_po_details(bot_reply)
#         # Attempt to parse the response as JSON
#         try:
#             structured_data = json.loads(bot_reply)
#         except json.JSONDecodeError:
#             structured_data = None  # If parsing fails, return null

#         # Update user PO details only if structured_data is valid
#         if structured_data:
#             user_po_details[user_id] = structured_data

#         # Append bot response to history
#         chat_histories[user_id].append(f"Bot: {bot_reply}")

#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "po_details": structured_data,
#             "chat_history":chat_histories,
#             "po_json":po_json # Will be null if parsing fails
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#02 March 2025#


#20 June 2024 #
from fastapi import FastAPI,Depends,HTTPException,status
from utils import gpt_response,extract_information,conversation,checkValidation,run_conversation,collect_invoice_data,bot_action,openaifunction,testModel
from pydantic import BaseModel;
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach
from database import engine,SessionLocal,get_db
from sqlalchemy.orm import Session
import models
from typing import List
from fastapi.middleware.cors import CORSMiddleware


Base.metadata.create_all(bind=engine)

import re;
import json;
import os
import signal

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3000/*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# def get_db():
#     try:
#         db = SessionLocal()
#         yield db
#     finally:
#         db.close()



@app.post("/adduser")
async def add_user(request:UserSchema,db:Session=Depends(get_db)):
    user = User(name=request.name,email=request.email,nickname=request.nickname)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.post("/poCreation/", status_code=status.HTTP_201_CREATED)
def create_po(poHeader: poHeaderCreate, db: Session = Depends(get_db)):
    db_poHeader = models.PoHeader(**poHeader.dict())
    db.add(db_poHeader)
    db.commit()
    db.refresh(db_poHeader)
    return db_poHeader

# Create a Class for a Teacher
@app.post("/poDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_poDetail(poDetailData: poDetailsCreate, db: Session = Depends(get_db)):
    
    db_poDetails = models.PoDetails(**poDetailData.dict())
    db.add(db_poDetails)
    db.commit()
    db.refresh(db_poDetails)
    return [db_poDetails]

@app.post("/invCreation/", status_code=status.HTTP_201_CREATED)
def create_invoice(invHeader: invHeaderCreate, db: Session = Depends(get_db)):
    db_invHeader = models.InvHeader(**invHeader.dict())
    db.add(db_invHeader)
    db.commit()
    db.refresh(db_invHeader)
    return db_invHeader

# Create a Class for a Teacher
@app.post("/invDetailsAdd/", status_code=status.HTTP_201_CREATED)
def create_invDetail(invDetailData: List[invDetailsCreate], db: Session = Depends(get_db)):
    for details in invDetailData:
        db_invDetails = models.InvDetails(
            itemId= details.itemId,
            itemQuantity= details.itemQuantity,
            itemDescription=details.itemDescription,
            itemCost= details.itemCost,
            totalItemCost= details.totalItemCost,
            poId=details.poId,
            invoiceNumber= details.invoiceNumber
        )
        db.add(db_invDetails)
        db.commit()
        db.refresh(db_invDetails)
    return {
        "message":"Items added Sucessfully!"
    }

@app.get("/poDetails/{po_id}")
def read_poDeatils(po_id: str, db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    if po is None:
        raise HTTPException(status_code=404, detail="po not found!")
    po_info = db.query(models.PoDetails).filter(models.PoDetails.poId == po_id).all()
    return { "po_header":po,"po_details":po_info}



@app.get("/ok")
async def ok_endpoint():
    # testModel()
    return {"message":"ok"}


@app.post("/clearData")
async def clearConversation(submitted:str):
    conversation.clear()
    submissionStatus = "not submitted"
    return {"conversation":conversation,"submissionStatus":"not submitted"}


@app.post("/creation/response")
async def generate_response(query:str):
    conversation.append('User: ' + query)
    output = gpt_response(query)
    conversation.append('Bot: ' + output)
    # action = run_conversation(query)
    test_model_reply=testModel(query)
    # res = openai.Completion.create(model="gpt-4-1106-preview", prompt=query + '\n\n###\n\n', max_tokens=1, temperature=0, logprobs=2)
    # action = res['choices'][0]['text']
    action='action'
    submissionStatus="not submitted"
    # patternInvoice = re.compile(r'\binvoice\b', re.IGNORECASE)
    bot_response_pattern = re.compile(r"Invoice\screated\ssuccessfully", re.IGNORECASE)

    # previousPatternInvoice=re.compile(r'\binvoice\b', re.IGNORECASE)
    # past_invoice_regex= r"(past\s+invoice|invoice\s+past|last\s+invoice|invoice\s+last|previous\s+invoice|invoice\s+previous)\s*"
    # invoice_regex = r"(create\s+invoice|invoice\s+create)\s*"
    # patternInvoice = re.compile(invoice_regex, re.IGNORECASE)
    # pastPatternInvoice=re.compile(past_invoice_regex,re.IGNORECASE)
    past_invoice_regex = r"(past\s+invoice|invoice\s+past|last\s+invoice|invoice\s+last|previous\s+invoice|invoice\s+previous)\s*"
    invoice_regex = r"(create\s+invoice|invoice\s+create|create\san\sinvoice)\s*"
    patternInvoice = re.compile(invoice_regex, re.IGNORECASE)
    pastPatternInvoice = re.compile(past_invoice_regex, re.IGNORECASE)
    bot_response_pattern = re.compile(r"Invoice\screated\ssuccessfully", re.IGNORECASE)
    for line in conversation:
        if line.startswith("User:"):
            user_input = line.split(":")[1].strip().lower()
            if re.search(pastPatternInvoice, user_input):
                action = "last invoice created"
                break
            elif re.search(patternInvoice, user_input):
                action = "create invoice"
                # break
            elif "create po" in user_input:
                action = "create PO"
                # break
        elif line.startswith("Bot:"):
            if re.search(bot_response_pattern, line):
                 submissionStatus = "submitted"
                #  conversation.clear()
                #  action='action'
            else:
                 submissionStatus="not submitted"
 
    pattern = r"invoice type:(.*?), date:(.*?), po number:(.*?), supplier id:(.*?), total amount:(.*?), total tax:(.*?), items:(.*?), quantity:(.*?)$"
    # invoice_details = {}

    # # Iterate through conversation to find invoice details
    # for line in conversation:
    #     match = re.match(r"User: invoice type:(.*?), date:(.*?), po number:(.*?), total amount:(.*?), total tax:(.*?), items:(.*?), quantity:(.*?)$", line)
    #     if match:
    #         invoice_details = {
    #             "invoice type": match.group(1).strip(),
    #             "date": match.group(2).strip(),
    #             "po number": match.group(3).strip(),
    #             "total amount": match.group(4).strip(),
    #             "total tax": match.group(5).strip(),
    #             "items": match.group(6).strip(),
    #             "quantity": match.group(7).strip()
    #         }
    #         break

    # # Convert invoice details to JSON format
    # invoice_json = json.dumps(invoice_details, indent=4)
    invoice_details = {
        "invoice type": None,
        "date": None,
        "po number": None,
        "supplier id":None,
        "total amount": None,
        "total tax": None,
        "items": None,
        "quantity": None
    }
    invoiceDatafromConversation=collect_invoice_data(conversation)
    # Iterate through conversation to find and update invoice details
    # for line in conversation:
    #     match = re.match(r"User: invoice type:(.*?), date:(.*?), po number:(.*?), total amount:(.*?), total tax:(.*?), items:(.*?), quantity:(.*?)$", line)
    #     if match:
    #         invoice_details.update({
    #             "invoice type": match.group(1).strip(),
    #             "date": match.group(2).strip(),
    #             "po number": match.group(3).strip(),
    #             "total amount": match.group(4).strip(),
    #             "total tax": match.group(5).strip(),
    #             "items": match.group(6).strip(),
    #             "quantity": match.group(7).strip()
    #         })
    # itemObject=[]
    
    regex_patterns = {
        "Invoice type": r"invoice\s+type\s*:\s*([^:]+)",
        "Date": r"date\s*:\s*(\d{2}/\d{2}/\d{4})",
        "PO number": r"po\s+number\s*:\s*(\w+)",
        "Supplier Id": r"supplier\s+id\s*:\s*(\w+)",
        "Total amount": r"total\s+amount\s*:\s*(\d+)",
        "Total tax": r"total\s+tax\s*:\s*(\d+)",
        "Items": r"items\s*:\s*(\w+)",
        "Quantity": r"quantity\s*:\s*(\d+)",
    }
        
               
    for line in conversation:
        
        if "User: " in line:
            match_invoice_type = re.match(r".*invoice\s*type\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
            # if match_invoice_type:
            #     invoice_details["invoice type"] = match_invoice_type.group(1).strip()

            # match_date = re.match(r".*date\s*\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
            # if match_date:
            #     invoice_details["date"] = match_date.group(1).strip()

            # match_po_number = re.match(r".*po\s*number\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
            # if match_po_number:
            #     invoice_details["po number"] = match_po_number.group(1).strip()

            # match_supplier_id =re.match(r".*supplier\s*id\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
            # if match_supplier_id:
            #     invoice_details["supplier id"] = match_supplier_id.group(1).strip()

            # match_total_amount = re.match(r".*total\s*amount\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
            # if match_total_amount:
            #     invoice_details["total amount"] = match_total_amount.group(1).strip()

            # match_total_tax = re.match(r".*total\s*tax\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
            # if match_total_tax:
            #     invoice_details["total tax"] = match_total_tax.group(1).strip()

            # # match_items = re.search(r'items:\s*(\[?.*?\]?)$', line)
            # # if match_items:
            # #     invoice_details["items"] = format(match_items.group(1))
            # match_items = re.match(r".*items: ?(.*?)(?:,|$)", line)
            # if match_items:
            #     # invoice_details["items"] = match_items.group()
            #     invoice_details["items"] = [item.strip() for item in match_items.group(1).split(',')]
            #     # itemObject=match_items.group


            # match_quantity = re.match(r".*quantity: ?(.*?)(?:,|$)", line)
            # if match_quantity:
            #     invoice_details["quantity"] = [quantity.strip() for quantity in match_quantity.group(1).split(',')]
        # elif "You can provide all the details at once, separated by commas. Here's an example format" in line:
        #     break
        elif "Bot: " in line:
            for detail, pattern in regex_patterns.items():
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        invoice_details[detail.lower()] = match.group(1) 
            # if "For example" in line:
            #     break
            # else:
            #     for detail, pattern in regex_patterns.items():
            #         match = re.search(pattern, line, re.IGNORECASE)
            #         if match:
            #             invoice_details[detail.lower()] = match.group(1) 



    invoice_json = json.dumps(invoice_details, indent=4)

    # pattern_bot = re.compile(r'"content": "Action: (creation|fetch)"')
#     bot_action_final=""
# # Iterate through messages and find matches
#     for message in bot_action:
#         content = message.get("content", "")
#         match = re.search(pattern_bot, content)
#         if match:
#             bot_action_final=match
#         else:
#             bot_action_final="No match"
      
    # print(action)
    return {"conversation":conversation,"invoice_json":invoice_json,"action":action,"submissionStatus":submissionStatus,"invoiceDatafromConversation":invoiceDatafromConversation,"test_model_reply":test_model_reply }
    # return {"conversation":conversation,"invoice_json":invoice_json,"action":action,"submissionStatus":submissionStatus,"invoiceDatafromConversation":invoiceDatafromConversation,"bot_action":bot_action }

#20 June 2024 #
# from fastapi import FastAPI,Depends,HTTPException,status
# from utils import gpt_response,extract_information,conversation,checkValidation,run_conversation
# from pydantic import BaseModel;
# from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails
# from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach
# from database import engine,SessionLocal,get_db
# from sqlalchemy.orm import Session
# import models
# from typing import List
# from fastapi.middleware.cors import CORSMiddleware


# Base.metadata.create_all(bind=engine)

# import re;
# import json;
# import os
# import signal

# app = FastAPI()

# origins = [
#     "http://localhost.tiangolo.com",
#     "https://localhost.tiangolo.com",
#     "http://localhost",
#     "http://localhost:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # def get_db():
# #     try:
# #         db = SessionLocal()
# #         yield db
# #     finally:
# #         db.close()



# @app.post("/adduser")
# async def add_user(request:UserSchema,db:Session=Depends(get_db)):
#     user = User(name=request.name,email=request.email,nickname=request.nickname)
#     db.add(user)
#     db.commit()
#     db.refresh(user)
#     return user

# @app.post("/poCreation/", status_code=status.HTTP_201_CREATED)
# def create_po(poHeader: poHeaderCreate, db: Session = Depends(get_db)):
#     db_poHeader = models.PoHeader(**poHeader.dict())
#     db.add(db_poHeader)
#     db.commit()
#     db.refresh(db_poHeader)
#     return db_poHeader

# # Create a Class for a Teacher
# @app.post("/poDetailsAdd/", status_code=status.HTTP_201_CREATED)
# def create_poDetail(poDetailData: poDetailsCreate, db: Session = Depends(get_db)):
    
#     db_poDetails = models.PoDetails(**poDetailData.dict())
#     db.add(db_poDetails)
#     db.commit()
#     db.refresh(db_poDetails)
#     return [db_poDetails]

# @app.post("/invCreation/", status_code=status.HTTP_201_CREATED)
# def create_invoice(invHeader: invHeaderCreate, db: Session = Depends(get_db)):
#     db_invHeader = models.InvHeader(**invHeader.dict())
#     db.add(db_invHeader)
#     db.commit()
#     db.refresh(db_invHeader)
#     return db_invHeader

# # Create a Class for a Teacher
# @app.post("/invDetailsAdd/", status_code=status.HTTP_201_CREATED)
# def create_invDetail(invDetailData: List[invDetailsCreate], db: Session = Depends(get_db)):
#     for details in invDetailData:
#         db_invDetails = models.InvDetails(
#             itemId= details.itemId,
#             itemQuantity= details.itemQuantity,
#             itemDescription=details.itemDescription,
#             itemCost= details.itemCost,
#             totalItemCost= details.totalItemCost,
#             poId=details.poId,
#             invoiceNumber= details.invoiceNumber
#         )
#         db.add(db_invDetails)
#         db.commit()
#         db.refresh(db_invDetails)
#     return {
#         "message":"Items added Sucessfully!"
#     }

# @app.get("/poDetails/{po_id}")
# def read_poDeatils(po_id: str, db: Session = Depends(get_db)):
#     po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
#     if po is None:
#         raise HTTPException(status_code=404, detail="po not found!")
#     po_info = db.query(models.PoDetails).filter(models.PoDetails.poId == po_id).all()
#     return { "po_header":po,"po_details":po_info}



# @app.get("/ok")
# async def ok_endpoint():
#     return {"message":"ok"}

# @app.post("/creation/response")
# async def generate_response(query:str):
#     conversation.append('User: ' + query)
#     output = gpt_response(query)
#     conversation.append('Bot: ' + output)
#     action='action'
#     submissionStatus="not submitted"
#     patternInvoice = re.compile(r'\binvoice\b', re.IGNORECASE)
#     bot_response_pattern = re.compile(r"Invoice\screated\ssuccessfully", re.IGNORECASE)

#     for line in conversation:
#         if line.startswith("User:"):
#             user_input = line.split(":")[1].strip().lower()
#             if re.search(patternInvoice, user_input):
#                 action = "create invoice"
#                 # break
#             elif "create po" in user_input:
#                 action = "create PO"
#                 # break
#         elif line.startswith("Bot:"):
#             if re.search(bot_response_pattern, line):
#                  submissionStatus = "submitted"
#                  break
 
#     pattern = r"invoice type:(.*?), date:(.*?), po number:(.*?), supplier id:(.*?), total amount:(.*?), total tax:(.*?), items:(.*?), quantity:(.*?)$"
#     # invoice_details = {}

#     # # Iterate through conversation to find invoice details
#     # for line in conversation:
#     #     match = re.match(r"User: invoice type:(.*?), date:(.*?), po number:(.*?), total amount:(.*?), total tax:(.*?), items:(.*?), quantity:(.*?)$", line)
#     #     if match:
#     #         invoice_details = {
#     #             "invoice type": match.group(1).strip(),
#     #             "date": match.group(2).strip(),
#     #             "po number": match.group(3).strip(),
#     #             "total amount": match.group(4).strip(),
#     #             "total tax": match.group(5).strip(),
#     #             "items": match.group(6).strip(),
#     #             "quantity": match.group(7).strip()
#     #         }
#     #         break

#     # # Convert invoice details to JSON format
#     # invoice_json = json.dumps(invoice_details, indent=4)
#     invoice_details = {
#         "invoice type": None,
#         "date": None,
#         "po number": None,
#         "supplier id":None,
#         "total amount": None,
#         "total tax": None,
#         "items": None,
#         "quantity": None
#     }

#     # Iterate through conversation to find and update invoice details
#     # for line in conversation:
#     #     match = re.match(r"User: invoice type:(.*?), date:(.*?), po number:(.*?), total amount:(.*?), total tax:(.*?), items:(.*?), quantity:(.*?)$", line)
#     #     if match:
#     #         invoice_details.update({
#     #             "invoice type": match.group(1).strip(),
#     #             "date": match.group(2).strip(),
#     #             "po number": match.group(3).strip(),
#     #             "total amount": match.group(4).strip(),
#     #             "total tax": match.group(5).strip(),
#     #             "items": match.group(6).strip(),
#     #             "quantity": match.group(7).strip()
#     #         })
#     # itemObject=[]
#     for line in conversation:
#         if "User: " in line:
#             match_invoice_type = re.match(r".*invoice\s*type\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
#             if match_invoice_type:
#                 invoice_details["invoice type"] = match_invoice_type.group(1).strip()

#             match_date = re.match(r".*date: ?(.*?)(?:,|$)", line)
#             if match_date:
#                 invoice_details["date"] = match_date.group(1).strip()

#             match_po_number = re.match(r".*po\s*number\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
#             if match_po_number:
#                 invoice_details["po number"] = match_po_number.group(1).strip()

#             match_supplier_id =re.match(r".*supplier\s*id\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
#             if match_supplier_id:
#                 invoice_details["supplier id"] = match_supplier_id.group(1).strip()

#             match_total_amount = re.match(r".*total\s*amount\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
#             if match_total_amount:
#                 invoice_details["total amount"] = match_total_amount.group(1).strip()

#             match_total_tax = re.match(r".*total\s*tax\s*: ?(.*?)(?:,|$)", line, re.IGNORECASE)
#             if match_total_tax:
#                 invoice_details["total tax"] = match_total_tax.group(1).strip()

#             # match_items = re.search(r'items:\s*(\[?.*?\]?)$', line)
#             # if match_items:
#             #     invoice_details["items"] = format(match_items.group(1))
#             match_items = re.match(r".*items: ?(.*?)(?:,|$)", line)
#             if match_items:
#                 # invoice_details["items"] = match_items.group()
#                 invoice_details["items"] = [item.strip() for item in match_items.group(1).split(',')]
#                 # itemObject=match_items.group


#             match_quantity = re.match(r".*quantity: ?(.*?)(?:,|$)", line)
#             if match_quantity:
#                 invoice_details["quantity"] = [quantity.strip() for quantity in match_quantity.group(1).split(',')]


#     invoice_json = json.dumps(invoice_details, indent=4)
#     # if None in invoice_details.values():
#     #   print("Not all details are filled.")
#     # else:
#     #   result = run_conversation(invoice_json)
#     #   print(result)
#     #   conversation.append('Bot: ' + result)

      
#     # print(action)
#     return {"conversation":conversation,"invoice_json":invoice_json,"action":action,"submissionStatus":submissionStatus,  }
