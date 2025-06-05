import base64
from fastapi import BackgroundTasks, FastAPI,Depends, Form,HTTPException,status
from fastapi_mail import FastMail, MessageSchema, MessageType
from insightGeneration import generate_supplier_insights,plot_all_graphs
from utils import gpt_response,extract_information,conversation,checkValidation,run_conversation,collect_invoice_data,bot_action,openaifunction,testModel,test_submission,extract_invoice_details,extract_text_from_pdf,extract_text_from_image,collect_invoice_data_from_file,extract_text_with_openai,categorize_invoice_details,client,categorize_invoice_details_new,previous_invoice_details,template_5,extract_details_gpt_vision,client_new,llm_gpt3,llm_gpt4,async_client ,template_5_new
from poUtils import template_PO,DEFAULT_PO_STRUCTURE,categorize_po_details,previous_po_details
from pydantic import BaseModel, EmailStr;
from models import Base, StoreDetails,User,PoDetails,PoHeader,InvHeader,InvDetails,Supplier,ShipmentHeader,ItemDiffs,ItemSupplier,ItemMaster,PromotionDetails,PromotionHeader
from schemas import ChatRequestUser, StoreDetailsSchema, UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach,ChatRequest,SupplierCreate,ShipmentHeader,ShipmentDetails,ItemDiffsSchema,ItemSupplierSchema,ItemMasterSchema,PromotionDetailsSchema,PromotionHeaderSchema
from schemas import ShipmentDetails as ShipmentDetailsSchema
from database import engine,SessionLocal,get_db
from promoUtils import extract_and_classify_promo_details, template_Promotion,categorize_promo_details,previous_promo_details,categorize_promo_details_fun_call,categorize_promo_details_new
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
from fastapi.responses import JSONResponse,Response
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
from send_email import conf 

import plotly.graph_objects as go
import plotly.io as pio
import io

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



@app.get("/plot")
async def plot_png():
    fill_rate = 100.0
    pending_rate = 0.0
    non_defective_rate = 98.6
    defective_rate = 1.4
    avg_delay = 2.0
    risk_score = 0.1
    graphs = await plot_all_graphs(fill_rate,pending_rate,non_defective_rate,defective_rate,avg_delay,risk_score)
    return {"graphs": graphs}
    
#Email Functionality
@app.post("/filenew")
async def send_file_new(
    file: UploadFile = File(...),
    email: EmailStr = Form(...),
    body: str = Form(...)
) -> JSONResponse:
    try:
        body_dict: Dict[str, Any] = json.loads(body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"message": "Invalid JSON in 'body' field."})

    message = MessageSchema(
        subject="Fastapi-Mail module",
        recipients=[email],
        template_body=body_dict,
        subtype=MessageType.html,
        attachments=[file]  # ✅ Pass UploadFile directly
    )

    fm = FastMail(conf)
    await fm.send_message(message, template_name="email-document.html")
    return JSONResponse(status_code=200, content={"message": "Email has been sent."})
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
#     await fm.send_message(message, template_name="email.html")
#     return JSONResponse(status_code=200, content={"message": "Email has been sent."})

class EmailSchema(BaseModel):
    email: List[EmailStr]

class EmailSchemaBody(BaseModel):
    email: List[EmailStr]
    body: Dict[str, Any]
    
class BodySchema(BaseModel):
    body: Dict[str, Any]
    
    
@app.post("/email-with-body")
async def send_with_template(email: EmailSchemaBody) -> JSONResponse:

    message = MessageSchema(
        subject="Fastapi-Mail module",
        recipients=email.dict().get("email"),
        template_body=email.dict().get("body"),
        subtype=MessageType.html,
        )

    fm = FastMail(conf)
    await fm.send_message(message, template_name="email.html") 
    return JSONResponse(status_code=200, content={"message": "email has been sent"})

#Store
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
    # Optionally, include the store IDs from the many-to-many relationship:
    stores = [store.storeId for store in promoHeader.stores]
    return {
        "promotion_header": promoHeader,
        "promotion_details": promoDetails,
        "storeIds": stores
    }

@app.get("/promotionDetails/{promotionId}", response_model=List[PromotionDetailsSchema])
def get_promotion_details(promotionId: str, db: Session = Depends(get_db)):
    promoDetails = db.query(models.PromotionDetails).filter(models.PromotionDetails.promotionId == promotionId).all()
    if not promoDetails:
        raise HTTPException(status_code=404, detail="No details found for this promotion")
    return promoDetails

@app.post("/promotionHeader/", status_code=status.HTTP_201_CREATED)
def create_promotion_header(promoHeader: PromotionHeaderSchema, db: Session = Depends(get_db)):
    promo_data = promoHeader.dict()
    store_ids = promo_data.pop("storeIds", [])

    # Retrieve store objects
    stores = db.query(models.StoreDetails).filter(models.StoreDetails.storeId.in_(store_ids)).all()

    # Validate all store IDs exist
    if len(stores) != len(store_ids):
        raise HTTPException(status_code=400, detail="One or more store IDs are invalid")

    # Create PromotionHeader instance
    db_promoHeader = models.PromotionHeader(**promo_data)
    
    # Associate stores explicitly
    db_promoHeader.stores.extend(stores)

    # Persist in DB
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

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
def fetch_store_id_from_query(store_query: str, db: Session) -> list:
    """
    Extract and fetch store ID(s) from the storedetails table based on the provided store_query.
    The store_query can be a store ID (e.g., "STORE001") or contain details like a city or state.
    """
    store_sql_prompt = f"""
    You are a SQL assistant for retail store data. The storedetails table has the following columns. Use only these columns for the query:
      - storeId
      - storeName
      - address
      - city
      - state
      - zipCode
      - phone

    Based on the following store query: {store_query}
    Generate a pure SQL query (without explanations, markdown formatting, or comments).
    1. Returns ONLY storeId column
    2. Filters using these priority rules in strict order:
    a. If explicitly requesting all stores (e.g., "Select all stores", "All locations"), return ALL storeIds
    b. Direct store ID match (e.g., 'STORE001') if present
    c. City/state/address matches (e.g., 'New Delhi')
    d. Phone/zip/name matches using LIKE for partial matches
    e. Return empty set if: 
        - Query mentions missing stores/locations 
        - Contains only promotional details without store info
        - Has placeholder text like "Not provided"
    3. Never assume store IDs unless explicitly specified

    ### **SQL Examples:**

    #### **Example 1: Select All Stores in New Delhi
    User: "Select all stores in New Delhi"
    ```sql
    Copy
    SELECT sd.storeId
    FROM storedetails sd
    WHERE sd.city = 'Delhi'
    ```
    #### **Example 2: Select All Items with 30% off from the FashionX Brand in the T-Shirt Department at STORE005
    User: "Create a promotion offering 30% off all yellow items from the FashionX Brand in the T-Shirt Department at STORE005, valid from 16/04/2025 until the end of May 2025."

    ```sql
    SELECT sd.storeId
    FROM storedetails sd
    WHERE sd.storeId = 'STORE005'
    ```
    #### **Example 3: Select all stores with storeId STORE006
    User: "stores: STORE006"

    ```sql
    SELECT sd.storeId
    FROM storedetails sd
    WHERE sd.storeId = 'STORE006'
    ```
    #### **Example 4: Select Stores for a Promotion with Specific Criteria where Stores: STORE007
    User: "Promotion Type: Simple, Hierarchy Type: Sub Class, Hierarchy Value: Full Sleeve, Brand: H&M, Items: ITEM001, ITEM002, Discount Type: % Off, Discount Value: 10, Start Date: 13/02/2025, End Date: 31/05/2025, Stores: STORE007"

    ```sql
    SELECT sd.storeId
    FROM storedetails sd
    WHERE sd.storeId = 'STORE007'
    ```
    
    #### **Example 5: Select Stores for a Buy One Get One Promotion where Store Id is STORE008
    User: "Simple, Class, Casuals, H&M, ITEM002, ITEM005, Buy One Get One Free, 0, 13/02/2025, 31/05/2025, STORE008"
    ```sql
    SELECT sd.storeId
    FROM storedetails sd
    WHERE sd.storeId = 'STORE008'
    ```
    
    #### **Example 6: Handling "All Stores" Selection
    User: "Select All stores"
    ```sql
    SELECT sd.storeId
    FROM storedetails sd
    ```
    
    **Example 7: Handling Missing Store Selection
    User: "Simple, Class, Casuals, H&M, ITEM002, ITEM005, Buy One Get One Free, 0, 13/02/2025, 31/05/2025. Missing Fields: Stores are not provided."
    ```sql
    SELECT sd.storeId
    FROM storedetails sd
    WHERE 1=0
    ```
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": store_sql_prompt},
                {"role": "user", "content": store_query}
            ],
            temperature=0.3,
            max_tokens=100
        )
        sql_query = response.choices[0].message.content.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        logging.info("Question in store id: %s", store_query)
        logging.info("fetch store id SQL query: %s", sql_query)
    
        if not sql_query.lower().startswith("select"):
            raise ValueError("Generated response is not a valid SQL query.")
    
        result = execute_mysql_query(sql_query)
        store_ids = []
        for row in result:
            if isinstance(row, dict) and "storeId" in row:
                store_ids.append(row["storeId"])
            elif isinstance(row, (list, tuple)) and len(row) > 0:
                store_ids.append(row[0])
        return store_ids
    except Exception as e:
        logging.error("Error executing store query: %s", str(e))
        return []

def extract_promo_entities(question: str, db: Session) -> dict:
    """
    Extract department, class, sub_class, brand, and store from the user's query using GPT.
    Also performs validation against the database and returns a "validation" sub-dictionary.
    
    Extract and validate required promotion entities from the user's query using GPT.
    The required entities are department, class, sub_class, brand, and store.
    Any additional details present in the query (e.g., Items or other metadata) should be returned unaltered.
    A "validation" sub-dictionary should also be added to indicate the validity of each required field.
    """
    prompt = f"""
    Analyze the following user query and extract the following entities:
    - Department: The department mentioned (e.g., "T-Shirt Department" → "T-Shirt")
    - Class: The class mentioned (e.g., "Formals Class" → "Formals")
    - Sub_Class: The sub class mentioned (e.g., "Half Sleeve Sub Class" → "Half Sleeve")
    - Brand: The brand mentioned (e.g., "FashionX Brand" → "FashionX")
    - Stores: The store ID or store name mentioned (e.g., "STORE006" or "New York Store")

    
    If the query does not mention any of the above entities: department, class, sub class, brand and store details, simply return the entire query without modifications and do not include their validation information.
    For each of these entities, verify if the value exists in the respective database columns. If an entity is not present or cannot be validated, set its value to null. Importantly, if the query includes additional details (for example, information about Items), do not modify or remove these details; return them exactly as received.
    
    Return a JSON object that includes at least the keys:
    - department
    - class
    - sub_class
    - brand
    - Stores
    
    Include any other data present in the query without changes.
    User query: {question}
    
    You may only reference the following tables and columns:
    - itemmaster: itemDepartment, itemClass, itemSubClass, brand
    - storedetails: storeId, storeName, address, city, state, zipCode, phone
    """
    try:
        store_ids = fetch_store_id_from_query(question, db)
        logging.info("Fetched Store IDs: %s", store_ids)
    except Exception as e:
        logging.error("Error fetching store IDs: %s", e)
        store_ids = [] # Default to empty list on error
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. Validate only the required promotion entities and return the full data as received."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        entities = json.loads(response.choices[0].message.content)
        # store_ids =  fetch_store_id_from_query(question, db)
        logging.info("Question in promo entity: %s, %s", question, entities)  
        #if store is found call fetch store function

        # Begin field validation
        validation_results = {}
        validation_errors = []

        if entities.get('department'):
            stmt = text("SELECT EXISTS(SELECT 1 FROM itemmaster WHERE itemDepartment LIKE :dept)")
            exists = db.execute(stmt, {'dept': f"{entities['department']}%"}).scalar()
            validation_results['department'] = bool(exists)
            if not exists:
                validation_errors.append(f"Department '{entities['department']}' does not exist.")

        if entities.get('class'):
            stmt = text("SELECT EXISTS(SELECT 1 FROM itemmaster WHERE itemClass = :class)")
            exists = db.execute(stmt, {'class': entities['class']}).scalar()
            validation_results['class'] = bool(exists)
            if not exists:
                validation_errors.append(f"Class '{entities['class']}' does not exist.")

        if entities.get('sub_class'):
            stmt = text("SELECT EXISTS(SELECT 1 FROM itemmaster WHERE itemSubClass = :subclass)")
            exists = db.execute(stmt, {'subclass': entities['sub_class']}).scalar()
            validation_results['sub_class'] = bool(exists)
            if not exists:
                validation_errors.append(f"Sub Class '{entities['sub_class']}' does not exist.")

        if entities.get('brand'):
            stmt = text("SELECT EXISTS(SELECT 1 FROM itemmaster WHERE brand = :brand)")
            exists = db.execute(stmt, {'brand': entities['brand']}).scalar()
            validation_results['brand'] = bool(exists)
            if not exists:
                validation_errors.append(f"Brand '{entities['brand']}' does not exist.")

        if not store_ids or not isinstance(store_ids, list) or len(store_ids) == 0:
            # validation_results['store'] = False
            validation_results['Stores'] = bool(store_ids)
            validation_errors.append("Store does not exist.")
        else:
            # validation_results['store'] = True
            validation_results['Stores'] = bool(store_ids)

        entities['Stores']= store_ids
        entities['validation'] = {
            'results': validation_results,
            'errors': validation_errors
        }
        logging.info("Final entities before return: %s", entities)
        return entities
    except Exception as e:
        logging.error("Error extracting promo entities: %s", e)
        return {
            'department': None,
            'class': None,
            'sub_class': None,
            'brand': None,
            'store': [],
            'validation': {
                'results': {},
                'errors': [str(e)]
            }
        }

def query_database_function_promo(question: str, db: Session) -> str:
    attr, value, attr_id = find_attributes(question)
    
    tables = ["itemmaster", "itemsupplier", "itemdiffs", "storedetails"]
    tables_str = ", ".join(tables)
    
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
      
    #### **5. Handling Multiple Predicate Filtering**  
    - If the user query contains more than one predicate for the same field (for example, multiple brands or multiple department values), the query should use the OR operator to combine these values.
    - For example:
        - **Multiple Brands:**  
          User: "Select all items from FashionX and Zara brands"  
          Expected SQL:  
          ```sql
          SELECT im.itemId
          FROM itemmaster im
          WHERE im.brand = 'FashionX' OR im.brand = 'Zara'
          ```
        - **Multiple Departments:**  
          User: "Select all items from T-Shirt and Shirt departments"  
          Expected SQL:  
          ```sql
          SELECT im.itemId
          FROM itemmaster im
          WHERE im.itemDepartment LIKE 'T-Shirt%' OR im.itemDepartment LIKE 'Shirt%'
          ```
        - **Multiple Sub Classes:**  
          User: "Select all items from Half and Full Sleeve Sub Classes"  
          Expected SQL:  
          ```sql
          SELECT im.itemId
          FROM itemmaster im
          WHERE im.itemSubClass LIKE 'Half Sleeve%' OR im.itemSubClass LIKE 'Full Sleeve%'
          ```
        - **Multiple Classes:**  
          User: "Select all items from Formals and Casuals Classes"  
          Expected SQL:  
          ```sql
          SELECT im.itemId
          FROM itemmaster im
          WHERE im.itemClass LIKE 'Formals%' OR im.itemClass LIKE 'Casuals%'
          ```
    #### **4. Handling Mixed Hierarchy Predicates (Across Different Fields)**  
    - If the user query specifies different hierarchy types (for instance, a department and a class), these conditions should be combined with an AND operator.
    - For example:
        - **Mixed Hierarchy Example:**  
            User: "Select all items from T-Shirt department and Casuals class"  
            Expected SQL:  
            ```sql
            SELECT im.itemId
            FROM itemmaster im
            WHERE im.itemDepartment LIKE 'T-Shirt%' AND im.itemClass LIKE 'Casuals%'
            ```
      
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
    #### **Example 5: Select Items Filtering on Multiple Brands**
    User: "Select all items from FashionX and Zara brands"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.brand = 'FashionX' OR im.brand = 'Zara'
    ```
    
    #### **Example 6: Select Items Filtering on Multiple Departments**
    User: "Select all items from T-Shirt and Shirt departments"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.itemDepartment LIKE 'T-Shirt%' OR im.itemDepartment LIKE 'Shirt%'
    ```
    
    #### **Example 7: Select Items Filtering on Multiple Sub Classes**
    User: "Select all items from Half and Full Sleeve Sub Classes"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.itemSubClass LIKE 'Half Sleeve%' OR im.itemSubClass LIKE 'Full Sleeve%'
    ```
    
    #### **Example 8: Select Items Filtering on Multiple Classes**
    User: "Select all items from Formals and Casuals Classes"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.itemClass LIKE 'Formals%' OR im.itemClass LIKE 'Casuals%'
    ```
    
    #### **Example 9: Mixed Hierarchy Conditions**
    User: "Select all items from T-Shirt department and Casuals class"
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.itemDepartment LIKE 'T-Shirt%' AND im.itemClass LIKE 'Casuals%'
    ```
    
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": promotion_sql_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0.5,
            max_tokens=150
        )
        logging.info("GPT response in query_database_function_promo: %s", response)
        mysql_query = response.choices[0].message.content.strip()
        mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()
        logging.info("Final SQL query: %s", mysql_query)
    
        if not mysql_query.lower().startswith(("select", "show")):
            return "Error: Generated response is not a valid SQL query."
        
        result = execute_mysql_query(mysql_query)
        items = []
        if result:
            for row in result:
                if isinstance(row, dict) and 'itemId' in row:
                    items.append(row['itemId'])
                elif isinstance(row, (list, tuple)) and len(row) > 0:
                    items.append(row[0])
        items_str = ', '.join(items) if items else 'No items found'
    
        return f"Items: [{items_str}]"
        # validation_info = f"\nValidation Results: {promo_entities.get('validation', {})}"
        # return f"Items: [{items_str}] {validation_info}"
    except Exception as e:
        logging.error("Query error: %s", str(e))
        return f"Error executing query: {str(e)}"
    
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

def detect_attribute(question: str, attribute_map: Dict) -> Tuple[List[str], List[str]]:
    """Optimized attribute detection for multiple attributes.
    
    Returns a tuple of two lists: (attributes, values). For example:
      Query: "Select all red and yellow items"
      Returns: (["color"], ["red", "yellow"])
      
      Query: "Select all red and Large items"
      Returns: (["color", "size"], ["red", "L"])
    """
    if not attribute_map:
        return ([], [])

    detected = {}  # key: attribute type, value: set of detected candidate values
    lower_question = question.lower()
    words = lower_question.split()

    # Iterate over each attribute type and its candidate values.
    for attr, candidates in attribute_map.items():
        for candidate in candidates.keys():
            # Use a simple check: if candidate is a substring in the question.
            if candidate in lower_question:
                detected.setdefault(attr, set()).add(candidate)
            else:
                # Otherwise, use fuzzy matching: check each word for a close match.
                for word in words:
                    matches = difflib.get_close_matches(word, [candidate], n=1, cutoff=0.6)
                    if matches:
                        detected.setdefault(attr, set()).add(candidate)
                        break

    # Fallback using prompt if nothing was detected.
    if not detected:
        attr_desc = "\n".join(
            f"- {attr}: {', '.join(values.keys())}"
            for attr, values in attribute_map.items()
        )
        prompt = f"""Analyze query: "{question}"
Identify attributes from:
{attr_desc}
Respond in JSON format: {{"attribute": [...], "value": [...]}}"""
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
        # Expect result["attribute"] and result["value"] to be lists.
        return (result.get("attribute", []), result.get("value", []))
    
    # Flatten the detected dictionary into two lists.
    attributes: List[str] = []
    values: List[str] = []
    for attr, vals in detected.items():
        for v in vals:
            attributes.append(attr)
            values.append(v)
    
    # Optional: refine using fuzzy matching on the candidate set so that e.g. "Large" can be mapped to "L" if needed.
    # (Implementation here depends on your database values.)
    logging.info("Detected attributes (pre-refinement): %s, values: %s", attributes, values)
    return (attributes, values)

def find_attributes(question: str) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[int]]]:
    """Find multiple attributes, values, and their IDs in a single optimized flow.

    Returns a tuple of three lists:
      (list_of_attributes, list_of_values, list_of_ids)
    For example:
      "Select all red and yellow items" ->
         (["color"], ["red", "yellow"], [id_red, id_yellow])
      "Select all red and Large items" ->
         (["color", "size"], ["red", "L"], [id_red, id_large])
    """
    print(f"\nProcessing query: '{question}'")
    
    # Get attribute metadata in a single query.
    attr_data = db_query(
        "SELECT diffType, diffId, MIN(id) AS id FROM itemdiffs GROUP BY diffType, diffId"
    )
    # attr_data = db_query(
    #     "SELECT diffType, diffId, id FROM itemdiffs GROUP BY diffType, diffId"
    # )
    
    if not attr_data:
        print("No attribute data found in database")
        return (None, None, None)

    # Build an optimized attribute map: {diffType: {diffId(lowercase): id}}
    attribute_map: Dict[str, Dict[str, int]] = {}
    for diff_type, diff_id, id_val in attr_data:
        key = diff_id.lower()  # Store candidate value as lowercase
        attribute_map.setdefault(diff_type, {})[key] = id_val

    print(f"Attribute map keys: {list(attribute_map.keys())}")
    
    # Detect attributes using our modified function.
    detected_attrs, detected_values = detect_attribute(question, attribute_map)
    if not detected_attrs or not detected_values:
        return (None, None, None)

    # Look up each detected value in the attribute map to get corresponding IDs.
    detected_ids: List[int] = []
    for attr, val in zip(detected_attrs, detected_values):
        id_val = attribute_map.get(attr, {}).get(val)
        if id_val is None:
            logging.warning("No ID found for attribute %s with value %s", attr, val)
            # You may decide to either skip or return None.
            detected_ids.append(-1)  # or use a placeholder value
        else:
            detected_ids.append(id_val)

    logging.info("Find attributes: %s, values: %s, IDs: %s", detected_attrs, detected_values, detected_ids)
    return (detected_attrs, detected_values, detected_ids)

llm_gpt4 = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.7)
# conversation_states = defaultdict(list)
previous_promo_details = defaultdict(dict)  
promo_states = defaultdict(dict)
user_promo_details={}
promo_email_cache = defaultdict(set)

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

def db_query_insights(query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Generic database query executor with error handling."""
    db = None # Initialize db to None
    try:
        db = next(get_db())
        # Use .mappings().fetchall() to get list of dict-like objects directly
        result = db.execute(text(query), params or {}).mappings().fetchall()
        # Alternatively, if using older SQLAlchemy or prefer explicit dict conversion:
        # result_proxy = db.execute(text(query), params or {})
        # result = [dict(row) for row in result_proxy.fetchall()]
        return result
    except Exception as e:
        # Log the actual error!
        logging.error(f"Database error executing query: {query} with params: {params}", exc_info=True)
        return []
    finally:
        if db:
            # It's generally better practice to manage the session lifecycle
            # using a context manager within the calling scope (e.g., using the
            # contextmanager get_db provides), rather than closing here,
            # but keeping db.close() based on your original structure.
            try:
                db.close()
            except Exception as close_err:
                logging.error(f"Error closing DB session: {close_err}", exc_info=True)

@app.get("/supplier-risk-insights")
async def supplier_risk(supplierId:str):
    insights=await generate_supplier_insights(supplierId,db_query_insights)
    # logging.info("promo_entities success: %s", await generate_supplier_insights(supplierId,db_query_insights))
    # def fig_to_bytes(fig):
    #     buf = io.BytesIO()
    #     fig.write_image(buf, format="png")
    #     return buf.getvalue()
    # graphs=insights["graph_data"]
    # # Render each figure to raw bytes
    # bar_bytes   = fig_to_bytes(graphs["bar_fig"])
    # pie_bytes   = fig_to_bytes(graphs["pie_fig"])
    # gauge_bytes = fig_to_bytes(graphs["gauge_fig"])
    # delay_bytes = fig_to_bytes(graphs["delay_fig"])

    # # Base-64 encode & add a data URI prefix so it's directly displayable in browsers
    # def to_data_uri(img_bytes):
    #     b64 = base64.b64encode(img_bytes).decode("ascii")
    #     return f"data:image/png;base64,{b64}"

    # return JSONResponse(
    #     content={
    #         "bar_chart"   : to_data_uri(bar_bytes),
    #         "pie_chart"   : to_data_uri(pie_bytes),
    #         "gauge_chart" : to_data_uri(gauge_bytes),
    #         "delay_chart" : to_data_uri(delay_bytes),
    #     }
    # )

    return {"insights":insights}

@app.post("/promo-chat")
async def handle_promotion_chat(request: dict):
    # Assuming request has keys "user_id" and "message"
    user_id = request.get("user_id")
    user_message = request.get("message")

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
            "description": "Execute database queries for item selection and retrieval using given details",
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
        },
        {
            "name": "entity_extraction_and_validation",
            "description": "Extract and validate entities like department, class, brand, and store details from the user query",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user query from which to extract promotion details including store information."
                    }
                },
                "required": ["question"]
            }
        }
]

        # First API call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=0.7,
            max_tokens=500
        )
        response_message = response.choices[0].message
        content_text = response_message.content  # may be None if function_call is triggered
        function_call = response_message.function_call
        query_called = False
        bot_reply = content_text

        if function_call and function_call.name == "query_database":
            args = json.loads(function_call.arguments)
            db = next(get_db())
            query_result = query_database_function_promo(args["question"], db)
            query_called = True

            # Append function result to the conversation
            messages.append({
                "role": "function", 
                "name": "query_database",
                "content": query_result
            })
            # Second API call: GPT processes the function's output
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content
            # Here, promo_argument is set to the processed GPT reply:
            promo_argument = bot_reply
            print("Processed promo argument:", promo_argument)
            
            try:
                promo_entities = extract_promo_entities(promo_argument, db)
                logging.info("promo_entities success: %s", promo_entities)
            except Exception as e:
                logging.error("Entity extraction failed: %s", str(e))
                promo_entities = {
                    'error': str(e),
                    'validation': {'errors': [str(e)]}
                }
            
            # Merge the validated store IDs into the promo argument.
            # Attempt to parse the original promo_argument as JSON.
            try:
                promo_dict = json.loads(promo_argument)
            except Exception as e:
                # If parsing fails, create a dict with the original promo_argument.
                promo_dict = {"query": promo_argument}
            
            # Override the "Stores" field with the validated store IDs
            promo_dict["Stores"] = promo_entities.get("Stores", promo_dict.get("Stores"))
            # Optionally, include validation details as well.
            promo_dict["validation"] = promo_entities.get("validation", {})
            
            # Convert the updated promo details back to JSON string.
            updated_promo_argument = json.dumps(promo_dict)
            
            # Append the updated promo details to the conversation
            messages.append({
                "role": "function", 
                "name": "entity_extraction_and_validation",
                "content": updated_promo_argument
            })

            # Optionally, you can make a third API call if needed to incorporate validation:
            third_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = third_response.choices[0].message.content
        elif function_call and  function_call.name == "entity_extraction_and_validation":
            # Handle direct entity extraction function call.
            args = json.loads(function_call.arguments)
            db = next(get_db())
            promo_entities = extract_promo_entities(args["question"], db)
            messages.append({
                "role": "function", 
                "name": "entity_extraction_and_validation",
                "content": json.dumps(promo_entities)
            })
            # Follow-up call to allow GPT to process the extraction output.
            second_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = second_response.choices[0].message.content

        user_promo_details[user_id] = await categorize_promo_details_new(bot_reply, user_id)
        # user_promo_details[user_id] = await categorize_promo_details(bot_reply, user_id)
        promo_json = user_promo_details[user_id]
        promo_email=promo_json["Email"]
        # if(promo_email):
        #     promo_email_cache[user_id].add(promo_email)

        classification_result = classify_query(user_message, conversation)

        promo_states[user_id].append(f"Bot: {bot_reply}")

        submissionStatus = "in_progress"
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Promotion created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"       
        
        if promo_email:
        # only if we’ve never seen it before do we override
            if promo_email not in promo_email_cache[user_id]:
                submissionStatus = "pending"
            promo_email_cache[user_id].add(promo_email)
        


        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": promo_states,
            "promo_json": promo_json,
            "submissionStatus": submissionStatus,
            "query_called": query_called,
            "classification_result": classification_result,
            "promo_email":promo_email
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def get_submission_status(bot_reply: str) -> str:
    if "Would you like to submit" in bot_reply:
        return "pending"
    elif "Promotion created successfully" in bot_reply:
        return "submitted"
    elif "I want to change something" in bot_reply:
        return "cancelled"
    return "in_progress"
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

# @app.post("/chat")
# async def chat_with_po_assistant(request: ChatRequest):
#     user_id = request.user_id
#     user_message = request.message

#     # Maintain user session
#     if user_id not in chat_histories:
#         chat_histories[user_id] = []
#         user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()
#         user_supplier_cache[user_id] = set()
    
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

#         # Retain po_json from previous interaction if query_called is True
#         if not query_called:
#             user_po_details[user_id] = await categorize_po_details(bot_reply, user_id)

#         po_json = user_po_details[user_id]  # Assign retained po_json

#         chat_histories[user_id].append(f"Bot: {bot_reply}")
#         print("PO JSON:", po_json, "User ID:", user_id)

#         # supplier_id= po_json.get("Supplier ID", "")
#         # print("Supplier ID: ",supplier_id,po_json["Supplier ID"])
#         # # Fetch PO items only if PO number is not empty and not fetched before
#         # if supplier_id and supplier_id not in user_supplier_cache[user_id]:
#         #     lead_time=fetch_lead_time(supplier_id)
#         #     print("Inside supplier loop, lead time: ",supplier_id,lead_time)
#         #     if lead_time:  # Only process if po_items is not empty
#         #         messages.append({
#         #             "role": "user",  # Simulate user input
#         #             "content": f"lead time: {lead_time}"
#         #             #change
#         #         })
#         #         # 
#         #         # Mark this PO as processed
#         #         user_supplier_cache[user_id].add(supplier_id)
#         #         print("Messages in po chat: ",messages)
#         #         # Second API call with updated template
#         #         second_response = client.chat.completions.create(
#         #             model="gpt-3.5-turbo",
#         #             messages=messages,
#         #             temperature=0.7,
#         #             max_tokens=500
#         #         )
#         #         bot_reply = second_response.choices[0].message.content

#         # Determine submission status
#         if "Would you like to submit" in bot_reply:
#             submissionStatus = "pending"
#         elif "Purchase Order created successfully" in bot_reply:
#             submissionStatus = "submitted"
#         elif "I want to change something" in bot_reply:
#             submissionStatus = "cancelled"
#         else:
#             submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
#         print("PO submission status:", submissionStatus)

#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "chat_history": chat_histories,
#             "po_json": po_json,  # Retains values if query_called is True
#             "submissionStatus": submissionStatus
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

def details_validator_po(db: Session, supplier_id: str = None, item_ids: list = None):
    """
    Validates Supplier ID and Item IDs against the database.
    - Supplier ID is checked against the 'suppliers' table.
    - Item IDs are checked for existence in 'itemmaster'.
    """
    validation_results = {
        "supplier_id_provided": supplier_id,
        "supplier_exists": None, # None indicates not checked or error, False means checked and not found
        "items_provided_count": len(item_ids) if item_ids else 0,
        "all_items_exist": None, # None indicates not checked or error (e.g. empty item_ids list)
        "item_validation_details": [] # List of dicts: {'item_id': id, 'exists': True/False}
    }

    if not supplier_id and not item_ids:
        validation_results["message"] = "No supplier ID or item IDs provided for validation."
        return validation_results

    # Validate Supplier ID
    if supplier_id:
        try:
            query_supplier = text("SELECT 1 FROM suppliers WHERE supplierId = :supplier_id LIMIT 1")
            result_supplier = db.execute(query_supplier, {"supplier_id": supplier_id}).fetchone()
            validation_results["supplier_exists"] = bool(result_supplier)
        except Exception as e:
            print(f"Error validating supplier ID {supplier_id}: {e}")
            validation_results["supplier_exists"] = False # Consider it as not existing on error
            validation_results["supplier_validation_error"] = str(e)

    # Validate Item IDs
    if item_ids:
        all_found = True
        processed_ids = set() # To handle duplicate item_ids in the input list efficiently

        for item_id in item_ids:
            if not item_id or item_id in processed_ids: # Skip empty or already processed IDs
                if not item_id and validation_results["all_items_exist"] is None and not any(d['item_id'] == item_id for d in validation_results["item_validation_details"]):
                     # If an empty item ID is passed, mark as not existing.
                     validation_results["item_validation_details"].append({'item_id': item_id, 'exists': False})
                     all_found = False
                continue
            
            item_detail = {'item_id': item_id, 'exists': False}
            try:
                # Check if item exists in any of the relevant tables
                query_item = text("""
                    SELECT 1 FROM (
                        SELECT itemId FROM itemmaster WHERE itemId = :item_id
                    ) AS combined_items LIMIT 1
                """)
                result_item = db.execute(query_item, {"item_id": item_id}).fetchone()
                if result_item:
                    item_detail['exists'] = True
                else:
                    all_found = False # Mark false if any item doesn't exist
            except Exception as e:
                print(f"Error validating item ID {item_id}: {e}")
                all_found = False # Mark false on error for this item
                item_detail['item_validation_error'] = str(e)
            
            validation_results["item_validation_details"].append(item_detail)
            processed_ids.add(item_id)
        
        validation_results["all_items_exist"] = all_found if item_ids else None # Only set if item_ids were provided
    elif not item_ids and validation_results["items_provided_count"] == 0 :
        validation_results["all_items_exist"] = None # Or True, if no items to check means all (non-existent) items are valid.

    return validation_results


def details_validator_invoice(db: Session, item_ids: list = None):
    """
    Validates Item IDs against the database.
    - Item IDs are checked for existence in 'itemmaster'.
    """
    validation_results = {
        "items_provided_count": len(item_ids) if item_ids else 0,
        "all_items_exist": None, # None indicates not checked or error (e.g. empty item_ids list)
        "item_validation_details": [] # List of dicts: {'item_id': id, 'exists': True/False}
    }
    # Validate Item IDs
    if item_ids:
        all_found = True
        processed_ids = set() # To handle duplicate item_ids in the input list efficiently

        for item_id in item_ids:
            if not item_id or item_id in processed_ids: # Skip empty or already processed IDs
                if not item_id and validation_results["all_items_exist"] is None and not any(d['item_id'] == item_id for d in validation_results["item_validation_details"]):
                     # If an empty item ID is passed, mark as not existing.
                     validation_results["item_validation_details"].append({'item_id': item_id, 'exists': False})
                     all_found = False
                continue
            
            item_detail = {'item_id': item_id, 'exists': False}
            try:
                # Check if item exists in any of the relevant tables
                query_item = text("""
                    SELECT 1 FROM (
                        SELECT itemId FROM itemmaster WHERE itemId = :item_id
                    ) AS combined_items LIMIT 1
                """)
                result_item = db.execute(query_item, {"item_id": item_id}).fetchone()
                if result_item:
                    item_detail['exists'] = True
                else:
                    all_found = False # Mark false if any item doesn't exist
            except Exception as e:
                print(f"Error validating item ID {item_id}: {e}")
                all_found = False # Mark false on error for this item
                item_detail['item_validation_error'] = str(e)
            
            validation_results["item_validation_details"].append(item_detail)
            processed_ids.add(item_id)
        
        validation_results["all_items_exist"] = all_found if item_ids else None # Only set if item_ids were provided
    elif not item_ids and validation_results["items_provided_count"] == 0 :
        validation_results["all_items_exist"] = None # Or True, if no items to check means all (non-existent) items are valid.

    return validation_results

# def details_validator_invoice(db: Session, po_number: str = None, item_ids: list = None):
#     """
#     Validates Purchase Order Number and Item IDs against the database.
#     - Purchase Order Number is checked against the 'podetails' table.
#     - Item IDs are checked for existence in 'itemmaster'.
#     """
#     validation_results = {
#         "po_number_provided": po_number,
#         "po_exists": None, # None indicates not checked or error, False means checked and not found
#         "items_provided_count": len(item_ids) if item_ids else 0,
#         "all_items_exist": None, # None indicates not checked or error (e.g. empty item_ids list)
#         "item_validation_details": [] # List of dicts: {'item_id': id, 'exists': True/False}
#     }

#     if not po_number and not item_ids:
#         validation_results["message"] = "No Purchase Order Number or item IDs provided for validation."
#         return validation_results

#     # Validate Purchase Order ID
#     if po_number:
#         try:
#             query_po = text("SELECT 1 FROM podetails WHERE poId= :po_number LIMIT 1")
#             result_po = db.execute(query_po, {"po_number": po_number}).fetchone()
#             validation_results["po_exists"] = bool(result_po)
#         except Exception as e:
#             print(f"Error validating Purchase Order ID {po_number}: {e}")
#             validation_results["po_exists"] = False # Consider it as not existing on error
#             validation_results["po_validation_error"] = str(e)

#     # Validate Item IDs
#     if item_ids:
#         all_found = True
#         processed_ids = set() # To handle duplicate item_ids in the input list efficiently

#         for item_id in item_ids:
#             if not item_id or item_id in processed_ids: # Skip empty or already processed IDs
#                 if not item_id and validation_results["all_items_exist"] is None and not any(d['item_id'] == item_id for d in validation_results["item_validation_details"]):
#                      # If an empty item ID is passed, mark as not existing.
#                      validation_results["item_validation_details"].append({'item_id': item_id, 'exists': False})
#                      all_found = False
#                 continue
            
#             item_detail = {'item_id': item_id, 'exists': False}
#             try:
#                 # Check if item exists in any of the relevant tables
#                 query_item = text("""
#                     SELECT 1 FROM (
#                         SELECT itemId FROM itemmaster WHERE itemId = :item_id
#                     ) AS combined_items LIMIT 1
#                 """)
#                 result_item = db.execute(query_item, {"item_id": item_id}).fetchone()
#                 if result_item:
#                     item_detail['exists'] = True
#                 else:
#                     all_found = False # Mark false if any item doesn't exist
#             except Exception as e:
#                 print(f"Error validating item ID {item_id}: {e}")
#                 all_found = False # Mark false on error for this item
#                 item_detail['item_validation_error'] = str(e)
            
#             validation_results["item_validation_details"].append(item_detail)
#             processed_ids.add(item_id)
        
#         validation_results["all_items_exist"] = all_found if item_ids else None # Only set if item_ids were provided
#     elif not item_ids and validation_results["items_provided_count"] == 0 :
#         validation_results["all_items_exist"] = None # Or True, if no items to check means all (non-existent) items are valid.

#     return validation_results

#PO Chatbot Functions
user_supplier_cache = {} #initialize in case lead time logic needs to be implemented
chat_histories = {}
user_po_details = {}
po_email_cache = defaultdict(set)

# @app.post("/chat")
# async def chat_with_po_assistant(request: ChatRequest):
#     user_id = request.user_id
#     user_message = request.message

#     # Maintain user session
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
#         },{
#             "name":"details_validation",
#             "description":"Validates the received data with the database using SQL Queries",
#         }
        
#         ]

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

#         # Retain po_json from previous interaction if query_called is True
#         if not query_called:
#             user_po_details[user_id] = await categorize_po_details(bot_reply, user_id)

#         po_json = user_po_details[user_id]  # Assign retained po_json
#         chat_histories[user_id].append(f"Bot: {bot_reply}")
#         print("PO JSON:", po_json, "User ID:", user_id)
#         po_email=po_json["Email"]
#         items=po_json["Items"]
#         item_ids=[item['Item ID'] for item in items]
#         supplier_id=po_json["Supplier ID"]
#         # if(po_email):
#         #     po_email_cache[user_id].add(po_email)

#         # Determine submission status
#         if "Would you like to submit" in bot_reply:
#             submissionStatus = "pending"
#         elif "Purchase Order created successfully" in bot_reply:
#             submissionStatus = "submitted"
#         elif "I want to change something" in bot_reply:
#             submissionStatus = "cancelled"
#         # elif po_email and po_email not in po_email_cache[user_id]:
#         #     submissionStatus = "pending"
#         else:
#             submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
#         if po_email:
#         # only if we’ve never seen it before do we override
#             if po_email not in po_email_cache[user_id]:
#                 submissionStatus = "pending"
#             po_email_cache[user_id].add(po_email)
#         print("PO submission status:", submissionStatus)

#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "chat_history": chat_histories,
#             "po_json": po_json,  # Retains values if query_called is True
#             "submissionStatus": submissionStatus,
#             "po_email":po_email

#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
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
        },
        {
                "name": "details_validation", # LLM function name
                "description": "Validates Supplier ID and a list of Item IDs against the database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "supplier_id": {
                            "type": "string",
                            "description": "The Supplier ID to validate (optional)."
                        },
                        "item_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of Item IDs to validate (optional)."
                        }
                    },
                    # No 'required' fields means LLM can call it with supplier_id, item_ids, or both, or neither.
                    # The Python function details_validator_po handles these cases.
                }
            }
        
        ]

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
        elif function_call.name == "details_validation":
            args = json.loads(function_call.arguments)
            supplier_id_to_validate = args.get("supplier_id")
            item_ids_to_validate = args.get("item_ids", []) # Default to empty list

            db_session: Session = next(get_db())
            try:
                validation_result = details_validator_po(db_session, supplier_id_to_validate, item_ids_to_validate)
                validation_info_for_llm = json.dumps(validation_result) # For LLM context
                print("LLM-triggered Validation Outcome:", validation_result)
            finally:
                db_session.close()
            
                        # Append function response to messages
            messages.append({
                "role": "function", 
                "name": "details_validation",
                "content": validation_info_for_llm
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
        po_email=po_json["Email"]
        items=po_json["Items"]
        item_ids=[item['Item ID'] for item in items]
        supplier_id=po_json["Supplier ID"]
        # if(po_email):
        #     po_email_cache[user_id].add(po_email)

        # Determine submission status
        if "Would you like to submit" in bot_reply:
            submissionStatus = "pending"
        elif "Purchase Order created successfully" in bot_reply:
            submissionStatus = "submitted"
        elif "I want to change something" in bot_reply:
            submissionStatus = "cancelled"
        # elif po_email and po_email not in po_email_cache[user_id]:
        #     submissionStatus = "pending"
        else:
            submissionStatus = "in_progress"  # Default state if no clear intent is detected
        
        if po_email:
        # only if we’ve never seen it before do we override
            if po_email not in po_email_cache[user_id]:
                submissionStatus = "pending"
            po_email_cache[user_id].add(po_email)
        print("PO submission status:", submissionStatus)

        return {
            "user_id": user_id,
            "bot_reply": bot_reply,
            "chat_history": chat_histories,
            "po_json": po_json,  # Retains values if query_called is True
            "submissionStatus": submissionStatus,
            "po_email":po_email

        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat") # Assuming 'app' is your FastAPI app instance
# async def chat_with_po_assistant(request: ChatRequest): # Assuming ChatRequest is defined
#     user_id = request.user_id
#     user_message = request.message

#     # Maintain user session (assuming chat_histories, user_po_details, DEFAULT_PO_STRUCTURE are defined)
#     if user_id not in chat_histories:
#         chat_histories[user_id] = []
#         user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy() # Ensure DEFAULT_PO_STRUCTURE is defined
    
#     chat_histories[user_id].append(f"User: {user_message}")
#     conversation = "\n".join(chat_histories[user_id])

#     try:
#         messages = [
#             {"role": "system", "content": template_PO}, # Ensure template_PO is defined
#             {"role": "user", "content": conversation}
#         ]

#         # Define functions for the LLM
#         # If the LLM should trigger validation, define parameters for details_validation
#         functions = [
#             {
#                 "name": "query_database",
#                 "description": "Retrieve data from the database using SQL queries based on a natural language question.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "question": {
#                             "type": "string",
#                             "description": "Natural language question requiring database data to answer."
#                         }
#                     },
#                     "required": ["question"]
#                 }
#             },
#             {
#                 "name": "details_validation", # LLM function name
#                 "description": "Validates Supplier ID and a list of Item IDs against the database.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": {
#                         "supplier_id": {
#                             "type": "string",
#                             "description": "The Supplier ID to validate (optional)."
#                         },
#                         "item_ids": {
#                             "type": "array",
#                             "items": {"type": "string"},
#                             "description": "A list of Item IDs to validate (optional)."
#                         }
#                     },
#                     # No 'required' fields means LLM can call it with supplier_id, item_ids, or both, or neither.
#                     # The Python function details_validator_po handles these cases.
#                 }
#             }
#         ]

#         response = client.chat.completions.create( # Assuming 'client' is your OpenAI client
#             model="gpt-4o",
#             messages=messages,
#             functions=functions,
#             function_call="auto",
#             temperature=0.7,
#             max_tokens=500
#         )

#         response_message = response.choices[0].message
#         bot_reply = response_message.content
#         function_call = response_message.function_call
#         query_called = False
#         validation_info_for_llm = None # To store validation results if LLM calls it

#         # Handle function calls from LLM
#         if function_call:
#             if function_call.name == "query_database":
#                 args = json.loads(function_call.arguments)
#                 query_result_json = query_database_function(args["question"]) # Expecting JSON string
#                 messages.append({
#                     "role": "function",
#                     "name": "query_database",
#                     "content": query_result_json # query_result must be a string
#                 })
#                 query_called = True
            
#             elif function_call.name == "details_validation":
#                 args = json.loads(function_call.arguments)
#                 supplier_id_to_validate = args.get("supplier_id")
#                 item_ids_to_validate = args.get("item_ids", []) # Default to empty list

#                 db_session: Session = next(get_db())
#                 try:
#                     validation_result = details_validator_po(db_session, supplier_id_to_validate, item_ids_to_validate)
#                     validation_info_for_llm = json.dumps(validation_result) # For LLM context
#                     print("LLM-triggered Validation Outcome:", validation_result)
#                 finally:
#                     db_session.close()
                
#                 messages.append({
#                     "role": "function",
#                     "name": "details_validation",
#                     "content": validation_info_for_llm
#                 })
#                 query_called = True # Treat as a turn that needs LLM's follow-up

#             # If a function was called (query_database or details_validation), get LLM's response
#             if query_called: # Renamed from if function_call for clarity
#                 second_response = client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=messages,
#                     temperature=0.7,
#                     max_tokens=500
#                 )
#                 bot_reply = second_response.choices[0].message.content

#         # Update po_json if no query/validation function was called by LLM in this turn
#         # (or if you want to update it regardless, adjust this logic)
#         if not query_called: # query_called indicates LLM interaction, po_json usually from LLM's content
#             # Assuming categorize_po_details updates user_po_details[user_id]
#             # Ensure categorize_po_details is an async function if awaited
#             # user_po_details[user_id] = await categorize_po_details(bot_reply, user_id)
#             pass # Re-evaluate this logic based on categorize_po_details behavior

#         po_json = user_po_details.get(user_id, DEFAULT_PO_STRUCTURE.copy())
#         chat_histories[user_id].append(f"Bot: {bot_reply}")
#         print("PO JSON after LLM interaction:", po_json, "User ID:", user_id)

#         # ---- Internal Validation Step (after LLM populates po_json) ----
#         # This validation runs on the po_json potentially populated/updated by the LLM.
#         # It's separate from the LLM-callable 'details_validation' function.
#         internal_validation_performed = False
#         if po_json: # Ensure po_json is not None
#             supplier_id_from_json = po_json.get("Supplier ID")
#             items_from_json = po_json.get("Items", [])
#             item_ids_from_json = [item.get('Item ID') for item in items_from_json if item.get('Item ID')]

#             # Perform internal validation if IDs are present and maybe if not just validated by LLM
#             if supplier_id_from_json or item_ids_from_json:
#                 # Avoid re-validating if LLM just did it with the same data, unless necessary
#                 # This simple check assumes LLM call would populate validation_info_for_llm
#                 # A more robust check might compare the IDs.
#                 # if not (function_call and function_call.name == "details_validation"):
#                 db_session_internal_val: Session = next(get_db())
#                 try:
#                     internal_validation_outcome = details_validator_po(db_session_internal_val, supplier_id_from_json, item_ids_from_json)
#                     print("Internal Validation Outcome:", internal_validation_outcome)
#                     internal_validation_performed = True
#                     # You can use internal_validation_outcome to:
#                     # 1. Modify `bot_reply` to inform the user about validation issues.
#                     #    Example:
#                     #    if internal_validation_outcome.get("supplier_exists") is False:
#                     #        bot_reply += f"\nNOTICE: The Supplier ID '{supplier_id_from_json}' could not be validated."
#                     #    if internal_validation_outcome.get("all_items_exist") is False:
#                     #        invalid_items = [d['item_id'] for d in internal_validation_outcome.get("item_validation_details", []) if d['exists'] is False]
#                     #        if invalid_items:
#                     #            bot_reply += f"\nNOTICE: The following Item IDs could not be validated: {', '.join(invalid_items)}."
#                     # 2. Adjust `submissionStatus`.
#                     # 3. Log the information for backend processing.
#                 finally:
#                     db_session_internal_val.close()

#         # Determine submission status (existing logic)
#         po_email = po_json.get("Email")
#         # Ensure po_email_cache is defined and handled correctly for the user_id
#         # if user_id not in po_email_cache: po_email_cache[user_id] = set()
        
#         submissionStatus = "in_progress" # Default
#         if "Would you like to submit" in bot_reply:
#             submissionStatus = "pending"
#         elif "Purchase Order created successfully" in bot_reply:
#             submissionStatus = "submitted"
#         elif "I want to change something" in bot_reply: # Or similar phrases indicating cancellation/changes
#             submissionStatus = "cancelled"
        
#         # This po_email logic seems to override above, ensure it's the desired behavior
#         if po_email:
#             if po_email not in po_email_cache.get(user_id, set()): # Use .get for safety
#                 submissionStatus = "pending"
#             po_email_cache.setdefault(user_id, set()).add(po_email) # Ensure set exists
        
#         print("PO submission status:", submissionStatus)

#         return_payload = {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "chat_history": chat_histories.get(user_id), # Use .get for safety
#             "po_json": po_json,
#             "submissionStatus": submissionStatus,
#             "po_email": po_email
#         }
#         if internal_validation_performed: # Optionally include internal validation results
#             return_payload["internal_validation_results"] = internal_validation_outcome
#         if validation_info_for_llm: # If LLM triggered validation
#              return_payload["llm_validation_results"] = json.loads(validation_info_for_llm)

#         return return_payload

#     except Exception as e:
#         print(f"Error in chat_with_po_assistant: {e}") # Log the error
#         # from fastapi import HTTPException (ensure imported)
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

# @app.post("/invCreation/", status_code=status.HTTP_201_CREATED)
# def create_invoice(invHeader: invHeaderCreate, db: Session = Depends(get_db)):
#     db_invHeader = models.InvHeader(**invHeader.dict())
#     db.add(db_invHeader)
#     db.commit()
#     db.refresh(db_invHeader)
#     return db_invHeader

@app.post("/invCreation/", status_code=status.HTTP_201_CREATED)
def create_invoice(inv: invHeaderCreate, db: Session = Depends(get_db)):
    store_ids = inv.storeIds
    data = inv.dict(exclude={"storeIds"})
    db_inv = models.InvHeader(**data)
    db.add(db_inv)
    db.commit()
    db.refresh(db_inv)
    stores = db.query(models.StoreDetails).filter(models.StoreDetails.storeId.in_(store_ids)).all()
    db_inv.stores.extend(stores)
    db.commit()
    db.refresh(db_inv)

    return db_inv
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
    returned_attributes =await categorize_promo_details_new(query, "string")
    # returned_attributes = fetch_store_id_from_query(query, db)
    # returned_attributes = query_database_function_promo(query, db)
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
async def clearConversation():
    # conversation.clear()
    # chat_histories.clear()
    # previous_invoice_details.clear()
    # previous_po_details.clear()
    # previous_promo_details.clear()
    # submissionStatus = "not submitted"
    previous_promo_details.clear()  
    promo_states.clear()
    user_promo_details.clear()
    promo_email_cache.clear()

    user_supplier_cache.clear() #initialize in case lead time logic needs to be implemented
    chat_histories.clear()
    user_po_details.clear()
    po_email_cache.clear()

    invoice_chat_histories.clear()
    user_invoice_details.clear()
    user_po_cache.clear()
    invoice_email_cache.clear()
    return {"conversation":conversation,"submissionStatus":"not submitted","chat_history":chat_histories}

@app.post("/clearDataNew")
async def clearConversationNew(request: ChatRequestUser):
    user_id = request.user_id
    # email_value = previous_po_details[user_id].get("Email")
    # previous_po_details[user_id].clear()
    # if email_value is not None:
    #     previous_po_details[user_id]["Email"] = email_value
    print("clear data before:", chat_histories, previous_invoice_details, previous_promo_details)

    # 1) --- Purchase-Order module ---
    # chat_histories is a dict of lists
    if user_id in chat_histories:
        chat_histories[user_id].clear()

    # user_po_details is a dict of dicts
    if user_id in user_po_details:
        po_email = user_po_details[user_id].get("Email")
        user_po_details[user_id].clear()
        if po_email is not None:
            user_po_details[user_id]["Email"] = po_email

    # Also reset the per-user email cache if you want to force “pending” next time
    po_email_cache[user_id].clear()


    # 2) --- Promotion module ---
    # promo_states is a dict of lists
    if user_id in promo_states:
        promo_states[user_id].clear()

    # user_promo_details is a dict of dicts
    if user_id in user_promo_details:
        promo_email = user_promo_details[user_id].get("Email")
        user_promo_details[user_id].clear()
        if promo_email is not None:
            user_promo_details[user_id]["Email"] = promo_email

    if user_id in promo_email_cache:
        promo_email_cache[user_id].clear()


    # 3) --- Invoice module ---
    # invoice_chat_histories is a dict of lists
    if user_id in invoice_chat_histories:
        invoice_chat_histories[user_id].clear()

    # user_invoice_details is a dict of dicts
    if user_id in user_invoice_details:
        inv_email = user_invoice_details[user_id].get("Email")
        user_invoice_details[user_id].clear()
        if inv_email is not None:
            user_invoice_details[user_id]["Email"] = inv_email

    # Reset PO-lookup cache & email cache for invoice
    if user_id in user_po_cache:
        user_po_cache[user_id].clear()

    if user_id in invoice_email_cache:
        invoice_email_cache[user_id].clear()

    print("clear data after:", chat_histories, previous_invoice_details, previous_promo_details)
    # print("email and po:",email_value,previous_po_details)
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
def fetch_po_item_ids(po_number: str) -> list:
    """Fetch PO items with error handling"""
    try:
        query = f"""
        SELECT itemId
        FROM podetails 
        WHERE poId = '{po_number}'
        """
        result = execute_mysql_query(query)
        return [dict(item) for item in result] if result else []
    except Exception as e:
        print(f"PO fetch error: {str(e)}")
        return []
    
def fetch_lead_time(supplier_id: str) -> list:
    """Fetch PO items with error handling"""
    try:
        query = f"""
        SELECT lead_time
        FROM suppliers 
        WHERE supplierId = '{supplier_id}'
        """
        result = execute_mysql_query(query)
        return [dict(item) for item in result] if result else []
    except Exception as e:
        print(f"PO fetch error: {str(e)}")
        return []
    
invoice_chat_histories = {}
user_invoice_details = {}
user_po_cache = {}
invoice_email_cache = defaultdict(set)
 

# @app.post("/creation/response_new") 
# async def generate_response_new(request: ChatRequest):
#     user_id = request.user_id
#     user_message = request.message

#     # Maintain user session for invoice creation
#     if user_id not in invoice_chat_histories:
#         invoice_chat_histories[user_id] = []
#         user_invoice_details[user_id] = {}
#         user_po_cache[user_id] = set()  # Initialize PO cache for the user

#     invoice_chat_histories[user_id].append(f"User: {user_message}")
#     conversation = "\n".join(invoice_chat_histories[user_id])

#     try:
#         # Prepare messages with system template
#         messages = [
#             {"role": "system", "content": template_5_new},
#             {"role": "user", "content": conversation}
#         ]

#         # Define available functions
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
#         },
#         {
#             "name": "details_validation", # LLM function name
#             "description": "Validates Purchase Order Number and a list of Item IDs against the database.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "po_number": {
#                         "type": "string",
#                         "description": "The Purchase Order Number to validate (optional)."
#                     },
#                     "item_ids": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": "A list of Item IDs to validate (optional)."
#                     }
#                 },
#                 # No 'required' fields means LLM can call it with supplier_id, item_ids, or both, or neither.
#                 # The Python function details_validator_po handles these cases.
#             }
#             }]

#         # First API call with function definition
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
#         function_call = response_message.function_call
#         query_called = False

#         # Handle function call
#         if function_call and function_call.name == "query_database":
#             print("Function call: query database")
#             args = json.loads(function_call.arguments)
#             query_result = query_database_function(args["question"])
#             query_called = True

#             # Append function response and make second call
#             messages.append({
#                 "role": "function",
#                 "name": "query_database",
#                 "content": query_result
#             })
#             # Second API call with function result
#             # second_response = client.chat.completions.create(
#             #     model="gpt-4o",
#             #     messages=messages,
#             #     temperature=0.7,
#             #     max_tokens=500
#             # )
#             # bot_reply = second_response.choices[0].message.content
#         elif function_call.name == "details_validation":
#             args = json.loads(function_call.arguments)
#             po_number_to_validate = args.get("po_number")
#             item_ids_to_validate = args.get("item_ids", []) # Default to empty list

#             db_session: Session = next(get_db())
#             try:
#                 validation_result = details_validator_invoice(db_session, po_number_to_validate, item_ids_to_validate)
#                 validation_info_for_llm = json.dumps(validation_result) # For LLM context
#                 print("LLM-triggered Validation Outcome:", validation_result)
#             finally:
#                 db_session.close()
            
#                         # Append function response to messages
#             messages.append({
#                 "role": "function", 
#                 "name": "details_validation",
#                 "content": validation_info_for_llm
#             })

#             # Second API call with function result
#             second_response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 temperature=0.7,
#                 max_tokens=500
#             )
#             bot_reply = second_response.choices[0].message.content

#         # Extract invoice details if no function was called
#         if not query_called:
#             user_invoice_details[user_id] = await categorize_invoice_details_new(bot_reply, user_id)

#         inv_details = user_invoice_details[user_id]
#         po_number = inv_details.get("PO Number", "")
#         print("PO Number: ",po_number)
#         # Fetch PO items only if PO number is not empty and not fetched before
#         po_items = []
#         if po_number and po_number not in user_po_cache[user_id]:
#             print("Inside po loop")
#             po_items = fetch_po_item_ids(po_number)
#             if po_items:  # Only process if po_items is not empty
#                 po_items_json = json.dumps({"po_items": po_items})
#                 po_item_ids = [item["itemId"] for item in po_items]
#                 po_item_ids_string = ",".join(po_item_ids)
#                 print("po_item_ids_string: ",po_item_ids_string,"po_items",po_items )
#                 messages.append({
#                     "role": "user",  # Simulate user input
#                     "content": f"Items: {po_item_ids_string}"
#                 })
#                 # 
#                 # Mark this PO as processed
#                 user_po_cache[user_id].add(po_number)

#                 # Second API call with updated template
#                 second_response = client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=messages,
#                     temperature=0.7,
#                     max_tokens=500
#                 )
#                 bot_reply = second_response.choices[0].message.content
#                 # if not query_called and "Total Amount" in bot_reply or "Email" in bot_reply:
#                 #     user_invoice_details[user_id] = await categorize_invoice_details_new(bot_reply, user_id)
#                 # inv_details = user_invoice_details[user_id]

#         # Update conversation history
#         invoice_chat_histories[user_id].append(f"Bot: {bot_reply}")
#         invoice_email=inv_details["Email"]

#         # Determine submission status from bot_reply
#         # submissionStatus = "not submitted"
#         # if "Would you like to submit" in bot_reply:
#         #     submissionStatus = "pending"
#         # elif "Invoice created successfully" in bot_reply:
#         #     submissionStatus = "submitted"
#         # elif "I want to change something" in bot_reply:
#         #     submissionStatus = "cancelled"
#         # else:
#         #     submissionStatus = "in_progress"

#         # Additional processing from original implementation
#         test_model_reply = testModel(user_message, bot_reply)
#         form_submission = test_submission(bot_reply)
#         # if invoice_email and invoice_email not in invoice_email_cache[user_id]:
#         #     form_submission = "pending"
        
#         # if invoice_email:
#         #     invoice_email_cache[user_id].add(invoice_email)
#         if invoice_email:
#         # only if we’ve never seen it before do we override
#             if invoice_email not in invoice_email_cache[user_id]:
#                 form_submission = "pending"
#             invoice_email_cache[user_id].add(invoice_email)
        
#         # Determine action type
#         action = 'action'
#         past_pattern = re.compile(r"(past\s+invoice|invoice\s+past|last\s+invoice)", re.IGNORECASE)
#         create_pattern = re.compile(r"(create\s+invoice|invoice\s+create)", re.IGNORECASE)
        
#         for line in invoice_chat_histories[user_id]:
#             if line.startswith("User:"):
#                 user_input = line.split(":", 1)[1].strip()
#                 if past_pattern.search(user_input):
#                     action = "last invoice created"
#                 elif create_pattern.search(user_input):
#                     action = "create invoice"
#                 elif "create po" in user_input.lower():
#                     action = "create PO"

#         return {
#             "user_id": user_id,
#             "bot_reply": bot_reply,
#             "chat_history": invoice_chat_histories[user_id],
#             "conversation": invoice_chat_histories[user_id],
#             "invoice_json": inv_details,
#             "action": action,
#             "submissionStatus": form_submission,
#             "po_items": po_items,  # Appended PO items if fetched
#             "test_model_reply": test_model_reply,
#             "invoiceDatafromConversation": inv_details,
#             "invoice_email":invoice_email
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/creation/response_new") 
async def generate_response_new(request: ChatRequest):
    user_id = request.user_id
    user_message = request.message

    # Maintain user session for invoice creation
    if user_id not in invoice_chat_histories:
        invoice_chat_histories[user_id] = []
        user_invoice_details[user_id] = {}
        user_po_cache[user_id] = set()  # Initialize PO cache for the user

    invoice_chat_histories[user_id].append(f"User: {user_message}")
    conversation = "\n".join(invoice_chat_histories[user_id])

    try:
        # Prepare messages with system template
        messages = [
            {"role": "system", "content": template_5_new},
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
            print("Function call: query database")
            args = json.loads(function_call.arguments)
            query_result = query_database_function(args["question"])
            query_called = True

            # Append function response and make second call
            messages.append({
                "role": "function",
                "name": "query_database",
                "content": query_result
            })
            # Second API call with function result
            # second_response = client.chat.completions.create(
            #     model="gpt-4o",
            #     messages=messages,
            #     temperature=0.7,
            #     max_tokens=500
            # )
            # bot_reply = second_response.choices[0].message.content

        # Extract invoice details if no function was called
        if not query_called:
            user_invoice_details[user_id] = await categorize_invoice_details_new(bot_reply, user_id)

        inv_details = user_invoice_details[user_id]
        raw_items = inv_details.get("Items","")            # e.g. [{"Item ID": "ID123", …}, …]
        print("Fetched ids: ",raw_items)
        item_ids = [item['Item ID'] for item in raw_items]
# Print the new list of item IDs
        print("Item IDs: ", item_ids)
        if(item_ids):
                try:
                    db_session: Session = next(get_db())
                    item_validation=details_validator_invoice(db_session,item_ids)
                    print("Item validations: ",item_validation)
                    second_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500
                    )
                    messages.append({
                        "role": "user",  # Simulate user input
                        "content": f"Item Validations: {item_validation}"
                    })
                    bot_reply = second_response.choices[0].message.content
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
                finally:
                    db_session.close()

        po_number = inv_details.get("PO Number", "")
        print("PO Number: ",po_number)
        # Fetch PO items only if PO number is not empty and not fetched before
        po_items = []
        if po_number and po_number not in user_po_cache[user_id]:
            print("Inside po loop")
            po_items = fetch_po_item_ids(po_number)
            if po_items:  # Only process if po_items is not empty
                po_items_json = json.dumps({"po_items": po_items})
                po_item_ids = [item["itemId"] for item in po_items]
                po_item_ids_string = ",".join(po_item_ids)
                print("po_item_ids_string: ",po_item_ids_string,"po_items",po_items )
                messages.append({
                    "role": "user",  # Simulate user input
                    "content": f"Items: {po_item_ids_string}"
                })
                # 
                # Mark this PO as processed
                user_po_cache[user_id].add(po_number)

                # Second API call with updated template
                second_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                bot_reply = second_response.choices[0].message.content
                # if not query_called and "Total Amount" in bot_reply or "Email" in bot_reply:
                #     user_invoice_details[user_id] = await categorize_invoice_details_new(bot_reply, user_id)
                # inv_details = user_invoice_details[user_id]

        # Update conversation history
        invoice_chat_histories[user_id].append(f"Bot: {bot_reply}")
        invoice_email=inv_details["Email"]
        test_model_reply = testModel(user_message, bot_reply)
        form_submission = test_submission(bot_reply)
        # if invoice_email and invoice_email not in invoice_email_cache[user_id]:
        #     form_submission = "pending"
        
        # if invoice_email:
        #     invoice_email_cache[user_id].add(invoice_email)
        if invoice_email:
        # only if we’ve never seen it before do we override
            if invoice_email not in invoice_email_cache[user_id]:
                form_submission = "pending"
            invoice_email_cache[user_id].add(invoice_email)
        
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
            "po_items": po_items,  # Appended PO items if fetched
            "test_model_reply": test_model_reply,
            "invoiceDatafromConversation": inv_details,
            "invoice_email":invoice_email
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

