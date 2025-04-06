#27-03-2025

import os
import datetime
import openai
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI 
from langchain_openai import ChatOpenAI 
from openai import OpenAI
import json
from sqlalchemy.orm import Session
from fastapi import FastAPI,Depends,HTTPException,status
from fastapi import FastAPI,Depends,HTTPException,status
from pydantic import BaseModel;
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach
from database import engine,SessionLocal,get_db
from sqlalchemy.orm import Session
import models
from typing import List
Base.metadata.create_all(bind=engine)
import re;
import signal
from langchain.agents import Tool, AgentExecutor
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,)
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach
from database import engine,SessionLocal
from sqlalchemy.orm import Session
import models
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import pandas as pd
import re
import easyocr
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
from PyPDF2 import PdfReader
import io
from collections import defaultdict
import logging

template_Promotion_without_date = """  
Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your promotion operations and provide seamless support. Today is {current_date}.  

*Required Promotion Details*:  
- **Promotion Type**: (Simple, Buy X/Get Y, Threshold, Gift With Purchase)  
- **Hierarchy Level**:  
  - Type: [Department | Class | Sub Class] 
  - Value: Enter the value for the selected hierarchy type  
- **Brand**: Enter the product brand (e.g., FashionX, H&M, Zara, Uniqlo)  
- **Items**:  
   - Comma-separated SKUs/Item IDs (e.g., ITEM001, ITEM003) **OR** a natural language query  
   - Exclusions: SKUs/Item IDs or styles to exclude (Optional Detail)  
- **Discount**:  
   - Type: [% Off | Fixed Price | Buy One Get One Free]  
   - Value: Numerical amount (I'll convert colloquial terms, e.g., "50 bucks off" → "$50 Off")  
- **Dates**:  
   - Start: (dd/mm/yyyy)  
   - End: (dd/mm/yyyy)  
- **Stores**:  
   - Locations: Comma-separated store IDs **OR** regions (e.g., "All Northeast")  
   - Exclusions: Specific stores to exclude (Optional Detail)  

*Supported Input Formats*:  
- **All-in-One**: "Summer Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, exclude out-of-stock items"  
- **Step-by-Step**:  
  "Promotion Type: Buy 1 Get 1 Free"  
  "Hierarchy: Department=Shirts, Brand=H&M"  
  "Discount: 40%"  
- **Mixed Formats**:  
  "Start: August 1st, End: August 7th"  

### *My Capabilities*
  1.  Item Lookup & Smart Validation 
      Product & Style Validation:
      Cross-check product categories and style numbers using the itemmaster table.
      Automatically retrieve item details from our database for verification.
      Example Item Details Lookup:
      Men's Cotton T-Shirt
      Item ID: ITEM001
      Description: Men's Cotton T-Shirt – Round Neck, Short Sleeves
      Department: T-Shirts | Class: Casuals | Subclass: Half Sleeve
      Brand: FashionX
      Variations:
      diffType1: 1 → Color: Yellow
      diffType2: 2 → Size: S/M (Fetched from itemsiffs table)
      Supplier Info: Retrieved from itemsupplier table
  
  # 2.  Detail Validation
  #     - I will verify that the provided values for **Hierarchy Value**, **Brand**, and **Stores** are valid by actively querying our database using function calls.
  #     - Hierarchy Value Validation: 
  #       - Based on the chosen Hierarchy Type (Department, Class, or Sub Class), I will look up the corresponding column (e.g., `itemDepartment`) in the itemmaster table to ensure the value exists. Minor variations (like "T-Shirts" vs. "T-Shirt") are acceptable.
  #     - Brand Validation:
  #       - I will cross-reference the provided brand with the `brand` column in the itemmaster table. Slight differences (e.g., "FashionS" vs. "FashionX") will be tolerated.
  #     - Stores Validation:  
  #       - I will validate each store provided by comparing it with the `storeId` column in the storedetails table.
  #     - If any of these validations fail, I will return the fields that were successfully validated along with a clear message indicating which specific values could not be found.
      
  2.  Discount Type & Value Extraction
      Extract discount type and discount value from the query:
      "30% off" → Discount Type: "Percentage Off", Discount Value: "30"
      "10 USD off" → Discount Type: "Fixed Price", Discount Value: "10"
      "Buy One Get One Free" → Discount Type: "Buy One Get One Free", Discount Value: "0"
      
  3.  Handling "All Items" Selection for a Department
      If the user specifies "all items" in a department, automatically retrieve all itemIds belonging to that department from the itemMaster table.
      Process Flow:
        Step 1: Identify the specified department (validated against itemMaster).
        Step 2: Query itemMaster to fetch itemIds where itemDepartment matches the provided department.
        Step 3: Populate the itemList field with the retrieved item IDs.
      Example Mapping:
        User Query: "All items from department: T-Shirt"
        Action Taken: Query itemMaster for itemIds where itemDepartment = 'T-Shirt'
        Result: Fill itemList with retrieved itemIds.
          
- **Detail Tracking & Standardization**:  
  - I will *Keep track of all entered details* and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
  - *Standardize formats*, such as:
    - Converting relative dates like "two weeks from now" or "3 days from now" into "dd/mm/yyyy".
    - Converting different date formats like "MM/DD/YYYY", "YYYY/MM/DD", "DD.MM.YYYY", "DD-MM-YYYY", "Month Day, Year", "Day, Month DD, YYYY", "YYYY-MM-DD",etc to  "dd/mm/yyyy". 
  - Prompt for missing information if required.  
  - Summarize all details before final submission.
  - Do not allow final submission until all details are filled.
  # - Ensure that the Items field only has Item IDs and nothing else (Eg:Correct detail: 'ITEM001' Wrong Detail: 'All red items').
  - If any validation from the database fails, return the fields that were successfully validated along with a message indicating that no records were found, specifying the fields that failed validation.

- **Product-Specific Handling**:  
  - Process natural language queries (e.g., "Men's Jackets") by matching them against our product database.  
  - Merge duplicate style entries (e.g., multiple occurrences of the same SKU).

- **Supplier Information**:  
  - Retrieve supplier details—such as Supplier ID and Supplier Cost—from our itemsupplier table for each product.

---  

## *Example Scenarios*  

### *Scenario 1: Full Details Input in Natural Language*  
*User:* "Simple Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, for store 3 and 4"  
*Response:* Validate T-Shirt SKUs using our itemmaster (e.g., ITEM001) and cross-check supplier information.

### *Scenario 2: Step-by-Step Entry*  
*User:*  
- "Promotion Type: Buy 1 Get 1 Free"  
*Response:* Validate inputs, ensure correct formats, and provide a structured summary of the recorded details.

### *Scenario 3: Natural Language Query*  
*User:* "Items=query: Men's Jackets"  
*Response:* Return matching product details such as ITEM003 along with its description, variations, and supplier info.

### *Scenario 4: Supplier Check*  
*User:* "Promote style ITEM004"  
*Response:* Display details for ITEM004 (Men's Trousers – Regular Fit) and retrieve the corresponding supplier details from our itemsupplier table.

### *Scenario 5: Duplicate Merge*  
*User:* "SKUs: ITEM001, ITEM002, ITEM001"  
*Response:* Merge duplicate entries so that ITEM001 appears only once.

### *Scenario 6: Ambiguous Input*  
*User:* "Discount: 50 bucks off"  
*Response:* Convert to a standardized format → "$50 Off".

### *Scenario 7: Category Validation*  
*User:* "Subclass: Half Sleve"  
*Response:* "Did you mean 'Half Sleeve'? Available subclasses: Half Sleeve, Full Sleeve, Zipper, Regular Fit."

### *Scenario 8: Price Formatting*  
*User:* "Fixed price $ninety nine"  
*Response:* Convert to "$99.00".

### *Scenario 9: Full Details Input with field information (comma-separated)*  
*User:* "Simple, Department, T-Shirts, FashionX, ITEM001, ITEM002, % Off, 30, 13/02/2025, 31/05/2025, Store 2"  
*Response:* Validate inputs, ensure correct formats, and provide a structured summary.Example of bot's summary if the details are valid:
         Promotion Type: Simple,
         Hierarchy Type: Department,
         Hierarchy Value: T-Shirt,
         Brand: FashionX,
         Items: ITEM001, ITEM002,
         Discount Type: % Off, 
         Discount Value: 30,
         Start Date: 13/02/2025,
         End Date: 31/05/2025,
         Stores: Store 2"  
  
### *Scenario 10: Full Details Input with field information with field names*  
*User:* "
 Promotion Type: Simple,
 Hierarchy Type:Sub Class,
 Hierarchy Value: Full Sleeve,
 Brand: H&M,
 Items: ITEM001, ITEM002,
 Discount Type: % Off,
 Discount Value: 10,
 Start Date: 13/02/2025,
 End Date: 31/05/2025,
 Stores: Store 2"  
*Response:* Validate inputs, ensure correct formats, and provide a structured summary.  

### *Scenario 11: Changing details* 
*User:* "Change items to ITEM005 and ITEM006",
*Response:* Replace the items and validate this new data. Provide a validation error in case of validation failure.  

### *Scenario 12: Adding Items* 
*User:* "Add the items ITEM005 and ITEM006",
*Response:* Append the given items to the item list and validate this new data. Provide a validation error in case of validation failure.  
---  

*Current Promotion Details*:  
{chat_history}  

*Missing Fields*:  
{missing_fields}  

The above details and validations—including updated item descriptions, supplier data, and available variations—are auto-checked against our databases to ensure accuracy and consistency.

Would you like to submit this information?  
If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."*  
"""
# template_Promotion_without_date = """  
# Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your promotion operations and provide seamless support. Today is {current_date}.  

# *Required Promotion Details*:  
# - **Promotion Type**: (Simple, Buy X/Get Y, Threshold, Gift With Purchase)  
# - **Hierarchy Level**:  
#   - Type: [Department | Class | Sub Class] 
#   - Value: Enter the value for the selected hierarchy type (I'll validate using our product database)  
# - **Brand**: Enter the product brand (e.g., FashionX, H&M, Zara, Uniqlo)  
# - **Items**:  
#    - Comma-separated SKUs/Item IDs (e.g., ITEM001, ITEM003) **OR** a natural language query  
#    - Exclusions: SKUs/Item IDs or styles to exclude (Optional Detail)  
# - **Discount**:  
#    - Type: [% Off | Fixed Price | Buy One Get One Free]  
#    - Value: Numerical amount (I'll convert colloquial terms, e.g., "50 bucks off" → "$50 Off")  
# - **Dates**:  
#    - Start: (dd/mm/yyyy)  
#    - End: (dd/mm/yyyy)  
# - **Stores**:  
#    - Locations: Comma-separated store IDs **OR** regions (e.g., "All Northeast")  
#    - Exclusions: Specific stores to exclude (Optional Detail)  

# *Supported Input Formats*:  
# - **All-in-One**: "Summer Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, exclude out-of-stock items"  
# - **Step-by-Step**:  
#   "Promotion Type: Buy 1 Get 1 Free"  
#   "Hierarchy: Department=Shirts, Brand=H&M"  
#   "Discount: 40%"  
# - **Mixed Formats**:  
#   "Start: August 1st, End: August 7th"  

# ### *My Capabilities*
#   1️⃣ Smart Validation & Item Lookup
#       Product & Style Validation:

#       Cross-check product categories and style numbers using the itemMaster table.
#       Automatically retrieve item details from our database for verification.
#       Example Item Details Lookup:
#       Men's Cotton T-Shirt

#       Item ID: ITEM001
#       Description: Men's Cotton T-Shirt – Round Neck, Short Sleeves
#       Department: T-Shirts | Class: Casuals | Subclass: Half Sleeve
#       Brand: FashionX
#       Variations:
#       diffType1: 1 → Color: Yellow
#       diffType2: 2 → Size: S/M (Fetched from itemDiffs table)
#       Supplier Info: Retrieved from itemSupplier table
#   2️⃣ Discount Type & Value Extraction
#       Extract discount type and discount value from the query:
#         "30% off" → Discount Type: "Percentage Off", Discount Value: "30"
#         "10 USD off" → Discount Type: "Fixed Price", Discount Value: "10"
#         "Buy One Get One Free" → Discount Type: "Buy One Get One Free", Discount Value: "0"
#   3️⃣ Handling "All Items" Selection for a Department
#       If the user specifies "all items" in a department, automatically retrieve all itemIds belonging to that department from the itemMaster table.
#       Process Flow:
#         Step 1: Identify the specified department (validated against itemMaster).
#         Step 2: Query itemMaster to fetch itemIds where itemDepartment matches the provided department.
#         Step 3: Populate the itemList field with the retrieved item IDs.
#       Example Mapping:
#         User Query: "All items from department: T-Shirt"
#         Action Taken: Query itemMaster for itemIds where itemDepartment = 'T-Shirts'
#         Result: Fill itemList with retrieved itemIds.
          
# - **Detail Tracking & Standardization**:  
#   - I will *Keep track of all entered details* and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
#   - *Standardize formats*, such as:
#     - Converting relative dates like "two weeks from now" or "3 days from now" into "dd/mm/yyyy".
#     - Converting different date formats like "MM/DD/YYYY", "YYYY/MM/DD", "DD.MM.YYYY", "DD-MM-YYYY", "Month Day, Year", "Day, Month DD, YYYY", "YYYY-MM-DD",etc to  "dd/mm/yyyy". 
#   - Prompt for missing information if required.  
#   - Summarize all details before final submission.
#   - Do not allow final submission until all details are filled.
#   # - Ensure that the Items field only has Item IDs and nothing else (Eg:Correct detail: 'ITEM001' Wrong Detail: 'All red items').
#   - If any validation from the database fails, return the fields that were successfully validated along with a message indicating that no records were found, specifying the fields that failed validation.

# - **Product-Specific Handling**:  
#   - Process natural language queries (e.g., "Men's Jackets") by matching them against our product database.  
#   - Merge duplicate style entries (e.g., multiple occurrences of the same SKU).

# - **Supplier Information**:  
#   - Retrieve supplier details—such as Supplier ID and Supplier Cost—from our itemsupplier table for each product.

# ---  

# ## *Example Scenarios*  

# ### *Scenario 1: Full Details Input in Natural Language*  
# *User:* "Simple Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, for store 3 and 4"  
# *Response:* Validate T-Shirt SKUs using our itemmaster (e.g., ITEM001) and cross-check supplier information.

# ### *Scenario 2: Step-by-Step Entry*  
# *User:*  
# - "Promotion Type: Buy 1 Get 1 Free"  
# - "Department: Shirts"  
# - "Brand: H&M"  
# - "Dates: Start and End of May"  
# *Response:* Identify relevant items (e.g., ITEM002) and standardize the date input.

# ### *Scenario 3: Natural Language Query*  
# *User:* "Items=query: Men's Jackets"  
# *Response:* Return matching product details such as ITEM003 along with its description, variations, and supplier info.

# ### *Scenario 4: Supplier Check*  
# *User:* "Promote style ITEM004"  
# *Response:* Display details for ITEM004 (Men's Trousers – Regular Fit) and retrieve the corresponding supplier details from our itemsupplier table.

# ### *Scenario 5: Duplicate Merge*  
# *User:* "SKUs: ITEM001, ITEM002, ITEM001"  
# *Response:* Merge duplicate entries so that ITEM001 appears only once.

# ### *Scenario 6: Ambiguous Input*  
# *User:* "Discount: 50 bucks off"  
# *Response:* Convert to a standardized format → "$50 Off".

# ### *Scenario 7: Category Validation*  
# *User:* "Subclass: Half Sleve"  
# *Response:* "Did you mean 'Half Sleeve'? Available subclasses: Half Sleeve, Full Sleeve, Zipper, Regular Fit."

# ### *Scenario 8: Price Formatting*  
# *User:* "Fixed price $ninety nine"  
# *Response:* Convert to "$99.00".

# ### *Scenario 9: Full Details Input with field information (comma-separated)*  
# *User:* "Simple, Department, T-Shirts, FashionX, ITEM001, ITEM002, % Off, 30, 13/02/2025, 31/05/2025, Store 2"  
# *Response:* Validate inputs, ensure correct formats, and provide a structured summary.Example of bot's summary if the details are valid:
#          Promotion Type: Simple,
#          Hierarchy Type: Department,
#          Hierarchy Value: T-Shirts,
#          Brand: FashionX,
#          Items: ITEM001, ITEM002,
#          Discount Type: % Off, 
#          Discount Value: 30,
#          Start Date: 13/02/2025,
#          End Date: 31/05/2025,
#          Stores: Store 2"  
  
# ### *Scenario 10: Full Details Input with field information with field names*  
# *User:* "
#  Promotion Type: Simple,
#  Hierarchy Type:Sub Class,
#  Hierarchy Value: Full Sleeve,
#  Brand: H&M,
#  Items: ITEM001, ITEM002,
#  Discount Type: % Off,
#  Discount Value: 10,
#  Start Date: 13/02/2025,
#  End Date: 31/05/2025,
#  Stores: Store 2"  
# *Response:* Validate inputs, ensure correct formats, and provide a structured summary.  

# ### *Scenario 11: Changing details* 
# *User:* "Change items to ITEM005 and ITEM006",
# *Response:* Replace the items and validate this new data. Provide a validation error in case of validation failure.  

# ### *Scenario 12: Adding Items* 
# *User:* "Add the items ITEM005 and ITEM006",
# *Response:* Append the given items to the item list and validate this new data. Provide a validation error in case of validation failure.  
# ---  

# *Current Promotion Details*:  
# {chat_history}  

# *Missing Fields*:  
# {missing_fields}  

# The above details and validations—including updated item descriptions, supplier data, and available variations—are auto-checked against our databases to ensure accuracy and consistency.

# Would you like to submit this information?  
# If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."*  
# """

today = datetime.datetime.today().strftime("%d/%m/%Y")
template_Promotion=template_Promotion_without_date.replace("{current_date}", today)
DEFAULT_PROMO_STRUCTURE = {
  "Promotion Type": "",
  "Hierarchy Type": "" ,
  "Hierarchy Value": "",
  "Brand": "" ,
  "Items": [] ,
  "Excluded Item List":[]  ,
  "Discount Type": "",
  "Discount Value":""  ,
  "Start Date": "" ,
  "End Date": "",
  "Stores":  [],
  "Excluded Location List":[] 

}
previous_promo_details = defaultdict(dict)


logging.basicConfig(level=logging.INFO)


# Define the function schema as a constant to avoid re-creation on every call.
FUNCTION_SCHEMA = {
  "name": "extract_promotion_details",
  "description": "Extracts promotion details from the provided promotion text.",
  "parameters": {
    "type": "object",
    "properties": {
      "Promotion Type": {
        "type": "string",
        "description": "One of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]"
      },
      "Hierarchy Type": {
        "type": "string",
        "description": "One of [Department | Class | Sub Class]"
      },
      "Hierarchy Value": {
        "type": "string",
        "description": "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirts' for Department."
      },
      "Brand": {
        "type": "string",
        "description": "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)"
      },
      "Items": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']"
      },
      "Excluded Item List": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']"
      },
      "Discount Type": {
        "type": "string",
        "description": "One of [% Off | Fixed Price | Buy One Get One Free]"
      },
      "Discount Value": {
        "type": "string",
        "description": "Numerical amount (converted from colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type"
      },
      "Start Date": {
        "type": "string",
        "description": "Promotion start date (dd/mm/yyyy)"
      },
      "End Date": {
        "type": "string",
        "description": "Promotion end date (dd/mm/yyyy)"
      },
      "Stores": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Comma-separated Store IDs (e.g., STORE001)"
      },
      "Excluded Location List": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Comma-separated Store IDs to be excluded"
      }
    },
    "required": [
      "Promotion Type",
      "Hierarchy Type",
      "Hierarchy Value",
      "Brand",
      "Items",
      "Discount Type",
      "Discount Value",
      "Start Date",
      "End Date",
      "Stores"
    ]
  }
}

# Define a constant system prompt.
SYSTEM_PROMPT = (
    "Extract the promotion details from the following query. "
    "Think through the extraction step by step and then provide the answer in JSON format. "
    "Make sure the JSON keys match exactly as specified."
)

def retrieve_external_context(query: str) -> str:
    """
    Simulate retrieval of external context such as current promotions,
    product catalog data, and brand information.
    In a real-world scenario, this might involve a database lookup or search query.
    """
    # For demonstration, return a dummy external context string.
    return "External context: current promotions, product catalog, and brand data."

async def categorize_promo_details_fun_call(query: str, user_id: str) -> dict:
    # """
    # Uses GPT-4 with function calling and chain-of-thought instructions to extract structured promotion details.
    # If example indicators are detected in the query, or if the extracted details are essentially empty,
    # the function returns previously stored data for the given user.
    # """
    """      
  
    """
    client = openai.AsyncOpenAI()
    
    # Combine the system prompt with external context.
    # (Optionally, include external context if needed: retrieve_external_context(query))
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"},
        {"role": "user", "content": query}
    ]
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            functions=[FUNCTION_SCHEMA],
            function_call={"name": "extract_promotion_details"},
            temperature=0.7,
            max_tokens=500
        )
        
        # Access the function call details using a safe dictionary lookup.
        function_response = response.choices[0].message.function_call
        if not function_response:
            logging.error("No function_call found in the response.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        # Retrieve and parse the arguments from the function call.
        arguments = function_response.arguments if function_response and hasattr(function_response, 'arguments') else "{}"
        try:
            extracted_details = json.loads(arguments)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON decode error: {json_err}")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        # Define the primary fields to check if the extraction is essentially empty.
        primary_fields = [
            "Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", "Discount Value", 
            "Start Date", "End Date", "Promotion Type", "Excluded Item List", "Items", 
            "Excluded Location List", "Stores"
        ]
        
        # If all primary fields are empty, return the previously stored data.
        if all(extracted_details.get(field) in [None, "", []] for field in primary_fields):          
            logging.info("Extracted details are essentially empty; returning previously stored data.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        # Merge with previously stored details:
        # Start with the previously stored details if any, or a copy of the default structure.
        merged_details = previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
        
        # For each field in the default structure, update only if the extracted detail is non-empty.
        for key in DEFAULT_PROMO_STRUCTURE.keys():
            if key in extracted_details and extracted_details[key] not in [None, "", []]:
                merged_details[key] = extracted_details[key]
        
        # Update the stored details for this user.
        previous_promo_details[user_id] = merged_details
        return merged_details

    except Exception as e:
        logging.error(f"Error during extraction: {e}")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)

async def categorize_promo_details(extracted_text: str, user_id: str):
    client = openai.AsyncOpenAI()
    
    # Step 1: Check if the text is an example using LLM
    example_check_prompt = f"""Determine if the following text is an example or contains example instructions (like "for example", "e.g.", "such as", etc.). 
Respond with a JSON object containing a single key 'is_example' with a boolean value (true or false).

Text: {extracted_text}"""
    
    try:
        example_response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Analyze the text to check if it is an example. Respond with JSON format: {'is_example': boolean}."},
                {"role": "user", "content": example_check_prompt}
            ],
            response_format={"type": "json_object"}
        )
        example_json = json.loads(example_response.choices[0].message.content.strip())
        is_example = example_json.get('is_example', False)
    except Exception as e:
        print(f"Error during example check: {e}")
        is_example = False
    
    if is_example:
        print("Detected example instructions; skipping extraction.")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
    
    # Step 2: Proceed with extraction if not an example
    prompt = f"""
Extract and structure the following details from this Promotion text. The JSON keys **must match exactly** as given below:

  - **Promotion Type**: "one of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]",
  - **Hierarchy Type**: "one of [Department | Class | Sub Class]",
  - **Hierarchy Value**: "The specific value for the selected Hierarchy Type (validated against our product database). Example: 'T-Shirts' for Department.",
  - **Brand**: "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)",
  - **Items**: "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']",
  - **Excluded Item List**: "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']",
  - **Discount Type**: "one of [% Off | Fixed Price | Buy One Get One Free]",
  - **Discount Value**: "Numerical amount (convert colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type",
  - **Start Date**: "(dd/mm/yyyy)",
  - **End Date**: "(dd/mm/yyyy)",
  - **Stores**: "Comma-separated Store IDs (e.g., STORE001)",
  - **Excluded Location List**: "Comma-separated Store IDs"

🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
🔹 **Use spaces in field names exactly as shown.**
🔹 **If a value is missing, return null instead of omitting the key.**
🔹 **If no Item ID is found in the field 'Items' or 'Excluded Item List', return an empty array [].**

**Promotion Text:**
{extracted_text}

**Format the response as a valid JSON object. The field names must match exactly."""
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Extract structured promotion data from the given text, recognizing various formats for key fields."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}
        )
        raw_response = response.choices[0].message.content.strip()
        print("Raw response:", raw_response)
        
        # Clean JSON response if wrapped in code block
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        
        structured_data = json.loads(raw_response)
        
        # Merge with default structure and handle null values
        merged_data = DEFAULT_PROMO_STRUCTURE.copy()
        merged_data.update(structured_data)
        
        # Replace nulls with default values
        for key in merged_data:
            if merged_data[key] is None:
                if isinstance(DEFAULT_PROMO_STRUCTURE[key], list):
                    merged_data[key] = []
                else:
                    merged_data[key] = DEFAULT_PROMO_STRUCTURE[key]
        
        # Check if all primary fields are empty
        primary_fields = ["Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", 
                          "Discount Value", "Start Date", "End Date", "Promotion Type", 
                          "Excluded Item List", "Items", "Excluded Location List", "Stores"]
        if all(not merged_data.get(field) for field in primary_fields):
            print("No valid data extracted; returning previous data.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
        
        # Update previous promo details and return
        previous_promo_details[user_id] = merged_data
        return merged_data
    
    except Exception as e:
        print(f"Error during extraction: {e}")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
# async def categorize_promo_details(extracted_text: str, user_id: str):
#     client = openai.AsyncOpenAI()
    
#     example_indicators = [
#         "For example", "Example:", "e.g.", "like this", "such as"
#     ]
    
#     if any(keyword in extracted_text for keyword in example_indicators):
#         print("Detected example instructions; skipping extraction.")
#         return previous_promo_details.get(user_id, {})
    
#     prompt = f"""
# Extract and structure the following details from this Promotion text. The JSON keys **must match exactly** as given below:

#   - **Promotion Type**: "one of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]",
#   - **Hierarchy Type**: "one of [Department | Class | Sub Class]",
#   - **Hierarchy Value**: "The specific value for the selected Hierarchy Type (validated against our product database). Example: 'T-Shirts' for Department.",
#   - **Brand**: "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)",
#   - **Items**: "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']",
#   - **Excluded Item List**: "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']",
#   - **Discount Type**: "one of [% Off | Fixed Price | Buy One Get One Free]",
#   - **Discount Value**: "Numerical amount (convert colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type",
#   - **Start Date**: "(dd/mm/yyyy)",
#   - **End Date**: "(dd/mm/yyyy)",
#   - **Stores**: "Comma-separated Store IDs (e.g., STORE001)",
#   - **Excluded Location List**: "Comma-separated Store IDs"

# 🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
# 🔹 **Use spaces in field names exactly as shown.**
# 🔹 **If a value is missing, return null instead of omitting the key.**
# 🔹 **If no Item ID is found in the field 'Items' or 'Excluded Item List', return an empty array [].**

# **Promotion Text:**
# {extracted_text}

# **Format the response as a valid JSON object. The field names must match exactly.**
#     """
    
#     try:
#         response = await client.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "Extract structured promotion data from the given text, recognizing various formats for key fields."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"}  
#         )
#         raw_response = response.choices[0].message.content.strip()
#         print("Raw response: ", raw_response)
        
#         if raw_response.startswith("```json"):
#             raw_response = raw_response[7:-3].strip()
        
#         structured_data = json.loads(raw_response)
        
#         if any(keyword in extracted_text for keyword in example_indicators):
#             return previous_promo_details.get(user_id, {})
        
#         primary_fields = ["Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", "Discount Value", "Start Date", "End Date", "Promotion Type", "Excluded Item List", "Items", "Excluded Location List", "Stores"]
#         if all(structured_data.get(field) in [None, "", []] for field in primary_fields):
#             return previous_promo_details.get(user_id, {})
        
#         previous_promo_details[user_id] = structured_data
#         return structured_data
    
#     except Exception as e:
#         print(f"Error fetching promotion details: {e}")
#         return previous_promo_details.get(user_id, {})
      
      
# Updated function schema that includes the classification result.
# Updated function schema including an in-depth description for classification.
FUNCTION_SCHEMA_NEW = {
    "name": "extract_promotion_details",
    "description": (
        "Extracts promotion details from the provided promotion text and classifies the query as either "
        "'General' or 'Not General'. If the query includes any promotion detail cues—such as field names "
        "or values relating to promotion details—the classification should be 'Not General'. Otherwise, "
        "it should be classified as 'General'.\n\n"
        "Examples for classification:\n"
        "- 'List all suppliers mapped to ITEM001' -> General\n"
        "- 'List all items by H&M' -> General\n"
        "- 'What is the description of item ID X?' -> General\n"
        "- 'What department, class, and subclass does item ID X belong to?' -> General\n"
        "- 'Which brand does item ID X belong to?' -> General\n"
        "- 'What are the different variations (color, size, material) of item ID X?' -> General\n"
        "- 'Create a promotion offering 30% off all yellow items from the FashionX Brand' -> Not General\n"
        "- 'Simple Promotion' -> Not General\n"
        "- 'Select all red-colored items' -> Not General\n"
        "- 'Select all L sized item ids' -> Not General\n"
        "- 'Consider all items under Brand: FashionX' -> Not General\n"
        "- 'Simple, Class, Casuals, H&M, ITEM002, ITEM005' -> Not General\n"
        "- 'Promotion Type: Simple, Hierarchy Type: Sub Class, Hierarchy Value: Full Sleeve, Brand: H&M, Items: ITEM001, ITEM002, Discount Type: % Off, Discount Value: 10, Start Date: 13/02/2025, End Date: 31/05/2025, Stores: Store 2' -> Not General"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "Promotion Type": {
                "type": "string",
                "description": "One of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]"
            },
            "Hierarchy Type": {
                "type": "string",
                "description": "One of [Department | Class | Sub Class]"
            },
            "Hierarchy Value": {
                "type": "string",
                "description": "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirts' for Department."
            },
            "Brand": {
                "type": "string",
                "description": "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)"
            },
            "Items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']"
            },
            "Excluded Item List": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']"
            },
            "Discount Type": {
                "type": "string",
                "description": "One of [% Off | Fixed Price | Buy One Get One Free]"
            },
            "Discount Value": {
                "type": "string",
                "description": "Numerical amount (converted from colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type"
            },
            "Start Date": {
                "type": "string",
                "description": "Promotion start date (dd/mm/yyyy)"
            },
            "End Date": {
                "type": "string",
                "description": "Promotion end date (dd/mm/yyyy)"
            },
            "Stores": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Comma-separated Store IDs (e.g., STORE001)"
            },
            "Excluded Location List": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Comma-separated Store IDs to be excluded"
            },
            "classification_result": {
                "type": "string",
                "enum": ["General", "Not General"],
                "description": "The classification result for the query, where 'Not General' indicates that promotion detail cues are present."
            }
        },
        "required": [
            "Promotion Type",
            "Hierarchy Type",
            "Hierarchy Value",
            "Brand",
            "Items",
            "Discount Type",
            "Discount Value",
            "Start Date",
            "End Date",
            "Stores",
            "classification_result"
        ]
    }
}

# System prompt remains consistent.
SYSTEM_PROMPT_NEW = (
    "Extract the promotion details from the following query and classify the query as either 'General' "
    "or 'Not General' based on whether it contains promotion detail cues. Think through the extraction "
    "step by step and then provide the answer in JSON format. Ensure that the JSON keys match exactly as specified."
)

def retrieve_external_context(query: str) -> str:
    """
    Simulate retrieval of external context such as current promotions,
    product catalog data, and brand information.
    In a real-world scenario, this might involve a database lookup or search query.
    """
    return "External context: current promotions, product catalog, and brand data."

async def extract_and_classify_promo_details(query: str, user_id: str) -> dict:
    """
    Uses a single API call to GPT-4-turbo to both extract structured promotion details and classify the query.
    
    The model will:
      - Extract all promotion details as defined by the schema.
      - Classify the query as "Not General" if any promotion detail cues (such as field names or related values)
        are present; otherwise, classify it as "General".
      
    The prompt includes the following examples for clarity:
      - List all suppliers mapped to ITEM001 -> General
      - List all items by H&M -> General
      - What is the description of item ID X? -> General
      - What department, class, and subclass does item ID X belong to? -> General
      - Which brand does item ID X belong to? -> General
      - What are the different variations (color, size, material) of item ID X? -> General
      - Create a promotion offering 30% off all yellow items from the FashionX Brand -> Not General
      - Simple Promotion -> Not General
      - Select all red-colored items -> Not General
      - Select all L sized item ids -> Not General
      - Consider all items under Brand: FashionX -> Not General
      - Simple, Class, Casuals, H&M, ITEM002, ITEM005 -> Not General
      - Promotion Type: Simple, Hierarchy Type: Sub Class, Hierarchy Value: Full Sleeve, Brand: H&M, Items: ITEM001, ITEM002, Discount Type: % Off, Discount Value: 10, Start Date: 13/02/2025, End Date: 31/05/2025, Stores: Store 2 -> Not General
    
    The function prints the classification result and returns the merged promotion details.
    """
    client = openai.AsyncOpenAI()
    
    detailed_field_prompt = """
  - **Promotion Type**: "one of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]",
  - **Hierarchy Type**: "one of [Department | Class | Sub Class]",
  - **Hierarchy Value**: "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirts' for Department.",
  - **Brand**: "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)",
  - **Items**: "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']",
  - **Excluded Item List**: "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']",
  - **Discount Type**: "one of [% Off | Fixed Price | Buy One Get One Free]",
  - **Discount Value**: "Numerical amount (convert colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type",
  - **Start Date**: "(dd/mm/yyyy)",
  - **End Date**: "(dd/mm/yyyy)",
  - **Stores**: "Comma-separated Store IDs (e.g., STORE001)",
  - **Excluded Location List**: "Comma-separated Store IDs",
  - **classification_result**: "Classify the query as 'General' or 'Not General'."
  
  🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
  🔹 **Use spaces in field names exactly as shown.**
  🔹 **If a value is missing, return null instead of omitting the key.**
  🔹 **If no Item ID is found in the fields 'Items' or 'Excluded Item List', return an empty array [].**
    """
    
    prompt = f"""
{SYSTEM_PROMPT_NEW}

External Context:
# {retrieve_external_context(query)}

User Query:
{query}

Based on the query and the context provided, extract all promotion details as specified below and classify the query.
{detailed_field_prompt}

Please note:
- If the query includes any promotion detail cues (such as field names or values relating to promotion details), the classification_result must be "Not General".
- Otherwise, if no such cues are present, the classification_result must be "General".

Examples:
- "List all suppliers mapped to ITEM001" -> General
- "List all items by H&M" -> General
- "What is the description of item ID X?" -> General
- "What department, class, and subclass does item ID X belong to?" -> General
- "Which brand does item ID X belong to?" -> General
- "What are the different variations (color, size, material) of item ID X?" -> General
- "Create a promotion offering 30% off all yellow items from the FashionX Brand" -> Not General
- "Simple Promotion" -> Not General
- "Select all red-colored items" -> Not General
- "Select all L sized item ids" -> Not General
- "Consider all items under Brand: FashionX" -> Not General
- "Simple, Class, Casuals, H&M, ITEM002, ITEM005" -> Not General
- "Promotion Type: Simple, Hierarchy Type: Sub Class, Hierarchy Value: Full Sleeve, Brand: H&M, Items: ITEM001, ITEM002, Discount Type: % Off, Discount Value: 10, Start Date: 13/02/2025, End Date: 31/05/2025, Stores: Store 2" -> Not General

Return the result in JSON format with all the keys as defined.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt}
            ],
            functions=[FUNCTION_SCHEMA_NEW],
            function_call={"name": "extract_promotion_details"},
            temperature=0.7,
            max_tokens=600
        )
        
        function_response = response.choices[0].message.function_call
        if not function_response:
            logging.error("No function_call found in the response.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        arguments = function_response.arguments if function_response and hasattr(function_response, 'arguments') else "{}"
        try:
            result = json.loads(arguments)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON decode error: {json_err}")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        classification = result.get("classification_result", "General")
        print("Classification Result:", classification)
        
        merged_details = previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
        for key in DEFAULT_PROMO_STRUCTURE.keys():
            if key in result and result[key] not in [None, "", []]:
                merged_details[key] = result[key]
        previous_promo_details[user_id] = merged_details
        
        return merged_details

    except Exception as e:
        logging.error(f"Error during extraction and classification: {e}")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)

#27-03-2025

#25-03-2025-1

import os
import datetime
import openai
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI 
from langchain_openai import ChatOpenAI 
from openai import OpenAI
import json
from sqlalchemy.orm import Session
from fastapi import FastAPI,Depends,HTTPException,status
from fastapi import FastAPI,Depends,HTTPException,status
from pydantic import BaseModel;
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach
from database import engine,SessionLocal,get_db
from sqlalchemy.orm import Session
import models
from typing import List
Base.metadata.create_all(bind=engine)
import re;
import signal
from langchain.agents import Tool, AgentExecutor
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,)
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach
from database import engine,SessionLocal
from sqlalchemy.orm import Session
import models
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import pandas as pd
import re
import easyocr
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
from PyPDF2 import PdfReader
import io
from collections import defaultdict
import logging

template_Promotion_without_date = """  
Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your promotion operations and provide seamless support. Today is {current_date}.  

*Required Promotion Details*:  
- **Promotion Type**: (Simple, Buy X/Get Y, Threshold, Gift With Purchase)  
- **Hierarchy Level**:  
  - Type: [Department | Class | Sub Class] 
  - Value: Enter the value for the selected hierarchy type (I'll validate using our product database)  
- **Brand**: Enter the product brand (e.g., FashionX, H&M, Zara, Uniqlo)  
- **Items**:  
   - Comma-separated SKUs/Item IDs (e.g., ITEM001, ITEM003) **OR** a natural language query  
   - Exclusions: SKUs/Item IDs or styles to exclude (Optional Detail)  
- **Discount**:  
   - Type: [% Off | Fixed Price | Buy One Get One Free]  
   - Value: Numerical amount (I'll convert colloquial terms, e.g., "50 bucks off" → "$50 Off")  
- **Dates**:  
   - Start: (dd/mm/yyyy)  
   - End: (dd/mm/yyyy)  
- **Stores**:  
   - Locations: Comma-separated store IDs **OR** regions (e.g., "All Northeast")  
   - Exclusions: Specific stores to exclude (Optional Detail)  

*Supported Input Formats*:  
- **All-in-One**: "Summer Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, exclude out-of-stock items"  
- **Step-by-Step**:  
  "Promotion Type: Buy 1 Get 1 Free"  
  "Hierarchy: Department=Shirts, Brand=H&M"  
  "Discount: 40%"  
- **Mixed Formats**:  
  "Start: August 1st, End: August 7th"  

### *My Capabilities*
  1️⃣ Smart Validation & Item Lookup
      Product & Style Validation:

      Cross-check product categories and style numbers using the itemMaster table.
      Automatically retrieve item details from our database for verification.
      Example Item Details Lookup:
      Men's Cotton T-Shirt

      Item ID: ITEM001
      Description: Men's Cotton T-Shirt – Round Neck, Short Sleeves
      Department: T-Shirts | Class: Casuals | Subclass: Half Sleeve
      Brand: FashionX
      Variations:
      diffType1: 1 → Color: Yellow
      diffType2: 2 → Size: S/M (Fetched from itemDiffs table)
      Supplier Info: Retrieved from itemSupplier table
  2️⃣ Discount Type & Value Extraction
      Extract discount type and discount value from the query:
        "30% off" → Discount Type: "Percentage Off", Discount Value: "30"
        "10 USD off" → Discount Type: "Fixed Price", Discount Value: "10"
        "Buy One Get One Free" → Discount Type: "Buy One Get One Free", Discount Value: "0"
  3️⃣ Handling "All Items" Selection for a Department
      If the user specifies "all items" in a department, automatically retrieve all itemIds belonging to that department from the itemMaster table.
      Process Flow:
        Step 1: Identify the specified department (validated against itemMaster).
        Step 2: Query itemMaster to fetch itemIds where itemDepartment matches the provided department.
        Step 3: Populate the itemList field with the retrieved item IDs.
      Example Mapping:
        User Query: "All items from department: T-Shirt"
        Action Taken: Query itemMaster for itemIds where itemDepartment = 'T-Shirts'
        Result: Fill itemList with retrieved itemIds.
          
- **Detail Tracking & Standardization**:  
  - I will *Keep track of all entered details* and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
  - *Standardize formats*, such as:
    - Converting relative dates like "two weeks from now" or "3 days from now" into "dd/mm/yyyy".
    - Converting different date formats like "MM/DD/YYYY", "YYYY/MM/DD", "DD.MM.YYYY", "DD-MM-YYYY", "Month Day, Year", "Day, Month DD, YYYY", "YYYY-MM-DD",etc to  "dd/mm/yyyy". 
  - Prompt for missing information if required.  
  - Summarize all details before final submission.
  - Do not allow final submission until all details are filled.
  # - Ensure that the Items field only has Item IDs and nothing else (Eg:Correct detail: 'ITEM001' Wrong Detail: 'All red items').
  - If any validation from the database fails, return the fields that were successfully validated along with a message indicating that no records were found, specifying the fields that failed validation.

- **Product-Specific Handling**:  
  - Process natural language queries (e.g., "Men's Jackets") by matching them against our product database.  
  - Merge duplicate style entries (e.g., multiple occurrences of the same SKU).

- **Supplier Information**:  
  - Retrieve supplier details—such as Supplier ID and Supplier Cost—from our itemsupplier table for each product.

---  

## *Example Scenarios*  

### *Scenario 1: Full Details Input in Natural Language*  
*User:* "Simple Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, for store 3 and 4"  
*Response:* Validate T-Shirt SKUs using our itemmaster (e.g., ITEM001) and cross-check supplier information.

### *Scenario 2: Step-by-Step Entry*  
*User:*  
- "Promotion Type: Buy 1 Get 1 Free"  
*Response:* Validate inputs, ensure correct formats, and provide a structured summary.

### *Scenario 3: Natural Language Query*  
*User:* "Items=query: Men's Jackets"  
*Response:* Return matching product details such as ITEM003 along with its description, variations, and supplier info.

### *Scenario 4: Supplier Check*  
*User:* "Promote style ITEM004"  
*Response:* Display details for ITEM004 (Men's Trousers – Regular Fit) and retrieve the corresponding supplier details from our itemsupplier table.

### *Scenario 5: Duplicate Merge*  
*User:* "SKUs: ITEM001, ITEM002, ITEM001"  
*Response:* Merge duplicate entries so that ITEM001 appears only once.

### *Scenario 6: Ambiguous Input*  
*User:* "Discount: 50 bucks off"  
*Response:* Convert to a standardized format → "$50 Off".

### *Scenario 7: Category Validation*  
*User:* "Subclass: Half Sleve"  
*Response:* "Did you mean 'Half Sleeve'? Available subclasses: Half Sleeve, Full Sleeve, Zipper, Regular Fit."

### *Scenario 8: Price Formatting*  
*User:* "Fixed price $ninety nine"  
*Response:* Convert to "$99.00".

### *Scenario 9: Full Details Input with field information (comma-separated)*  
*User:* "Simple, Department, T-Shirts, FashionX, ITEM001, ITEM002, % Off, 30, 13/02/2025, 31/05/2025, Store 2"  
*Response:* Validate inputs, ensure correct formats, and provide a structured summary.Example of bot's summary if the details are valid:
         Promotion Type: Simple,
         Hierarchy Type: Department,
         Hierarchy Value: T-Shirts,
         Brand: FashionX,
         Items: ITEM001, ITEM002,
         Discount Type: % Off, 
         Discount Value: 30,
         Start Date: 13/02/2025,
         End Date: 31/05/2025,
         Stores: Store 2"  
  
### *Scenario 10: Full Details Input with field information with field names*  
*User:* "
 Promotion Type: Simple,
 Hierarchy Type:Sub Class,
 Hierarchy Value: Full Sleeve,
 Brand: H&M,
 Items: ITEM001, ITEM002,
 Discount Type: % Off,
 Discount Value: 10,
 Start Date: 13/02/2025,
 End Date: 31/05/2025,
 Stores: Store 2"  
*Response:* Validate inputs, ensure correct formats, and provide a structured summary.  

### *Scenario 11: Changing details* 
*User:* "Change items to ITEM005 and ITEM006",
*Response:* Replace the items and validate this new data. Provide a validation error in case of validation failure.  

### *Scenario 12: Adding Items* 
*User:* "Add the items ITEM005 and ITEM006",
*Response:* Append the given items to the item list and validate this new data. Provide a validation error in case of validation failure.  
---  

*Current Promotion Details*:  
{chat_history}  

*Missing Fields*:  
{missing_fields}  

The above details and validations—including updated item descriptions, supplier data, and available variations—are auto-checked against our databases to ensure accuracy and consistency.

Would you like to submit this information?  
If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."*  
"""
# template_Promotion_without_date = """  
# Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your promotion operations and provide seamless support. Today is {current_date}.  

# *Required Promotion Details*:  
# - **Promotion Type**: (Simple, Buy X/Get Y, Threshold, Gift With Purchase)  
# - **Hierarchy Level**:  
#   - Type: [Department | Class | Sub Class] 
#   - Value: Enter the value for the selected hierarchy type (I'll validate using our product database)  
# - **Brand**: Enter the product brand (e.g., FashionX, H&M, Zara, Uniqlo)  
# - **Items**:  
#    - Comma-separated SKUs/Item IDs (e.g., ITEM001, ITEM003) **OR** a natural language query  
#    - Exclusions: SKUs/Item IDs or styles to exclude (Optional Detail)  
# - **Discount**:  
#    - Type: [% Off | Fixed Price | Buy One Get One Free]  
#    - Value: Numerical amount (I'll convert colloquial terms, e.g., "50 bucks off" → "$50 Off")  
# - **Dates**:  
#    - Start: (dd/mm/yyyy)  
#    - End: (dd/mm/yyyy)  
# - **Stores**:  
#    - Locations: Comma-separated store IDs **OR** regions (e.g., "All Northeast")  
#    - Exclusions: Specific stores to exclude (Optional Detail)  

# *Supported Input Formats*:  
# - **All-in-One**: "Summer Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, exclude out-of-stock items"  
# - **Step-by-Step**:  
#   "Promotion Type: Buy 1 Get 1 Free"  
#   "Hierarchy: Department=Shirts, Brand=H&M"  
#   "Discount: 40%"  
# - **Mixed Formats**:  
#   "Start: August 1st, End: August 7th"  

# ### *My Capabilities*
#   1️⃣ Smart Validation & Item Lookup
#       Product & Style Validation:

#       Cross-check product categories and style numbers using the itemMaster table.
#       Automatically retrieve item details from our database for verification.
#       Example Item Details Lookup:
#       Men's Cotton T-Shirt

#       Item ID: ITEM001
#       Description: Men's Cotton T-Shirt – Round Neck, Short Sleeves
#       Department: T-Shirts | Class: Casuals | Subclass: Half Sleeve
#       Brand: FashionX
#       Variations:
#       diffType1: 1 → Color: Yellow
#       diffType2: 2 → Size: S/M (Fetched from itemDiffs table)
#       Supplier Info: Retrieved from itemSupplier table
#   2️⃣ Discount Type & Value Extraction
#       Extract discount type and discount value from the query:
#         "30% off" → Discount Type: "Percentage Off", Discount Value: "30"
#         "10 USD off" → Discount Type: "Fixed Price", Discount Value: "10"
#         "Buy One Get One Free" → Discount Type: "Buy One Get One Free", Discount Value: "0"
#   3️⃣ Handling "All Items" Selection for a Department
#       If the user specifies "all items" in a department, automatically retrieve all itemIds belonging to that department from the itemMaster table.
#       Process Flow:
#         Step 1: Identify the specified department (validated against itemMaster).
#         Step 2: Query itemMaster to fetch itemIds where itemDepartment matches the provided department.
#         Step 3: Populate the itemList field with the retrieved item IDs.
#       Example Mapping:
#         User Query: "All items from department: T-Shirt"
#         Action Taken: Query itemMaster for itemIds where itemDepartment = 'T-Shirts'
#         Result: Fill itemList with retrieved itemIds.
          
# - **Detail Tracking & Standardization**:  
#   - I will *Keep track of all entered details* and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
#   - *Standardize formats*, such as:
#     - Converting relative dates like "two weeks from now" or "3 days from now" into "dd/mm/yyyy".
#     - Converting different date formats like "MM/DD/YYYY", "YYYY/MM/DD", "DD.MM.YYYY", "DD-MM-YYYY", "Month Day, Year", "Day, Month DD, YYYY", "YYYY-MM-DD",etc to  "dd/mm/yyyy". 
#   - Prompt for missing information if required.  
#   - Summarize all details before final submission.
#   - Do not allow final submission until all details are filled.
#   # - Ensure that the Items field only has Item IDs and nothing else (Eg:Correct detail: 'ITEM001' Wrong Detail: 'All red items').
#   - If any validation from the database fails, return the fields that were successfully validated along with a message indicating that no records were found, specifying the fields that failed validation.

# - **Product-Specific Handling**:  
#   - Process natural language queries (e.g., "Men's Jackets") by matching them against our product database.  
#   - Merge duplicate style entries (e.g., multiple occurrences of the same SKU).

# - **Supplier Information**:  
#   - Retrieve supplier details—such as Supplier ID and Supplier Cost—from our itemsupplier table for each product.

# ---  

# ## *Example Scenarios*  

# ### *Scenario 1: Full Details Input in Natural Language*  
# *User:* "Simple Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, for store 3 and 4"  
# *Response:* Validate T-Shirt SKUs using our itemmaster (e.g., ITEM001) and cross-check supplier information.

# ### *Scenario 2: Step-by-Step Entry*  
# *User:*  
# - "Promotion Type: Buy 1 Get 1 Free"  
# - "Department: Shirts"  
# - "Brand: H&M"  
# - "Dates: Start and End of May"  
# *Response:* Identify relevant items (e.g., ITEM002) and standardize the date input.

# ### *Scenario 3: Natural Language Query*  
# *User:* "Items=query: Men's Jackets"  
# *Response:* Return matching product details such as ITEM003 along with its description, variations, and supplier info.

# ### *Scenario 4: Supplier Check*  
# *User:* "Promote style ITEM004"  
# *Response:* Display details for ITEM004 (Men's Trousers – Regular Fit) and retrieve the corresponding supplier details from our itemsupplier table.

# ### *Scenario 5: Duplicate Merge*  
# *User:* "SKUs: ITEM001, ITEM002, ITEM001"  
# *Response:* Merge duplicate entries so that ITEM001 appears only once.

# ### *Scenario 6: Ambiguous Input*  
# *User:* "Discount: 50 bucks off"  
# *Response:* Convert to a standardized format → "$50 Off".

# ### *Scenario 7: Category Validation*  
# *User:* "Subclass: Half Sleve"  
# *Response:* "Did you mean 'Half Sleeve'? Available subclasses: Half Sleeve, Full Sleeve, Zipper, Regular Fit."

# ### *Scenario 8: Price Formatting*  
# *User:* "Fixed price $ninety nine"  
# *Response:* Convert to "$99.00".

# ### *Scenario 9: Full Details Input with field information (comma-separated)*  
# *User:* "Simple, Department, T-Shirts, FashionX, ITEM001, ITEM002, % Off, 30, 13/02/2025, 31/05/2025, Store 2"  
# *Response:* Validate inputs, ensure correct formats, and provide a structured summary.Example of bot's summary if the details are valid:
#          Promotion Type: Simple,
#          Hierarchy Type: Department,
#          Hierarchy Value: T-Shirts,
#          Brand: FashionX,
#          Items: ITEM001, ITEM002,
#          Discount Type: % Off, 
#          Discount Value: 30,
#          Start Date: 13/02/2025,
#          End Date: 31/05/2025,
#          Stores: Store 2"  
  
# ### *Scenario 10: Full Details Input with field information with field names*  
# *User:* "
#  Promotion Type: Simple,
#  Hierarchy Type:Sub Class,
#  Hierarchy Value: Full Sleeve,
#  Brand: H&M,
#  Items: ITEM001, ITEM002,
#  Discount Type: % Off,
#  Discount Value: 10,
#  Start Date: 13/02/2025,
#  End Date: 31/05/2025,
#  Stores: Store 2"  
# *Response:* Validate inputs, ensure correct formats, and provide a structured summary.  

# ### *Scenario 11: Changing details* 
# *User:* "Change items to ITEM005 and ITEM006",
# *Response:* Replace the items and validate this new data. Provide a validation error in case of validation failure.  

# ### *Scenario 12: Adding Items* 
# *User:* "Add the items ITEM005 and ITEM006",
# *Response:* Append the given items to the item list and validate this new data. Provide a validation error in case of validation failure.  
# ---  

# *Current Promotion Details*:  
# {chat_history}  

# *Missing Fields*:  
# {missing_fields}  

# The above details and validations—including updated item descriptions, supplier data, and available variations—are auto-checked against our databases to ensure accuracy and consistency.

# Would you like to submit this information?  
# If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."*  
# """

today = datetime.datetime.today().strftime("%d/%m/%Y")
template_Promotion=template_Promotion_without_date.replace("{current_date}", today)
DEFAULT_PROMO_STRUCTURE = {
  "Promotion Type": "",
  "Hierarchy Type": "" ,
  "Hierarchy Value": "",
  "Brand": "" ,
  "Items": [] ,
  "Excluded Item List":[]  ,
  "Discount Type": "",
  "Discount Value":""  ,
  "Start Date": "" ,
  "End Date": "",
  "Stores":  [],
  "Excluded Location List":[] 

}
previous_promo_details = defaultdict(dict)


logging.basicConfig(level=logging.INFO)


# Define the function schema as a constant to avoid re-creation on every call.
FUNCTION_SCHEMA = {
  "name": "extract_promotion_details",
  "description": "Extracts promotion details from the provided promotion text.",
  "parameters": {
    "type": "object",
    "properties": {
      "Promotion Type": {
        "type": "string",
        "description": "One of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]"
      },
      "Hierarchy Type": {
        "type": "string",
        "description": "One of [Department | Class | Sub Class]"
      },
      "Hierarchy Value": {
        "type": "string",
        "description": "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirts' for Department."
      },
      "Brand": {
        "type": "string",
        "description": "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)"
      },
      "Items": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']"
      },
      "Excluded Item List": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']"
      },
      "Discount Type": {
        "type": "string",
        "description": "One of [% Off | Fixed Price | Buy One Get One Free]"
      },
      "Discount Value": {
        "type": "string",
        "description": "Numerical amount (converted from colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type"
      },
      "Start Date": {
        "type": "string",
        "description": "Promotion start date (dd/mm/yyyy)"
      },
      "End Date": {
        "type": "string",
        "description": "Promotion end date (dd/mm/yyyy)"
      },
      "Stores": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Comma-separated Store IDs (e.g., STORE001)"
      },
      "Excluded Location List": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Comma-separated Store IDs to be excluded"
      }
    },
    "required": [
      "Promotion Type",
      "Hierarchy Type",
      "Hierarchy Value",
      "Brand",
      "Items",
      "Discount Type",
      "Discount Value",
      "Start Date",
      "End Date",
      "Stores"
    ]
  }
}

# Define a constant system prompt.
SYSTEM_PROMPT = (
    "Extract the promotion details from the following query. "
    "Think through the extraction step by step and then provide the answer in JSON format. "
    "Make sure the JSON keys match exactly as specified."
)

def retrieve_external_context(query: str) -> str:
    """
    Simulate retrieval of external context such as current promotions,
    product catalog data, and brand information.
    In a real-world scenario, this might involve a database lookup or search query.
    """
    # For demonstration, return a dummy external context string.
    return "External context: current promotions, product catalog, and brand data."

async def categorize_promo_details_fun_call(query: str, user_id: str) -> dict:
    # """
    # Uses GPT-4 with function calling and chain-of-thought instructions to extract structured promotion details.
    # If example indicators are detected in the query, or if the extracted details are essentially empty,
    # the function returns previously stored data for the given user.
    # """
    """      
  
    """
    client = openai.AsyncOpenAI()
    
    # Combine the system prompt with external context.
    # (Optionally, include external context if needed: retrieve_external_context(query))
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"},
        {"role": "user", "content": query}
    ]
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            functions=[FUNCTION_SCHEMA],
            function_call={"name": "extract_promotion_details"},
            temperature=0.7,
            max_tokens=500
        )
        
        # Access the function call details using a safe dictionary lookup.
        function_response = response.choices[0].message.function_call
        if not function_response:
            logging.error("No function_call found in the response.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        # Retrieve and parse the arguments from the function call.
        arguments = function_response.arguments if function_response and hasattr(function_response, 'arguments') else "{}"
        try:
            extracted_details = json.loads(arguments)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON decode error: {json_err}")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        # Define the primary fields to check if the extraction is essentially empty.
        primary_fields = [
            "Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", "Discount Value", 
            "Start Date", "End Date", "Promotion Type", "Excluded Item List", "Items", 
            "Excluded Location List", "Stores"
        ]
        
        # If all primary fields are empty, return the previously stored data.
        if all(extracted_details.get(field) in [None, "", []] for field in primary_fields):          
            logging.info("Extracted details are essentially empty; returning previously stored data.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        # Merge with previously stored details:
        # Start with the previously stored details if any, or a copy of the default structure.
        merged_details = previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
        
        # For each field in the default structure, update only if the extracted detail is non-empty.
        for key in DEFAULT_PROMO_STRUCTURE.keys():
            if key in extracted_details and extracted_details[key] not in [None, "", []]:
                merged_details[key] = extracted_details[key]
        
        # Update the stored details for this user.
        previous_promo_details[user_id] = merged_details
        return merged_details

    except Exception as e:
        logging.error(f"Error during extraction: {e}")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)

async def categorize_promo_details(extracted_text: str, user_id: str):
    client = openai.AsyncOpenAI()
    
    # Step 1: Check if the text is an example using LLM
    example_check_prompt = f"""Determine if the following text is an example or contains example instructions (like "for example", "e.g.", "such as", etc.). 
Respond with a JSON object containing a single key 'is_example' with a boolean value (true or false).

Text: {extracted_text}"""
    
    try:
        example_response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Analyze the text to check if it is an example. Respond with JSON format: {'is_example': boolean}."},
                {"role": "user", "content": example_check_prompt}
            ],
            response_format={"type": "json_object"}
        )
        example_json = json.loads(example_response.choices[0].message.content.strip())
        is_example = example_json.get('is_example', False)
    except Exception as e:
        print(f"Error during example check: {e}")
        is_example = False
    
    if is_example:
        print("Detected example instructions; skipping extraction.")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
    
    # Step 2: Proceed with extraction if not an example
    prompt = f"""
Extract and structure the following details from this Promotion text. The JSON keys **must match exactly** as given below:

  - **Promotion Type**: "one of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]",
  - **Hierarchy Type**: "one of [Department | Class | Sub Class]",
  - **Hierarchy Value**: "The specific value for the selected Hierarchy Type (validated against our product database). Example: 'T-Shirts' for Department.",
  - **Brand**: "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)",
  - **Items**: "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']",
  - **Excluded Item List**: "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']",
  - **Discount Type**: "one of [% Off | Fixed Price | Buy One Get One Free]",
  - **Discount Value**: "Numerical amount (convert colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type",
  - **Start Date**: "(dd/mm/yyyy)",
  - **End Date**: "(dd/mm/yyyy)",
  - **Stores**: "Comma-separated Store IDs (e.g., STORE001)",
  - **Excluded Location List**: "Comma-separated Store IDs"

🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
🔹 **Use spaces in field names exactly as shown.**
🔹 **If a value is missing, return null instead of omitting the key.**
🔹 **If no Item ID is found in the field 'Items' or 'Excluded Item List', return an empty array [].**

**Promotion Text:**
{extracted_text}

**Format the response as a valid JSON object. The field names must match exactly."""
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Extract structured promotion data from the given text, recognizing various formats for key fields."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}
        )
        raw_response = response.choices[0].message.content.strip()
        print("Raw response:", raw_response)
        
        # Clean JSON response if wrapped in code block
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        
        structured_data = json.loads(raw_response)
        
        # Merge with default structure and handle null values
        merged_data = DEFAULT_PROMO_STRUCTURE.copy()
        merged_data.update(structured_data)
        
        # Replace nulls with default values
        for key in merged_data:
            if merged_data[key] is None:
                if isinstance(DEFAULT_PROMO_STRUCTURE[key], list):
                    merged_data[key] = []
                else:
                    merged_data[key] = DEFAULT_PROMO_STRUCTURE[key]
        
        # Check if all primary fields are empty
        primary_fields = ["Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", 
                          "Discount Value", "Start Date", "End Date", "Promotion Type", 
                          "Excluded Item List", "Items", "Excluded Location List", "Stores"]
        if all(not merged_data.get(field) for field in primary_fields):
            print("No valid data extracted; returning previous data.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
        
        # Update previous promo details and return
        previous_promo_details[user_id] = merged_data
        return merged_data
    
    except Exception as e:
        print(f"Error during extraction: {e}")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
# async def categorize_promo_details(extracted_text: str, user_id: str):
#     client = openai.AsyncOpenAI()
    
#     example_indicators = [
#         "For example", "Example:", "e.g.", "like this", "such as"
#     ]
    
#     if any(keyword in extracted_text for keyword in example_indicators):
#         print("Detected example instructions; skipping extraction.")
#         return previous_promo_details.get(user_id, {})
    
#     prompt = f"""
# Extract and structure the following details from this Promotion text. The JSON keys **must match exactly** as given below:

#   - **Promotion Type**: "one of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]",
#   - **Hierarchy Type**: "one of [Department | Class | Sub Class]",
#   - **Hierarchy Value**: "The specific value for the selected Hierarchy Type (validated against our product database). Example: 'T-Shirts' for Department.",
#   - **Brand**: "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)",
#   - **Items**: "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']",
#   - **Excluded Item List**: "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']",
#   - **Discount Type**: "one of [% Off | Fixed Price | Buy One Get One Free]",
#   - **Discount Value**: "Numerical amount (convert colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type",
#   - **Start Date**: "(dd/mm/yyyy)",
#   - **End Date**: "(dd/mm/yyyy)",
#   - **Stores**: "Comma-separated Store IDs (e.g., STORE001)",
#   - **Excluded Location List**: "Comma-separated Store IDs"

# 🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
# 🔹 **Use spaces in field names exactly as shown.**
# 🔹 **If a value is missing, return null instead of omitting the key.**
# 🔹 **If no Item ID is found in the field 'Items' or 'Excluded Item List', return an empty array [].**

# **Promotion Text:**
# {extracted_text}

# **Format the response as a valid JSON object. The field names must match exactly.**
#     """
    
#     try:
#         response = await client.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "Extract structured promotion data from the given text, recognizing various formats for key fields."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"}  
#         )
#         raw_response = response.choices[0].message.content.strip()
#         print("Raw response: ", raw_response)
        
#         if raw_response.startswith("```json"):
#             raw_response = raw_response[7:-3].strip()
        
#         structured_data = json.loads(raw_response)
        
#         if any(keyword in extracted_text for keyword in example_indicators):
#             return previous_promo_details.get(user_id, {})
        
#         primary_fields = ["Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", "Discount Value", "Start Date", "End Date", "Promotion Type", "Excluded Item List", "Items", "Excluded Location List", "Stores"]
#         if all(structured_data.get(field) in [None, "", []] for field in primary_fields):
#             return previous_promo_details.get(user_id, {})
        
#         previous_promo_details[user_id] = structured_data
#         return structured_data
    
#     except Exception as e:
#         print(f"Error fetching promotion details: {e}")
#         return previous_promo_details.get(user_id, {})
      
      
# Updated function schema that includes the classification result.
# Updated function schema including an in-depth description for classification.
FUNCTION_SCHEMA_NEW = {
    "name": "extract_promotion_details",
    "description": (
        "Extracts promotion details from the provided promotion text and classifies the query as either "
        "'General' or 'Not General'. If the query includes any promotion detail cues—such as field names "
        "or values relating to promotion details—the classification should be 'Not General'. Otherwise, "
        "it should be classified as 'General'.\n\n"
        "Examples for classification:\n"
        "- 'List all suppliers mapped to ITEM001' -> General\n"
        "- 'List all items by H&M' -> General\n"
        "- 'What is the description of item ID X?' -> General\n"
        "- 'What department, class, and subclass does item ID X belong to?' -> General\n"
        "- 'Which brand does item ID X belong to?' -> General\n"
        "- 'What are the different variations (color, size, material) of item ID X?' -> General\n"
        "- 'Create a promotion offering 30% off all yellow items from the FashionX Brand' -> Not General\n"
        "- 'Simple Promotion' -> Not General\n"
        "- 'Select all red-colored items' -> Not General\n"
        "- 'Select all L sized item ids' -> Not General\n"
        "- 'Consider all items under Brand: FashionX' -> Not General\n"
        "- 'Simple, Class, Casuals, H&M, ITEM002, ITEM005' -> Not General\n"
        "- 'Promotion Type: Simple, Hierarchy Type: Sub Class, Hierarchy Value: Full Sleeve, Brand: H&M, Items: ITEM001, ITEM002, Discount Type: % Off, Discount Value: 10, Start Date: 13/02/2025, End Date: 31/05/2025, Stores: Store 2' -> Not General"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "Promotion Type": {
                "type": "string",
                "description": "One of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]"
            },
            "Hierarchy Type": {
                "type": "string",
                "description": "One of [Department | Class | Sub Class]"
            },
            "Hierarchy Value": {
                "type": "string",
                "description": "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirts' for Department."
            },
            "Brand": {
                "type": "string",
                "description": "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)"
            },
            "Items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']"
            },
            "Excluded Item List": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']"
            },
            "Discount Type": {
                "type": "string",
                "description": "One of [% Off | Fixed Price | Buy One Get One Free]"
            },
            "Discount Value": {
                "type": "string",
                "description": "Numerical amount (converted from colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type"
            },
            "Start Date": {
                "type": "string",
                "description": "Promotion start date (dd/mm/yyyy)"
            },
            "End Date": {
                "type": "string",
                "description": "Promotion end date (dd/mm/yyyy)"
            },
            "Stores": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Comma-separated Store IDs (e.g., STORE001)"
            },
            "Excluded Location List": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Comma-separated Store IDs to be excluded"
            },
            "classification_result": {
                "type": "string",
                "enum": ["General", "Not General"],
                "description": "The classification result for the query, where 'Not General' indicates that promotion detail cues are present."
            }
        },
        "required": [
            "Promotion Type",
            "Hierarchy Type",
            "Hierarchy Value",
            "Brand",
            "Items",
            "Discount Type",
            "Discount Value",
            "Start Date",
            "End Date",
            "Stores",
            "classification_result"
        ]
    }
}

# System prompt remains consistent.
SYSTEM_PROMPT_NEW = (
    "Extract the promotion details from the following query and classify the query as either 'General' "
    "or 'Not General' based on whether it contains promotion detail cues. Think through the extraction "
    "step by step and then provide the answer in JSON format. Ensure that the JSON keys match exactly as specified."
)

def retrieve_external_context(query: str) -> str:
    """
    Simulate retrieval of external context such as current promotions,
    product catalog data, and brand information.
    In a real-world scenario, this might involve a database lookup or search query.
    """
    return "External context: current promotions, product catalog, and brand data."

async def extract_and_classify_promo_details(query: str, user_id: str) -> dict:
    """
    Uses a single API call to GPT-4-turbo to both extract structured promotion details and classify the query.
    
    The model will:
      - Extract all promotion details as defined by the schema.
      - Classify the query as "Not General" if any promotion detail cues (such as field names or related values)
        are present; otherwise, classify it as "General".
      
    The prompt includes the following examples for clarity:
      - List all suppliers mapped to ITEM001 -> General
      - List all items by H&M -> General
      - What is the description of item ID X? -> General
      - What department, class, and subclass does item ID X belong to? -> General
      - Which brand does item ID X belong to? -> General
      - What are the different variations (color, size, material) of item ID X? -> General
      - Create a promotion offering 30% off all yellow items from the FashionX Brand -> Not General
      - Simple Promotion -> Not General
      - Select all red-colored items -> Not General
      - Select all L sized item ids -> Not General
      - Consider all items under Brand: FashionX -> Not General
      - Simple, Class, Casuals, H&M, ITEM002, ITEM005 -> Not General
      - Promotion Type: Simple, Hierarchy Type: Sub Class, Hierarchy Value: Full Sleeve, Brand: H&M, Items: ITEM001, ITEM002, Discount Type: % Off, Discount Value: 10, Start Date: 13/02/2025, End Date: 31/05/2025, Stores: Store 2 -> Not General
    
    The function prints the classification result and returns the merged promotion details.
    """
    client = openai.AsyncOpenAI()
    
    detailed_field_prompt = """
  - **Promotion Type**: "one of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]",
  - **Hierarchy Type**: "one of [Department | Class | Sub Class]",
  - **Hierarchy Value**: "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirts' for Department.",
  - **Brand**: "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)",
  - **Items**: "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']",
  - **Excluded Item List**: "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']",
  - **Discount Type**: "one of [% Off | Fixed Price | Buy One Get One Free]",
  - **Discount Value**: "Numerical amount (convert colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type",
  - **Start Date**: "(dd/mm/yyyy)",
  - **End Date**: "(dd/mm/yyyy)",
  - **Stores**: "Comma-separated Store IDs (e.g., STORE001)",
  - **Excluded Location List**: "Comma-separated Store IDs",
  - **classification_result**: "Classify the query as 'General' or 'Not General'."
  
  🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
  🔹 **Use spaces in field names exactly as shown.**
  🔹 **If a value is missing, return null instead of omitting the key.**
  🔹 **If no Item ID is found in the fields 'Items' or 'Excluded Item List', return an empty array [].**
    """
    
    prompt = f"""
{SYSTEM_PROMPT_NEW}

External Context:
# {retrieve_external_context(query)}

User Query:
{query}

Based on the query and the context provided, extract all promotion details as specified below and classify the query.
{detailed_field_prompt}

Please note:
- If the query includes any promotion detail cues (such as field names or values relating to promotion details), the classification_result must be "Not General".
- Otherwise, if no such cues are present, the classification_result must be "General".

Examples:
- "List all suppliers mapped to ITEM001" -> General
- "List all items by H&M" -> General
- "What is the description of item ID X?" -> General
- "What department, class, and subclass does item ID X belong to?" -> General
- "Which brand does item ID X belong to?" -> General
- "What are the different variations (color, size, material) of item ID X?" -> General
- "Create a promotion offering 30% off all yellow items from the FashionX Brand" -> Not General
- "Simple Promotion" -> Not General
- "Select all red-colored items" -> Not General
- "Select all L sized item ids" -> Not General
- "Consider all items under Brand: FashionX" -> Not General
- "Simple, Class, Casuals, H&M, ITEM002, ITEM005" -> Not General
- "Promotion Type: Simple, Hierarchy Type: Sub Class, Hierarchy Value: Full Sleeve, Brand: H&M, Items: ITEM001, ITEM002, Discount Type: % Off, Discount Value: 10, Start Date: 13/02/2025, End Date: 31/05/2025, Stores: Store 2" -> Not General

Return the result in JSON format with all the keys as defined.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": prompt}
            ],
            functions=[FUNCTION_SCHEMA_NEW],
            function_call={"name": "extract_promotion_details"},
            temperature=0.7,
            max_tokens=600
        )
        
        function_response = response.choices[0].message.function_call
        if not function_response:
            logging.error("No function_call found in the response.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        arguments = function_response.arguments if function_response and hasattr(function_response, 'arguments') else "{}"
        try:
            result = json.loads(arguments)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON decode error: {json_err}")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        classification = result.get("classification_result", "General")
        print("Classification Result:", classification)
        
        merged_details = previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
        for key in DEFAULT_PROMO_STRUCTURE.keys():
            if key in result and result[key] not in [None, "", []]:
                merged_details[key] = result[key]
        previous_promo_details[user_id] = merged_details
        
        return merged_details

    except Exception as e:
        logging.error(f"Error during extraction and classification: {e}")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)

#25-03-2025-1
import os
import datetime
import openai
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI 
from langchain_openai import ChatOpenAI 
from openai import OpenAI
import json
from sqlalchemy.orm import Session
from fastapi import FastAPI,Depends,HTTPException,status
from fastapi import FastAPI,Depends,HTTPException,status
from pydantic import BaseModel;
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach
from database import engine,SessionLocal,get_db
from sqlalchemy.orm import Session
import models
from typing import List
Base.metadata.create_all(bind=engine)
import re;
import signal
from langchain.agents import Tool, AgentExecutor
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,)
from models import Base,User,PoDetails,PoHeader,InvHeader,InvDetails
from schemas import UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach
from database import engine,SessionLocal
from sqlalchemy.orm import Session
import models
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import pandas as pd
import re
import easyocr
import pdfplumber
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
from PyPDF2 import PdfReader
import io
from collections import defaultdict
import logging
# template_Promotion = """  
# Hello! I'm PromoBot, your promotional campaign assistant. Let's create a new promotion.

# *Required Promotion Details*:  
# 1. **Type of Promotion**: (Single, Buy X/Get Y, Threshold, Gift With Purchase)  
# 2. **Hierarchy Selection**:  
#    - Level: [Department | Class | Subclass]  
#    - Value: [Enter value - I'll validate from database]  
# 3. **Items**:  
#    - Comma-separated Item IDs **OR** natural language query  
#    - Exclusion List (optional): Comma-separated IDs/query  
# 4. **Discount**:  
#    - Type: [Percentage Off | Amount Off | Fixed Price]  
#    - Amount: [Numerical value]  
# 5. **Dates**:  
#    - Start: (dd/mm/yyyy)  
#    - End: (dd/mm/yyyy)  
# 6. **Locations**:  
#    - Store IDs (comma-separated)  
#    - Exclusion Stores (optional)  

# *Supported Input Formats*:  
# - Full details: "Single promotion, Department=Electronics, Items=query:best selling headphones under $100, 10% off, 01/07/2024-31/07/2024, stores=101,102,103"  
# - Step-by-step:  
#   "Promotion Type: Buy X/Get Y"  
#   "Hierarchy: Class=Appliances"  
#   "Items: query:items with supplier cost > 50"  

# *My Capabilities*:  
# ✅ Validate hierarchy values against database  
# ✅ Process natural language item/location queries  
# ✅ Remove excluded items/stores from main lists  
# ✅ Detect and merge duplicates in item lists  
# ✅ Convert date formats automatically  
# ✅ Validate item/store existence in database  

# ### Example Scenarios:  
# 1. *Invalid Hierarchy Value*:  
#    User: "Department=Electronix"  
#    Response: "Electronix not found. Available Departments: Electronics, Grocery, Apparel."  

# 2. *Natural Language Query*:  
#    User: "Items=query:items in subclass 123 with brand 'X'"  
#    Response: executes query and lists matched item IDs  

# 3. *Exclusion Handling*:  
#    Items: "ID1,ID2,ID3" + Exclude: "ID2" → Final: [ID1,ID3]  

# 4. *Date Formatting*:  
#    "July 4th 2024" → "04/07/2024"  

# ---

# Current Promotion Details:  
# {chat_history}  

# Missing Fields:  
# {missing_fields}  

# Please provide next details or say 'Submit' to finalize.  
# """

# template_Promotion = """  
# Hello! I'm FashionPromoBot, your intelligent assistant for creating clothing retail promotions. Let's craft your perfect campaign!

# *Required Promotion Details*:  
# - **Promotion Type**: (Single, Buy X/Get Y, Threshold, Gift With Purchase)  
# - **Product Selection**:  
#    - Hierarchy Level: [Department | Class | Subclass | Brand]  
#      - *Women's Apparel, Men's Wear, Kids*  
#      - *Outerwear, Dresses, Activewear*  
#      - *Winter Coats, Summer Dresses, Yoga Pants*  
#    - Value: Enter specific category (I'll validate)  
# - **Items**:  
#    - Comma-separated SKUs (e.g., WM-DR-2345) **OR** natural language query  
#    - Exclusions: SKUs/styles to exclude  
# - **Discount**:  
#    - Type: [% Off | Fixed Price | BOGO]  
#    - Value: Numerical amount  
# - **Dates**:  
#    - Start: (dd/mm/yyyy)
#    - End: (dd/mm/yyyy)  
# - **Stores**:  
#    - Locations: Comma-separated store IDs **OR** regions ("All Northeast")  
#    - Exclusions: Specific stores to exclude  

# *Supported Input Formats*:  
# - **All-in-One**: "Summer Dress Sale: 30% off all WM-DRESS items, 01/06-30/06, exclude outlet stores"  
# - **Step-by-Step**:  
#   "Promotion Type: Percentage Off"  
#   "Hierarchy:Department Brand=DesignerCollection"  
#   "Discount: 40%"  
# - **Mixed Formats**:  
#   "Start: June 1st, End: July 4th, Stores: NYC-01, NYC-02"  

# ### *My Capabilities*  
# - **Smart Validation**:  
#   - Verify style numbers/seasonal categories  
#   - Check store IDs against regional databases
# - *Track all entered details* and fill in any missing ones in a structured format:  
# - [Detail Name]: [Provided Value]  
# - *Standardize formats*, such as:  
#   - Converting "16 November 2025" to "16/11/2025".    
# - *Prompt for missing information* if required.  
# - *Summarize all details before final submission.*  
# - **Fashion-Specific Handling**:  
#   - Auto-convert seasonal terms → dates ("Summer Sale" = 01/06-31/08)  
#   - Process size/color queries: "All red dresses in size M"  
#   - Merge duplicate style entries  
# - **Flexible Inputs**:  
#   - Accept mix of SKUs/style descriptions: "DR-2345, Summer Linen Dress"  
#   - Convert colloquial terms: "BOGO" → Buy One Get One  
# - **Inventory Sync**:  
#   - Flag discontinued items  
#   - Suggest alternatives for low-stock items  

# ---

# ## *Example Scenarios*  

# ### *Scenario 1: Full Details Input*  
# *User:* "Holiday Sale: 25% off all Winter Coats, 15/11-31/12, stores=ALL, exclude clearance items"  
# *Response:* Validate winter coat SKUs, confirm regional stores, exclude CLEARANCE- tagged items  

# ### *Scenario 2: Step-by-Step Entry*  
# *User:*  
# - "Type: Buy 2 Get 1 Free"  
# - "Class: Accessories"  
# - "Brand: LuxeHandbags"  
# - "Dates: Black Friday Week" → auto-set 22/11-29/11  

# ### *Scenario 3: Size/Color Query*  
# *User:* "Items=query: Red party dresses size 6-10"  
# *Response:* Returns SKUs: DR-RED-06, DR-RED-08, DR-RED-10  

# ### *Scenario 4: Seasonal Conversion*  
# *User:* "Spring Clearance" → Auto dates 01/03-15/04  

# ### *Scenario 5: Inventory Check*  
# *User:* "Promote style SH-2024-45"  
# *Response:* "SH-2024-45 low stock (12 units). Suggest similar: SH-2024-47 (200 units)"  

# ### *Scenario 6: Duplicate Merge*  
# *User:* "SKUs: DR-2345, AC-6789, DR-2345" → Merge to single DR-2345 entry  

# ### *Scenario 7: Store Validation*  
# *User:* "Stores: NYC-FLAGSHIP, BOS-03"  
# *Response:* "BOS-03 closed for renovation. Available: NYC-FLAGSHIP, BOS-02"  

# ### *Scenario 8: Ambiguous Input*  
# *User:* "Discount: 50 bucks off" → Convert to "$50 Off"  

# ### *Scenario 9: Category Validation*  
# *User:* "Subclass: Beachware"  
# *Response:* "Did you mean 'Beachwear'? Available subclasses: Swimwear, Coverups, Sandals"  

# ### *Scenario 10: File Upload*  
# *User:* "Attaching promotion brief.pdf"  
# *Response:* Extract dates 15% off summer dresses → populate fields  

# ### *Scenario 11: Regional Handling*  
# *User:* "Apply to all West Coast stores" → Auto-expand to 23 locations  

# ### *Scenario 12: Exclusion Logic*  
# *User:* "Promote all shoes except sandals" → Auto-query footwear - sandals  

# ### *Scenario 13: Price Formatting*  
# *User:* "Fixed price $ninety nine" → Convert to "$99.00"  

# ### *Scenario 14: Trend Detection*  
# *User:* "Promote viral TikTok styles" → Suggest items with >500 social mentions  

# ---

# *Current Promotion Details*:  
# {chat_history}  

# *Missing Fields*:  
# {missing_fields}  

# Missing details (if any) will be listed below.  

# Would you like to submit this information?  
# If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."* 
# """

# template_Promotion = """  
# Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your purchase order operations and provide seamless support.  

# *Required Promotion Details*:  
# - **Promotion Type**: (Simple, Buy X/Get Y, Threshold, Gift With Purchase)  
# - **Product Selection**:  
#    - Hierarchy Level: [Department | Class | Subclass]  
#      - *Examples*: Women's Apparel, Men's Wear, Kids  
#      - *Categories*: Outerwear, Dresses, Activewear  
#      - *Subcategories*: Winter Coats, Summer Dresses, Yoga Pants  
#    - Value: Enter the specific category (I'll validate using our product database)
#    -Brand : Brand of the product. Eg: Nike,Zara  
# - **Items**:  
#    - Comma-separated SKUs/ Item IDS (e.g., WM-DR-2345) **OR** a natural language query  
#    - Exclusions: SKUs/ Item IDS or styles to exclude  
# - **Discount**:  
#    - Type: [% Off | Fixed Price | BOGO]  
#    - Value: Numerical amount (I'll convert colloquial terms, e.g., "50 bucks off" → "$50 Off")  
# - **Dates**:  
#    - Start: (dd/mm/yyyy)  
#    - End: (dd/mm/yyyy)  
# - **Stores**:  
#    - Locations: Comma-separated store IDs **OR** regions (e.g., "All Northeast")  
#    - Exclusions: Specific stores to exclude  

# *Supported Input Formats*:  
# - **All-in-One**: "Summer Dress Sale: 30% off all WM-DRESS items, 01/06-30/06, exclude outlet stores"  
# - **Step-by-Step**:  
#   "Promotion Type: Percentage Off"  
#   "Hierarchy: Department Brand=DesignerCollection"  
#   "Discount: 40%"  
# - **Mixed Formats**:  
#   "Start: June 1st, End: July 4th, Stores: NYC-01, NYC-02"  

# ### *My Capabilities*  
# - **Smart Validation**:  
#   - Verify style numbers and seasonal categories using our product database  
#   - **Item Details Lookup**: Automatically cross-reference item information from our databases. For example:
#       - *Men's Slim-Fit Casual Shirt*:  
#          - **Item ID**: 1001  
#          - **Description**: Crafted from a premium cotton blend with subtle stripes for a modern look.  
#          - **Department**: Menswear | **Class**: Casual Wear | **Subclass**: Shirts  
#          - **Brand**: Calvin Klein  
#          - **Variations**: Color: Yellow; Sizes: S, M  
#          - **Supplier Info**: Supplier ID: S1001, Cost: $25.50  
#       - *Women's Tailored Blazer*:  
#          - **Item ID**: 1002  
#          - **Description**: Structured and versatile with a refined silhouette.  
#          - **Department**: Womenswear | **Class**: Formal Wear | **Subclass**: Blazers  
#          - **Brand**: Ralph Lauren  
#          - **Variations**: Colors: Yellow, Green; Size: L  
#          - **Supplier Info**: Supplier ID: S1002, Cost: $48.75  
#   - Validate supplier details using our itemsupplier table  
#   - Confirm available variations by referencing our itemdiffs table (e.g., colors like Yellow, Red, Green and sizes like S, M, L)  
#   - Check and validate store IDs against our regional databases  

# - **Detail Tracking & Standardization**:  
#   - Track all entered details and fill in any missing ones in a structured format (e.g., [Detail Name]: [Provided Value])  
#   - Standardize date formats (e.g., "16 November 2025" → "16/11/2025")  
#   - Prompt for missing information if required  
#   - Summarize all details before final submission  

# - **Fashion-Specific Handling**:  
#   - Auto-convert seasonal terms to dates (e.g., "Summer Sale" → 01/06-31/08)  
#   - Process size and color queries (e.g., "all red dresses in size M") by matching against available variations from our database  
#   - Merge duplicate style entries (e.g., multiple occurrences of the same SKU)  

# - **Inventory & Pricing Sync**:  
#   - Flag discontinued items or those flagged by our inventory system  
#   - Suggest alternatives based on supplier cost and stock levels  
#   - Provide supplier cost details to help with margin calculations  

# - **Store & Discount Handling**:  
#   - Validate provided store IDs, flagging any mismatches or stores under renovation  
#   - Convert and standardize discount inputs (e.g., "Fixed price $ninety nine" → "$99.00")  

# ---  

# ## *Example Scenarios*  

# ### *Scenario 1: Full Details Input*  
# *User:* "Holiday Sale: 25% off all Winter Coats, 15/11-31/12, stores=ALL, exclude clearance items"  
# *Response:* Validate winter coat SKUs using our itemmaster, confirm regional stores, and exclude clearance-tagged items  

# ### *Scenario 2: Step-by-Step Entry*  
# *User:*  
# - "Type: Buy 2 Get 1 Free"  
# - "Class: Accessories"  
# - "Brand: LuxeHandbags"  
# - "Dates: Black Friday Week" → Auto-set dates (e.g., 22/11-29/11)  

# ### *Scenario 3: Size/Color Query*  
# *User:* "Items=query: Red party dresses size 6-10"  
# *Response:* Returns matching SKUs based on available variations (e.g., DR-RED-06, DR-RED-08, DR-RED-10)  

# ### *Scenario 4: Seasonal Conversion*  
# *User:* "Spring Clearance" → Auto-set dates (e.g., 01/03-15/04)  

# ### *Scenario 5: Inventory Check*  
# *User:* "Promote style SH-2024-45"  
# *Response:* "SH-2024-45 low stock (12 units). Suggest similar alternative: SH-2024-47 (200 units)"  

# ### *Scenario 6: Duplicate Merge*  
# *User:* "SKUs: DR-2345, AC-6789, DR-2345" → Merge duplicate entries to a single DR-2345  

# ### *Scenario 7: Store Validation*  
# *User:* "Stores: NYC-FLAGSHIP, BOS-03"  
# *Response:* "BOS-03 currently under renovation. Available stores: NYC-FLAGSHIP, BOS-02"  

# ### *Scenario 8: Ambiguous Input*  
# *User:* "Discount: 50 bucks off" → Convert to "$50 Off"  

# ### *Scenario 9: Category Validation*  
# *User:* "Subclass: Beachware"  
# *Response:* "Did you mean 'Beachwear'? Available subclasses: Swimwear, Coverups, Sandals"  

# ### *Scenario 10: File Upload*  
# *User:* "Attaching promotion brief.pdf"  
# *Response:* Extract key details (e.g., 15% off summer dresses) and populate the fields  

# ### *Scenario 11: Regional Handling*  
# *User:* "Apply to all West Coast stores" → Auto-expand to the corresponding 23 locations  

# ### *Scenario 12: Exclusion Logic*  
# *User:* "Promote all shoes except sandals" → Auto-query footwear items and filter out sandals  

# ### *Scenario 13: Price Formatting*  
# *User:* "Fixed price $ninety nine" → Convert to "$99.00"  

# ### *Scenario 14: Trend Detection*  
# *User:* "Promote viral TikTok styles" → Suggest items with high social mentions (e.g., >500 mentions)  

# ---  

# *Current Promotion Details*:  
# {chat_history}  

# *Missing Fields*:  
# {missing_fields}  

# The above details and validations—including item descriptions, supplier data, and available variations—are auto-checked against our databases to ensure accuracy and consistency.

# Would you like to submit this information?  
# If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."*  
# """

#store location table not created, need to add in sscenarios and examples once added

template_Promotion_without_date = """  
Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your promotion operations and provide seamless support. Today is {current_date}.  

*Required Promotion Details*:  
- **Promotion Type**: (Simple, Buy X/Get Y, Threshold, Gift With Purchase)  
- **Hierarchy Level**:  
  - Type: [Department | Class | Sub Class] 
  - Value: Enter the value for the selected hierarchy type (I'll validate using our product database)  
- **Brand**: Enter the product brand (e.g., FashionX, H&M, Zara, Uniqlo)  
- **Items**:  
   - Comma-separated SKUs/Item IDs (e.g., ITEM001, ITEM003) **OR** a natural language query  
   - Exclusions: SKUs/Item IDs or styles to exclude (Optional Detail)  
- **Discount**:  
   - Type: [% Off | Fixed Price | Buy One Get One Free]  
   - Value: Numerical amount (I'll convert colloquial terms, e.g., "50 bucks off" → "$50 Off")  
- **Dates**:  
   - Start: (dd/mm/yyyy)  
   - End: (dd/mm/yyyy)  
- **Stores**:  
   - Locations: Comma-separated store IDs **OR** regions (e.g., "All Northeast")  
   - Exclusions: Specific stores to exclude (Optional Detail)  

*Supported Input Formats*:  
- **All-in-One**: "Summer Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, exclude out-of-stock items"  
- **Step-by-Step**:  
  "Promotion Type: Buy 1 Get 1 Free"  
  "Hierarchy: Department=Shirts, Brand=H&M"  
  "Discount: 40%"  
- **Mixed Formats**:  
  "Start: August 1st, End: August 7th"  

### *My Capabilities*
  1️⃣ Smart Validation & Item Lookup
      Product & Style Validation:

      Cross-check product categories and style numbers using the itemMaster table.
      Automatically retrieve item details from our database for verification.
      Example Item Details Lookup:
      Men's Cotton T-Shirt

      Item ID: ITEM001
      Description: Men's Cotton T-Shirt – Round Neck, Short Sleeves
      Department: T-Shirts | Class: Casuals | Subclass: Half Sleeve
      Brand: FashionX
      Variations:
      diffType1: 1 → Color: Yellow
      diffType2: 2 → Size: S/M (Fetched from itemDiffs table)
      Supplier Info: Retrieved from itemSupplier table
  2️⃣ Discount Type & Value Extraction
      Extract discount type and discount value from the query:
        "30% off" → Discount Type: "Percentage Off", Discount Value: "30"
        "10 USD off" → Discount Type: "Fixed Price", Discount Value: "10"
        "Buy One Get One Free" → Discount Type: "Buy One Get One Free", Discount Value: "0"
  3️⃣ Handling "All Items" Selection for a Department
      If the user specifies "all items" in a department, automatically retrieve all itemIds belonging to that department from the itemMaster table.
      Process Flow:
        Step 1: Identify the specified department (validated against itemMaster).
        Step 2: Query itemMaster to fetch itemIds where itemDepartment matches the provided department.
        Step 3: Populate the itemList field with the retrieved item IDs.
      Example Mapping:
        User Query: "All items from department: T-Shirt"
        Action Taken: Query itemMaster for itemIds where itemDepartment = 'T-Shirts'
        Result: Fill itemList with retrieved itemIds.
          
- **Detail Tracking & Standardization**:  
  - I will *Keep track of all entered details* and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
  - *Standardize formats*, such as:
    - Converting relative dates like "two weeks from now" or "3 days from now" into "dd/mm/yyyy".
    - Converting different date formats to  "dd/mm/yyyy" like "16 November 2025" to "16/11/2025" 
  - Prompt for missing information if required.  
  - Summarize all details before final submission.
  - Do not allow final submission until all details are filled.
  # - Ensure that the Items field only has Item IDs and nothing else (Eg:Correct detail: 'ITEM001' Wrong Detail: 'All red items').
  - If any validation from the database fails, return the fields that were successfully validated along with a message indicating that no records were found, specifying the fields that failed validation.

- **Product-Specific Handling**:  
  - Process natural language queries (e.g., "Men's Jackets") by matching them against our product database.  
  - Merge duplicate style entries (e.g., multiple occurrences of the same SKU).

- **Supplier Information**:  
  - Retrieve supplier details—such as Supplier ID and Supplier Cost—from our itemsupplier table for each product.

---  

## *Example Scenarios*  

### *Scenario 1: Full Details Input in Natural Language*  
*User:* "Simple Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, for store 3 and 4"  
*Response:* Validate T-Shirt SKUs using our itemmaster (e.g., ITEM001) and cross-check supplier information.

### *Scenario 2: Step-by-Step Entry*  
*User:*  
- "Promotion Type: Buy 1 Get 1 Free"  
- "Department: Shirts"  
- "Brand: H&M"  
- "Dates: Start and End of May"  
*Response:* Identify relevant items (e.g., ITEM002) and standardize the date input.

### *Scenario 3: Natural Language Query*  
*User:* "Items=query: Men's Jackets"  
*Response:* Return matching product details such as ITEM003 along with its description, variations, and supplier info.

### *Scenario 4: Supplier Check*  
*User:* "Promote style ITEM004"  
*Response:* Display details for ITEM004 (Men's Trousers – Regular Fit) and retrieve the corresponding supplier details from our itemsupplier table.

### *Scenario 5: Duplicate Merge*  
*User:* "SKUs: ITEM001, ITEM002, ITEM001"  
*Response:* Merge duplicate entries so that ITEM001 appears only once.

### *Scenario 6: Ambiguous Input*  
*User:* "Discount: 50 bucks off"  
*Response:* Convert to a standardized format → "$50 Off".

### *Scenario 7: Category Validation*  
*User:* "Subclass: Half Sleve"  
*Response:* "Did you mean 'Half Sleeve'? Available subclasses: Half Sleeve, Full Sleeve, Zipper, Regular Fit."

### *Scenario 8: Price Formatting*  
*User:* "Fixed price $ninety nine"  
*Response:* Convert to "$99.00".

### *Scenario 9: Full Details Input with field information (comma-separated)*  
*User:* "Simple, Department, T-Shirts, FashionX, ITEM001, ITEM002, % Off, 30, 13/02/2025, 31/05/2025, Store 2"  
*Response:* Validate inputs, ensure correct formats, and provide a structured summary.Example of bot's summary if the details are valid:
         Promotion Type: Simple,
         Hierarchy Type: Department,
         Hierarchy Value: T-Shirts,
         Brand: FashionX,
         Items: ITEM001, ITEM002,
         Discount Type: % Off, 
         Discount Value: 30,
         Start Date: 13/02/2025,
         End Date: 31/05/2025,
         Stores: Store 2"  
  
### *Scenario 10: Full Details Input with field information with field names*  
*User:* "
 Promotion Type: Simple,
 Hierarchy Type:Sub Class,
 Hierarchy Value: Full Sleeve,
 Brand: H&M,
 Items: ITEM001, ITEM002,
 Discount Type: % Off,
 Discount Value: 10,
 Start Date: 13/02/2025,
 End Date: 31/05/2025,
 Stores: Store 2"  
*Response:* Validate inputs, ensure correct formats, and provide a structured summary.  

### *Scenario 11: Changing details* 
*User:* "Change items to ITEM005 and ITEM006",
*Response:* Replace the items and validate this new data. Provide a validation error in case of validation failure.  

### *Scenario 12: Adding Items* 
*User:* "Add the items ITEM005 and ITEM006",
*Response:* Append the given items to the item list and validate this new data. Provide a validation error in case of validation failure.  
---  

*Current Promotion Details*:  
{chat_history}  

*Missing Fields*:  
{missing_fields}  

The above details and validations—including updated item descriptions, supplier data, and available variations—are auto-checked against our databases to ensure accuracy and consistency.

Would you like to submit this information?  
If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."*  
"""
today = datetime.datetime.today().strftime("%d/%m/%Y")
template_Promotion=template_Promotion_without_date.replace("{current_date}", today)
DEFAULT_PROMO_STRUCTURE = {
  
    # "promotionType": "",
    # "hierarchy": {"level": "", "value": ""},
    # "items": [],
    # "excluded_items": [],
    # "discount": {"type": "", "amount": 0},
    # "dates": {"start": "", "end": ""},
    # "locations": [],
    # "excluded_locations": [],
    # "status": "draft",
  "Promotion Type": "",
  "Hierarchy Type": "" ,
  "Hierarchy Value": "",
  "Brand": "" ,
  "Items": [] ,
  "Excluded Item List":[]  ,
  "Discount Type": "",
  "Discount Value":""  ,
  "Start Date": "" ,
  "End Date": "",
  "Stores":  [],
  "Excluded Location List":[] 

}
previous_promo_details = defaultdict(dict)


logging.basicConfig(level=logging.INFO)


# Define the function schema as a constant to avoid re-creation on every call.
FUNCTION_SCHEMA = {
  "name": "extract_promotion_details",
  "description": "Extracts promotion details from the provided promotion text.",
  "parameters": {
    "type": "object",
    "properties": {
      "Promotion Type": {
        "type": "string",
        "description": "One of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]"
      },
      "Hierarchy Type": {
        "type": "string",
        "description": "One of [Department | Class | Sub Class]"
      },
      "Hierarchy Value": {
        "type": "string",
        "description": "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirts' for Department."
      },
      "Brand": {
        "type": "string",
        "description": "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)"
      },
      "Items": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']"
      },
      "Excluded Item List": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']"
      },
      "Discount Type": {
        "type": "string",
        "description": "One of [% Off | Fixed Price | Buy One Get One Free]"
      },
      "Discount Value": {
        "type": "string",
        "description": "Numerical amount (converted from colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type"
      },
      "Start Date": {
        "type": "string",
        "description": "Promotion start date (dd/mm/yyyy)"
      },
      "End Date": {
        "type": "string",
        "description": "Promotion end date (dd/mm/yyyy)"
      },
      "Stores": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Comma-separated Store IDs (e.g., STORE001)"
      },
      "Excluded Location List": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Comma-separated Store IDs to be excluded"
      }
    },
    "required": [
      "Promotion Type",
      "Hierarchy Type",
      "Hierarchy Value",
      "Brand",
      "Items",
      "Discount Type",
      "Discount Value",
      "Start Date",
      "End Date",
      "Stores"
    ]
  }
}

# Define a constant system prompt.
SYSTEM_PROMPT = (
    "Extract the promotion details from the following query. "
    "Think through the extraction step by step and then provide the answer in JSON format. "
    "Make sure the JSON keys match exactly as specified."
)

def retrieve_external_context(query: str) -> str:
    """
    Simulate retrieval of external context such as current promotions or product catalog data.
    In a real-world scenario, this might involve a database lookup or search query.
    """
    # For demonstration, return a dummy external context string.
    return "External context: current promotions, product catalog, and brand data."

async def categorize_promo_details_fun_call(query: str, user_id: str) -> dict:
    """
    Uses GPT-4 with function calling and chain-of-thought instructions to extract structured promotion details.
    If example indicators are detected in the query, or if the extracted details are essentially empty,
    the function returns previously stored data for the given user.
    """
    client = openai.AsyncOpenAI()
    
    # Combine the system prompt with external context.
    external_context = retrieve_external_context(query)
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n{external_context}"},
        {"role": "user", "content": query}
    ]
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            functions=[FUNCTION_SCHEMA],
            function_call={"name": "extract_promotion_details"},
            temperature=0.7,
            max_tokens=500
        )
        
        # Access the function call details using a safe dictionary lookup.
        function_response = response.choices[0].message.function_call
        if not function_response:
            logging.error("No function_call found in the response.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        # Retrieve and parse the arguments from the function call.
        arguments = function_response.arguments if function_response and hasattr(function_response, 'arguments') else "{}"
        try:
            extracted_details = json.loads(arguments)
        except json.JSONDecodeError as json_err:
            logging.error(f"JSON decode error: {json_err}")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        # Optional: Map or validate the extracted details against DEFAULT_PROMO_STRUCTURE if needed.
        # return extracted_details
        primary_fields = ["Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", "Discount Value", 
                          "Start Date", "End Date", "Promotion Type", "Excluded Item List", "Items", 
                          "Excluded Location List", "Stores"
        ]
        if all(extracted_details.get(field) in [None, "", []] for field in primary_fields):          
          logging.info("Extracted details are essentially empty; returning previously stored data.")
          return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)
        
        # Update the previous promotion details for this user.
        previous_promo_details[user_id] = extracted_details
        return extracted_details

    except Exception as e:
        logging.error(f"Error during extraction: {e}")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE)

async def categorize_promo_details(extracted_text: str, user_id: str):
    client = openai.AsyncOpenAI()
    
    example_indicators = [
        "For example", "Example:", "e.g.", "like this", "such as"
    ]
    
    if any(keyword in extracted_text for keyword in example_indicators):
        print("Detected example instructions; skipping extraction.")
        return previous_promo_details.get(user_id, {})
    
    prompt = f"""
Extract and structure the following details from this Promotion text. The JSON keys **must match exactly** as given below:

  - **Promotion Type**: "one of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]",
  - **Hierarchy Type**: "one of [Department | Class | Sub Class]",
  - **Hierarchy Value**: "The specific value for the selected Hierarchy Type (validated against our product database). Example: 'T-Shirts' for Department.",
  - **Brand**: "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)",
  - **Items**: "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']",
  - **Excluded Item List**: "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']",
  - **Discount Type**: "one of [% Off | Fixed Price | Buy One Get One Free]",
  - **Discount Value**: "Numerical amount (convert colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type",
  - **Start Date**: "(dd/mm/yyyy)",
  - **End Date**: "(dd/mm/yyyy)",
  - **Stores**: "Comma-separated Store IDs (e.g., STORE001)",
  - **Excluded Location List**: "Comma-separated Store IDs"

🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
🔹 **Use spaces in field names exactly as shown.**
🔹 **If a value is missing, return null instead of omitting the key.**
🔹 **If no Item ID is found in the field 'Items' or 'Excluded Item List', return an empty array [].**

**Promotion Text:**
{extracted_text}

**Format the response as a valid JSON object. The field names must match exactly.**
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Extract structured promotion data from the given text, recognizing various formats for key fields."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}  
        )
        raw_response = response.choices[0].message.content.strip()
        print("Raw response: ", raw_response)
        
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        
        structured_data = json.loads(raw_response)
        
        if any(keyword in extracted_text for keyword in example_indicators):
            return previous_promo_details.get(user_id, {})
        
        primary_fields = ["Hierarchy Type", "Hierarchy Value", "Brand", "Discount Type", "Discount Value", "Start Date", "End Date", "Promotion Type", "Excluded Item List", "Items", "Excluded Location List", "Stores"]
        if all(structured_data.get(field) in [None, "", []] for field in primary_fields):
            return previous_promo_details.get(user_id, {})
        
        previous_promo_details[user_id] = structured_data
        return structured_data
    
    except Exception as e:
        print(f"Error fetching promotion details: {e}")
        return previous_promo_details.get(user_id, {})

# template_Promotion = """  
# Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your promotion operations and provide seamless support.

# *Required Promotion Details*:  
# - **Promotion Type**: (Simple, Buy X/Get Y, Threshold, Gift With Purchase)  
# - **Product Selection**:  
#   - **Hierarchy Level**:  
#    - Type: [Department | Class | Subclass] 
#    - Value: Enter the value for the selected hierarchy type (I'll validate using our product database)  
#   - **Brand**: Enter the product brand (e.g., FashionX, H&M, Zara, Uniqlo)  
# - **Items**:  
#    - Comma-separated SKUs/Item IDs (e.g., ITEM001, ITEM003) **OR** a natural language query  
#    - Exclusions: SKUs/Item IDs or styles to exclude (Optional Detail)  
# - **Discount**:  
#    - Type: [% Off | Fixed Price | BOGO]  
#    - Value: Numerical amount (I'll convert colloquial terms, e.g., "50 bucks off" → "$50 Off")  
# - **Dates**:  
#    - Start: (dd/mm/yyyy)  
#    - End: (dd/mm/yyyy)  
# - **Stores**:  
#    - Locations: Comma-separated store IDs **OR** regions (e.g., "All Northeast")  
#    - Exclusions: Specific stores to exclude (Optional Detail)  

# *Supported Input Formats*:  
# - **All-in-One**: "Summer Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, exclude out-of-stock items"  
# - **Step-by-Step**:  
#   "Promotion Type: Buy 1 Get 1 Free"  
#   "Hierarchy: Department=Shirts, Brand=H&M"  
#   "Discount: 40%"  
# - **Mixed Formats**:  
#   "Start: August 1st, End: August 7th"  

# ### *My Capabilities*
#   1️⃣ Smart Validation & Item Lookup
#       Product & Style Validation:

#       Cross-check product categories and style numbers using the itemMaster table.
#       Automatically retrieve item details from our database for verification.
#       Example Item Details Lookup:
#       Men's Cotton T-Shirt

#       Item ID: ITEM001
#       Description: Men's Cotton T-Shirt – Round Neck, Short Sleeves
#       Department: T-Shirts | Class: Casuals | Subclass: Half Sleeve
#       Brand: FashionX
#       Variations:
#       diffType1: 1 → Color: Yellow
#       diffType2: 2 → Size: S/M (Fetched from itemDiffs table)
#       Supplier Info: Retrieved from itemSupplier table
#       Men's Cotton Shirt

#       Item ID: ITEM002
#       Description: Men's Cotton Shirt – Full Sleeves
#       Department: Shirts | Class: Casuals | Subclass: Full Sleeve
#       Brand: H&M
#       Variations:
#       diffType1: 1 → Color: Blue
#       diffType2: 6 → Size: L
#       Supplier Info: Retrieved from itemSupplier table
#   2️⃣ Discount Type & Value Extraction
#       Extract discount type and discount value from the query:
#         "30% off" → Discount Type: "Percentage Off", Discount Value: "30"
#         "10 USD off" → Discount Type: "Fixed Price", Discount Value: "10"
#         "Buy One Get One Free" → Discount Type: "Buy One Get One Free", Discount Value: "0"
#   3️⃣ Handling "All Items" Selection for a Department
#       If the user specifies "all items" in a department, automatically retrieve all itemIds belonging to that department from the itemMaster table.
#       Process Flow:
#         Step 1: Identify the specified department (validated against itemMaster).
#         Step 2: Query itemMaster to fetch itemIds where itemDepartment matches the provided department.
#         Step 3: Populate the itemList field with the retrieved item IDs.
#       Example Mapping:
#         User Query: "All items from department: T-Shirt"
#         Action Taken: Query itemMaster for itemIds where itemDepartment = 'T-Shirts'
#         Result: Fill itemList with retrieved itemIds.  
# # - **Smart Validation**:  
# #   - Verify product categories and style numbers using our itemmaster table.  
# #   - **Item Details Lookup**: Automatically cross-reference item information from our databases. For example:
# #       - *Men's Cotton T-Shirt*:  
# #          - **Item ID**: ITEM001  
# #          - **Description**: Men's Cotton T-Shirt – Round Neck, Short Sleeves  
# #          - **Department**: T-Shirts | **Class**: Casuals | **Subclass**: Half Sleeve  
# #          - **Brand**: FashionX  
# #          - **Variations**: diffType1: 1, diffType2: 2, diffType3: 3 (e.g., corresponding to color "yellow" and sizes "S/M" from our itemdiffs table)  
# #          - **Supplier Info**: Retrieved from our itemsupplier table  
# #       - *Men's Cotton Shirt*:  
# #          - **Item ID**: ITEM002  
# #          - **Description**: Men's Cotton Shirt – Full Sleeves  
# #          - **Department**: Shirts | **Class**: Casuals | **Subclass**: Full Sleeve  
# #          - **Brand**: H&M  
# #          - **Variations**: diffType1: 1, diffType2: 6, diffType3: 5  
# #          - **Supplier Info**: Retrieved from our itemsupplier table   
# #   - Validate product details against our itemmaster and itemdiffs tables.
# #   -Fetch Discount Type and Value from the query. Examples:
# #     1) 30% off: Discount Type="Percentage Off" Discount Value="30"
# #     2) 10 USD off: Discount Type="Fixed Price" Discount Value="10"
# #     3) Buy One Get One Free: Discount Type="Buy One Get One Free" Discount Value="0"
# #   -If the user specifies "all items" in a department, map it to retrieving all `itemId`s from `itemMaster`
# #    where `itemDepartment` matches the validated department and return the list inside itemList detail.
# #     - Example: "all items from department: T Shirt" → Retrieve `itemId`s from `itemMaster` where `itemDepartment` is equal to 'T-Shirts' and fill itemList with the retrieved itemIds`.

# - **Detail Tracking & Standardization**:  
#   - Track all entered details and prompt to fill in any missing ones in a structured format (e.g., [Detail Name]: [Provided Value]).  
#   - Standardize date formats (e.g., "August 1st, 2025" → "01/08/2025").  
#   - Prompt for missing information if required.  
#   - Summarize all details before final submission.
#   - Do not allow final submission until all details are filled.

# - **Product-Specific Handling**:  
#   - Process natural language queries (e.g., "Men's Jackets") by matching them against our product database.  
#   - Merge duplicate style entries (e.g., multiple occurrences of the same SKU).

# - **Supplier Information**:  
#   - Retrieve supplier details—such as Supplier ID and Supplier Cost—from our itemsupplier table for each product.

# ---  

# ## *Example Scenarios*  

# ### *Scenario 1: Full Details Input*  
# *User:* "Summer Promo: 20% off all T-Shirts from FashionX, 01/07-31/07, exclude out-of-stock items"  
# *Response:* Validate T-Shirt SKUs using our itemmaster (e.g., ITEM001) and cross-check supplier information.

# ### *Scenario 2: Step-by-Step Entry*  
# *User:*  
# - "Promotion Type: Buy 1 Get 1 Free"  
# - "Department: Shirts"  
# - "Brand: H&M"  
# - "Dates: Back-to-School Week"  
# *Response:* Identify relevant items (e.g., ITEM002) and standardize the date input.

# ### *Scenario 3: Natural Language Query*  
# *User:* "Items=query: Men's Jackets"  
# *Response:* Return matching product details such as ITEM003 along with its description, variations, and supplier info.

# ### *Scenario 4: Supplier Check*  
# *User:* "Promote style ITEM004"  
# *Response:* Display details for ITEM004 (Men's Trousers – Regular Fit) and retrieve the corresponding supplier details from our itemsupplier table.

# ### *Scenario 5: Duplicate Merge*  
# *User:* "SKUs: ITEM001, ITEM002, ITEM001"  
# *Response:* Merge duplicate entries so that ITEM001 appears only once.

# ### *Scenario 6: Ambiguous Input*  
# *User:* "Discount: 50 bucks off"  
# *Response:* Convert to a standardized format → "$50 Off".

# ### *Scenario 7: Category Validation*  
# *User:* "Subclass: Half Sleve"  
# *Response:* "Did you mean 'Half Sleeve'? Available subclasses: Half Sleeve, Full Sleeve, Zipper, Regular Fit."

# ### *Scenario 8: Price Formatting*  
# *User:* "Fixed price $ninety nine"  
# *Response:* Convert to "$99.00".

# ---  

# *Current Promotion Details*:  
# {chat_history}  

# *Missing Fields*:  
# {missing_fields}  

# The above details and validations—including updated item descriptions, supplier data, and available variations—are auto-checked against our databases to ensure accuracy and consistency.

# Would you like to submit this information?  
# If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."*  
# """

# async def categorize_promo_details(extracted_text: str, user_id: str):
#     client = openai.AsyncOpenAI()
    
#     # Define keywords that indicate examples rather than actual user-provided data
#     example_indicators = [
#         "For example", "Example:", "e.g.", "like this", "such as"
#     ]
    
#     # Check if the extracted text contains example indicators
#     if any(keyword in extracted_text for keyword in example_indicators):
#         print("Detected example instructions; skipping extraction.")
#         return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
    
#     # Prompt to enforce exact JSON structure
#     prompt = f"""
# Extract and structure the following promotion details from the provided text. The JSON keys **must match exactly** as given below:

# - **Type** → (e.g., "Percentage Off", "Buy One Get One", "Flat Discount", etc.)
# - **Hierarchy** → (structure with the following keys):  
#   - **Level** → (e.g., "Category", "Brand", "Product")  
#   - **Value** → (e.g., "Shirts", "Nike", "Product ABC")
# - **Items** → (list of item identifiers, each representing a product eligible for the promotion)
# - **Excluded Items** → (list of item identifiers, each representing a product excluded from the promotion)
# - **Discount** → (structure with the following keys):  
#   - **Type** → (e.g., "Flat", "Percentage", "BOGO")
#   - **Amount** → (numeric value representing the discount amount or percentage)
# - **Dates** → (structure with the following keys):  
#   - **Start** → (formatted as dd/mm/yyyy, representing the promotion start date)
#   - **End** → (formatted as dd/mm/yyyy, representing the promotion end date)
# - **Locations** → (list of locations where the promotion is applicable, e.g., stores or regions)
# - **Excluded Locations** → (list of locations where the promotion is not applicable)
# - **Status** → (e.g., "Active", "Draft", "Completed")

# 🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
# 🔹 **Use spaces in field names exactly as shown.**
# 🔹 **If a value is missing, return null instead of omitting the key.**

# **Promotion Text:**
# {extracted_text}

# **Format the response as a valid JSON object. The field names must match exactly.**
# """
#     # prompt = f"""
#     # Extract and structure promotion details from the following text into this exact JSON format:
#     # {json.dumps(DEFAULT_PROMO_STRUCTURE, indent=2)}

#     # Follow these rules:
#     # 1. Use EXACT field names shown (including underscores)
#     # 2. Maintain nested structure
#     # 3. For dates use dd/mm/yyyy format
#     # 4. For missing values use: 
#     #    - "" for strings
#     #    - [] for lists
#     #    - {{}} for objects with default fields
#     # 5. Convert all item/location lists to arrays of strings

#     # Text to analyze:
#     # {extracted_text}

#     # Return ONLY valid JSON matching the structure above.
#     # """
    
#     try:
#         response = await client.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "Extract promotion details into exact JSON structure."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"}
#         )
#         raw_response = response.choices[0].message.content.strip()
        
#         # Clean response if wrapped in code blocks
#         if raw_response.startswith("```json"):
#             raw_response = raw_response[7:-3].strip()
        
#         parsed_data = json.loads(raw_response)
        
#         # Merge with default structure
#         structured_data = DEFAULT_PROMO_STRUCTURE.copy()
        
#         # Deep merge for nested structures
#         def merge_nested(target, source):
#             for key, value in source.items():
#                 if isinstance(value, dict):
#                     node = target.setdefault(key, {})
#                     merge_nested(node, value)
#                 elif isinstance(value, list):
#                     target[key] = value.copy()
#                 else:
#                     target[key] = value

#         merge_nested(structured_data, parsed_data)

#         # Validate extracted lists aren't examples
#         list_fields = ["items", "excluded_items", "locations", "excluded_locations"]
#         for field in list_fields:
#             structured_data[field] = [
#                 item for item in structured_data[field]
#                 if not any(keyword in str(item) for keyword in example_indicators)
#             ]

#         # Check if essential fields are empty
#         essential_fields_empty = (
#             not structured_data["type"] and
#             not structured_data["hierarchy"]["level"] and
#             not structured_data["hierarchy"]["value"] and
#             not structured_data["discount"]["type"] and
#             structured_data["discount"]["amount"] == 0 and
#             not any(structured_data["items"]) and
#             not any(structured_data["locations"])
#         )

#         if essential_fields_empty:
#             return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())

#         # Store and return new data
#         previous_promo_details[user_id] = structured_data
#         return structured_data

#     except Exception as e:
#         print(f"Error extracting promo details: {e}")
#         return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
# async def categorize_promo_details(extracted_text: str, user_id: str):
#     client = openai.AsyncOpenAI()
    
#     # Enhanced example detection
#     example_indicators = [
#         "For example", "Example:", "e.g.", "like this", "such as", "specifically:"
#     ]
    
#     if any(keyword.lower() in extracted_text.lower() for keyword in example_indicators):
#         print("Detected example instructions; skipping extraction.")
#         return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
    
#     # Improved prompt with explicit field mappings
#     prompt = f"""
#     Extract promotion details from this text and structure into JSON matching exactly these fields:
#     {json.dumps(DEFAULT_PROMO_STRUCTURE, indent=2)}

#     Follow these mappings from text to JSON:
#     - "Promotion Type" → "type"
#     - "Hierarchy Level" → "hierarchy.level"
#     - "Category"/"Brand"/etc → "hierarchy.value"
#     - "Items" → "items" (extract only Item IDs as strings)
#     - "Discount" → Parse percentage/amount into "discount.type" and "discount.amount"
#     - Date range → Split into "dates.start" and "dates.end"
#     - "Stores"/"Locations" → "locations" (store names as strings)

#     Text: {extracted_text}

#     Rules:
#     1. Convert all percentages to numeric values (e.g., "30%" → 30)
#     2. For dates use dd/mm/yyyy format
#     3. For item lists, extract only the Item IDs
#     4. Maintain exact JSON structure including empty fields
#     """

#     try:
#         response = await client.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "Convert promotion details to exact JSON format. Extract specific IDs and values."},
#                 {"role": "user", "content": prompt},
#             ],
#             response_format={"type": "json_object"}
#         )
        
#         raw_response = response.choices[0].message.content.strip()
#         if raw_response.startswith("```json"):
#             raw_response = raw_response[7:-3].strip()
        
#         parsed_data = json.loads(raw_response)

#         # Post-processing for specific fields
#         # Convert percentage discount types
#         if "percentage" in parsed_data.get("discount", {}).get("type", "").lower():
#             parsed_data["discount"]["type"] = "Percentage"
        
#         # Extract only Item IDs from complex item descriptions
#         if "items" in parsed_data:
#             parsed_data["items"] = [
#                 item.split(":")[0].strip() if ":" in item else item
#                 for item in parsed_data["items"]
#             ]

#         # Merge with default structure
#         structured_data = DEFAULT_PROMO_STRUCTURE.copy()
        
#         def deep_merge(target, source):
#             for key, value in source.items():
#                 if isinstance(value, dict):
#                     node = target.setdefault(key, {})
#                     deep_merge(node, value)
#                 else:
#                     if key in target:
#                         if isinstance(target[key], list) and isinstance(value, list):
#                             target[key].extend(value)
#                         else:
#                             target[key] = value
#                     else:
#                         target[key] = value
        
#         deep_merge(structured_data, parsed_data)

#         # Store and return
#         previous_promo_details[user_id] = structured_data
#         return structured_data

#     except Exception as e:
#         print(f"Error in promo extraction: {e}")
#         return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())