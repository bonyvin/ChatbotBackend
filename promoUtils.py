
import copy
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
  - Value: Enter the value for the selected hierarchy type (You can enter multiple values separated by commas, for example: "T-Shirt, Shirt")  
  - **Mixed Hierarchy Predicate Example**: If you wish to combine different hierarchy conditions (for example, Class=Casuals and Department=T-Shirt), please specify each field accordingly.  
- **Brand**: Enter the product brand (e.g., FashionX, H&M, Zara, Uniqlo). You may specify multiple brands separated by commas (e.g., "FashionX, Zara").  
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
   - **Note:** If the query contains phrases such as "all stores" or similar, automatically record the field as **"All Stores"**. This value will then be processed further to return the actual store IDs.  

*Supported Input Formats*:  
- **All-in-One**: "Summer Promo: 20% off all T-Shirt from FashionX, 01/07-31/07, exclude out-of-stock items"  
- **Step-by-Step**:  
  "Promotion Type: Buy 1 Get 1 Free"  
  "Hierarchy: Department=Shirt, Brand=H&M"  
  "Discount: 40%"  
- **Mixed Formats**:  
  "Start: August 1st, End: August 7th"
- If the query contains store-related terms (e.g., "store", "location", "all stores"), call the function 'entity_extraction_and_validation'.
- If the query mentions product-related details (e.g., "SKU", "item", "T-Shirt", "red"), or both product and store details, call the function 'query_database'.  

### *My Capabilities*
1. **Item Lookup & Smart Validation**  
   - **Automatic Trigger Conditions:**  
       - Activated when detecting any item-related input, such as:
         - Specific Item IDs (e.g., ITEM001, ITEM002)
         - Product descriptors like "size", "color", "description", etc.
         - Phrases like “all items” within a department context  
   - **Validation Process:**  
       1. Call `query_database` for any item-related input.
       2. Cross-check the returned details against the itemMaster table and related tables (e.g., itemdiffs for differentiators, itemsupplier for supplier info).
       3. Handle three scenarios:  
            ✅ **Valid Items**: Display verified item details including Item IDs.  
            ❌ **Invalid Items**: Flag errors with suggestions on how to correct the input.  
            ❓ **Ambiguous Inputs**: Request clarification if the provided details can’t uniquely identify an item.  
   - **Automatic Validation Checks:**  
       - After any item input, always:  
         1. Display the extracted item details (ID, description, category info, etc.).  
         2. Show the validation status (✅/❌).  
         3. Offer alternatives or request clarification for ambiguous or invalid entries.
       - Block promotion submission until item validation passes.
   
2. **Discount Type & Value Extraction**  
   - Extract discount information from the query:  
         "30% off" → Discount Type: "Percentage Off", Discount Value: "30"  
         "10 USD off" → Discount Type: "Fixed Price", Discount Value: "10"  
         "Buy One Get One Free" → Discount Type: "Buy One Get One Free", Discount Value: "0"
      
3. **Handling "All Items" Selection for a Department**  
   - If the user specifies "all items" in a department, automatically retrieve all itemIds belonging to that department from the itemMaster table.
   - **Process Flow:**  
        1. Call `query_database` to identify the specified department (validated against itemMaster).  
        2. Query itemMaster to fetch itemIds where itemDepartment matches the provided department.  
        3. Populate the itemList field with the retrieved item IDs.
        
4. **Store Location Processing**  
   - **Automatic Trigger Conditions:**  
        - Activated when detecting any store-related input, including:  
          - Specific Store IDs (e.g., STORE001, STORE002)  
          - Location terms (city, state, region)  
          - Phrases like "all stores", "these locations", "exclude [area]"  
   - **Validation Process:**  
        1. Call `entity_extraction_and_validation` for any store-related input.  
        2. Cross-check extracted stores against the storedetails table.  
        3. Handle three scenarios:  
            ✅ **Valid Stores**: Display verified store IDs.  
            ❌ **Invalid Stores**: Flag errors with suggestions.  
            ❓ **Ambiguous Locations**: Request clarification (e.g., "Did you mean New York City or New York State?").  
        4. After validating stores, always replace phrases like "All Stores" with the actual validated store IDs in the summary.  
   - **Automatic Validation Checks:**  
        - After any store input, always:  
            1. Display extracted store IDs along with their validation status (✅/❌).  
            2. Provide alternatives for invalid entries.  
        - Block promotion submission until store validation passes.

        
5. **Date Validation**  
   - Ensure that the start date is equal to or greater than {current_date}.
          
- **Detail Tracking & Standardization**:  
  - I will *Keep track of all entered details* and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
#    - **Important**: Whenever I record a valid detail from any field, I will immediately display the recorded detail in a summary along with my response and the previously recorded and missing fields. For example, if the user provides "Promotion Type: Simple," my reply will include "Promotion Type: Simple" in the summary section along with all the previously recorded and missing fields.  
    - **Important**:  
     1. **Immediately Display Recorded Details**: Whenever the user provides a valid input, record and **immediately display** that information in the response. This should include:
        - The field just filled by the user (e.g., "Promotion Type: Simple").
        - All previously recorded details.
     2. **Show Missing Fields**: Always include a list of **missing fields** (details that the user has not yet provided). This allows the user to know what is still required.
        - Missing fields should be shown clearly with labels like: "Hierarchy Level (Type and Value for Department, Class, or Sub Class)," "Brand," "Items," etc.
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
*User:* "Simple Promo: 20% off all T-Shirt from FashionX, 01/07-31/07, for store 3 and 4"  
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
*Response:* Display details for ITEM004 (Men's Trousers - Regular Fit) and retrieve the corresponding supplier details from our itemsupplier table.

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
*User:* "Simple, Department, T-Shirt, FashionX, ITEM001, ITEM002, % Off, 30, 13/02/2025, 31/05/2025, Store 2"  
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

### *Scenario 13: Exclusions* 
*User:* "Excluded Stores are STORE002 and STORE003 and Excluded Items are ITEM003,ITEM004",
*Response:*
Recorded Details:
Excluded Stores:STORE002 , STORE003 
Excluded Items: ITEM003,ITEM004
*Validate inputs, ensure correct formats, and provide a structured summary*

### *Scenario 14: All items* 
*User:* "Create a promotion on all items"
*Action:* Trigger the `query_database` function call and return the received store ids
*Response:*
Recorded Details:
Items: ITEM003,ITEM004
...other fields.
*Validate inputs, ensure correct formats, and provide a structured summary*

### *Scenario 15: All stores* 
*User:* "Create a promotion across all stores"
*Action:* Trigger the 'extract_promo_entities' function call and return the received store ids
*Response:*
Recorded Details:
Stores:STORE002 , STORE003 
Items: ITEM003,ITEM004
...other fields.
*Validate inputs, ensure correct formats, and provide a structured summary*

### *Scenario 16: Add both items and stores* 
*User:* "Create simple promotion for all items across all stores."
*Action:* Trigger the `query_database` function call which in turn calls the 'extract_promo_entities' function and return the received item ids and store ids
*Response:*
Recorded Details:
Stores:STORE002 , STORE003 
Items: ITEM003,ITEM004
...other fields.
*Validate inputs, ensure correct formats, and provide a structured summary*

### *Scenario 17: Multiple Predicates for a Single Field* 
- **Multiple Brands:**  
    User: "Select all items from FashionX and Zara brands"
    *Response:*  
    Recorded Details:
    Brand: FashionX, Zara
    Items: (Populated from SQL query result)
    Final SQL Query:  
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.brand = 'FashionX' OR im.brand = 'Zara'
    ```
- **Multiple Departments:**  
    User: "Select all items from T-Shirt and Shirt departments"  
    *Response:*  
    Recorded Details:
    Hierarchy Type: Department
    Hierarchy Value: T-Shirt, Shirt
    Items: (Populated from SQL query result)
    Final SQL Query: 
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.itemDepartment LIKE 'T-Shirt%' OR im.itemDepartment LIKE 'Shirt%'
    ```
- **Multiple Sub Classes:**  
    User: "Select all items from Half and Full Sleeve Sub Classes"  
    *Response:* 
    Recorded Details:
    Hierarchy Type: Sub Class 
    Hierarchy Value: Half Sleeve, Full Sleeve
    Items: (Populated from SQL query result) 
    Final SQL Query: 
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.itemSubClass LIKE 'Half Sleeve%' OR im.itemSubClass LIKE 'Full Sleeve%'
    ```
- **Multiple Classes:**  
    User: "Select all items from Formals and Casuals Classes"  
    *Response:* 
    Recorded Details:
    Hierarchy Type: Class 
    Hierarchy Value: Formals, Casuals
    Items: (Populated from SQL query result)  
    Final SQL Query:  
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.itemClass LIKE 'Formals%' OR im.itemClass LIKE 'Casuals%'
    ```
### *Scenario 17: Mixed Hierarchy Conditions* 
    User: "Select all items from T-Shirt department and Casuals class" 
    *Response:*  
    Recorded Details:
    Hierarchy Type: Department, Class
    Hierarchy Value: T-Shirt, Casuals
    Items: (Populated from SQL query result)  
    Final SQL Query:  
    ```sql
    SELECT im.itemId
    FROM itemmaster im
    WHERE im.itemDepartment LIKE 'T-Shirt%' AND im.itemClass LIKE 'Casuals%'
    ```
---  

*Current Promotion Details*:  
{chat_history}  

*Missing Fields*:  
{missing_fields}  

Would you like to submit this information?  
If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."*
Upon receiving a 'Yes' response, inquire whether the user would like the document sent to their email and request their email address.
"""
# template_Promotion_without_date = """  
# Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your promotion operations and provide seamless support. Today is {current_date}.  

# *Required Promotion Details*:  
# - **Promotion Type**: (Simple, Buy X/Get Y, Threshold, Gift With Purchase)  
# - **Hierarchy Level**:  
#   - Type: [Department | Class | Sub Class] 
#   - Value: Enter the value for the selected hierarchy type  
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
# - **All-in-One**: "Summer Promo: 20% off all T-Shirt from FashionX, 01/07-31/07, exclude out-of-stock items"  
# - **Step-by-Step**:  
#   "Promotion Type: Buy 1 Get 1 Free"  
#   "Hierarchy: Department=Shirt, Brand=H&M"  
#   "Discount: 40%"  
# - **Mixed Formats**:  
#   "Start: August 1st, End: August 7th"  

# ### *My Capabilities*
#   1.  Item Lookup & Smart Validation 
#       Product & Style Validation:
#       Cross-check product categories and style numbers using the itemmaster table.
#       Automatically retrieve item details from our database for verification. Call `query_database` for item lookup and validation.
#       Example Item Details Lookup:
#       Men's Cotton T-Shirt
#       Item ID: ITEM001
#       Description: Men's Cotton T-Shirt – Round Neck, Short Sleeves
#       Department: T-Shirt | Class: Casuals | Subclass: Half Sleeve
#       Brand: FashionX
#       Variations:
#       diffType1: 1 → Color: Yellow
#       diffType2: 2 → Size: S/M (Fetched from itemsiffs table)
#       Supplier Info: Retrieved from itemsupplier table
   
#   2.  Discount Type & Value Extraction
#       Extract discount type and discount value from the query:
#       "30% off" → Discount Type: "Percentage Off", Discount Value: "30"
#       "10 USD off" → Discount Type: "Fixed Price", Discount Value: "10"
#       "Buy One Get One Free" → Discount Type: "Buy One Get One Free", Discount Value: "0"
      
#   3.  Handling "All Items" Selection for a Department
#       If the user specifies "all items" in a department, automatically retrieve all itemIds belonging to that department from the itemMaster table.
#       Process Flow:
#         Step 1: Call `query_database` and identify the specified department (validated against itemMaster).
#         Step 2: Query itemMaster to fetch itemIds where itemDepartment matches the provided department.
#         Step 3: Populate the itemList field with the retrieved item IDs.
        
#       Example Mapping:
#         User Query: "All items from department: T-Shirt"
#         Action Taken: Query itemMaster for itemIds where itemDepartment = 'T-Shirt'
#         Result: Fill itemList with retrieved itemIds.
# #   3.  Handling "All Items" Selection for a Department
# #       If the user specifies "all items" in a department, automatically retrieve all itemIds belonging to that department from the itemMaster table.
# #       Process Flow:
# #         Step 1: Identify the specified department (validated against itemMaster).
# #         Step 2: Query itemMaster to fetch itemIds where itemDepartment matches the provided department.
# #         Step 3: Populate the itemList field with the retrieved item IDs.
# #       Example Mapping:
# #         User Query: "All items from department: T-Shirt"
# #         Action Taken: Query itemMaster for itemIds where itemDepartment = 'T-Shirt'
# #         Result: Fill itemList with retrieved itemIds.
        
#     4. **Store Location Processing**  
#     - **Automatic Trigger Conditions**:  
#         - Immediate activation when detecting any of:  
#         - Store IDs (e.g., STORE001, STORE002)  
#         - Location terms (city, state, region)  
#         - Phrases like "all stores", "these locations", "exclude [area]"  
#     - **Validation Process**:  
#         1. Call `entity_extraction_and_validation` for ANY store-related input  
#         2. Cross-check extracted stores against storedetails table  
#         3. Handle three scenarios:  
#             ✅ **Valid Stores**: Display verified store IDs  
#             ❌ **Invalid Stores**: Flag errors with suggestions  
#             ❓ **Ambiguous Locations**: Request clarification (e.g., "Did you mean New York City or State?")  
#     - **Automatic Validation Checks**:  
#         - After ANY store input, always:  
#             1. Display extracted store IDs  
#             2. Show validation status (✅/❌)  
#             3. Provide alternatives for invalid entries  
#         - Block promotion submission until store validation passes  
#     5. Date Validation
#         Make sure that the start date is equal to or greater than {current_date}.
          
# - **Detail Tracking & Standardization**:  
#   - I will *Keep track of all entered details* and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
# #    - **Important**: Whenever I record a valid detail from any field, I will immediately display the recorded detail in a summary along with my response and the previously recorded and missing fields. For example, if the user provides "Promotion Type: Simple," my reply will include "Promotion Type: Simple" in the summary section along with all the previously recorded and missing fields.  
#     - **Important**:  
#      1. **Immediately Display Recorded Details**: Whenever the user provides a valid input, record and **immediately display** that information in the response. This should include:
#         - The field just filled by the user (e.g., "Promotion Type: Simple").
#         - All previously recorded details.
#      2. **Show Missing Fields**: Always include a list of **missing fields** (details that the user has not yet provided). This allows the user to know what is still required.
#         - Missing fields should be shown clearly with labels like: "Hierarchy Level (Type and Value for Department, Class, or Sub Class)," "Brand," "Items," etc.
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
# *User:* "Simple Promo: 20% off all T-Shirt from FashionX, 01/07-31/07, for store 3 and 4"  
# *Response:* Validate T-Shirt SKUs using our itemmaster (e.g., ITEM001) and cross-check supplier information.

# ### *Scenario 2: Step-by-Step Entry*  
# *User:*  
# - "Promotion Type: Buy 1 Get 1 Free"  
# *Response:* Validate inputs, ensure correct formats, and provide a structured summary of the recorded details.

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
# *User:* "Simple, Department, T-Shirt, FashionX, ITEM001, ITEM002, % Off, 30, 13/02/2025, 31/05/2025, Store 2"  
# *Response:* Validate inputs, ensure correct formats, and provide a structured summary.Example of bot's summary if the details are valid:
#          Promotion Type: Simple,
#          Hierarchy Type: Department,
#          Hierarchy Value: T-Shirt,
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

# ### *Scenario 12: Exclusions* 
# *User:* "Excluded Stores are STORE002 and STORE003 and Excluded Items are ITEM003,ITEM004",
# *Response:*
# Recorded Details:
# Excluded Stores:STORE002 , STORE003 
# Excluded Items: ITEM003,ITEM004
# *Validate inputs, ensure correct formats, and provide a structured summary*

# ---  

# *Current Promotion Details*:  
# {chat_history}  

# *Missing Fields*:  
# {missing_fields}  

# Would you like to submit this information?  
# If you respond with 'Yes', I'll confirm with *"Promotion created successfully. Thank you for choosing us."*  
# """

today = datetime.datetime.today().strftime("%d/%m/%Y")
template_Promotion=template_Promotion_without_date.replace("{current_date}", today)
DEFAULT_PROMO_STRUCTURE = {
  "Promotion Type": "",
  "Hierarchy Type": [] ,
  "Hierarchy Value": [],
  "Brand": [] ,
  "Items": [] ,
  "Excluded Item List":[]  ,
  "Discount Type": "",
  "Discount Value":""  ,
  "Start Date": "" ,
  "End Date": "",
  "Stores":  [],
  "Excluded Location List":[] ,
  "Email":""

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
        "description": "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirt' for Department."
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
      "Excluded Items": {
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
      "Excluded Stores": {
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
            "Start Date", "End Date", "Promotion Type", "Excluded Items", "Items", 
            "Excluded Stores", "Stores"
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
  - **Hierarchy Value**: "The specific value for the selected Hierarchy Type (validated against our product database). Example: 'T-Shirt' for Department.",
  - **Brand**: "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)",
  - **Items**: "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']",
  - **Excluded Items**: "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']",
  - **Discount Type**: "one of [% Off | Fixed Price | Buy One Get One Free]",
  - **Discount Value**: "Numerical amount (convert colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type",
  - **Start Date**: "(dd/mm/yyyy)",
  - **End Date**: "(dd/mm/yyyy)",
  - **Stores**: "Array of Store IDs formatted as ['STORE001', 'STORE002']"
  - **Excluded Stores**:"Array of Store IDs formatted as ['STORE001', 'STORE002']"

🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
🔹 **Use spaces in field names exactly as shown.**
🔹 **If a value is missing, return null instead of omitting the key.**
🔹 **If no Item ID is found in the field 'Items' or 'Excluded Items',or if  Store ID is found in the field 'Stores' or 'Excluded Stores' return an empty array [].**

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
                          "Excluded Items", "Items", "Excluded Stores", "Stores"]
        if all(not merged_data.get(field) for field in primary_fields):
            print("No valid data extracted; returning previous data.")
            return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
        
        # Update previous promo details and return
        previous_promo_details[user_id] = merged_data
        return merged_data
    
    except Exception as e:
        print(f"Error during extraction: {e}")
        return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())

# async def categorize_promo_details_new(extracted_text: str, user_id: str):
#     client = openai.AsyncOpenAI()
    
#     # Single combined prompt:
#     prompt = f"""
# Extract the following promotion details from the provided text and for each field determine if the value is merely an example instruction. For each field, return an object with two keys:
#   - "value": the extracted field value (or null if missing).
#   - "is_example": true if the extracted value appears to be just an example instruction (e.g. containing phrases like "for example", "e.g.", "such as"), false otherwise.
  
# The fields and their expected formats are:
  
#   - **Promotion Type**: one of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]
#   - **Hierarchy Type**:  An array containing one or more of the following: [Department, Class, Sub Class]
#   - **Hierarchy Value**: An array of one or more specific values for the selected Hierarchy Type (e.g., for Department, ["T-Shirt", "Shirt"])
#   - **Brand**: An array of product brands (e.g., ["FashionX", "H&M", "Zara", "Uniqlo"])
#   - **Items**: Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']
#   - **Excluded Items**: Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']
#   - **Discount Type**: one of [% Off | Fixed Price | Buy One Get One Free]
#   - **Discount Value**: Numerical amount (convert colloquial terms such as "50 bucks off" to "$50 Off")
#   - **Start Date**: (dd/mm/yyyy)
#   - **End Date**: (dd/mm/yyyy)
#   - **Stores**: Array of Store IDs formatted as ['STORE001', 'STORE002']
#   - **Excluded Stores**: Array of Store IDs formatted as ['STORE001', 'STORE002']
#   - **Email**: A valid email address where the email format follows standard conventions (e.g., user@example.com).

# Use the exact field names as given above.

# **Promotion Text:**
# {extracted_text}

# Return the response as a valid JSON object where each key is one of the fields and its value is an object in the form:
#     "Field Name": {{ "value": "<extracted_value>", "is_example": "<true/false>" }}
  
# If a field's value is missing, return null (or an empty array for fields expected to be arrays) and set "is_example" to false.
#     """
    
#     try:
#         response = await client.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "Extract promotion details with per-field example flags as described."},
#                 {"role": "user", "content": prompt}
#             ],
#             response_format={"type": "json_object"}
#         )
#         raw_response = response.choices[0].message.content.strip()
#         print("Raw combined response:", raw_response)
        
#         # Clean JSON response if wrapped in a code block
#         if raw_response.startswith("```json"):
#             raw_response = raw_response[7:-3].strip()
        
#         extracted_data = json.loads(raw_response)
#     except Exception as e:
#         print(f"Error during combined extraction: {e}")
#         return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
    
#     # Process the returned structure:
#     # For each field, if "is_example" is true, set its value to null (or [] for array types)
#     processed_data = {}
#     for field, details in extracted_data.items():
#         is_example = details.get("is_example", False)
#         value = details.get("value")
        
#         # Determine the expected type from the default structure.
#         default_value = DEFAULT_PROMO_STRUCTURE.get(field)
#         if is_example:
#             # For example fields, use the default empty value.
#             processed_data[field] = [] if isinstance(default_value, list) else None
#         else:
#             processed_data[field] = value if value is not None else ( [] if isinstance(default_value, list) else None )
    
#     # Merge processed data with previous details (if needed) ensuring all keys are present
#     merged_data = DEFAULT_PROMO_STRUCTURE.copy()
#     merged_data.update(processed_data)
    
#     # Optional: Check if all primary fields are empty. If so, fallback to previous details.
#     primary_fields = [
#         "Promotion Type", "Hierarchy Type", "Hierarchy Value", "Brand", 
#         "Items", "Excluded Items", "Discount Type", "Discount Value", 
#         "Start Date", "End Date", "Stores", "Excluded Stores","Email"
#     ]
#     if all(not merged_data.get(field) for field in primary_fields):
#         print("No valid data extracted; returning previous data.")
#         return previous_promo_details.get(user_id, DEFAULT_PROMO_STRUCTURE.copy())
    
#     previous_promo_details[user_id] = merged_data
#     return merged_data

async def categorize_promo_details_new(extracted_text: str, user_id: str):
    client = openai.AsyncOpenAI()
    
    # Single combined prompt:
    prompt = f"""
Extract the following promotion details from the provided text and for each field determine if the value is merely an example instruction. For each field, return an object with two keys:
  - "value": the extracted field value (or null if missing).
  - "is_example": true if the extracted value appears to be just an example instruction (e.g. containing phrases like "for example", "e.g.", "such as"), false otherwise.
  
The fields and their expected formats are:
  
  - **Promotion Type**: one of [Simple | Buy X/Get Y | Threshold | GWP (Gift with Purchase)]
  - **Hierarchy Type**:  An array containing one or more of the following: [Department, Class, Sub Class]
  - **Hierarchy Value**: An array of one or more specific values for the selected Hierarchy Type (e.g., for Department, ["T-Shirt", "Shirt"])
  - **Brand**: An array of product brands (e.g., ["FashionX", "H&M", "Zara", "Uniqlo"])
  - **Items**: Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']
  - **Excluded Items**: Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']
  - **Discount Type**: one of [% Off | Fixed Price | Buy One Get One Free]
  - **Discount Value**: Numerical amount (convert colloquial terms such as "50 bucks off" to "$50 Off")
  - **Start Date**: (dd/mm/yyyy)
  - **End Date**: (dd/mm/yyyy)
  - **Stores**: Array of Store IDs formatted as ['STORE001', 'STORE002']
  - **Excluded Stores**: Array of Store IDs formatted as ['STORE001', 'STORE002']
  - **Email**: A valid email address where the email format follows standard conventions (e.g., user@example.com).

Use the exact field names as given above.

**Promotion Text:**
{extracted_text}

Return the response as a valid JSON object where each key is one of the fields and its value is an object in the form:
    "Field Name": {{ "value": "<extracted_value>", "is_example": "<true/false>" }}
  
If a field's value is missing, return null (or an empty array for fields expected to be arrays) and set "is_example" to false.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Extract promotion details with per-field example flags as described."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content.strip()
        # strip any ```json ... ``` wrappers
        if raw.startswith("```"):
            raw = raw.split("```", 2)[-1].rsplit("```", 1)[0].strip()

        extracted_data = json.loads(raw)

    except Exception as e:
        print(f"[categorize_promo] extraction error: {e!r}")
        # fallback: full default or last saved
        return previous_promo_details.get(user_id, copy.deepcopy(DEFAULT_PROMO_STRUCTURE))

    # 2) If *all* fields are null/empty in the LLM output → immediately fallback
    all_empty = all(
        (details.get("is_example", False) or details.get("value") in (None, [], "")) 
        for details in extracted_data.values()
    )
    if all_empty:
        print("[categorize_promo] all fields empty → returning previous data.")
        return previous_promo_details.get(user_id, copy.deepcopy(DEFAULT_PROMO_STRUCTURE))

    # 3) Clean & map into our structure
    try:
        processed = {}
        for field, details in extracted_data.items():
            is_ex = details.get("is_example", False)
            val   = details.get("value")

            default = DEFAULT_PROMO_STRUCTURE.get(field)
            if is_ex:
                # example entries get reset
                processed[field] = [] if isinstance(default, list) else None
            else:
                # preserve explicit nulls vs empty lists
                if val is None:
                    processed[field] = [] if isinstance(default, list) else None
                else:
                    processed[field] = val

        # 4) Merge on a fresh deep-copy of defaults
        merged = copy.deepcopy(DEFAULT_PROMO_STRUCTURE)
        merged.update(processed)

        # 5) Final sanity-check: if *still* nothing, fallback
        primaries = list(DEFAULT_PROMO_STRUCTURE.keys())
        if all(not merged[f] for f in primaries):
            print("[categorize_promo] nothing valid after clean → fallback")
            return previous_promo_details.get(user_id, copy.deepcopy(DEFAULT_PROMO_STRUCTURE))

        # Save and return
        previous_promo_details[user_id] = merged
        return merged

    except Exception as e:
        print(f"[categorize_promo] processing error: {e!r}")
        return previous_promo_details.get(user_id, copy.deepcopy(DEFAULT_PROMO_STRUCTURE))


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
                "description": "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirt' for Department."
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
            "Excluded Items": {
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
            "Excluded Stores": {
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
  - **Hierarchy Value**: "The specific value for the selected Hierarchy Type (validated against our product database), e.g., 'T-Shirt' for Department.",
  - **Brand**: "Product brand (e.g., FashionX, H&M, Zara, Uniqlo)",
  - **Items**: "Array of SKUs/Item IDs formatted as ['ITEM001', 'ITEM002']",
  - **Excluded Items**: "Array of SKUs/Item IDs formatted as ['ITEM003', 'ITEM004']",
  - **Discount Type**: "one of [% Off | Fixed Price | Buy One Get One Free]",
  - **Discount Value**: "Numerical amount (convert colloquial terms such as '50 bucks off' to '$50 Off') for the mentioned Discount Type",
  - **Start Date**: "(dd/mm/yyyy)",
  - **End Date**: "(dd/mm/yyyy)",
  - **Stores**: "Comma-separated Store IDs (e.g., STORE001)",
  - **Excluded Stores**: "Comma-separated Store IDs",
  - **classification_result**: "Classify the query as 'General' or 'Not General'."
  
  🔹 **Ensure that the JSON response strictly follows the exact key names provided.**
  🔹 **Use spaces in field names exactly as shown.**
  🔹 **If a value is missing, return null instead of omitting the key.**
  🔹 **If no Item ID is found in the fields 'Items' or 'Excluded Items', return an empty array [].**
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
