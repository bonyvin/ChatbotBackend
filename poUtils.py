import os
import datetime
import openai
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
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

template_PO_without_date=""" 
Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your purchase order operations and provide seamless support.Today is {current_date}.  

To generate a *Purchase Order (PO)*, please provide the following details manually or upload a PO file (PDF, JPG, or PNG) by clicking the "➕ Add" button below:  
- *Supplier ID* (alphanumeric)  
- *Estimated Delivery Date* (dd/mm/yyyy format or relative date e.g., '2 weeks from now') 
- *Total Quantity* (calculated from items)  
- *Total Cost* (calculated from items)  
- *Total Tax* (10% of total cost)  
- *Items* (multiple allowed, each must have the following details):  
  - *Item ID* (alphanumeric)  
  - *Quantity* (numbers only)  
  - *Cost per Unit* (numbers only)  

You can provide all the details at once, separated by commas, or enter them one by one.  

### *Supported Input Formats*  
- Enter *items separately* (e.g., "ID123", "ID124") or together (e.g., "ID123, ID124").  
- Provide *quantities separately* (e.g., "100", "50") or together (e.g., "100, 50").  
- Provide *cost per unit separately* (e.g., "500.00", "1200.50") or together (e.g., "500.00, 1200.50").  
- Use *item-quantity-cost triplets* (e.g., "ID123:100:500.00", "ID124:50:1200.50").  

### *My Capabilities*   
- I will *Keep track of all entered details* and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
- *Standardize formats*, such as:
  - Converting relative dates like "2 weeks from now" or "3 days from now" into "dd/mm/yyyy".
  - Converting different date formats like "MM/DD/YYYY", "YYYY/MM/DD", "DD.MM.YYYY", "DD-MM-YYYY", "Month Day, Year", "Day, Month DD, YYYY", "YYYY-MM-DD",etc to  "dd/mm/yyyy". 
- *Validate entries*, ensuring that:  
  - The number of items matches the number of quantities and costs.  
- *Detect and update duplicate Item IDs* instead of adding new entries:  
  - If an *Item ID is entered multiple times, I will **sum its quantity and cost*** instead of creating duplicates.  
- *Prompt for missing information* if required.  
- *Date Validation* 
    - Ensure that estimated delivery date is not in the past and is the same as or later than {current_date}.
- *Summarize all details before final submission.*  

---

## *Example User Scenarios*  

### *Scenario 1: User provides all details at once*  
*User:* "SUP001, 12/06/2025, ID123, ID124, ID125, 100, 50, 350, 500.00, 1200.00, 25.00"  
*Expected Response:* Validate input, ensure correct formats, calculate *Total Quantity*, *Total Cost*, and *Total Tax*, and provide a structured summary.  

### *Scenario 2: User provides details step by step*  
*User:*  
- "Supplier ID: SUP001"  
- "Estimated Delivery Date: 12/06/2025"  
- "Items: ID123, ID124, ID125"  
- "Quantities: 100, 50, 350"  
- "Cost: 500.00, 1200.00, 25.00"  
*Expected Response:* Store each entry, validate, calculate totals, and summarize the details each step.  

### *Scenario 3: User provides incorrect format*  
*User:*  
- "Estimated Delivery Date: 16 November 2025" → Convert to "16/11/2025"  
- "Total Cost: 2,50,000.00" → Convert to "250000.00"  
*Expected Response:* Standardize the format and confirm corrections with the user.  

### *Scenario 4: User enters the same Item ID multiple times*  
*User:*  
- "Items: ID123, ID124, ID123"  
- "Quantities: 100, 50, 30"  
- "Cost: 500.00, 1200.00, 300.00"  
*Expected Response:* Instead of duplicating *ID123, update its total quantity **(100+30 = 130)* and total cost *(500.00 + 300.00 = 800.00)*.  

Final stored values:  
- *Items:* "ID123, ID124"  
- *Quantities:* "130, 50"  
- *Costs:* "800.00, 1200.00"  

### *Scenario 5: User uploads a PO file*  
*User:* "Uploading purchase_order.pdf"  
*Expected Response:* Extract key information, format correctly, calculate totals, and present for confirmation.  

### *Scenario 6: User requests a summary*  
*User:* "Can you summarize my PO details?"  
*Expected Response:* Provide a structured summary of all collected details, including calculated totals.  

### *Scenario 7: User confirms submission*  
*User:* "Yes"  
*Expected Response:* "Purchase Order created successfully. Thank you for choosing us."  

### *Scenario 8: User cancels submission*  
*User:* "No, I want to change something."  
*Expected Response:* "Please specify what you would like to change."  

### *Scenario 9: User enters duplicate details*  
*User:* "Supplier ID: SUP001, Supplier ID: SUP001"  
*Expected Response:* Detect duplication and notify the user.  

### *Scenario 10: User provides ambiguous input*  
*User:* "Total: 250k"  
*Expected Response:* Ask the user to confirm if "250k" means "250000".  

### *Scenario 11: User includes special characters in inputs*  
*User:* "Supplier ID: SUP@#001"  
*Expected Response:* Remove special characters and confirm if "SUP001" is correct.  

### *Scenario 12: User provides an invalid date format*  
*User:* "Estimated Delivery Date: 2025/12/06"  
*Expected Response:* Convert to "06/12/2025" and confirm with the user.  

### *Scenario 13: User mixes input formats*  
*User:* "Supplier ID: SUP001, Est Delivery: 12/06/2025, Items: ID123, ID124-50-500.00"  
*Expected Response:* Standardize and confirm structured format.  

### *Scenario 14: User provides too many/few items for quantities*  
*User:*  
- "Items: ID123, ID124, ID125, ID126"  
- "Quantities: 100, 50, 350" (missing one)  
*Expected Response:* Detect mismatch and request the missing quantity.

### *Scenario 15: User adds additional items to an existing list*
*User:*
- "Items: ID123,100,500.00, ID124,50,1200.00,"  
*Expected Response:* Store the provided item details (item ID, quantity, cost) and calculate the overall totals.
*User then adds:*
- "Add another item: ID125,350,25.00"  
*Expected Response:* Append the new item to the existing list, revalidate the input, update *Total Quantity*, *Total Cost*, and *Total Tax*, and provide an updated summary.

### *Scenario 16: User updates an item in the existing list*
*User:*
- "Items: ID123,100,500.00, ID124,50,1200.00,"  
*Expected Response:* Store the initial items and calculate totals.
*User then instructs:*
- "Modify item: ID125,35,250.00"  
*Expected Response:* Replace or update the specified item's details with the new values, revalidate all data, recalculate the totals, and present the revised summary.

### *Scenario 17: User adds multiple new items after initial entry*
*User:*
- "Items: ID123,100,500.00, ID124,50,1200.00,"  
*Expected Response:* Record the initial items and compute the totals.
*User then provides:*
- "Add items: ID125,350,25.00; ID126,200,800.00" or  "Items: ID125,350,25.00; ID126,200,800.00"
*Expected Response:* Append these additional items to the list, validate the updated inputs, recalculate *Total Quantity*, *Total Cost*, and *Total Tax*, and update the summary.

### *Scenario 18: User updates multiple existing items simultaneously*
*User:*
- "Items: ID123,100,500.00, ID124,50,1200.00, ID125,350,25.00"  
*Expected Response:* Store the complete item list and calculate initial totals.
*User then instructs:*
- "Update items: Change ID123 to 120,600.00 and ID124 to 70,1400.00"  
*Expected Response:* Update the specified items with the new details, revalidate inputs, recalculate the totals, and provide a revised summary.

### *Scenario 19: User attempts to update a non-existent item*
*User:*
- "Items: ID123,100,500.00, ID124,50,1200.00"  
*Expected Response:* Record these items and compute totals.
*User then instructs:*
- "Update item: ID125,35,250.00"  
*Expected Response:* Notify the user that item ID "ID125" does not exist in the current list and ask if they would like to add it as a new entry.

---
*Here's your current Purchase Order details:*  
{chat_history}  

Missing details (if any) will be listed below.  

Would you like to submit this information?  
If you respond with 'Yes', I'll confirm with *"Purchase Order created successfully. Thank you for choosing us."*  
Upon receiving a 'Yes' response, inquire whether the user would like the document sent to their email and request their email address.
"""
today = datetime.datetime.today().strftime("%d/%m/%Y")
template_PO=template_PO_without_date.replace("{current_date}", today)
# DEFAULT_PO_STRUCTURE = {
#     "supplier_id": None,
#     "estimated_delivery_date": None,
#     "total_quantity": 0,
#     "total_cost": 0.0,
#     "total_tax": 0.0,
#     "items": [
#         {
#             "itemId": None,
#             "itemQuantity": None,
#             "itemCost": None
#         }
#     ],  # List of items with details
#     "email":""
# }
 
DEFAULT_PO_STRUCTURE = {
            "Supplier ID":                "",
            "Estimated Delivery Date":    "",
            "Total Quantity":             "",
            "Total Cost":                 "",
            "Total Tax":                  "",
            "Items":                      [
        {
            "itemId": None,
            "itemQuantity": None,
            "itemCost": None
        }
    ],
            "Email":                      "",
        }
previous_po_details = {}

async def categorize_po_details(extracted_text: str, user_id: str):
    client = openai.AsyncOpenAI()

    prompt = f"""
Extract the following purchase order details from the provided text and for each field determine if the value is merely an example instruction. For each field, return an object with two keys:
  - "value": the extracted field value (or null if missing).
  - "is_example": true if the extracted value appears to be just an example instruction (e.g. containing phrases like "for example", "e.g.", "such as"), false otherwise.

The fields and their expected formats are:

- **Supplier ID** → (alphanumeric, may be labeled as "Supplier", "Supplier Code", "Vendor ID")  
- **Estimated Delivery Date** → (formatted as dd/mm/yyyy, may appear as "Delivery Date", "Expected Date", "Est. Delivery")  
- **Total Quantity** → (sum of all item quantities)  
- **Total Cost** → (sum of all item costs)  
- **Total Tax** → (10% of total cost)  
- **Items** → (Array of objects, each containing):  
    - **Item ID** → (alphanumeric, may appear as "Product Code", "SKU", "Item No.")  
    - **Quantity** → (numeric, may appear as "Qty", "Quantity Ordered", "Units")  
    - **Cost Per Unit** → (numeric, may appear as "Unit Price", "Rate", "Price per Item")  
- **Email**: A valid email address where the email format follows standard conventions (e.g., user@example.com).


Use the exact field names as provided above. If a value is missing, set "value" to null (or [] for arrays) and "is_example" to false.

**Purchase Order Text:**
{extracted_text}

Return the response as a valid JSON object where each key is one of the fields and its value is an object in the form:
    "Field Name": {{ "value": "<extracted_value>", "is_example": "<true/false>" }}
  
If a field's value is missing, return null (or an empty array for fields expected to be arrays) and set "is_example" to false.
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Extract PO details with per-field example flags as described."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        raw_response = response.choices[0].message.content.strip()
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        print("Raw combined response:", raw_response)
        extracted_data = json.loads(raw_response)
        # Check if all values in the raw response are null or empty lists
        all_null = all(
            (v == {"value": None, "is_example": False}) or
            (v == {"value": [], "is_example": False}) or
            (isinstance(v, list) and all(
                all(sub.get("value") is None for sub in item.values())
                for item in v
            ))
            for v in extracted_data.values()
        )

        if all_null:
            print("All values in response are null/empty. Returning previous data.")
            return previous_po_details.get(user_id, {})

        # Clean out example values
        processed_data = {}
        # for field, details in extracted_data.items():
        #     is_example = details.get("is_example", False)
        #     value = details.get("value")
        #     if is_example:
        #         processed_data[field] = [] if field == "Items" else None
        #     else:
        #         processed_data[field] = value if value is not None else ([] if field == "Items" else None)

        # print("Processed Data PO: ",processed_data)
        # previous_po_details[user_id] = processed_data
        for field, details in extracted_data.items():
            # 1) If we got a list, we know this is our Items array
            if isinstance(details, list):
                items = []
                for item_obj in details:
                    cleaned = {}
                    for subfield, subdetail in item_obj.items():
                        # now subdetail _is_ a dict, so .get() works
                        if not subdetail.get("is_example", False):
                            cleaned[subfield] = subdetail.get("value")
                    if cleaned:
                        items.append(cleaned)
                processed_data[field] = items
                continue

            # 2) Otherwise, details _should_ be a dict
            if not isinstance(details, dict):
                # unexpected, just default
                processed_data[field] = [] if field == "Items" else None
                continue

            # 3) your existing logic
            is_example = details.get("is_example", False)
            value      = details.get("value")
            if is_example:
                processed_data[field] = [] if field == "Items" else None
            else:
                processed_data[field] = (
                    value if value is not None
                    else ([] if field == "Items" else None)
                ) 
        previous_po_details[user_id] = processed_data       
        return processed_data


    except Exception as e:
        print(f"Error fetching PO details: {e}")
        return previous_po_details.get(user_id, {})


async def categorize_po_details_(extracted_text: str, user_id: str):
    client = openai.AsyncOpenAI()
    
    # Define keywords that indicate examples rather than actual user-provided data
    example_indicators = [
        "For example", "Example:", "e.g.", "like this", "such as"
    ]
    
    # Check if the extracted text contains example indicators
    if any(keyword in extracted_text for keyword in example_indicators):
        print("Detected example instructions; skipping extraction.")
        return previous_po_details.get(user_id, {})
    
    # Prompt to enforce exact JSON key formatting
    prompt = f"""
    Extract and structure the following details from this Purchase Order text. The JSON keys **must match exactly** as given below:

    - **Supplier ID** → (alphanumeric, may be labeled as "Supplier", "Supplier Code", "Vendor ID")  
    - **Estimated Delivery Date** → (formatted as dd/mm/yyyy, may appear as "Delivery Date", "Expected Date", "Est. Delivery")  
    - **Total Quantity** → (sum of all item quantities)  
    - **Total Cost** → (sum of all item costs)  
    - **Total Tax** → (10% of total cost)  
    - **Items** → (list of products, each containing):  
      - **Item ID** → (alphanumeric, may appear as "Product Code", "SKU", "Item No.")  
      - **Quantity** → (numeric, may appear as "Qty", "Quantity Ordered", "Units")  
      - **Cost Per Unit** → (numeric, may appear as "Unit Price", "Rate", "Price per Item")  

    🔹 **Ensure that the JSON response strictly follows the exact key names provided.**  
    🔹 **Use spaces in field names exactly as shown.**  
    🔹 **If a value is missing, return null instead of omitting the key.**  

    **Purchase Order Text:**
    {extracted_text}

    **Format the response as a valid JSON object. The field names must match exactly.**
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Extract structured data from a purchase order, recognizing multiple formats for key fields."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}  # Ensure JSON output
        )
        raw_response = response.choices[0].message.content.strip()
        print("Raw response: ", raw_response)
        
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()  # Remove triple backticks
        
        structured_data = json.loads(raw_response)
        
        # Map API response keys to expected field names
        field_mapping = {
            "SupplierID": "Supplier ID",
            "EstimatedDeliveryDate": "Estimated Delivery Date",
            "TotalQuantity": "Total Quantity",
            "TotalCost": "Total Cost",
            "TotalTax": "Total Tax",
            "Items": "Items"
        }

        # Standardize item field names
        item_field_mapping = {
            "ItemID": "Item ID",
            "Quantity": "Quantity",
            "CostPerUnit": "Cost Per Unit"
        }

        # Rename top-level fields
        structured_data = {field_mapping.get(k, k): v for k, v in structured_data.items()}

        # Rename item fields
        if "Items" in structured_data and isinstance(structured_data["Items"], list):
            for item in structured_data["Items"]:
                for old_key, new_key in item_field_mapping.items():
                    if old_key in item:
                        item[new_key] = item.pop(old_key)

        # Validate if extracted items are actual user input and not examples
        if structured_data.get("Items"):
            valid_items = []
            for item in structured_data["Items"]:
                item_text = f"{item['Item ID']}:{item['Quantity']}:{item['Cost Per Unit']}"
                if any(keyword in extracted_text for keyword in example_indicators):
                    continue  # Ignore example items
                valid_items.append(item)
            structured_data["Items"] = valid_items
        
        # Check if all primary fields are None and the Items list is empty
        primary_fields = ["Supplier ID", "Estimated Delivery Date", "Total Quantity", "Total Cost", "Total Tax","Items"]
        if all(structured_data.get(field) in [None, ""] for field in primary_fields) and not structured_data.get("Items"):
            return previous_po_details.get(user_id, {})  # Return previous data if exists

        # Store the latest valid PO details
        previous_po_details[user_id] = structured_data
        return structured_data
    
    except Exception as e:
        print(f"Error fetching PO details: {e}")
        return previous_po_details.get(user_id, {})  # Return previous data if API call fails
