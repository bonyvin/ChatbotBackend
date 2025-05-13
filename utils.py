import os
import datetime
import openai
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
# from openai import OpenAI,AsyncOpenAI
import json
from pdf2image import convert_from_path
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

import base64
import csv
from pathlib import Path
import pytesseract
import docx
from openai import OpenAI, AsyncOpenAI, api_key
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

HF_TOKEN = os.getenv('HF_TOKEN')
FINE_TUNED_OPENAI_API_KEY=os.getenv('FINE_TUNED_OPENAI_API_KEY')

conversation = []

def read_poDeatilsPO(po_id: str, db: Session = Depends(get_db)):
    po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
    return po


bot_action=""


client = OpenAI(api_key=OPENAI_API_KEY)
client_new = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0.7)
sync_client = OpenAI(api_key=OPENAI_API_KEY)
async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

llm_gpt4 = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-4-1106-preview",
    temperature=0.7
)

llm_gpt3 = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.0
)

def testModel(query,output):
    # Initial query to determine the action
    ft_model = 'ft:gpt-3.5-turbo-0125:personal::9dzqEHs0'
    response = client.chat.completions.create(
        model=ft_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    response_message = response.choices[0].message.content
    if response_message == "Invoice Creation":
        print("Creation")
        return "Creation"
    elif response_message == "Invoice Fetch" or response_message == "Invoice Fetching" :
        print("Fetch")
        return "Fetch"
    # else:
    #     print("No matches")
    #     return "No matches"
    print("Response Message: ",response_message)
    # Follow-up query to check if details have been submitted
    # response = client.chat.completions.create(
    #     model=ft_model,
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": query},
    #         {"role": "assistant", "content": output}
    #     ]
    # )
    # follow_up_response_message = response.choices[0].message.content

    # if follow_up_response_message == "Invoice Submission":
    #     print("Submission")
    #     return "Submission"
    # else:
    #     print(action)
    #     return action

def extract_text_with_openai2(file: UploadFile):
    file_content = file.file.read()
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "Extract key information from the provided document."},
            {"role": "user", "content": "Please extract text from this file."},
        ],
        file=file_content
    )
    return response.choices[0].message.content

 
template_5_without_date="""
Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your invoice operations and provide seamless support.Today is {current_date}. 

# To generate an invoice, please provide the following details manually or upload an invoice file (PDF, JPG, or PNG) by clicking the "➕ Add" button below:
# - PO number (alphanumeric)
# - Invoice type (one of [Merchandise | Non - Merchandise | Debit Note | Credit Note])
# - Date (dd/mm/yyyy format or relative date e.g., '2 weeks from now') 
# - Invoice Number (alphanumeric only)
# - Items (alphanumeric, multiple allowed)
# - Quantities (numbers only, must match item count)
# - Invoice Cost (numbers only, must match item count)

To generate an invoice, please provide the following details manually or upload an invoice file (PDF, JPG, or PNG) by clicking the "➕ Add" button below. **Every detail must be recorded as: [Detail Name]: [Provided Value]**.

**Required Details:**
- **PO Number:** (alphanumeric; may appear as "PO ID", "Purchase Order Id", or "PO No.")
- **Invoice Type:** (must be one of: Merchandise, Non-Merchandise, Debit Note, Credit Note. Variations such as "merch", "non merch", "debit", or "credit" will be mapped accordingly.)
- **Date:** (enter in dd/mm/yyyy format. Relative dates like "2 weeks from now" will be converted automatically. For example, "16 November 2025" becomes "16/11/2025".)
- **Invoice Number:** (alphanumeric only; remove spaces and special characters. For instance, "INV 1234" should be recorded as "INV1234".)
- **Items:** Provide each item in a list format. **For each item, record the details exactly as follows:**
    - **Item ID:** (alphanumeric; may also appear as "Product Code", "SKU", or "Item No.")
    - **Quantity:** (numeric)
    - **Invoice Cost:** (numeric)

Note: **Total Amount** and **Total Tax** will be automatically calculated from the provided item details. You do not need to supply them.

You can provide all the details at once, separated by commas, or enter them one by one.  
I support flexible formats for Items, Quantities, and Invoice Costs:
- Enter items separately (e.g., "ID123", "ID124") or together (e.g., "ID123, ID124").
- Provide quantities separately (e.g., "122", "43") or together (e.g., "122, 43").
- Provide invoice costs separately (e.g., "500.00", "1200.50") or together (e.g., "500.00, 1200.50").
- Use item-quantity-cost triplets (e.g., "ID123:5:500.00", "ID124-10-1200.50")

I will:
- Keep track of all entered details and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
    - - **Important**: Whenever I record a valid detail from any field, I will immediately display the recorded detail in a summary along with my response and the previously recorded and missing fields.
- *Standardize formats*, such as:
  - Converting relative dates like "2 weeks from now" or "3 days from now" into "dd/mm/yyyy".
  - Converting different date formats like "MM/DD/YYYY", "YYYY/MM/DD", "DD.MM.YYYY", "DD-MM-YYYY", "Month Day, Year", "Day, Month DD, YYYY", "YYYY-MM-DD",etc to  "dd/mm/yyyy". 
- Validate that the number of items matches the number of quantities and invoice costs.
- Prompt for any missing or incomplete information.
- Summarize all details before submission, including the **computed Total Tax** and **Total Amount**.
- *If an Item ID is entered more than once, I will automatically update its quantity and invoice cost instead of adding a duplicate entry.*
- *Date Validation* - Ensure that start date is equal to or greater than {current_date}.
---

## Example User Scenarios

### *Scenario 1: User provides all details at once*  
*User:*  
"PO12345, Merchandise invoice, 12/06/2025, INV5678, ID123, ID124, ID125, 5, 10, 3, 500.00, 1000.00, 600.00"  

*Expected Response:*  
- Validate the format and ensure that the number of items, quantities, and invoice costs match.
- Standardize the date format.
- Calculate the totals:  
  - For ID123: 5 × 500.00 = 2500.00  
  - For ID124: 10 × 1000.00 = 10000.00  
  - For ID125: 3 × 600.00 = 1800.00  
  - **Sum of items cost** = 2500.00 + 10000.00 + 1800.00 = 14800.00  
  - **Total Tax** = 10% of 14800.00 = 1480.00  
  - **Total Amount** = 14800.00 + 1480.00 = 16280.00  
- Summarize the details:
    - PO Number: PO12345
    - Invoice Type: Merchandise invoice
    - Date: 12/06/2025
    - Invoice Number: INV5678
    - Items: ID123, ID124, ID125
    - Quantities: 5, 10, 3
    - Invoice Costs: 500.00, 1000.00, 600.00
    - Total Tax: 1480.00
    - Total Amount: 16280.00  
---

### *Scenario 2: User provides details step by step*  
*User:*  
- "PO number: PO12345"  
- "Invoice type: Merchandise invoice"  
- "Date: 12/06/2025"  
- "Invoice Number: INV5678"  
- "Items: ID123, ID124, ID125"  
- "Quantities: 5, 10, 3"  
- "Invoice Cost: 500.00, 1000.00, 600.00"  

*Expected Response:*  
- Store each value as it’s provided.
- Validate that the counts for items, quantities, and invoice costs match.
- Compute the totals (as in Scenario 1).  
- Present a summary including:  
  - **Total Tax: 1480.00**  
  - **Total Amount: 16280.00**

---

### *Scenario 3: User provides incorrect format*  
*User:*  
- "Date: 16 November 2025" → Should be converted to "16/11/2025"  
- "Invoice Number: INV 5678" → Should be "INV5678" (remove spaces)  
- (If a total amount is provided like "15,000.50", it will be ignored since totals are calculated automatically)  

*Expected Response:*  
- Correct and confirm the standardized formats:  
  - Date becomes "16/11/2025"  
  - Invoice Number becomes "INV5678"  
- Inform the user that the total values will be recalculated based on the item details.

---

### *Scenario 4: User provides incomplete details*  
*User:*  
- "PO number: PO12345"  
- "Invoice type: Merchandise invoice"  
- "Date: 12/06/2025"  
- "Invoice Number: INV5678"  
- "Items: ID123, ID124"  
- "Quantities: 5"  
- "Invoice Cost: 500.00, 1000.00"  

*Expected Response:*  
- Detect the missing quantity for ID124.
- Ask the user to provide the missing quantity before proceeding with the calculation.

---

### *Scenario 5: User enters the same Item ID multiple times*  
*User:*  
- "Items: ID123, ID124, ID123"  
- "Quantities: 5, 10, 3"  
- "Invoice Cost: 500.00, 1000.00, 300.00"  

*Expected Response:*  
- Instead of adding duplicate ID123 entries, update its quantity by summing the values:  
  - New quantity for ID123 = 5 + 3 = 8  
  - New invoice cost for ID123 = 500.00 + 300.00 = 800.00  
- Final output:  
  - Items: ID123, ID124  
  - Quantities: 8, 10  
  - Invoice Costs: 800.00, 1000.00  
- Compute totals based on these final values.

---

### *Scenario 6: User uploads an invoice file*  
*User:* "Uploading invoice.pdf"  

*Expected Response:*  
- Extract the relevant details from the file.
- Standardize the formats and calculate the **Total Tax** and **Total Amount** automatically.
- Present the extracted and computed details to the user for confirmation.

---

### *Scenario 7: User requests a summary*  
*User:* "Can you summarize my invoice details?"  

*Expected Response:*  
- Provide a structured summary of all collected details, including the computed **Total Tax** and **Total Amount**.

---

### *Scenario 8: User confirms submission*  
*User:* "Yes"  

*Expected Response:*  
"Invoice created successfully. Thank you for choosing us."

---

### *Scenario 9: User cancels submission*  
*User:* "No, I want to change something."  

*Expected Response:*  
"Please specify what you would like to change."

---

### *Scenario 10: User enters duplicate details*  
*User:* "PO number: PO12345, PO number: PO12345"  

*Expected Response:*  
- Detect duplication and inform the user that duplicate entries are not allowed.

---

### *Scenario 11: User provides ambiguous input*  
*User:* "Total: 15k"  

*Expected Response:*  
- Ask the user to confirm if "15k" means "15000" before proceeding.
- (Note: Since totals are auto-calculated, clarify that any such input will be disregarded in favor of computed values.)

---

### *Scenario 12: User includes special characters in inputs*  
*User:* "Invoice Number: INV@#5678"  

*Expected Response:*  
- Remove special characters and confirm with the user that the corrected invoice number is "INV5678".

---

### *Scenario 13: User provides an invalid date format*  
*User:* "Date: 2025/12/06"  

*Expected Response:*  
- Convert to the correct format, e.g., "06/12/2025", and confirm the change with the user.

---

### *Scenario 14: User mixes input formats*  
*User:* "PO12345, Invoice Number: INV5678, Date: 12/06/2025, Items: ID123, ID124-10-500.00"  

*Expected Response:*  
- Standardize all inputs and extract the details.
- Calculate the totals from the item details.
- Present a structured summary for confirmation.

---

### *Scenario 15: User provides too many/few items for quantities*  
*User:*  
"Items: ID123, ID124, ID125, ID126"  
"Quantities: 5, 10, 3"  

*Expected Response:*  
- Detect the mismatch between the number of items and quantities.
- Request the missing or extra information before proceeding.

---

### *Scenario 16: Validation*  
*User provides:*  
- PO Number: PO123  
- Invoice Type: Merchandise  
- Date: 16/06/2025  
- Invoice Number: INVV9990  
- Items: ITEM01, ITEM02, ITEM03  
- Quantities: 1, 2, 3  
- Invoice Costs: 2000, 1000, 500  

*Calculation:*  
- For ITEM01: 1 × 2000 = 2000  
- For ITEM02: 2 × 1000 = 2000  
- For ITEM03: 3 × 500 = 1500  
- **Sum of items cost** = 2000 + 2000 + 1500 = 5500  
- **Total Tax** = 10% of 5500 = 550  
- **Total Amount** = 5500 + 550 = 6050  

*Expected Response:*  
Let's validate the details:  
- The number of items matches the number of quantities and invoice costs.
- Here are your invoice details:
   - PO Number: PO123
   - Invoice Type: Merchandise
   - Date: 16/06/2025
   - Invoice Number: INVV9990
   - Items: ITEM01, ITEM02, ITEM03
   - Quantities: 1, 2, 3
   - Invoice Costs: 2000, 1000, 500
   - Total Tax: 550
   - Total Amount: 6050

# Here's your current invoice details:  
# {chat_history}  

Missing details (if any) will be listed below.  

Would you like to submit this information?  
If you respond with 'Yes', I'll confirm with "Invoice created successfully. Thank you for choosing us."


"""
template_5_without_date_new="""
Hello and welcome! I'm ExpX, your dedicated assistant. I'm here to streamline your invoice operations and provide seamless support.Today is {current_date}. 
To generate an invoice, please provide the following details manually or upload an invoice file (PDF, JPG, or PNG) by clicking the "➕ Add" button below. **Every detail must be recorded as: [Detail Name]: [Provided Value]**.

**Required Details:**
- **PO Number:** (alphanumeric; may appear as "PO ID", "Purchase Order Id", or "PO No.")
- **Invoice Type:** (must be one of: Merchandise, Non-Merchandise, Debit Note, Credit Note. Variations such as "merch", "non merch", "debit", or "credit" will be mapped accordingly.)
- **Date:** (enter in dd/mm/yyyy format. Relative dates like "2 weeks from now" will be converted automatically. For example, "16 November 2025" becomes "16/11/2025".)
- **Invoice Number:** (alphanumeric only; remove spaces and special characters. For instance, "INV 1234" should be recorded as "INV1234".)
- **Items:** Provide each item in a list format. **For each item, record the details exactly as follows:**
    - **Item ID:** (alphanumeric; may also appear as "Product Code", "SKU", or "Item No.")
    - **Quantity:** (numeric)
    - **Invoice Cost:** (numeric)

Note: **Total Amount** and **Total Tax** will be automatically calculated from the provided item details. You do not need to supply them.

You can provide all the details at once, separated by commas, or enter them one by one.  
I support flexible formats for Items, Quantities, and Invoice Costs:
- Enter items separately (e.g., "ID123", "ID124") or together (e.g., "ID123, ID124").
- Provide quantities separately (e.g., "122", "43") or together (e.g., "122, 43").
- Provide invoice costs separately (e.g., "500.00", "1200.50") or together (e.g., "500.00, 1200.50").
- Use item-quantity-cost triplets (e.g., "ID123:5:500.00", "ID124-10-1200.50")

I will:
- Keep track of all entered details and fill in any missing ones in the same structured format. Each detail will be recorded as: [Detail Name]: [Provided Value], ensuring consistency with the format outlined above.
- *Standardize formats*, such as:
  - Converting relative dates like "2 weeks from now" or "3 days from now" into "dd/mm/yyyy".
  - Converting different date formats like "MM/DD/YYYY", "YYYY/MM/DD", "DD.MM.YYYY", "DD-MM-YYYY", "Month Day, Year", "Day, Month DD, YYYY", "YYYY-MM-DD",etc to  "dd/mm/yyyy". 
- Validate that the number of items matches the number of quantities and invoice costs.
- Prompt for any missing or incomplete information.
- Summarize all details before submission, including the **computed Total Tax** and **Total Amount**.
- *If an Item ID is entered more than once, I will automatically update its quantity and invoice cost instead of adding a duplicate entry.*
- I will add item ids retrieved from the PO : {po_item_list} to **Item ID**  of **Items**.
---

## Example User Scenarios

### *Scenario 1: User provides all details at once*  
*User:*  
"PO12345, Merchandise invoice, 12/06/2025, INV5678, ID123, ID124, ID125, 5, 10, 3, 500.00, 1000.00, 600.00"  

*Expected Response:*  
- Validate the format and ensure that the number of items, quantities, and invoice costs match.
- Standardize the date format.
- Calculate the totals:  
  - For ID123: 5 × 500.00 = 2500.00  
  - For ID124: 10 × 1000.00 = 10000.00  
  - For ID125: 3 × 600.00 = 1800.00  
  - **Sum of items cost** = 2500.00 + 10000.00 + 1800.00 = 14800.00  
  - **Total Tax** = 10% of 14800.00 = 1480.00  
  - **Total Amount** = 14800.00 + 1480.00 = 16280.00  
- Summarize the details:
    - PO Number: PO12345
    - Invoice Type: Merchandise invoice
    - Date: 12/06/2025
    - Invoice Number: INV5678
    - Items: ID123, ID124, ID125
    - Quantities: 5, 10, 3
    - Invoice Costs: 500.00, 1000.00, 600.00
    - Total Tax: 1480.00
    - Total Amount: 16280.00  
---

### *Scenario 2: User provides details step by step*  
*User:*  
- "PO number: PO12345"  
- "Invoice type: Merchandise invoice"  
- "Date: 12/06/2025"  
- "Invoice Number: INV5678"  
- "Items: ID123, ID124, ID125"  
- "Quantities: 5, 10, 3"  
- "Invoice Cost: 500.00, 1000.00, 600.00"  

*Expected Response:*  
- Store each value as it’s provided.
- Validate that the counts for items, quantities, and invoice costs match.
- Compute the totals (as in Scenario 1).  
- Present a summary including:  
  - **Total Tax: 1480.00**  
  - **Total Amount: 16280.00**

---

### *Scenario 3: User provides incorrect format*  
*User:*  
- "Date: 16 November 2025" → Should be converted to "16/11/2025"  
- "Invoice Number: INV 5678" → Should be "INV5678" (remove spaces)  
- (If a total amount is provided like "15,000.50", it will be ignored since totals are calculated automatically)  

*Expected Response:*  
- Correct and confirm the standardized formats:  
  - Date becomes "16/11/2025"  
  - Invoice Number becomes "INV5678"  
- Inform the user that the total values will be recalculated based on the item details.

---

### *Scenario 4: User provides incomplete details*  
*User:*  
- "PO number: PO12345"  
- "Invoice type: Merchandise invoice"  
- "Date: 12/06/2025"  
- "Invoice Number: INV5678"  
- "Items: ID123, ID124"  
- "Quantities: 5"  
- "Invoice Cost: 500.00, 1000.00"  

*Expected Response:*  
- Detect the missing quantity for ID124.
- Ask the user to provide the missing quantity before proceeding with the calculation.

---

### *Scenario 5: User enters the same Item ID multiple times*  
*User:*  
- "Items: ID123, ID124, ID123"  
- "Quantities: 5, 10, 3"  
- "Invoice Cost: 500.00, 1000.00, 300.00"  

*Expected Response:*  
- Instead of adding duplicate ID123 entries, update its quantity by summing the values:  
  - New quantity for ID123 = 5 + 3 = 8  
  - New invoice cost for ID123 = 500.00 + 300.00 = 800.00  
- Final output:  
  - Items: ID123, ID124  
  - Quantities: 8, 10  
  - Invoice Costs: 800.00, 1000.00  
- Compute totals based on these final values.

---

### *Scenario 6: User uploads an invoice file*  
*User:* "Uploading invoice.pdf"  

*Expected Response:*  
- Extract the relevant details from the file.
- Standardize the formats and calculate the **Total Tax** and **Total Amount** automatically.
- Present the extracted and computed details to the user for confirmation.

---

### *Scenario 7: User requests a summary*  
*User:* "Can you summarize my invoice details?"  

*Expected Response:*  
- Provide a structured summary of all collected details, including the computed **Total Tax** and **Total Amount**.

---

### *Scenario 8: User confirms submission*  
*User:* "Yes"  

*Expected Response:*  
"Invoice created successfully. Thank you for choosing us."

---

### *Scenario 9: User cancels submission*  
*User:* "No, I want to change something."  

*Expected Response:*  
"Please specify what you would like to change."

---

### *Scenario 10: User enters duplicate details*  
*User:* "PO number: PO12345, PO number: PO12345"  

*Expected Response:*  
- Detect duplication and inform the user that duplicate entries are not allowed.

---

### *Scenario 11: User provides ambiguous input*  
*User:* "Total: 15k"  

*Expected Response:*  
- Ask the user to confirm if "15k" means "15000" before proceeding.
- (Note: Since totals are auto-calculated, clarify that any such input will be disregarded in favor of computed values.)

---

### *Scenario 12: User includes special characters in inputs*  
*User:* "Invoice Number: INV@#5678"  

*Expected Response:*  
- Remove special characters and confirm with the user that the corrected invoice number is "INV5678".

---

### *Scenario 13: User provides an invalid date format*  
*User:* "Date: 2025/12/06"  

*Expected Response:*  
- Convert to the correct format, e.g., "06/12/2025", and confirm the change with the user.

---

### *Scenario 14: User mixes input formats*  
*User:* "PO12345, Invoice Number: INV5678, Date: 12/06/2025, Items: ID123, ID124-10-500.00"  

*Expected Response:*  
- Standardize all inputs and extract the details.
- Calculate the totals from the item details.
- Present a structured summary for confirmation.

---

### *Scenario 15: User provides too many/few items for quantities*  
*User:*  
"Items: ID123, ID124, ID125, ID126"  
"Quantities: 5, 10, 3"  

*Expected Response:*  
- Detect the mismatch between the number of items and quantities.
- Request the missing or extra information before proceeding.

---

### *Scenario 16: Validation*  
*User provides:*  
- PO Number: PO123  
- Invoice Type: Merchandise  
- Date: 16/06/2025  
- Invoice Number: INVV9990  
- Items: ITEM01, ITEM02, ITEM03  
- Quantities: 1, 2, 3  
- Invoice Costs: 2000, 1000, 500  

*Calculation:*  
- For ITEM01: 1 × 2000 = 2000  
- For ITEM02: 2 × 1000 = 2000  
- For ITEM03: 3 × 500 = 1500  
- **Sum of items cost** = 2000 + 2000 + 1500 = 5500  
- **Total Tax** = 10% of 5500 = 550  
- **Total Amount** = 5500 + 550 = 6050  

*Expected Response:*  
Let's validate the details:  
- The number of items matches the number of quantities and invoice costs.
- Here are your invoice details:
   - PO Number: PO123
   - Invoice Type: Merchandise
   - Date: 16/06/2025
   - Invoice Number: INVV9990
   - Items: ITEM01, ITEM02, ITEM03
   - Quantities: 1, 2, 3
   - Invoice Costs: 2000, 1000, 500
   - Total Tax: 550
   - Total Amount: 6050

# Here's your current invoice details:  
# {chat_history}  

Missing details (if any) will be listed below.  

Would you like to submit this information?  
If you respond with 'Yes', I'll confirm with "Invoice created successfully. Thank you for choosing us."


"""
today = datetime.datetime.today().strftime("%d/%m/%Y")
template_5=template_5_without_date.replace("{current_date}", today)
template_5_new=template_5_without_date_new.replace("{current_date}", today)
# system_message_prompt = SystemMessagePromptTemplate.from_template(template_5)
# human_template="{query}"
# human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
# chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])

# chat = ChatOpenAI(model="gpt-4o",temperature=0.7)

# memory = ConversationBufferMemory(memory_key="chat_history")

def gpt_response(query):

    chain = LLMChain(llm = chat, prompt = chat_prompt, memory = memory)
    action = determine_action(query)
    if action:
        memory.chat_memory.add_user_message(f"Action: {action}")
    response = chain.run(query)

    return response

# bot_action=memory.chat_memory.messages
# test_model_reply=""

def test_submission(query):
    # print("Form submission query:",query)
    ft_model = 'ft:gpt-3.5-turbo-0125:personal::A4RdItSX'
    # ft_model = 'ft:gpt-3.5-turbo-0125:personal::9dzqEHs0'
    # ft_model = 'ft:gpt-3.5-turbo-0125:personal::9cWmjxLZ'
    response = client.chat.completions.create(
        model=ft_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": query}
        ]
    )
    follow_up_response_message = response.choices[0].message.content

    if follow_up_response_message == "Invoice Submission":
        print("test_submission: Submission")
        return "submitted"
    elif follow_up_response_message == "Invoice Progressing":
        print("Progressing test_submission")
    elif follow_up_response_message=="Invoice Fetch":
        print("Fetch test_submission")
    else:
        print("test_submission:No matches")
        return "not submitted"


def openaifunction():
    # file_path = 'finetuning_excel_format.xlsx'
    # df = pd.read_excel(file_path)
    # # Create a list to store the JSONL data
    # jsonl_data = []
    # # Convert each row to the JSONL format
    # for index, row in df.iterrows():
    #     prompt = row['prompt']
    #     completion = row['completion']
    #     message_pair = {
    #         "messages": [
    #             {"role": "user", "content": prompt},
    #             {"role": "assistant", "content": completion}
    #         ]
    #     }  
    #     jsonl_data.append(message_pair)

    # # Define the output JSONL file path
    # output_file_path = 'finetuning_json_format.json'
    # # Write the JSONL data to the file
    # with open(output_file_path, 'w') as jsonl_file:
    #     for entry in jsonl_data:
    #         jsonl_file.write(json.dumps(entry) + '\n')
    # print(f"JSONL file has been created at {output_file_path}")


    input_jsonl_file = 'finetuning_json_format.json'
    client = OpenAI(
    api_key=FINE_TUNED_OPENAI_API_KEY
    )
    file = client.files.create(
    file=open(input_jsonl_file, "rb"),
    purpose="fine-tune"
    )
    print("File has been uploaded to OpenAI with id ", file.id)

    ft_job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo"
    )
    print("Fine Tune Job has been created with id ", ft_job.id)

def extract_information(conversation, pattern):
    for line in conversation:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def determine_action(query):
    if "create" in query.lower():
        return "creation"
    elif "fetch" in query.lower() or "get details" in query.lower():
        return "fetch"
    return None

previous_invoice_details = {}
async def categorize_invoice_details_new(extracted_text: str, user_id: str):
    client = openai.AsyncOpenAI()

    prompt = f"""
Extract the following invoice details from the provided text and for each field determine if the value is merely an example instruction. For each field, return an object with two keys:
  - "value": the extracted field value (or null if missing).
  - "is_example": true if the extracted value appears to be just an example instruction (e.g. containing phrases like "for example", "e.g.", "such as"), false otherwise.

The fields and their expected formats are:

- **PO Number** (alphanumeric, may appear as "PO ID", "Purchase Order Id","PO No.")
- **Invoice Number** (alphanumeric, may appear as "Invoice ID", "Bill No.")
- **Invoice Type**: (Normalize to one of [Merchandise, Non - Merchandise, Debit Note, Credit Note]. Accept variations or shorthand inputs such as "merch", "non merch", "debit", or "credit" and map them to the correct option.)
- **Date** (formatted as dd/mm/yyyy, may be labeled as "Invoice Date", "Billing Date")
- **Total Amount** (sum of item costs)
- **Total Tax** (10% of total amount)
- **Items**:(Array of objects, each containing):
    - **Item ID** (alphanumeric, may appear as "Product Code", "SKU", "Item No.")
    - **Quantity** (numeric, may appear as "Qty", "Quantity Ordered", "Units")
    - **Invoice Cost** (numeric, may appear as "Item Cost", "Total Cost per Item")

Use the exact field names as provided above. If a value is missing, set "value" to null (or [] for arrays) and "is_example" to false.

**Invoice Text:**
{extracted_text}

Return the response as a valid JSON object like:
"Field Name": {{ "value": ..., "is_example": true/false }}
"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Extract invoice details with per-field example flags as described."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        raw_response = response.choices[0].message.content.strip()
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        extracted_data = json.loads(raw_response)

        processed_data = {}
        for field, details in extracted_data.items():
            is_example = details.get("is_example", False)
            value = details.get("value")
            if is_example:
                processed_data[field] = [] if field == "Items" else None
            else:
                processed_data[field] = value if value is not None else ([] if field == "Items" else None)

        previous_invoice_details[user_id] = processed_data
        return processed_data

    except Exception as e:
        print(f"Error fetching invoice details: {e}")
        return previous_invoice_details.get(user_id, {})
    
async def categorize_invoice_details_new_(extracted_text: str, user_id: str):
    client = openai.AsyncOpenAI()
    example_indicators = ["For example", "Example:", "e.g.", "like this", "such as"]
    
    if any(keyword in extracted_text for keyword in example_indicators):
        print("Detected example instructions; skipping extraction.")
        return previous_invoice_details.get(user_id, {})
    
    prompt = f"""
    Extract and structure the following details from this Invoice text. Accept different variations and short forms for each field:
    - **PO Number** (alphanumeric, may appear as "PO ID", "Purchase Order Id","PO No.")
    - **Invoice Number** (alphanumeric, may appear as "Invoice ID", "Bill No.")
    - **Invoice Type** (normalize the extracted value to exactly one of the following: Merchandise, Non - Merchandise, Debit Note, Credit Note. Accept variations or shorthand inputs such as "merch", "non merch", "debit", or "credit" and map them to the correct option.)
    - **Date** (formatted as dd/mm/yyyy, may be labeled as "Invoice Date", "Billing Date")
    - **Total Amount** (sum of item costs)
    - **Total Tax** (10% of total amount)
    - **Items** (multiple allowed, each item should have):
      - **Item ID** (alphanumeric, may appear as "Product Code", "SKU", "Item No.")
      - **Quantity** (numeric, may appear as "Qty", "Quantity Ordered", "Units")
      - **Invoice Cost** (numeric, may appear as "Item Cost", "Total Cost per Item")
    
    🔹 **Ensure that example items are excluded from the extraction.**
    🔹 **Ensure that the JSON response strictly follows the exact key names provided.**  
    🔹 **Use spaces in field names exactly as shown.**  
    🔹 **If a value is missing, return null instead of omitting the key.**
      
    
    **Invoice Text:**
    {extracted_text}
    
    **Format the response as a valid JSON object. If a field is missing, return null for that field.**
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Extract structured data from an invoice, recognizing multiple formats for key fields."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}
        )
        raw_response = response.choices[0].message.content.strip()
        print("Raw response:", raw_response)
        
        if raw_response.startswith("```json"):
            raw_response = raw_response[7:-3].strip()
        
        structured_data = json.loads(raw_response)
        
        # Validate if extracted items are actual user input and not examples
        if structured_data.get("Items"):
            valid_items = []
            for item in structured_data["Items"]:
                item_text = f"{item['Item ID']}:{item['Quantity']}:{item['Invoice Cost']}"
                if any(keyword in extracted_text for keyword in example_indicators):
                    continue  # Ignore example items
                valid_items.append(item)
            structured_data["Items"] = valid_items
        
        primary_fields = ["PO Number","Invoice Number", "Invoice Type", "Date", "Total Amount", "Total Tax","Items"]
        if all(structured_data.get(field) in [None, ""] for field in primary_fields) and not structured_data.get("Items"):
            return previous_invoice_details.get(user_id, {})
        
        previous_invoice_details[user_id] = structured_data
        return structured_data
    
    except Exception as e:
        print(f"Error fetching invoice details: {e}")
        return previous_invoice_details.get(user_id, {})

def collect_invoice_data(chat_history):
    invoice_data = {}

    for message in chat_history:
        if "Bot: " in message:
            # Extract invoice type
            match_invoice_type = re.search(r"\*\*Invoice Type:\*\*\s*(.+)", message)
            if match_invoice_type:
                invoice_data["invoice type"] = match_invoice_type.group(1).strip()

            # Extract date
            match_date = re.search(r"\*\*Date:\*\*\s*(\d{2}/\d{2}/\d{4})", message)
            if match_date:
                invoice_data["date"] = match_date.group(1).strip()

            # Extract PO number
            match_po_number = re.search(r"\*\*PO Number:\*\*\s*(\w+)", message)
            if match_po_number:
                invoice_data["po number"] = match_po_number.group(1).strip()

            # Extract Supplier Id
            match_supplier_id = re.search(r"\*\*Supplier Id:\*\*\s*(\w+)", message)
            if match_supplier_id:
                invoice_data["supplier id"] = match_supplier_id.group(1).strip()

            # Extract Total Amount (handles commas)
            match_total_amount = re.search(r"\*\*Total Amount:\*\*\s*([\d,]+)", message)
            if match_total_amount:
                invoice_data["total amount"] = match_total_amount.group(1).strip()

            # Extract Total Tax
            match_total_tax = re.search(r"\*\*Total Tax:\*\*\s*([\d,]+)", message)
            if match_total_tax:
                invoice_data["total tax"] = match_total_tax.group(1).strip()

            # Extract Items (Supports multiple formats)
            # match_items = re.search(r"\*\*Items:\*\*\s*([\w,\s]+?)(?=\n\d+|\Z)", message)
            # match_items = re.search(r"\*\*Items:\*\*\s*([\w,\s]+)", message)
            # if match_items:
            #     items_list = [item.strip() for item in match_items.group(1).split(",")]
            #     invoice_data["items"] = items_list

            # # Extract Quantities (Handles spaces correctly)
            # match_quantities = re.search(r"\*\*Quantities:\*\*\s*([\d,\s]+)", message)
            # if match_quantities:
            #     quantity_list = [qty.strip() for qty in match_quantities.group(1).split(",")]
            #     invoice_data["quantities"] = quantity_list

            items_pattern = r"\*\*Items\:\*\*\s*(.*?)\n"
            match_items = re.findall(items_pattern, message)
            if match_items:
                invoice_data["items"] = match_items[-1].strip()

            # Extract Quantities (take the last occurrence)
            quantity_pattern = r"\*\*Quantities\:\*\*\s*(.*?)\n"
            match_quantities = re.findall(quantity_pattern, message)
            if match_quantities:
                invoice_data["quantities"] = match_quantities[-1].strip()

            # Extract Invoice Cost (take the last occurrence)
            invoice_cost_pattern = r"\*\*Invoice Costs?\:\*\*\s*(.*?)\n"
            match_invoice_costs = re.findall(invoice_cost_pattern, message)
            if match_invoice_costs:
                invoice_data["invoiceCost"] = match_invoice_costs[-1].strip()

            # items_pattern = r"\*\*Items\:\*\*\s*(.*?)\n"
            # match_items = re.search(items_pattern, message)
            # if match_items:
            #     items_list = match_items.group(1).strip().split(", ")
            #     items_str = ", ".join(items_list)
            #     invoice_data["items"] = items_str

            # # Extract Quantities
            # quantity_pattern = r"\*\*Quantities\:\*\*\s*(.*?)\n"
            # match_quantity = re.search(quantity_pattern, message)
            # if match_quantity:
            #     quantity_list = match_quantity.group(1).strip().split(", ")
            #     quantity_str = ", ".join(quantity_list)
            #     invoice_data["quantities"] = quantity_str
            # quantity_pattern = r"\*\*Invoice Costs?\:\*\*\s*(.*?)\n"
            # match_quantity = re.search(quantity_pattern, message)
            # if match_quantity:
            #     quantity_list = match_quantity.group(1).strip().split(", ")
            #     quantity_str = ", ".join(quantity_list)
            #     invoice_data["invoiceCost"] = quantity_str

    return invoice_data
    
def collect_invoice_data1(chat_history):
    invoice_data = {}
    for message in chat_history:
        if "Bot: " in message:

            match_invoice_type = re.match(r".*invoice\s*type\s*:? ?(.*?)(?:\n|$)", message, re.IGNORECASE)
            if match_invoice_type:
                invoice_type = match_invoice_type.group(1).strip()
                invoice_data["Invoice type"] = invoice_type
            match_date = re.match(r".*date\s*\s*: ?(.*?)(?:,|$)", message, re.IGNORECASE)
            if match_date:
                invoice_data["Date"] = message.split(":")[2].strip()
            match_po_number = re.match(r".*po\s*number\s*: ?(.*?)(?:,|$)", message, re.IGNORECASE)
            if match_po_number:
                invoice_data["PO number"] = message.split(":")[2].strip()

            match_supplier_id =re.match(r".*supplier\s*id\s*: ?(.*?)(?:,|$)", message, re.IGNORECASE)
            if match_supplier_id:
                invoice_data["Supplier Id"] = message.split(":")[2].strip()

            match_total_amount = re.match(r".*total\s*amount\s*: ?(.*?)(?:,|$)", message, re.IGNORECASE)
            if match_total_amount :
                invoice_data["Total amount"] = message.split(":")[2].strip()

            match_total_tax = re.match(r".*total\s*tax\s*: ?(.*?)(?:,|$)", message, re.IGNORECASE)
            if match_total_tax:
                invoice_data["Total tax"] = message.split(":")[2].strip()
            items_pattern = r"Items: (.+)\n"
            match_items = re.search(items_pattern, message)
            if match_items:
                items_list = match_items.group(1).strip().split(", ")
                items_str = ", ".join(items_list)
                invoice_data["Items"] = items_str
            
            quantity_pattern = r"Quantities: (.+)\n"
            match_quantity = re.search(quantity_pattern, message)
            if match_quantity:
                quantity_list = match_quantity.group(1).strip().split(", ")
                quantity_str = ", ".join(quantity_list)
                invoice_data["Quantities"] = quantity_str
            # quantity_pattern = r"Quantity: (.+)\n"
            # match_quantity = re.search(quantity_pattern, message)
            # if match_quantity:
            #     quantity_list = match_quantity.group(1).strip().split(", ")
            #     quantity_str = ", ".join(quantity_list)
            #     invoice_data["Quantity"] = quantity_str

    return invoice_data

def checkValidation(params):
    return "dbsdcbxbcxb"

def run_conversation(data):

   # Step 1: send the conversation and available functions to the model

   messages = [{"role": "user", "content": data}]

   tools = [

       {

           "type": "function",

           "function": {

            "name": "checkValidation",

            "description": """To validate the all details like po number, item number and quantity from Database. If every thing is correct it will ask

            whether you to submit this detail or not.

            """,

            "parameters": {

                "type": "object",

                "properties": {

                    "bot_action": {

                        "type": "string",

                        "description": "Whether we want to create or fetch previous Inovice detail",

                    }

                },

                "required": ["bot_action"],

            },

        }

       }

 

   ]

   response = client.chat.completions.create(

       model="gpt-4-1106-preview",

       messages=messages,

       tools=tools,

       tool_choice="auto",  # auto is default, but we'll be explicit

   )

   response_message = response.choices[0].message

#    print(response_message)

   tool_calls = response_message.tool_calls

   # Step 2: check if the model wanted to call a function

   if tool_calls:

       # Step 3: call the function

       # Note: the JSON response may not always be valid; be sure to handle errors

       available_functions = {

            "checkValidation": checkValidation

       }  # only one function in this example, but you can have multiple

       messages.append(response_message)  # extend conversation with assistant's reply

       # Step 4: send the info for each function call and function response to the model

       for tool_call in tool_calls:

           function_name = tool_call.function.name

           function_to_call = available_functions[function_name]

           function_args = json.loads(tool_call.function.arguments)

           function_response = function_to_call(

              function_args

           )

           messages.append(

               {

                   "tool_call_id": tool_call.id,

                   "role": "tool",

                   "name": function_name,

                   "content": function_response,

               }

           )  # extend conversation with function response

       second_response = client.chat.completions.create(

           model="gpt-4-1106-preview",

           messages=messages,

       )  # get a new response from the model where it can see the function response

       return second_response.choices[0].message.content

reader = easyocr.Reader(["en"])  # Initialize EasyOCR for English

def extract_text_from_image(image: Image):
    """Extract text from an image using EasyOCR."""
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    result = reader.readtext(img_bytes, detail=0)
    return "\n".join(result)

def extract_text_from_pdf(pdf_bytes: bytes):
    """Extract text from a PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text if text.strip() else "No text extracted"

def extract_invoice_details(text):
    """Extract structured invoice details from text."""
    details = {
        "invoice type": None,
        "date": None,
        "po number": None,
        "supplier id": None,
        "total amount": None,
        "total tax": None,
        "items": [],
        "quantities": [],
        "invoiceCost": []
    }

    # Extract Invoice Type
    match = re.search(r'Invoice Type:\s*(\w+)', text, re.IGNORECASE)
    if match:
        details["invoice type"] = match.group(1).strip()

    # Extract Date
    match = re.search(r'(\d{1,2}\s+\w+\s+\d{4})', text)
    if match:
        details["date"] = match.group(1).strip()

    # Extract PO Number
    match = re.search(r'PO Number\.\s*(\S+)', text, re.IGNORECASE)
    if match:
        details["po number"] = match.group(1).strip()

    # Extract Supplier ID
    match = re.search(r'Supplier ID\.\s*(\S+)', text, re.IGNORECASE)
    if match:
        details["supplier id"] = match.group(1).strip()

    # Extract Total Amount
    match = re.findall(r'Total\s*[\n\s]*([\d,]+)', text, re.IGNORECASE)
    if match:
        details["total amount"] = match[-1].strip().replace(",", "")

    # Extract Total Tax
    match = re.search(r'Tax.*?(\d+)', text, re.IGNORECASE)
    if match:
        details["total tax"] = match.group(1).strip()

    # Extract Items
    item_matches = re.findall(r'(ITEM\d+)', text)
    details["items"] = ", ".join(item_matches) if item_matches else "Please enter manually"

    # Extract Quantities
    quantity_matches = re.findall(r'(ITEM\d+)\s+(\d+)', text)
    if quantity_matches:
        quantities = [q[1] for q in quantity_matches]  # Get only the first number after each item
        details["quantities"] = ", ".join(quantities)
    else:
        details["quantities"] = "Please enter manually"
    invoice_cost_matches = re.findall(r'(ITEM\d+)\s+\d+\s+(\d+)', text)
    if invoice_cost_matches:
        invoice_costs = [cost[1] for cost in invoice_cost_matches]  # Get only the invoice cost values
        details["invoiceCost"] = ", ".join(invoice_costs)
    else:
        details["invoiceCost"] = "Please enter manually"

    # Check for missing fields
    missing_fields = [key for key, value in details.items() if not value]
    if missing_fields:
        details["Missing Fields"] = f"Please fill manually: {', '.join(missing_fields)}"

    return details

def collect_invoice_data_from_file(chat_history):
    invoice_data = {}

    for message in chat_history:
            # Extract invoice type
        match_invoice_type = re.search(r"\*\*Invoice Type:\*\*\s*(.+)", message)
        if match_invoice_type:
            invoice_data["invoice type"] = match_invoice_type.group(1).strip()

        # Extract date
        match_date = re.search(r"\*\*Date:\*\*\s*(\d{2}/\d{2}/\d{4})", message)
        if match_date:
            invoice_data["date"] = match_date.group(1).strip()

        # Extract PO number
        match_po_number = re.search(r"\*\*PO Number:\*\*\s*(\w+)", message)
        if match_po_number:
            invoice_data["po number"] = match_po_number.group(1).strip()

        # Extract Supplier Id
        match_supplier_id = re.search(r"\*\*Supplier Id:\*\*\s*(\w+)", message)
        if match_supplier_id:
            invoice_data["supplier id"] = match_supplier_id.group(1).strip()

        # Extract Total Amount (handles commas)
        match_total_amount = re.search(r"\*\*Total Amount:\*\*\s*([\d,]+)", message)
        if match_total_amount:
            invoice_data["total amount"] = match_total_amount.group(1).strip()

        # Extract Total Tax
        match_total_tax = re.search(r"\*\*Total Tax:\*\*\s*([\d,]+)", message)
        if match_total_tax:
            invoice_data["total tax"] = match_total_tax.group(1).strip()

        # Extract Items (Supports multiple formats)
        match_items = re.search(r"\*\*Items:\*\*\s*([\w,\s]+)", message)
        if match_items:
            items_list = [item.strip() for item in match_items.group(1).split(",")]
            invoice_data["items"] = items_list

        # Extract Quantities (Handles spaces correctly)
        match_quantities = re.search(r"\*\*Quantities:\*\*\s*([\d,\s]+)", message)
        if match_quantities:
            quantity_list = [qty.strip() for qty in match_quantities.group(1).split(",")]
            invoice_data["quantities"] = quantity_list

    return invoice_data

async def extract_details_gpt_vision(file_path:str):
    """
    Extract structured information from various file types using GPT-4 and OCR.
    
    Args:
        file_path (str): Path to the input file 
        (supports .png, .jpg, .pdf, .docx, .txt, .csv, .xlsx, .xls)
    
    Returns:
        str: JSON-formatted extracted information
    """
    file_ext = Path(file_path).suffix.lower()
    
    # Process images using GPT-4 Vision
    if file_ext in ('.png', '.jpg', '.jpeg'):
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all relevant details from this image and return as JSON."},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/{file_ext[1:]};base64,{base64_image}"
                    }
                ]
            }],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        return response.choices[0].message.content

    # Process text-based and spreadsheet documents
    elif file_ext in ('.pdf', '.docx', '.txt', '.csv', '.xlsx', '.xls'):
        text = ""
        
        if file_ext == '.pdf':
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
            if not text.strip():
                images = convert_from_path(file_path)
                for img in images:
                    text += pytesseract.image_to_string(img)
        
        elif file_ext == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        
        elif file_ext == '.txt':
            with open(file_path, 'r') as f:
                text = f.read()
        
        elif file_ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                text = f"CSV Data:\nColumns: {', '.join(reader.fieldnames)}\n"
                for i, row in enumerate(reader, 1):
                    row_text = ', '.join([f"{k}: {v}" for k, v in row.items()])
                    text += f"Row {i}: {row_text}\n"
        
        elif file_ext in ('.xlsx', '.xls'):
            df_dict = pd.read_excel(file_path, sheet_name=None)
            text = "Excel Data:\n"
            for sheet_name, df in df_dict.items():
                text += f"\nSheet: {sheet_name}\n"
                text += df.to_string(index=False) + "\n"

        # Process extracted text with GPT-4
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{
                "role": "user",
                "content": f"""Extract structured information from this document. 
                Include key entities, dates, amounts, and relationships. 
                Return as JSON. Document content: {text[:15000]}"""  # Truncate to 15k chars
            }],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

async def extract_text_with_openai(file: UploadFile):
    file_content = await file.read()
    extracted_text = ""

    if file.content_type == "application/pdf":
        pdf_reader = PdfReader(io.BytesIO(file_content))
        extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    else:
        extracted_text = "This is an image file. Please describe its content."

    client = openai.AsyncOpenAI()  # Use AsyncOpenAI for async support
    
    response = await client.chat.completions.create(  # Correct async API call
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Extract key information from the provided document."},
            {"role": "user", "content": extracted_text},
        ]
    )
    
    return response.choices[0].message.content

async def categorize_invoice_details(extracted_text: str):
    client = openai.AsyncOpenAI()

    prompt = f"""
    Extract and structure the following details from this invoice text:
    - Invoice type (e.g., Merchandise invoice)
    - Date (dd/mm/yyyy format)
    - PO number (alphanumeric)
    - Supplier ID (alphanumeric only)
    - Total amount (numbers only)
    - Total tax (numbers only)
    - Items (alphanumeric, multiple allowed)
    - Quantities (numbers only, must match item count)
    - Invoice Cost (numbers only, must match item count)

    Invoice Text:
    {extracted_text}

    Provide the response as a JSON object.
    """

    response = await client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Extract structured data from invoices."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"}   # FIXED: Correct response format
    )

    structured_data = json.loads(response.choices[0].message.content)
    return structured_data

