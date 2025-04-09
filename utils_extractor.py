import os
import datetime
import openai
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
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
from langchain import LLMChain
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
from langchain import PromptTemplate
import pandas as pd
import re
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

HF_TOKEN = os.getenv('HF_TOKEN')

data = "Invoice type: Debit Note Date: 26/06/2024 PO number: PO123 Supplier Id: SUP1123 Total amount: 6700 Total tax: 342 Items: ID123, ID124 Quantity: 2, 4"
def checkValidation(invoice_detail):
    return json.dumps({"invoice_detail":invoice_detail})
    

def run_conversation(data,db: Session = Depends(get_db)):

   client = OpenAI(api_key=OPENAI_API_KEY)

   # Step 1: send the conversation and available functions to the model

   messages = [{"role": "user", "content": data}]

#    tools = [

#        {

#            "type": "function",

#            "function": {

#             "name": "checkValidation",

#             "description": """To validate the all details like po number, item number and quantity from Database. If every thing is correct it will ask

#             whether you to submit this detail or not.

#             """,

#             "parameters": {

#                 "type": "object",

#                 "properties": {

#                     "invoicetype": {

#                         "type": "string",

#                         "description": "Type of invoice e.g Merchandise Invoice",

#                     },

#                     "datetime": {

#                         "type": "string",

#                         "description": "Data in which we want to create an invoice",

#                     },

#                     "ponumber": {

#                         "type": "string",

#                         "description": """po number is alphanumeric value

#                             """

#                     },

#                     "totalamount": {

#                         "type": "string",

#                         "description": """Total amount of the invoice order

#                             """

#                     },

#                      "totaltax": {

#                         "type": "string",

#                         "description": """Total tax of the invoice order

#                             """

#                     },
#                      "supplierid": {

#                         "type": "string",

#                         "description": """Supplier Id of the invoice order

#                             """

#                     },

#                      "items": {

#                         "type": "string",

#                         "description": """It can multiple items using comman separated eg ID123,ID124

#                             """

#                     },

#                      "quantity": {

#                         "type": "string",

#                         "description": """It is the quantity associated to each items eg 10,10

#                             """

#                     },

#                 },

#                 # "required": [],
#                 # "required": ["invoicetype","datetime","ponumber","totalamount","total_tax","supplier_id","items","quantity"],

#             },

#         }

#        }

 

#    ]

#    response = client.chat.completions.create(

#        model="gpt-4-1106-preview",

#        messages=messages,

#        tools=tools,

#        tool_choice="auto",  # auto is default, but we'll be explicit

#    )

   tools = [
        { 
            "type": "function",
            "function": {
                "name": "checkValidation",
                "description": """To validate all details like PO number, item number, and quantity from Database. If everything is correct, it will ask whether you want to submit this detail or not.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "invoicetype": {
                            "type": "string",
                            "description": "Type of invoice e.g Merchandise Invoice"
                        },
                        "datetime": {
                            "type": "string",
                            "description": "Date on which we want to create an invoice"
                        },
                        "ponumber": {
                            "type": "string",
                            "description": "PO number is alphanumeric value"
                        },
                        "totalamount": {
                            "type": "string",
                            "description": "Total amount of the invoice order"
                        },
                        "totaltax": {
                            "type": "string",
                            "description": "Total tax of the invoice order"
                        },
                        "supplierid": {
                            "type": "string",
                            "description": "Supplier ID of the invoice order"
                        },
                        "items": {
                            "type": "string",
                            "description": "It can include multiple items using comma separated values e.g. ID123,ID124"
                        },
                        "quantity": {
                            "type": "string",
                            "description": "Quantity associated with each item e.g. 10,10"
                        }
                    }
                }
            }
        }
    ]

   def checkValidation(invoicetype=None, datetime=None, ponumber=None, totalamount=None, totaltax=None, supplierid=None, items=None, quantity=None,db: Session = Depends(get_db)):
        # po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == ponumber).first()
        details = json.loads(json.dumps(invoicetype))
        # print("Po Number",details["ponumber"])
        # findPoDetails(details["ponumber"])
        
        # if details["ponumber"]=='PO123':
        #     return "Po is not found!"
        # else:
        #     return "Po is found!"
        # result = {}
        # if invoicetype is not None:
        #     result['invoicetype'] = invoicetype
        # if datetime is not None:
        #     result['datetime'] = datetime
        # if ponumber is not None:
        #     result['ponumber'] = ponumber
        # if totalamount is not None:
        #     result['totalamount'] = totalamount
        # if totaltax is not None:
        #     result['totaltax'] = totaltax
        # if supplierid is not None:
        #     result['supplierid'] = supplierid
        # if items is not None:
        #     result['items'] = items
        # if quantity is not None:
        #     result['quantity'] = quantity
        # return result

    # Example usage
   response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )

   response_message = response.choices[0].message

   print(response_message)

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
        
       print("Extractor info",messages[2]['content'])
    #    second_response = client.chat.completions.create(

    #        model="gpt-4-1106-preview",

    #        messages=messages,

    #    )  # get a new response from the model where it can see the function response

    #    return second_response.choices[0].message
       return messages[2]['content']

    

   

print(run_conversation(""))
