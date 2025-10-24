# --- FastAPI Application ---
import json
import logging
import traceback
from typing import Dict,Tuple, Optional, Any, AsyncIterator
from fastapi import BackgroundTasks, FastAPI,Depends, Form,HTTPException,status
import uuid
from insightGeneration import generate_supplier_insights
from main import db_query_insights
from pydantic import BaseModel, EmailStr;
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from promo_llm import ChatRequestPromotion,stream_response_generator_promotion,app_runnable_promotion 
from promo_llm_agentic import app_runnable_promotion_agentic
from invoice_llm import ChatRequestInvoice,stream_response_generator_invoice,app_runnable_invoice
from po_llm import ChatRequestPurchaseOrder,stream_response_generator_purchase_order,app_runnable_purchase_order
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    AIMessageChunk, # Import AIMessageChunk for type checking stream
)
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from send_email import conf 
from models import Base, StoreDetails,User,PoDetails,PoHeader,InvHeader,InvDetails,Supplier,ShipmentHeader,ItemDiffs,ItemSupplier,ItemMaster,PromotionDetails,PromotionHeader
from schemas import ChatRequestUser, StoreDetailsSchema, UserSchema,poHeaderCreate,poDetailsCreate,invHeaderCreate,invDetailsCreate,poDetailsSearch,invDetailsSerach,ChatRequest,SupplierCreate,ShipmentHeader,ShipmentDetails,ItemDiffsSchema,ItemSupplierSchema,ItemMasterSchema,PromotionDetailsSchema,PromotionHeaderSchema
from schemas import ShipmentDetails as ShipmentDetailsSchema
from database import engine,SessionLocal,get_db
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from PIL import Image
from io import BytesIO
import models
from fastapi_mail import FastMail, MessageSchema, MessageType
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse

app = FastAPI(
    title="LangGraph Chatbot API",
    description="API endpoint for a LangChain chatbot using LangGraph, detail extraction, SQL generation, and streaming.",
)
#Common Functions:
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

#All Endpoints
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

# @app.get("/poDetails/{po_id}")
# def read_poDeatils(po_id: str, db: Session = Depends(get_db)):
#     po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
#     if po is None:
#         detail = "Po Number is not found in our database! Please add a valid PO number!"
#         conversation.append('Bot: ' + detail)
#         raise HTTPException(status_code=404, detail=conversation)
        
#     po_info = db.query(models.PoDetails).filter(models.PoDetails.poId == po_id).all()
#     return { "po_header":po,"po_details":po_info}

# @app.post("/invoiceValidation")
# def po_data_validations(po_id:str,detail:Dict[str,int],db: Session = Depends(get_db)):
#     po = db.query(models.PoHeader).filter(models.PoHeader.poNumber == po_id).first()
#     if po is None:
#         raise HTTPException(status_code=404, detail="PO is not found!")
#     for item,quantity in detail:
#         po_details = db.query(models.PoDetails).filter(models.PoDetails.itemId==item).first()
#         if po_details is None:
#             detail = "Item which you added is not present in this PO"
#             conversation.append('Bot: ' + detail)
#         raise HTTPException(status_code=404, detail=conversation)
#         if(po_details.itemQuantity>quantity):
#             detail = po_details.itemId + "quantity is excced according to PO quantity is" + po_details.itemQuantity
#             conversation.append('Bot: ' + detail)
#             raise HTTPException(status_code=404, detail=conversation)
#     return {"details":conversation}

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
    # returned_attributes =await categorize_promo_details_new(query, "string")
    # returned_attributes = fetch_store_id_from_query(query, db)
    # returned_attributes = query_database_function_promo(query, db)
    return {"message": "ok" }





@app.post("/clearData")
async def clearConversation():
    # previous_promo_details.clear()  
    # promo_states.clear()
    # user_promo_details.clear()
    # promo_email_cache.clear()

    # user_supplier_cache.clear() #initialize in case lead time logic needs to be implemented
    # chat_histories.clear()
    # user_po_details.clear()
    # po_email_cache.clear()

    # invoice_chat_histories.clear()
    # user_invoice_details.clear()
    # user_po_cache.clear()
    # invoice_email_cache.clear()
    # return {"conversation":conversation,"submissionStatus":"not submitted","chat_history":chat_histories}
    return {"message": "Cleared all conversation data."}

@app.post("/clearDataNew")
async def clearConversationNew(request: ChatRequestUser):
    # user_id = request.user_id
    # # email_value = previous_po_details[user_id].get("Email")
    # # previous_po_details[user_id].clear()
    # # if email_value is not None:
    # #     previous_po_details[user_id]["Email"] = email_value
    # print("clear data before:", chat_histories, previous_invoice_details, previous_promo_details)

    # # 1) --- Purchase-Order module ---
    # # chat_histories is a dict of lists
    # if user_id in chat_histories:
    #     chat_histories[user_id].clear()

    # # user_po_details is a dict of dicts
    # if user_id in user_po_details:
    #     po_email = user_po_details[user_id].get("Email")
    #     user_po_details[user_id].clear()
    #     if po_email is not None:
    #         user_po_details[user_id]["Email"] = po_email

    # # Also reset the per-user email cache if you want to force “pending” next time
    # po_email_cache[user_id].clear()


    # # 2) --- Promotion module ---
    # # promo_states is a dict of lists
    # if user_id in promo_states:
    #     promo_states[user_id].clear()

    # # user_promo_details is a dict of dicts
    # if user_id in user_promo_details:
    #     promo_email = user_promo_details[user_id].get("Email")
    #     user_promo_details[user_id].clear()
    #     if promo_email is not None:
    #         user_promo_details[user_id]["Email"] = promo_email

    # if user_id in promo_email_cache:
    #     promo_email_cache[user_id].clear()


    # # 3) --- Invoice module ---
    # # invoice_chat_histories is a dict of lists
    # if user_id in invoice_chat_histories:
    #     invoice_chat_histories[user_id].clear()

    # # user_invoice_details is a dict of dicts
    # if user_id in user_invoice_details:
    #     inv_email = user_invoice_details[user_id].get("Email")
    #     user_invoice_details[user_id].clear()
    #     if inv_email is not None:
    #         user_invoice_details[user_id]["Email"] = inv_email

    # # Reset PO-lookup cache & email cache for invoice
    # if user_id in user_po_cache:
    #     user_po_cache[user_id].clear()

    # if user_id in invoice_email_cache:
    #     invoice_email_cache[user_id].clear()

    # print("clear data after:", chat_histories, previous_invoice_details, previous_promo_details)
    # # print("email and po:",email_value,previous_po_details)
    # return {"conversation":conversation,"submissionStatus":"not submitted","chat_history":chat_histories}
    return {"message": "Cleared all conversation data."}



# @app.post("/uploadGpt/")
# async def upload_file(file: UploadFile = File(...)):
#     if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
#         raise HTTPException(status_code=400, detail="Unsupported file type")

#     extracted_text = await extract_text_with_openai(file)
#     structured_data = await categorize_invoice_details(extracted_text)

#     return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})

# @app.post("/uploadPo/")
# async def upload_file(file: UploadFile = File(...)):
#     if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
#         raise HTTPException(status_code=400, detail="Unsupported file type")

#     extracted_text = await extract_text_with_openai(file)
#     structured_data = await categorize_po_details(extracted_text,"admin")

#     return JSONResponse(content={"extracted_text": extracted_text, "structured_data": structured_data})
    
# @app.post("/uploadOpenAi/")
# async def upload_file(file: UploadFile = File(...)):
#     if file.content_type not in ["application/pdf", "image/png", "image/jpeg", "image/jpg"]:
#         raise HTTPException(status_code=400, detail="Unsupported file type")
    
#     extracted_text = extract_text_with_openai(file)
#     return JSONResponse(content={"extracted_text": extracted_text})

# @app.post("/upload/")
# async def upload_invoice(file: UploadFile = File(...)):
#     """API to upload an invoice file and extract details."""
#     if file.content_type not in ["image/png", "image/jpeg", "application/pdf", "text/plain"]:
#         raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, PDF, and TXT are supported.")

#     # Read file bytes
#     file_bytes = await file.read()

#     # Extract text based on file type
#     if file.content_type in ["image/png", "image/jpeg"]:
#         image = Image.open(BytesIO(file_bytes))
#         extracted_text = extract_text_from_image(image)
#     elif file.content_type == "application/pdf":
#         extracted_text = extract_text_from_pdf(file_bytes)
#     else:  # Text file
#         extracted_text = file_bytes.decode("utf-8")

#     # Process extracted text
#     invoice_details = extract_invoice_details(extracted_text)
#     invoice_data_from_conversation = {
#         "quantities": extract_invoice_details(extracted_text).get("quantities", []),
#         "items": extract_invoice_details(extracted_text).get("items", [])
#     }
#     # invoice_json=json.dumps(invoice_details)
#     # await generate_response(invoice_details)

#     return {"file_name": file.filename, "invoice_details": invoice_details,"invoice_data_from_conversation":invoice_data_from_conversation,"extracted_text":extracted_text}

# @app.post("/uploadPromo")  
# async def upload_promo(file: UploadFile = File(...)):
#     # Save the uploaded file to a temporary location
#     temp_file_path = f"temp_{file.filename}"
#     try:
#         with open(temp_file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # Call the extraction function
#         data = await extract_details_gpt_vision(temp_file_path)
#         result=await categorize_promo_details(data,"admin")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         # Remove the temporary file if it exists
#         if os.path.exists(temp_file_path):
#             os.remove(temp_file_path)
#     return {"structured_data": result}

 
#CHAT ENGINES:
#GENERAL
@app.post("/chat")
async def chat_endpoint(request: ChatRequestInvoice):
    """"""
#PROMOTION
# @app.post("/promotion_chat/")
# async def chat_endpoint_promotion(request: ChatRequestPromotion):
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

#         return StreamingResponse(
#             stream_response_generator_promotion(graph_stream), # Pass the graph stream to the generator
#             media_type="text/event-stream" # Use text/event-stream for Server-Sent Events
#             # media_type="text/plain" # Or application/jsonl if streaming JSON chunks
#         )

#     except Exception as e:
#         print(f"!!! ERROR invoking graph for thread {thread_id}: {e} !!!")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")

@app.post("/promotion_chat/")
async def chat_endpoint_promotion(request: ChatRequestPromotion):
    """
    Receives a ChatRequestPromotion containing a user message and optional thread_id.
    Invokes the LangGraph app for promotion chat, first running ainvoke() to log final outputs,
    then streams the LLM response back to the client using astream_events().
    The response is streamed as Server-Sent Events (SSE) with media_type "text/event-stream".
    Input: JSON body matching ChatRequestPromotion schema.
    Output: StreamingResponse of LLM-generated events.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n--- [Thread: {thread_id}] Received message: '{user_message}' ---")

    input_message = HumanMessage(content=user_message)
    input_state = {"messages": [input_message]}
    try:
        # -------------------------
        # Run graph and stream results to client (single run)
        # -------------------------
        graph_stream =app_runnable_promotion.astream_events(input_state, config)
        print(f"\n--- Graph stream: {graph_stream} ---")
        # Use your existing stream_response_generator_promotion which expects an async iterator of events.
        return StreamingResponse(
            stream_response_generator_promotion(graph_stream),
            media_type="text/event-stream"
        )

    except Exception as e:
        print(f"!!! ERROR invoking graph for thread {thread_id}: {e} !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")

@app.post("/promotion_chat_agentic/")
async def chat_endpoint_promotion_agentic(request: ChatRequestPromotion):
    """
    Receives a ChatRequestPromotion containing a user message and optional thread_id.
    Invokes the LangGraph app for promotion chat, first running ainvoke() to log final outputs,
    then streams the LLM response back to the client using astream_events().
    The response is streamed as Server-Sent Events (SSE) with media_type "text/event-stream".
    Input: JSON body matching ChatRequestPromotion schema.
    Output: StreamingResponse of LLM-generated events.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n--- [Thread: {thread_id}] Received message: '{user_message}' ---")

    input_message = HumanMessage(content=user_message)
    input_state = {"messages": [input_message]}
    try:
        # -------------------------
        # Run graph and stream results to client (single run)
        # -------------------------
        graph_stream =app_runnable_promotion_agentic.astream_events(input_state, config)
        print(f"\n--- Graph stream: {graph_stream} ---")
        # Use your existing stream_response_generator_promotion which expects an async iterator of events.
        return StreamingResponse(
            stream_response_generator_promotion(graph_stream),
            media_type="text/event-stream"
        )

    except Exception as e:
        print(f"!!! ERROR invoking graph for thread {thread_id}: {e} !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")

# @app.post("/promotion_extract_details/")
# async def get_extract_details_api(request: ChatRequestPromotion):
#     """
#     Run the graph for one turn and return only the extract_details output (no streaming).
#     """
#     user_message = request.message
#     thread_id = request.thread_id or str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}

#     print(f"\n--- [Thread: {thread_id}] Extract details request: '{user_message}' ---")

#     input_message = HumanMessage(content=user_message)
#     input_state = {"messages": [input_message]}

#     try:
#         # This will execute the graph fully and return the final state dict
#         final_state = await app_runnable_promotion.ainvoke(input_state, config)

#         # Depending on how your graph stores state, extract_details output
#         extract_node_data = final_state.get("extract_details")
#         print(f"--- Extract details from final state: {extract_node_data} ---")
#         user_intent = None
#         if "intent_classifier" in final_state:
#             user_intent = final_state["intent_classifier"].get("user_intent")
#             print(f"--- User intent from intent_classifier node: {user_intent} ---")
#         else:
#             user_intent = final_state.get("user_intent")
#             print(f"--- User intent from final state: {user_intent} ---")
#         if not extract_node_data:
#             # Sometimes node outputs are merged into final state keys
#             print("--- extract_details node output not found directly, checking final state keys ---",user_intent,final_state.get("extracted_details"),final_state)
#             extract_node_data = {
#                 "extracted_details": final_state.get("extracted_details"),
#                 "user_intent": user_intent
#             }

#         # Convert Pydantic models to dict
#         if hasattr(extract_node_data.get("extracted_details"), "dict"):
#             extract_node_data["extracted_details"] = extract_node_data["extracted_details"].dict()
#         if hasattr(extract_node_data.get("user_intent"), "dict"):
#             extract_node_data["user_intent"] = extract_node_data["user_intent"].dict()

#         return JSONResponse(content=extract_node_data)

#     except Exception as e:
#         print(f"!!! ERROR in extract_details API: {e} !!!")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e)) 

def _to_plain(obj):
    """Return a plain-serializable Python object from dict / pydantic v1/v2 / simple values."""
    if obj is None:
        return None
    # already a dict/list/primitive
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
    # Fallback: let FastAPI/jsonable_encoder try
    try:
        return jsonable_encoder(obj)
    except Exception:
        return str(obj)

# @app.post("/promotion_extract_details/")
# async def get_extract_details_api(request: ChatRequestPromotion):
#     """
#     Run the graph for one turn and return only the extract_details output (no streaming).
#     This instrumented version prints user_intent at many stages for debugging.
#     """
#     user_message = request.message
#     thread_id = request.thread_id or str(uuid.uuid4())
#     config = {"configurable": {"thread_id": thread_id}}

#     print(f"\n--- [Thread: {thread_id}] Extract details request: '{user_message}' ---")

#     input_message = HumanMessage(content=user_message)
#     input_state = {"messages": [input_message]}

#     try:
#         final_state = await app_runnable_promotion.ainvoke(input_state, config)
#         print("DEBUG: final_state keys:", list(final_state.keys()))
#         # Print repr for quick inspection (careful with large objects)
#         try:
#             print("DEBUG: final_state repr snippet:", repr(final_state)[:2000])
#         except Exception:
#             print("DEBUG: final_state repr: <unprintable>")

#         # Attempt to find user_intent in likely locations with debug prints
#         intent_candidates = {}
#         # 1) top-level 'user_intent'
#         top_user_intent = final_state.get("user_intent")
#         print("DEBUG: top-level final_state['user_intent']:", _to_plain(top_user_intent))
#         intent_candidates["top_user_intent"] = top_user_intent

#         # 2) top-level 'intent_classifier' block (if present)
#         intent_classifier = final_state.get("intent_classifier")
#         print("DEBUG: final_state['intent_classifier']:", _to_plain(intent_classifier))
#         if isinstance(intent_classifier, dict):
#             intent_candidates["intent_classifier.user_intent"] = intent_classifier.get("user_intent")

#         # 3) extract_details node output if present
#         extract_node_data = final_state.get("extract_details")
#         print("DEBUG: final_state['extract_details'] (raw):", _to_plain(extract_node_data))

#         if not extract_node_data:
#             # fallback: maybe outputs were flattened into top-level keys
#             extract_node_data = {
#                 "extracted_details": final_state.get("extracted_details"),
#                 "user_intent": final_state.get("user_intent") or (intent_classifier and intent_classifier.get("user_intent"))
#             }
#             print("DEBUG: constructed fallback extract_node_data:", _to_plain(extract_node_data))

#         # Normalize extract_node_data shape (could be pydantic model or dict)
#         if not isinstance(extract_node_data, dict):
#             try:
#                 extract_node_data_plain = _to_plain(extract_node_data)
#             except Exception:
#                 extract_node_data_plain = {"extracted_details": None, "user_intent": None}
#         else:
#             extract_node_data_plain = extract_node_data

#         print("DEBUG: extract_node_data_plain:", extract_node_data_plain)

#         # user_intent might itself be a Pydantic model; convert to plain
#         raw_user_intent = extract_node_data_plain.get("user_intent")
#         print("DEBUG: raw_user_intent (before conversion):", repr(raw_user_intent))
#         user_intent_plain = _to_plain(raw_user_intent)
#         print("DEBUG: user_intent_plain (after conversion):", user_intent_plain)

#         # extracted_details conversion
#         extracted_details_raw = extract_node_data_plain.get("extracted_details")
#         extracted_details_plain = _to_plain(extracted_details_raw)
#         print("DEBUG: extracted_details_plain:", extracted_details_plain)

#         payload = {
#             "extracted_details": extracted_details_plain,
#             "user_intent": user_intent_plain
#         }

#         print("DEBUG: Returning payload:", jsonable_encoder(payload))
#         return JSONResponse(content=payload)

#     except Exception as e:
#         print(f"!!! ERROR in extract_details API: {e} !!!")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/promotion_extract_details/")
async def get_extract_details_api(request: ChatRequestPromotion):
    """
    Retrieve extracted details from current conversation state.
    If message is empty, just return current state without processing.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- [Thread: {thread_id}] Extract details request: '{user_message}' ---")

    # If message is empty, just retrieve current state without invoking graph
    if not user_message or not user_message.strip():
        print("DEBUG: Empty message - retrieving current state only")
        try:
            # Get current state from checkpointer without invoking
            current_state = await app_runnable_promotion.aget_state(config)
            state_values = current_state.values if hasattr(current_state, 'values') else {}
            
            extracted_details_plain = _to_plain(state_values.get("extracted_details"))
            user_intent_plain = _to_plain(state_values.get("user_intent"))
            
            payload = {
                "extracted_details": extracted_details_plain,
                "user_intent": user_intent_plain
            }
            
            print("DEBUG: Returning current state payload:", jsonable_encoder(payload))
            return JSONResponse(content=payload)
            
        except Exception as e:
            print(f"!!! ERROR retrieving state: {e} !!!")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    # Original logic for non-empty messages
    input_message = HumanMessage(content=user_message)
    input_state = {"messages": [input_message]}

    try:
        final_state = await app_runnable_promotion.ainvoke(input_state, config)
        print("DEBUG: final_state keys:", list(final_state.keys()))
        # Print repr for quick inspection (careful with large objects)
        try:
            print("DEBUG: final_state repr snippet:", repr(final_state)[:2000])
        except Exception:
            print("DEBUG: final_state repr: <unprintable>")

        # Attempt to find user_intent in likely locations with debug prints
        intent_candidates = {}
        # 1) top-level 'user_intent'
        top_user_intent = final_state.get("user_intent")
        print("DEBUG: top-level final_state['user_intent']:", _to_plain(top_user_intent))
        intent_candidates["top_user_intent"] = top_user_intent

        # 2) top-level 'intent_classifier' block (if present)
        intent_classifier = final_state.get("intent_classifier")
        print("DEBUG: final_state['intent_classifier']:", _to_plain(intent_classifier))
        if isinstance(intent_classifier, dict):
            intent_candidates["intent_classifier.user_intent"] = intent_classifier.get("user_intent")

        # 3) extract_details node output if present
        extract_node_data = final_state.get("extract_details")
        print("DEBUG: final_state['extract_details'] (raw):", _to_plain(extract_node_data))

        if not extract_node_data:
            # fallback: maybe outputs were flattened into top-level keys
            extract_node_data = {
                "extracted_details": final_state.get("extracted_details"),
                "user_intent": final_state.get("user_intent") or (intent_classifier and intent_classifier.get("user_intent"))
            }
            print("DEBUG: constructed fallback extract_node_data:", _to_plain(extract_node_data))

        # Normalize extract_node_data shape (could be pydantic model or dict)
        if not isinstance(extract_node_data, dict):
            try:
                extract_node_data_plain = _to_plain(extract_node_data)
            except Exception:
                extract_node_data_plain = {"extracted_details": None, "user_intent": None}
        else:
            extract_node_data_plain = extract_node_data

        print("DEBUG: extract_node_data_plain:", extract_node_data_plain)

        # user_intent might itself be a Pydantic model; convert to plain
        raw_user_intent = extract_node_data_plain.get("user_intent")
        print("DEBUG: raw_user_intent (before conversion):", repr(raw_user_intent))
        user_intent_plain = _to_plain(raw_user_intent)
        print("DEBUG: user_intent_plain (after conversion):", user_intent_plain)

        # extracted_details conversion
        extracted_details_raw = extract_node_data_plain.get("extracted_details")
        extracted_details_plain = _to_plain(extracted_details_raw)
        print("DEBUG: extracted_details_plain:", extracted_details_plain)

        payload = {
            "extracted_details": extracted_details_plain,
            "user_intent": user_intent_plain
        }

        print("DEBUG: Returning payload:", jsonable_encoder(payload))
        return JSONResponse(content=payload)

    except Exception as e:
        print(f"!!! ERROR in extract_details API: {e} !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    # ... rest of your existing code ...

@app.post("/promotion_extract_details_agentic/")
async def get_extract_details_api_agentic(request: ChatRequestPromotion):
    """
    Retrieve extracted details from current conversation state.
    If message is empty, just return current state without processing.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- [Thread: {thread_id}] Extract details request: '{user_message}' ---")

    # If message is empty, just retrieve current state without invoking graph
    if not user_message or not user_message.strip():
        print("DEBUG: Empty message - retrieving current state only")
        try:
            # Get current state from checkpointer without invoking
            current_state = await app_runnable_promotion_agentic.aget_state(config)
            state_values = current_state.values if hasattr(current_state, 'values') else {}
            
            extracted_details_plain = _to_plain(state_values.get("extracted_details"))
            user_intent_plain = _to_plain(state_values.get("user_intent"))
            
            payload = {
                "extracted_details": extracted_details_plain,
                "user_intent": user_intent_plain
            }
            
            print("DEBUG: Returning current state payload:", jsonable_encoder(payload))
            return JSONResponse(content=payload)
            
        except Exception as e:
            print(f"!!! ERROR retrieving state: {e} !!!")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
    
    # Original logic for non-empty messages
    input_message = HumanMessage(content=user_message)
    input_state = {"messages": [input_message]}

    try:
        final_state = await app_runnable_promotion.ainvoke(input_state, config)
        print("DEBUG: final_state keys:", list(final_state.keys()))
        # Print repr for quick inspection (careful with large objects)
        try:
            print("DEBUG: final_state repr snippet:", repr(final_state)[:2000])
        except Exception:
            print("DEBUG: final_state repr: <unprintable>")

        # Attempt to find user_intent in likely locations with debug prints
        intent_candidates = {}
        # 1) top-level 'user_intent'
        top_user_intent = final_state.get("user_intent")
        print("DEBUG: top-level final_state['user_intent']:", _to_plain(top_user_intent))
        intent_candidates["top_user_intent"] = top_user_intent

        # 2) top-level 'intent_classifier' block (if present)
        intent_classifier = final_state.get("intent_classifier")
        print("DEBUG: final_state['intent_classifier']:", _to_plain(intent_classifier))
        if isinstance(intent_classifier, dict):
            intent_candidates["intent_classifier.user_intent"] = intent_classifier.get("user_intent")

        # 3) extract_details node output if present
        extract_node_data = final_state.get("extract_details")
        print("DEBUG: final_state['extract_details'] (raw):", _to_plain(extract_node_data))

        if not extract_node_data:
            # fallback: maybe outputs were flattened into top-level keys
            extract_node_data = {
                "extracted_details": final_state.get("extracted_details"),
                "user_intent": final_state.get("user_intent") or (intent_classifier and intent_classifier.get("user_intent"))
            }
            print("DEBUG: constructed fallback extract_node_data:", _to_plain(extract_node_data))

        # Normalize extract_node_data shape (could be pydantic model or dict)
        if not isinstance(extract_node_data, dict):
            try:
                extract_node_data_plain = _to_plain(extract_node_data)
            except Exception:
                extract_node_data_plain = {"extracted_details": None, "user_intent": None}
        else:
            extract_node_data_plain = extract_node_data

        print("DEBUG: extract_node_data_plain:", extract_node_data_plain)

        # user_intent might itself be a Pydantic model; convert to plain
        raw_user_intent = extract_node_data_plain.get("user_intent")
        print("DEBUG: raw_user_intent (before conversion):", repr(raw_user_intent))
        user_intent_plain = _to_plain(raw_user_intent)
        print("DEBUG: user_intent_plain (after conversion):", user_intent_plain)

        # extracted_details conversion
        extracted_details_raw = extract_node_data_plain.get("extracted_details")
        extracted_details_plain = _to_plain(extracted_details_raw)
        print("DEBUG: extracted_details_plain:", extracted_details_plain)

        payload = {
            "extracted_details": extracted_details_plain,
            "user_intent": user_intent_plain
        }

        print("DEBUG: Returning payload:", jsonable_encoder(payload))
        return JSONResponse(content=payload)

    except Exception as e:
        print(f"!!! ERROR in extract_details API: {e} !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    # ... rest of your existing code ...

#INVOICE
@app.post("/invoice_chat/")
async def chat_endpoint_invoice(request: ChatRequestInvoice):
    """
    Receives user message, invokes the LangGraph app, and streams the
    appropriate LLM response back to the client. Server-side processing
    (SQL, extraction) happens within the graph nodes.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n--- [Thread: {thread_id}] Received message: '{user_message}' ---")
    input_message = HumanMessage(content=user_message)
    input_state = {"messages": [input_message]} # Pass the new message in the list
    try:
        graph_stream = app_runnable_invoice.astream_events(input_state, config, version="v2")
        return StreamingResponse(
            stream_response_generator_invoice(graph_stream), # Pass the graph stream to the generator
            media_type="text/event-stream" # Use text/event-stream for Server-Sent Events
        )
    except Exception as e:
        print(f"!!! ERROR invoking graph for thread {thread_id}: {e} !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")

        # -------------------------
        # 2) Start a streaming run for the client
        # -------------------------
        streaming_graph_stream = app_runnable_promotion.astream_events(input_state, config, version="v2")

        # Use your existing stream_response_generator_promotion which expects an async iterator of events.
        return StreamingResponse(
            stream_response_generator_promotion(streaming_graph_stream),
            media_type="text/event-stream"
        )

    print(f"\n--- [Thread: {thread_id}] Received message: '{user_message}' ---")

    # Prepare input for the graph - the input is the list of messages
    input_message = HumanMessage(content=user_message)
    input_state = {"messages": [input_message]} # Pass the new message in the list

    try:
        # Use astream_events to get the stream of events from the graph execution
        graph_stream = app_runnable_purchase_order.astream_events(input_state, config, version="v2")

        # Return a StreamingResponse that iterates over the generator
        return StreamingResponse(
            stream_response_generator_purchase_order(graph_stream), # Pass the graph stream to the generator
            media_type="text/event-stream" # Use text/event-stream for Server-Sent Events
            # media_type="text/plain" # Or application/jsonl if streaming JSON chunks
        )

    except Exception as e:
        print(f"!!! ERROR invoking graph for thread {thread_id}: {e} !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")


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
