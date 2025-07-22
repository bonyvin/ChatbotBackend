# --- FastAPI Application ---
import traceback
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from promo_llm import ChatRequestPromotion,stream_response_generator_promotion,app_runnable_promotion,field_details
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

app = FastAPI(
    title="LangGraph Chatbot API",
    description="API endpoint for a LangChain chatbot using LangGraph, detail extraction, SQL generation, and streaming.",
)

#GENERAL
@app.post("/chat")
async def chat_endpoint(request: ChatRequestInvoice):
    """"""
#PROMOTION
@app.post("/promotion_chat/")
async def chat_endpoint_promotion(request: ChatRequestPromotion):
    """
    Receives user message, invokes the LangGraph app, and streams the
    appropriate LLM response back to the client. Server-side processing
    (SQL, extraction) happens within the graph nodes.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n--- [Thread: {thread_id}] Received message: '{user_message}' ---")

    # Prepare input for the graph - the input is the list of messages
    input_message = HumanMessage(content=user_message)
    input_state = {"messages": [input_message]} # Pass the new message in the list
    try:
        # Use astream_events to get the stream of events from the graph execution
        graph_stream = app_runnable_promotion.astream_events(input_state, config, version="v2")
        # Return a StreamingResponse that iterates over the generator
        return StreamingResponse(
            stream_response_generator_promotion(graph_stream), # Pass the graph stream to the generator
            media_type="text/event-stream" # Use text/event-stream for Server-Sent Events
            # media_type="text/plain" # Or application/jsonl if streaming JSON chunks
        )

    except Exception as e:
        print(f"!!! ERROR invoking graph for thread {thread_id}: {e} !!!")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing chat: {e}")

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

@app.post("/purchase_order_chat/")
async def chat_endpoint_purchase_order(request: ChatRequestPurchaseOrder):
    """
    Receives user message, invokes the LangGraph app, and streams the
    appropriate LLM response back to the client. Server-side processing
    (SQL, extraction) happens within the graph nodes.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

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
