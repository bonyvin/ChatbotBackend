import datetime
import os
import uuid
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse # Import StreamingResponse
from pydantic import AliasChoices, BaseModel, Field
from dotenv import load_dotenv
from typing import Dict, List, Optional, TypedDict, Annotated, Sequence, AsyncIterator # Typing for state, tools, and streaming
import operator
import traceback # For detailed error logging

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
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END,MessagesState,START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from fastapi.middleware.cors import CORSMiddleware
from llm_templates import template_Promotion_without_date
# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

print(f"LangSmith tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2') == 'true'}")
print(f"LangSmith project: {os.getenv('LANGCHAIN_PROJECT')}")

# --- Tool Definition ---
class ExtractedPromotionDetails(BaseModel):
    # Use validation_alias and AliasChoices for more flexible field naming in LLM output
    promotion_type: str | None = Field(None, validation_alias=AliasChoices('Promotion Type', 'promotion_type'))
    hierarchy_type: str | None = Field(None, validation_alias=AliasChoices('Hierarchy Type', 'hierarchy_type'))
    hierarchy_value: str | None = Field(None, validation_alias=AliasChoices('Hierarchy Value', 'hierarchy_value'))
    brand: str | None = Field(None, validation_alias=AliasChoices('Brand', 'brand'))
    items: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Items', 'items'))
    excluded_items: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Items', 'excluded_items'))
    discount_type: str | None = Field(None, validation_alias=AliasChoices('Discount Type', 'discount_type'))
    discount_value: str | None = Field(None, validation_alias=AliasChoices('Discount Value', 'discount_value')) # Keep as str to handle % vs fixed ambiguity
    start_date: str | None = Field(None, validation_alias=AliasChoices('Start Date', 'start_date'))
    end_date: str | None = Field(None, validation_alias=AliasChoices('End Date', 'end_date'))
    stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Stores', 'stores'))
    excluded_stores: List[str] = Field(default_factory=list, validation_alias=AliasChoices('Excluded Stores', 'excluded_stores'))

    class Config:
        populate_by_name = True # Allows using field names as well as aliases
        extra = 'ignore' # Ignore extra fields the LLM might hallucinate


# @tool(args_schema=PromotionDetails)
# def validate_promotion_details(
#     # ... (Tool function definition remains the same)
#     promotion_type: str,
#     discount_type: str,
#     discount_value: float,
#     start_date: str,
#     end_date: str,
#     stores_query: str,
#     hierarchy_level_type: str | None = None,
#     hierarchy_level_value: str | None = None,
#     brand: str | None = None,
#     items_query: str | None = None,
#     item_exclusions: str | None = None,
#     store_exclusions: str | None = None,
# ) -> str:
#     """
#     Validates the provided promotion details against the database.
#     Checks for conflicts, invalid SKUs/stores, date issues, etc.
#     Returns 'Valid' or a description of the validation error(s).
#     """
#     print(f"--- Tool Called: validate_promotion_details ---")
#     print(f"Received details: {locals()}")

#     # --- !!! Placeholder Database Logic !!! ---
#     errors = []
#     if items_query and "INVALID_SKU" in items_query:
#         errors.append("Invalid SKU detected in items query.")
#     if start_date == end_date:
#         errors.append("Start and end dates cannot be the same.")
#     # --- End Placeholder ---

#     if errors:
#         validation_result = f"Invalid: {'; '.join(errors)}"
#     else:
#         validation_result = "Valid"

#     print(f"Validation result: {validation_result}")
#     print(f"--- Tool Execution Finished ---")
#     return validation_result

# tools = [validate_promotion_details]

# --- LangChain/LangGraph Setup ---
# Ensure model is initialized with streaming=True
model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, streaming=True)
# model_with_tools = model.bind_tools(tools)

# Agent State definition remains the same
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- User's Prompt Template ---
today = datetime.datetime.today().strftime("%d/%m/%Y")

SYSTEM_PROMPT=template_Promotion_without_date.replace("{current_date}", today)

# 1. Initialize the Chat Model
model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# 2. Define the Prompt Template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 3. Define the LangGraph Workflow
workflow = StateGraph(MessagesState)

# Define the function that calls the model
async def call_model(state: MessagesState):
    """Invokes the LLM with the current state and system prompt."""
    messages = state.get("messages", [])
    if not isinstance(messages, list):
         print(f"Warning: state['messages'] was not a list: {type(messages)}. Resetting.")
         messages = []

    prompt_input = {"messages": messages}
    try:
        prompt = await prompt_template.ainvoke(prompt_input)
        # LangChain automatically traces this call to LangSmith if configured
        response = await model.ainvoke(prompt)
        return {"messages": [response]}
    except Exception as e:
        print(f"Error during model invocation: {e}")
        error_message = AIMessage(content=f"Sorry, an error occurred: {e}")
        
# Conditional edge logic remains the same
def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print("--- Decision: Route to Tools ---")
        return "call_tool"
    else:
        print("--- Decision: End Turn ---")
        return END

# Graph construction remains the same
# workflow = StateGraph(AgentState)
workflow.add_node("llm", call_model)
workflow.add_edge(START, "llm")
workflow.add_edge("llm", "__end__")

# Memory setup remains the same
memory = MemorySaver()

# Compile the graph
app_runnable = workflow.compile(checkpointer=memory)

# --- Detail Extractor Setup ---
extractor_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0) # Use low temp for extraction
# Bind the Pydantic model for structured output
extractor_llm_structured = extractor_llm.with_structured_output(
    ExtractedPromotionDetails,
    method="function_calling", # Or "json_mode" if preferred and model supports
    include_raw=False
)

async def extract_details_from_response(response_text: str) -> ExtractedPromotionDetails | None:
    """
    Uses a dedicated LLM call to extract structured promotion details
    from the main AI's response text.
    """
    print("\n--- Attempting Detail Extraction ---")
    if not response_text:
        print("--- No response text provided for extraction. Skipping. ---")
        return None

    # Simple instruction for the extractor LLM
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
You are an expert extraction system. Analyze the provided text, which is a response from a promotion creation assistant summarizing the current state of a promotion. Extract the promotion details mentioned into the structured format defined by the 'ExtractedPromotionDetails' function/tool.

- Today's date is {today}. Use this ONLY if needed to interpret relative dates mentioned IN THE TEXT (like "starts tomorrow"), standardizing to dd/mm/yyyy.
- Standardize date formats found in the text to dd/mm/yyyy.
- Extract lists for items, excluded items, stores, excluded stores.
- If a field is explicitly mentioned as 'Missing' or not present in the text, leave its value as null or empty list.
- Focus *only* on the details present in the text. Do not infer or add information not explicitly stated.
"""),
        ("human", "Extract promotion details from this text:\n\n```text\n{text_to_extract}\n```")
    ])

    # Create the extraction chain
    extraction_chain = extraction_prompt | extractor_llm_structured

    try:
        # Limit input text length for safety/cost
        max_len = 8000
        truncated_text = response_text[:max_len] + ("..." if len(response_text) > max_len else "")

        extracted_data = await extraction_chain.ainvoke({"text_to_extract": truncated_text})

        if extracted_data:
            print("--- Extraction Successful ---")
            return extracted_data
        else:
            print("--- Extraction returned no data. ---")
            return None

    except Exception as e:
        print(f"!!! ERROR during detail extraction: {e} !!!")
        traceback.print_exc()
        return None
    
# --- FastAPI Application ---
app = FastAPI(
    title="LangChain Chatbot API (Function Calling, Streaming with stream_mode='messages')",
    description="API endpoint for a LangChain chatbot using tools and streaming final message tokens.",
)
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
# --- API Request Model ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None

      
# --- Streaming Generator (Using astream_events) ---
async def stream_response_generator(app_runnable, input_state: dict, config: dict) -> AsyncIterator[str]:
    """
    Asynchronously streams LLM response chunks for the client using astream_events.
    """
    print("--- STREAM GENERATOR (for client) STARTED ---")
    full_response = ""
    try:
        async for event in app_runnable.astream_events(input_state, config, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    content_to_yield = chunk.content
                    full_response += content_to_yield
                    yield content_to_yield

            # Optional: Log end event
            elif kind == "on_chain_end":
                 pass # print(f"Stream Chain Ended for config: {config}")

    except Exception as e:
        print(f"!!! ERROR in stream_response_generator: {e} !!!")
        traceback.print_exc()
        yield f"\n\nStream error: {e}"
    finally:
        print(f"--- STREAM GENERATOR (for client) FINISHED ---")
        # print(f"--- Full response streamed to client ({config['configurable']['thread_id']}):\n{full_response}\n---")

@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    """
    Receives user message, gets AI response, performs server-side detail extraction,
    and streams the AI response back to the client.
    """
    user_message = request.message
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Prepare input for the graph
    input_message = HumanMessage(content=user_message)
    # Get current state to maintain conversation history
    # Use aget_state to fetch the history associated with the thread_id
    try:
        current_state = await app_runnable.aget_state(config)
        current_messages = current_state.values.get("messages", []) if current_state else []
        if not isinstance(current_messages, list):
             print(f"Warning: current_state messages was not a list: {type(current_messages)}. Resetting.")
             current_messages = []
    except Exception as e:
        print(f"Warning: Could not retrieve state for thread {thread_id}. Starting new history. Error: {e}")
        current_messages = []

    # Append the new user message to the history
    input_state = {"messages": current_messages + [input_message]}

    # --- Server-Side Extraction Step ---
    extracted_details_dict: Dict | None = None
    ai_response_text = ""
    try:
        print(f"\n--- [Thread: {thread_id}] Invoking main graph for full response (for extraction) ---")
        # Use ainvoke to get the complete final state *without* streaming
        final_state = await app_runnable.ainvoke(input_state, config)

        if final_state and final_state.get("messages"):
            final_messages = final_state["messages"]
            if isinstance(final_messages, list) and final_messages:
                # The last message in the state after invoke should be the AI's response
                last_message = final_messages[-1]
                if isinstance(last_message, AIMessage):
                    ai_response_text = last_message.content
                    print(f"--- [Thread: {thread_id}] Full AI response obtained for extraction (length: {len(ai_response_text)}) ---")

                    # Perform extraction on the full response text
                    extracted_details_obj = await extract_details_from_response(ai_response_text)

                    if extracted_details_obj:
                        # Convert Pydantic model to dict for logging/potential future use
                        extracted_details_dict = extracted_details_obj.model_dump(by_alias=False) # Use field names for logging
                        print("--- [Thread: {thread_id}] Extracted Details (Server-Side Log) ---")
                        print(json.dumps(extracted_details_dict, indent=2))
                        print("-------------------------------------------------------------")
                    else:
                        print(f"--- [Thread: {thread_id}] Detail extraction failed or returned no data. ---")
                else:
                     print(f"--- [Thread: {thread_id}] Last message in final_state was not AIMessage ({type(last_message)}). Cannot extract. ---")
            else:
                 print(f"--- [Thread: {thread_id}] No messages or invalid format in final_state. Cannot extract. ---")
        else:
            print(f"--- [Thread: {thread_id}] Failed to get final state from ainvoke. Cannot extract. ---")

    except Exception as e:
        print(f"!!! [Thread: {thread_id}] ERROR during invoke/extraction phase: {e} !!!")
        traceback.print_exc()
        # Decide if you want to raise an error or continue to streaming
        # raise HTTPException(status_code=500, detail=f"Error during processing: {e}") # Option 1: Fail fast
        # --- Streaming Step ---
    # Now, initiate the streaming response for the client.
    # This uses the *same* input_state, causing the graph to run again,
    # but this time the output is streamed via the generator.
    try:
        print(f"--- [Thread: {thread_id}] Initiating streaming response for client ---")
        return StreamingResponse(
            stream_response_generator(app_runnable, input_state, config),
            media_type="text/plain"
            # Note: The extracted_details_dict is logged server-side.
            # Sending it to the client alongside the stream requires a different API design
            # (e.g., WebSockets, multipart response, separate endpoint).
        )
    except Exception as e:
        print(f"!!! [Thread: {thread_id}] Error setting up stream: {e} !!!")
        traceback.print_exc()
        # If invoke/extraction succeeded but streaming fails, we might still want to inform the client
        raise HTTPException(status_code=500, detail=f"Internal Server Error during streaming setup: {e}")



# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "Welcome to the LangChain Chatbot API (Function Calling & Streaming Enabled)!"}

# --- To Run ---
# Ensure .env file has keys.
# Ensure requirements are installed (no changes needed from previous version).
# Run in terminal: uvicorn main:app --reload
