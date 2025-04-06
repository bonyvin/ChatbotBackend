from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os

# Load OpenAI API Key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key. Set OPENAI_API_KEY as an environment variable.")

# Initialize FastAPI app
app = FastAPI()

# Store user chat history & PO details (can be replaced with a database)
chat_histories = {}
user_po_details = {}

# Define request schema
class ChatRequest(BaseModel):
    user_id: str
    message: str

# Default PO structure (All fields set to null initially)
DEFAULT_PO_STRUCTURE = {
    "po_number": None,
    "supplier_id": None,
    "lead_time": None,
    "estimated_delivery_date": None,
    "total_quantity": None,
    "total_cost": None,
    "total_tax": None,
    "items": []  # List of items with details
}

# Define prompt template for structured PO creation
PROMPT_TEMPLATE = """
You are a Purchase Order (PO) assistant chatbot. Your job is to extract structured PO details from user messages.

### Required PO Fields:
1. *PO Number* (alphanumeric)
2. *Supplier ID* (alphanumeric)
3. *Lead Time* (in days)
4. *Estimated Delivery Date* (dd/mm/yyyy)
5. *Total Quantity* (numeric)
6. *Total Cost* (numeric)
7. *Total Tax* (numeric)
8. *Items* (List of dictionaries containing):
   - Item ID (alphanumeric)
   - Item Description (text)
   - Quantity (numeric)
   - Cost (numeric)

### Instructions:
- Extract available information from the user message.
- If any details are missing, set them as null.
- Return the output in *JSON format*.
- Ensure numerical values are properly formatted.
- If a user provides duplicate *Item IDs*, update the quantity & cost instead of adding duplicates.

### Example Output Format:
{
    "po_number": "PO12345",
    "supplier_id": "Supplier123",
    "lead_time": 10,
    "estimated_delivery_date": "12/06/2025",
    "total_quantity": 500,
    "total_cost": 15000,
    "total_tax": 1800,
    "items": [
        {"item_id": "ID123", "description": "Widget A", "quantity": 5, "cost": 500.00},
        {"item_id": "ID124", "description": "Widget B", "quantity": 10, "cost": 1000.00}
    ]
}
"""

@app.post("/chat")
async def chat_with_po_assistant(request: ChatRequest):
    """
    Handles chat requests and processes PO creation using OpenAI API.
    Returns a structured JSON object with missing values as null.
    """
    user_id = request.user_id
    user_message = request.message

    # Initialize chat history & PO details if not exists
    if user_id not in chat_histories:
        chat_histories[user_id] = []
        user_po_details[user_id] = DEFAULT_PO_STRUCTURE.copy()

    # Append user message to history
    chat_histories[user_id].append(f"User: {user_message}")

    # Create conversation history
    conversation = "\n".join(chat_histories[user_id])

    # Generate structured response from ChatGPT
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": PROMPT_TEMPLATE},
                {"role": "user", "content": conversation}
            ],
            temperature=0.7,
            max_tokens=500
        )

        bot_reply = response["choices"][0]["message"]["content"]

        # Convert ChatGPT response into JSON format
        import json
        try:
            structured_data = json.loads(bot_reply)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse response as JSON")

        # Update user PO details
        user_po_details[user_id] = structured_data

        # Append bot response to history
        chat_histories[user_id].append(f"Bot: {bot_reply}")

        return {"user_id": user_id, "po_details": structured_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

