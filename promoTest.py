import json
import logging
from fastapi import FastAPI, HTTPException
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import StructuredTool
from typing import Dict, List, Optional, Tuple
from database import engine,SessionLocal,get_db
from sqlalchemy.orm import Session
from database import get_db
from sqlalchemy.sql import text
from utils import gpt_response,extract_information,conversation,checkValidation,run_conversation,collect_invoice_data,bot_action,openaifunction,testModel,test_submission,extract_invoice_details,extract_text_from_pdf,extract_text_from_image,collect_invoice_data_from_file,extract_text_with_openai,categorize_invoice_details,client,categorize_invoice_details_new,previous_invoice_details,template_5,extract_details_gpt_vision,client_new,llm_gpt3,llm_gpt4,async_client 
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from typing import Dict, Optional
from promoUtils import template_Promotion, categorize_promo_details_fun_call


app = FastAPI()

# Global Variables
promo_states = {}
user_promo_details = {}

DEFAULT_PROMO_STRUCTURE = {
    "Hierarchy Type": None,
    "Hierarchy Value": None,
    "Brand": None,
    "Discount Type": None,
    "Discount Value": None,
    "Start Date": None,
    "End Date": None,
    "Promotion Type": None,
    "Excluded Item List": [],
    "Items": [],
    "Excluded Location List": [],
    "Stores": []
}

# Initialize LangChain Model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# ✅ **Fix: Query Classification Using RunnablePipeline**
def classify_query(user_message: str, conversation_context: str) -> str:
    """Classifies user queries as 'General' or 'Not General' based on promotion details."""
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a query classifier."),
        HumanMessage(
            content=(
                "Classify the following query based on the user's message and trimmed conversation context."
                "If the query contains any promotion-related fields like 'Brand', 'Discount Type', 'Start Date', etc., classify it as 'Not General'."
                "Otherwise, classify it as 'General'.\n\n"
                "**Context:** {context}\n"
                "**User Query:** {message}\n\n"
                "Respond in JSON format as: {\"query_type\": \"General\"} or {\"query_type\": \"Not General\"}"
            )
        )
    ])

    classify_chain = prompt_template | llm | JsonOutputParser()

    try:
        response = classify_chain.invoke({"context": conversation_context, "message": user_message})
        return response.get("query_type", "General")
    except Exception as e:
        logging.error(f"Error in classify_query: {e}")
        return "General"

# ✅ **Fix: Function Tool for Database Query Execution**
def query_database_tool(question: str) -> str:
    """Executes a database query and returns the result."""
    db_session = get_db()
    return query_database_function_promo(question, db_session)

query_database_function = StructuredTool.from_function(
    func=query_database_tool,
    name="query_database",
    description="Executes database queries to fetch relevant information."
)

# ✅ **Fix: Use `RunnablePipeline` for Chat Processing**
async def handle_promotion_chat_promo_test(request: Dict):
    """Handles user chat related to promotions."""
    
    user_id = request["user_id"]
    user_message = request["message"]

    # Maintain user session
    if user_id not in promo_states:
        promo_states[user_id] = []
        user_promo_details[user_id] = DEFAULT_PROMO_STRUCTURE.copy()

    promo_states[user_id].append(f"User: {user_message}")
    conversation = "\n".join(promo_states[user_id])

    try:
        # ✅ Fix: Use `RunnablePipeline`
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=template_Promotion),
            HumanMessage(content="{conversation}")
        ])

        chat_chain = prompt_template | llm

        # First response
        response_message = chat_chain.invoke({"conversation": conversation})

        # ✅ Fix: Ensure response is handled properly
        if isinstance(response_message, AIMessage):
            response_message = response_message.content  # Extract content from AIMessage

        function_call = None
        query_called = False

        # Checking function call support
        if isinstance(response_message, dict):
            function_call = response_message.get("function_call")

        if function_call and function_call.get("name") == "query_database":
            query_called = True
            db_response = query_database_function.invoke(function_call["arguments"]["question"])

            # ✅ Fix: RunnablePipeline for Second API Call
            messages = [
                SystemMessage(content=template_Promotion),
                HumanMessage(content=conversation),
                SystemMessage(content=f"Database Response: {db_response}")
            ]

            second_chat_chain = ChatPromptTemplate.from_messages(messages) | llm
            second_response = second_chat_chain.invoke({"conversation": conversation})

            if isinstance(second_response, AIMessage):
                response_message = second_response.content
            else:
                response_message = second_response  # Fallback in case it's a string

        # Categorize Promotion Details
        user_promo_details[user_id] = await categorize_promo_details_fun_call(response_message, user_id)
        promo_json = user_promo_details[user_id]

        # Classify query
        classification_result = classify_query(user_message, conversation)

        # Append bot response to chat history
        promo_states[user_id].append(f"Bot: {response_message}")

        # Determine Submission Status
        if "Would you like to submit" in response_message:
            submission_status = "pending"
        elif "Promotion created successfully" in response_message:
            submission_status = "submitted"
        elif "I want to change something" in response_message:
            submission_status = "cancelled"
        else:
            submission_status = "in_progress"

        return {
            "user_id": user_id,
            "bot_reply": response_message,
            "chat_history": promo_states,
            "promo_json": promo_json,
            "submissionStatus": submission_status,
            "query_called": query_called,
            "classification_result": classification_result
        }

    except Exception as e:
        logging.error(f"Error in handle_promotion_chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def query_database_function_promo(question: str, db: Session) -> str:
    # Use the optimized function to extract attribute details (e.g., color, size, or material)
    attr, value, attr_id = find_attributes(question)
    tables = ["itemmaster", "itemsupplier", "itemdiffs"]
    tables_str = ", ".join(tables)
    # If an attribute filter is detected, prepare an optimization hint to use direct equality.
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
    I have 3 tables in my database, namely:
    - `itemmaster` (aliased as `im`)  
    - `itemsupplier` (aliased as `isup`)  
    - `itemdiffs` (aliased as `idf`)  

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
        - **`itemId`**: Foreign key linking to `itemsMaster.itemId`, establishing the relationship between items and suppliers.   
    - `itemdiffs` (For Attribute Filtering):
        - **`id`**: Unique identifier for each differentiation type.  
        - **`diffType`**: The attribute type used to differentiate items (e.g., color, size, material).  
          - If the user wants to filter items by a specific attribute (e.g., color or size), the query should check `diffType` and retrieve the corresponding `diffId`.  
        - **`diffId`**: The actual differentiation value corresponding to `diffType`.  
          - Example: If `diffType = 'color'`, then `diffId` could be "Red"; if `diffType = 'size'`, then `diffId` could be "M".

    #### **3. Query Format & Execution**  
    - **Start queries from `itemmaster`** as the primary table.
    - Use **explicit JOINs** when needed (e.g., joining `itemsupplier` for cost-related queries).
    - **Return only valid SQL queries** without additional explanations or markdown formatting.
    - The generated SQL query should be fully structured and dynamically adapt to the user query.
    {optimization_hint}
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
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": promotion_sql_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.5,
        max_tokens=150
    )
    print("Query response:", response)
    mysql_query = response.choices[0].message.content.strip()
    mysql_query = mysql_query.replace("```sql", "").replace("```", "").strip()

    # Validate the generated SQL query
    if not mysql_query.lower().startswith(("select", "show")):
        return "Error: Generated response is not a valid SQL query."
    
    try:
        result = execute_mysql_query(mysql_query)
        print("Query result:", result)
        return json.dumps(result, default=str)
    except Exception as e:
        print("Query error:", {str(e)})
        return f"Error executing query: {str(e)}"
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

# Optimized core functions
def find_attributes(question: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Find attributes, values, and their ID in a single optimized flow."""
    print(f"\nProcessing query: '{question}'")
    
    # Get attribute metadata in single query
    attr_data = db_query(
        "SELECT diffType, diffId, id FROM itemdiffs GROUP BY diffType, diffId"
    )
    
    if not attr_data:
        print("No attribute data found in database")
        return (None, None, None)

    # Build optimized attribute structure {diffType: {diffId: id}}
    attribute_map = {}
    for diff_type, diff_id, id_val in attr_data:
        attribute_map.setdefault(diff_type, {})[diff_id] = id_val

    print(f"Attribute map: {attribute_map.keys()}")
    
    # Detect attributes using optimized function
    attr, value = detect_attribute(question, attribute_map)
    if not attr or not value:
        return (None, None, None)

    # Get ID from pre-loaded map
    item_id = attribute_map.get(attr, {}).get(value)
    return (attr, value, item_id)

def detect_attribute(question: str, attribute_map: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Optimized attribute detection with caching."""
    if not attribute_map:
        return (None, None)

    try:
        # Build prompt from pre-loaded data
        attr_desc = "\n".join(
            f"- {attr}: {', '.join(values)}" 
            for attr, values in attribute_map.items()
        )

        prompt = f"""Analyze query: "{question}"
        Identify attributes from:
        {attr_desc}
        Respond in JSON format: {{"attribute": "...", "value": "..."}}"""

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
        attr = result.get('attribute')
        value = result.get('value')

        # Validate against known data
        if not attribute_map.get(attr, {}).get(value):
            print(f"Invalid detection: {attr}/{value}")
            return (None, None)

        return (attr, value)

    except Exception as e:
        print(f"Detection error: {e}")
        return (None, None)
