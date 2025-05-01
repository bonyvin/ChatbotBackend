import base64
from datetime import datetime
import io
import logging
from typing import List, Tuple, Dict, Any, Callable
import os

from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI # Needed for API key example
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
    AIMessageChunk, # Import AIMessageChunk for type checking stream
)
import plotly.graph_objects as go
import plotly.io as pio

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Assume Langchain components are installed and imported ---
# Example (you'll need to install langchain, langchain-openai etc.):
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.schema import HumanMessage

# --- Calculation functions (calculate_fill_rate, etc.) ---
# (Keep these functions as they were defined in your previous code)
def calculate_fill_rate(po_ordered: float, delivered: float) -> Tuple[float, float]:
    """Calculates the fill rate as delivered percentage and pending percentage."""
    if po_ordered == 0:
        return 100.0, 0.0 # If nothing ordered, assume 100% fill rate (nothing pending)
    # Ensure inputs are floats for division
    po_ordered = float(po_ordered)
    delivered = float(delivered)
    fill_rate = min(100.0, (delivered / po_ordered) * 100) # Cap fill rate at 100%
    pending_rate = max(0.0, 100.0 - fill_rate) # Ensure pending rate isn't negative
    return fill_rate, pending_rate

def calculate_quality_metrics(total_received: int, total_damaged: int) -> Tuple[float, float]:
    """
    Calculates the non-defective and defective item percentage rates.
    Returns (non_defective_rate, defective_rate).
    """
    if total_received == 0:
        return 100.0, 0.0 # If nothing received, assume 0% defective
    # Ensure inputs are floats for division
    total_received = float(total_received)
    total_damaged = float(total_damaged)
    # Ensure damaged doesn't exceed received
    total_damaged = min(total_damaged, total_received)
    non_defective = total_received - total_damaged
    non_defective_rate = (non_defective / total_received) * 100
    defective_rate = 100.0 - non_defective_rate
    return non_defective_rate, defective_rate

def compute_risk_score(fill_rate: float, avg_delay: float, defective_rate: float) -> float:
    """
    Computes a composite risk score based on fill rate, average delay, and defective rate.
    Higher scores indicate higher risk. Score ranges roughly 0-5+.
    Weights can be adjusted based on business priority.
    """
    fill_rate = float(fill_rate)
    avg_delay = float(avg_delay)
    defective_rate = float(defective_rate)

    # --- Risk Component Calculation ---
    # Fill Rate Component: Higher risk for lower fill rate. Score 0-5.
    # 100% fill rate = 0 risk points. 0% fill rate = 5 risk points.
    fill_component = max(0, 5 - (fill_rate / 20.0))

    # Delay Component: Higher risk for longer delays. Score potentially > 5.
    # 0 days delay = 0 risk points. 20 days delay = 2 risk points.
    delay_component = max(0, avg_delay / 10.0) # Normalize delay, e.g., 1 point per 10 days

    # Quality Component: Higher risk for higher defective rate. Score 0-5.
    # 0% defective = 0 risk points. 100% defective = 5 risk points.
    quality_component = max(0, defective_rate / 20.0)

    # --- Weighted Sum ---
    # Example weights: Fill Rate (40%), Delay (30%), Quality (30%)
    risk_score = (fill_component * 0.4) + (delay_component * 0.3) + (quality_component * 0.3)

    # Clamp score to a practical range like 0-5 or 0-10 if desired, though raw score can be informative.
    # For this example, we'll round but not clamp strictly.
    return round(risk_score, 1)


# Type alias for clarity
DbQueryFunc = Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]

logging.basicConfig(level=logging.INFO) # Configure logging

# --- Database interaction functions ---
# (Keep get_total_ordered, get_total_delivered, get_quality_metrics_from_db)
# (Use the corrected get_average_delay_for_supplier from above)
def get_total_ordered(supplier_id: str, db_query_func: DbQueryFunc) -> int:
    logging.info(f"Fetching total ordered for supplier_id: '{supplier_id}'")
    query = """
    SELECT SUM(itemQuantity) AS total_ordered
    FROM podetails
    WHERE supplierId = :supplierId
    """
    params = {'supplierId': supplier_id}
    results = db_query_func(query, params)
    if results and results[0].get('total_ordered') is not None:
        total = results[0]['total_ordered']
        try:
            return int(float(total))
        except (ValueError, TypeError):
            logging.error(f"Could not convert total_ordered value '{total}' to int.")
            return 0
    else:
        logging.info(f"No results or NULL sum found for total_ordered, supplier_id: '{supplier_id}'")
        return 0

def get_total_delivered(supplier_id: str, db_query_func: DbQueryFunc) -> int:
    logging.info(f"Fetching total delivered for supplier_id: '{supplier_id}'")
    query = """
    SELECT SUM(sd.receivedItemQuantity) AS delivered_items
    FROM shipmentdetails sd
    JOIN shipmentheader sh ON sd.receiptId = sh.receiptId
    WHERE sh.poId IN (
        SELECT DISTINCT poId
        FROM podetails
        WHERE supplierId = :supplierId
    )
    """
    params = {'supplierId': supplier_id}
    results = db_query_func(query, params)
    if results and results[0].get('delivered_items') is not None:
        total = results[0]['delivered_items']
        try:
            return int(float(total))
        except (ValueError, TypeError):
            logging.error(f"Could not convert delivered_items value '{total}' to int.")
            return 0
    else:
        logging.info(f"No results or NULL sum found for total_delivered, supplier_id: '{supplier_id}'")
        return 0

def get_average_delay_for_supplier(supplier_id: str, db_query_func: DbQueryFunc) -> float:
    # Corrected version from above
    logging.info(f"Fetching average delay for supplier_id: '{supplier_id}'")
    query = """
    SELECT AVG(DATEDIFF(sh.receivedDate, sh.expectedDate)) AS avg_delay
    FROM shipmentHeader sh
    WHERE sh.poId IN (
        SELECT DISTINCT poId 
        FROM poDetails 
        WHERE supplierId = :supplierId
    )
     AND sh.receivedDate IS NOT NULL AND sh.expectedDate IS NOT NULL
     AND sh.receivedDate >= sh.expectedDate -- Only consider actual delays for average risk calculation
    """
    # Note: DATEDIFF syntax varies (e.g., JULIANDAY for SQLite, AGE for PostgreSQL)
    # Note: Added condition to only average non-negative delays for risk scoring.
    #       You might want a separate metric for average early days if needed.
    params = {'supplierId': supplier_id}
    results = db_query_func(query, params)
    if results and results[0].get('avg_delay') is not None:
        delay = results[0]['avg_delay']
        try:
            return float(delay)
        except (ValueError, TypeError):
            logging.error(f"Could not convert avg_delay value '{delay}' to float.")
            return 0.0
    else:
        logging.info(f"No results or NULL average found for (non-negative) delay, supplier_id: '{supplier_id}'")
        return 0.0 # Return 0 if no delayed shipments found


def get_quality_metrics_from_db(supplier_id: str, db_query_func: DbQueryFunc) -> Tuple[int, int]:
    logging.info(f"Fetching quality metrics for supplier_id: '{supplier_id}'")
    query = """
    SELECT
        SUM(sd.receivedItemQuantity) AS total_received,
        SUM(sd.damagedItemQuantity) AS total_damaged
    FROM shipmentdetails sd
    JOIN shipmentheader sh ON sd.receiptId = sh.receiptId
    WHERE sh.poId IN (
        SELECT DISTINCT poId
        FROM podetails
        WHERE supplierId = :supplierId
    )
    """
    params = {'supplierId': supplier_id}
    results = db_query_func(query, params)
    if results:
        # Use COALESCE logic within Python if SUM returns None (safer than relying on SQL COALESCE)
        total_received = results[0].get('total_received') or 0
        total_damaged = results[0].get('total_damaged') or 0
        try:
            return int(total_received), int(total_damaged)
        except (ValueError, TypeError) as e:
             logging.error(f"Could not convert quality metrics to int: {e}. Received: R={total_received}, D={total_damaged}")
             return 0,0
    else:
        logging.warning(f"No rows returned for quality metrics query, supplier_id: '{supplier_id}'")
        return 0, 0


# --- Placeholder function for Langchain Integration ---
def generate_llm_key_insights(metrics: Dict[str, Any]) -> str:
    """
    Generates key insights using an LLM (e.g., via Langchain).
    This is a placeholder implementation.

    Args:
        metrics: A dictionary containing calculated supplier metrics.
                 Expected keys: supplier_id, fill_rate, pending_rate, avg_delay,
                                non_defective_rate, defective_rate, risk_score, risk_level

    Returns:
        A formatted string containing 3-4 key insights.
    """
    logging.info(f"Generating LLM insights for supplier: {metrics.get('supplier_id')}")

    # 1. Define the Prompt Template
    prompt_template = """
You are a supply chain analyst reviewing supplier performance data.
Supplier ID: {supplier_id}
Performance Metrics:
- Fill Rate: {fill_rate:.1f}% ({pending_rate:.1f}% pending)
- Average Delivery Delay (when late): {avg_delay:.1f} days
- Quality (Defective Rate): {defective_rate:.1f}%
- Overall Risk Score: {risk_score} (Category: {risk_level})

Based *only* on these metrics, provide 3-4 concise key insights written in natural language, suitable for a business report. Focus on the implications of these numbers. Do not invent new information. Start each insight on a new line with a number and parenthesis (e.g., "1) ...").
"""

    # 2. Format the prompt with actual metrics
    try:
        prompt = prompt_template.format(**metrics)
    except KeyError as e:
        logging.error(f"Missing key in metrics dictionary for prompt formatting: {e}")
        return "Error: Could not format insights prompt due to missing metric data."

    # 3. **LANGCHAIN INTEGRATION POINT**
    #    Replace this section with your actual Langchain code.
    try:
        # --- Example Langchain Setup (Requires API Key & Installation) ---
        # Make sure OPENAI_API_KEY is set as an environment variable
        # os.environ["OPENAI_API_KEY"] = "your_api_key_here" # Or use other auth methods
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5, streaming=True, api_key=OPENAI_API_KEY)

        # --- Make the LLM Call ---
        response = llm.invoke([HumanMessage(content=prompt)])
        insights_text = response.content

        # --- Placeholder Implementation ---
        logging.warning("Langchain LLM call is currently a placeholder.")
        # Returning the formatted prompt for demonstration purposes.
        # Replace this with the actual result from the LLM call.
        # insights_text = (
        #     f"1) [Placeholder Insight 1 based on metrics for {metrics.get('supplier_id')}]\n"
        #     f"2) [Placeholder Insight 2 based on metrics for {metrics.get('supplier_id')}]\n"
        #     f"3) [Placeholder Insight 3 based on metrics for {metrics.get('supplier_id')}]\n"
        #     f"4) [Placeholder Insight 4 summarizing risk for {metrics.get('supplier_id')}]"
        # )
        # --- End Placeholder ---

        return insights_text.strip()

    except ImportError:
        logging.error("Langchain or related libraries not installed. Cannot generate LLM insights.")
        return "Error: Langchain libraries not found."
    except Exception as e:
        logging.error(f"Error during LLM insight generation: {e}", exc_info=True)
        return f"Error generating insights: {e}"


# --- Main insight generation function (Modified) ---
async def generate_supplier_insights(supplier_id: str, db_query_func: DbQueryFunc) -> dict:
    """
    Generates a risk assessment report for a given supplier, including LLM-generated insights.
    """
    # 1. Retrieve data
    total_ordered = get_total_ordered(supplier_id, db_query_func)
    total_delivered = get_total_delivered(supplier_id, db_query_func)
    avg_delay = get_average_delay_for_supplier(supplier_id, db_query_func) # Now calculates avg of actual delays
    total_received, total_damaged = get_quality_metrics_from_db(supplier_id, db_query_func)

    # 2. Calculate metrics
    fill_rate, pending_rate = calculate_fill_rate(total_ordered, total_delivered)
    non_defective_rate, defective_rate = calculate_quality_metrics(total_received, total_damaged)
    risk_score = compute_risk_score(fill_rate, avg_delay, defective_rate)

    # Determine risk level based on score
    risk_level = "Low"
    if risk_score > 3.5: # Example thresholds - adjust as needed
        risk_level = "High"
    elif risk_score > 1.5:
        risk_level = "Medium"

    # 3. Prepare metrics dictionary for LLM
    metrics_for_llm = {
        "supplier_id": supplier_id,
        "fill_rate": fill_rate,
        "pending_rate": pending_rate,
        "avg_delay": avg_delay,
        "non_defective_rate": non_defective_rate,
        "defective_rate": defective_rate,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "total_ordered": total_ordered, # Include raw numbers if needed for context
        "total_delivered": total_delivered,
        "total_received": total_received,
        "total_damaged": total_damaged
    }

    # 4. Generate LLM Insights (using the placeholder function)
    key_insights_text = generate_llm_key_insights(metrics_for_llm)

    # 5. Format the final report string
    assessment_summary = (
        f"Supplier Risk Assessment for ID: {supplier_id}\n"
        f"--------------------------------------------\n"
        f"- Fill Rate: {fill_rate:.1f}% delivered ({total_delivered}/{total_ordered} items), {pending_rate:.1f}% pending.\n"
        f"- On-Time Delivery: Average delay of {avg_delay:.1f} days (when late).\n" # Clarified metric
        f"- Quality: {non_defective_rate:.1f}% non-defective, {defective_rate:.1f}% defective (based on {total_received} received items).\n"
        f"- Supplier Risk Score: {risk_score} (Category: {risk_level}).\n"
    )

    final_report = assessment_summary + "\nKey Insights & Interpretation:\n" + key_insights_text
    graph_data= await plot_all_graphs(fill_rate,pending_rate,non_defective_rate,defective_rate,avg_delay,risk_score)
    fill_rate_dict={"fill_rate":f"{fill_rate:.1f}% delivered ({total_delivered}/{total_ordered} items)",
                    "pending_rate":f"{pending_rate:.1f}% pending."}
    # quality_dict={"non_defective_rate":f"{non_defective_rate:.1f}% non-defective",
    #               "defective_rate":}
    # print("Graph Data: ",graph_data)
    # insights={"fill_rate":f"{fill_rate:.1f}% delivered ({total_delivered}/{total_ordered} items), {pending_rate:.1f}% pending.\n",
    insights={"fill_rate_dict":fill_rate_dict,
              "delays":f"Average delay of {avg_delay:.1f} days (when late).",
              "quality":f"{non_defective_rate:.1f}% non-defective, {defective_rate:.1f}% defective (based on {total_received} received items).",
              "supplier_risk_score":f"{risk_score} (Category: {risk_level}).",
              "key_insights":f"{key_insights_text}",
              "graph_data":graph_data
              }
    logging.info("insights dict: %s", insights)

    # Optional: Add alerts based on thresholds if needed
    # if risk_level == "High":
    #     final_report += "\n\nALERT: High risk supplier identified."
    # if defective_rate > 10: # Example threshold
    #     final_report += "\nALERT: Quality issues exceed 10% threshold."

    # return final_report
    return insights


async def plot_all_graphs(fill_rate:float,pending_rate:float,non_defective_rate:float,defective_rate:float,avg_delay:float,risk_score:float):
    # --- constants to match your CSS tileImg width: 8.5rem ---
    REM_IN_PX    = 16
    TILE_REM     = 8.5
    # If you want a crisp retina graphic, multiply by 2 (or 3)
    SCALE_FACTOR = 2

    TILE_PX = int(REM_IN_PX * TILE_REM * SCALE_FACTOR)  # = 136px * 2 = 272px
    # --- 1) Delivery Performance Bar Chart ---
# --- 1) Delivery Performance Bar Chart ---
    bar_fig = go.Figure(data=[
        go.Bar(name='Fill Rate',    x=['Fill Rate'],    y=[fill_rate],    marker_color='#1c244b',   width=0.8),
        go.Bar(name='Pending Rate', x=['Pending Rate'], y=[pending_rate], marker_color='mediumturquoise', width=0.8)
    ])

    bar_fig.update_layout(
        template='plotly_white',
        barmode='group',
        width=TILE_PX*1.25,
        height=TILE_PX,
        margin=dict(l=20, r=20, t=40, b=30),
        # title=dict(text='Delivery Performance', font=dict(size=16)),
        legend=dict(font=dict(size=12)),
        xaxis=dict(title='Metric', title_font=dict(size=14), tickfont=dict(size=6.5)),
        yaxis=dict(title='% of Items', title_font=dict(size=14), tickfont=dict(size=12)),
        bargap=0.9  # Adjust this value to increase/decrease space between bars (0.1 - 1.0)
    )

    # --- 2) Quality Breakdown Pie Chart ---
    colors = ['#1c244b', 'mediumturquoise']
    pie_fig = go.Figure(data=[
        go.Pie(
            labels=['Non-Defective','Defective'],
            values=[non_defective_rate, defective_rate],
            hole=0.4,
            textinfo='percent',
            textfont=dict(size=14)
        )
    ])
    pie_fig.update_layout(
        template='plotly_white',
        width=TILE_PX*1.25,
        height= TILE_PX,
        margin= dict(l=20, r=20, t=40, b=20),
        title = dict(text='Quality Breakdown', font=dict(size=16)),
        legend= dict(font=dict(size=12))
    )
    pie_fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                    marker=dict(colors=colors, line=dict(color='#000000')))
    # --- 3) Supplier Risk Gauge Chart ---
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Supplier Risk Score", 'font':{'size':16}},
        number={'font':{'size':20}},
        gauge={
            'axis': {'range': [0, 1], 'tickfont':{'size':12}},
            'bar':  {'color': "mediumturquoise"},
            'steps': [
                {'range': [0,0.5], 'color': "#1c244b"},
                # {'range': [1.5,3.5], 'color': "#1c244b"},
                {'range': [0.5,1],   'color': "#1c244b"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    gauge_fig.update_layout(
        template='plotly_white',
        width=TILE_PX*1.25,
        height= TILE_PX,
        margin= dict(l=20, r=20, t=40, b=20)
    )

    # --- 4) Average Delivery Delay Box Plot ---
    delay_fig = go.Figure(data=[
        go.Box(
            y=[avg_delay],
            name="Delay",
            boxpoints=False,
            marker_color='#1c244b'
        )
    ])
    delay_fig.update_layout(
        template='plotly_white',
        width=TILE_PX*1.25,
        height= TILE_PX,
        margin= dict(l=20, r=20, t=40, b=30),
        title = dict(text='Avg Delay (days)', font=dict(size=16)),
        yaxis = dict(title='Days',      title_font=dict(size=14), tickfont=dict(size=12))
    )
    def fig_to_bytes(fig):
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        return buf.getvalue()
    # Render each figure to raw bytes
    bar_bytes   = fig_to_bytes(bar_fig)
    pie_bytes   = fig_to_bytes(pie_fig)
    gauge_bytes = fig_to_bytes(gauge_fig)
    delay_bytes = fig_to_bytes(delay_fig)

    # Base-64 encode & add a data URI prefix so it's directly displayable in browsers
    def to_data_uri(img_bytes):
        b64 = base64.b64encode(img_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"

    return {
            "bar_chart"   : to_data_uri(bar_bytes),
            "pie_chart"   : to_data_uri(pie_bytes),
            "gauge_chart" : to_data_uri(gauge_bytes),
            "delay_chart" : to_data_uri(delay_bytes),
        }
    
# async def plot_all_graphs(fill_rate:float,pending_rate:float,non_defective_rate:float,defective_rate:float,avg_delay:float,risk_score:float):
#     # 1. Bar Chart - Fill vs Pending
#     bar_fig = go.Figure(data=[
#         go.Bar(name='Fill Rate', x=['Fill Rate'], y=[fill_rate], marker_color='cyan'),
#         go.Bar(name='Pending Rate', x=['Pending Rate'], y=[pending_rate], marker_color='magenta')
#     ])
#     bar_fig.update_layout(
#         title='Delivery Performance',
#         yaxis_title='% of Items',
#         xaxis_title='Metric',
#         template='plotly_dark',
#         barmode='group'
#     )

#     # 2. Pie Chart - Quality Breakdown
#     pie_fig = go.Figure(data=[
#         go.Pie(labels=['Non-Defective', 'Defective'], values=[non_defective_rate, defective_rate], hole=0.4)
#     ])
#     pie_fig.update_layout(
#         title='Quality Breakdown of Received Items',
#         template='plotly_dark'
#     )

#     # 3. Gauge Chart - Supplier Risk Score
#     gauge_fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=risk_score,
#         title={'text': "Supplier Risk Score"},
#         gauge={
#             'axis': {'range': [0, 5]},
#             'bar': {'color': "blue"},
#             'steps': [
#                 {'range': [0, 1.5], 'color': "cyan"},
#                 {'range': [1.5, 3.5], 'color': 'cyan'},
#                 {'range': [3.5, 5], 'color': "cyan"},
#             ],
#             'threshold': {
#                 'line': {'color': "black", 'width': 4},
#                 'thickness': 0.75,
#                 'value': risk_score
#             }
#         }
#     ))
#     gauge_fig.update_layout(
#         template='plotly_dark'
#     )

#     # 4. Box Plot (for Delay) – simplified for average delay
#     delay_fig = go.Figure(data=[
#         go.Box(y=[avg_delay], name="Delivery Delay", boxpoints=False, marker_color='purple')
#     ])
#     delay_fig.update_layout(
#         title='Average Delivery Delay (days)',
#         yaxis_title='Days',
#         template='plotly_dark'
#     )
#     # return {
#     #     "bar_fig": bar_fig,
#     #     "pie_fig": pie_fig,
#     #     "gauge_fig": gauge_fig,
#     #     "delay_fig": delay_fig,
#     #     "status": "All charts generated successfully"
#     # }
#     def fig_to_bytes(fig):
#         buf = io.BytesIO()
#         fig.write_image(buf, format="png")
#         return buf.getvalue()
#     # Render each figure to raw bytes
#     bar_bytes   = fig_to_bytes(bar_fig)
#     pie_bytes   = fig_to_bytes(pie_fig)
#     gauge_bytes = fig_to_bytes(gauge_fig)
#     delay_bytes = fig_to_bytes(delay_fig)

#     # Base-64 encode & add a data URI prefix so it's directly displayable in browsers
#     def to_data_uri(img_bytes):
#         b64 = base64.b64encode(img_bytes).decode("ascii")
#         return f"data:image/png;base64,{b64}"

#     return {
#             "bar_chart"   : to_data_uri(bar_bytes),
#             "pie_chart"   : to_data_uri(pie_bytes),
#             "gauge_chart" : to_data_uri(gauge_bytes),
#             "delay_chart" : to_data_uri(delay_bytes),
#         }

# from datetime import datetime
# import logging
# from typing import List, Tuple, Dict, Any, Callable # Import Callable

# # Removed: from sqlalchemy import text - No longer needed here
# # Removed: from database import get_db - No longer needed here

# # --- Calculation functions (calculate_fill_rate, etc.) remain the same ---
# def calculate_fill_rate(po_ordered: float, delivered: float) -> Tuple[float, float]:
#     """Calculates the fill rate as delivered percentage and pending percentage."""
#     if po_ordered == 0:
#         return 0.0, 0.0
#     fill_rate = (delivered / po_ordered) * 100
#     pending_rate = 100 - fill_rate
#     return fill_rate, pending_rate

# def calculate_average_delay(expected_date: datetime, received_date: datetime) -> int:
#     """Calculates the number of days delay between expected and received dates."""
#     # This function isn't directly used by generate_supplier_insights as written,
#     # but kept for completeness if needed elsewhere. The DB query gets the avg directly.
#     return (received_date - expected_date).days

# def calculate_quality_metrics(total_received: int, total_damaged: int) -> Tuple[float, float]:
#     """
#     Calculates the non-defective and defective item percentage rates.
#     Returns (non_defective_rate, defective_rate).
#     """
#     if total_received == 0:
#         return 0.0, 0.0
#     non_defective = total_received - total_damaged
#     non_defective_rate = (non_defective / total_received) * 100
#     defective_rate = 100 - non_defective_rate
#     return non_defective_rate, defective_rate

# def compute_risk_score(fill_rate: float, avg_delay: float, defective_rate: float) -> float:
#     """
#     Computes a composite risk score based on fill rate, average delay, and defective rate.
#     Lower metric values yield a lower risk score. Adjust weights as necessary.
#     """
#     # Ensure inputs are floats to avoid potential type errors in calculation
#     fill_rate = float(fill_rate)
#     avg_delay = float(avg_delay)
#     defective_rate = float(defective_rate)

#     # Handle potential division by zero or negative rates if data/logic allows
#     fill_component = 5 - (fill_rate / 20) if fill_rate >= 0 else 5
#     delay_component = avg_delay / 10 if avg_delay >= 0 else 0
#     quality_component = defective_rate / 20 if defective_rate >= 0 else 0

#     # Weighted sum - ensure weights add up as intended (e.g., to 1.0)
#     risk_score = fill_component * 0.4 + delay_component * 0.3 + quality_component * 0.3
#     # Clamp risk score to a reasonable range if necessary, e.g., 0-5
#     return round(max(0, min(risk_score, 5)), 1)

# # Type alias for clarity
# DbQueryFunc = Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]

# logging.basicConfig(level=logging.INFO) # Configure logging

# # --- Database interaction functions now accept db_query_func ---

# def get_total_ordered(supplier_id: str, db_query_func: DbQueryFunc) -> int:
#     logging.info(f"Fetching total ordered for supplier_id: '{supplier_id}'")
#     # Reverted to Parameterized Query (SAFE!)
#     query = """
#     SELECT SUM(itemQuantity) AS total_ordered
#     FROM podetails
#     WHERE supplierId = :supplierId
#     """
#     params = {'supplierId': supplier_id}
#     # Use the passed-in db_query_func
#     results = db_query_func(query, params)

#     # Debugging prints can be removed in production
#     # print(f"DEBUG - Raw DB Results (Total Ordered): {results}")
#     # print(f"DEBUG - Query Template Used (Total Ordered): {query}")

#     if results and results[0].get('total_ordered') is not None:
#         total = results[0]['total_ordered']
#         try:
#             # Handle potential Decimal type from SUM
#             return int(float(total))
#         except (ValueError, TypeError):
#             logging.error(f"Could not convert total_ordered value '{total}' to int.")
#             return 0
#     else:
#         logging.info(f"No results or NULL sum found for total_ordered, supplier_id: '{supplier_id}'")
#         return 0

# def get_total_delivered(supplier_id: str, db_query_func: DbQueryFunc) -> int:
#     logging.info(f"Fetching total delivered for supplier_id: '{supplier_id}'")
#     # Reverted to Parameterized Query (SAFE!)
#     query = """
#     SELECT SUM(sd.receivedItemQuantity) AS delivered_items
#     FROM shipmentDetails sd
#     JOIN shipmentHeader sh ON sd.receiptId = sh.receiptId
#     WHERE sh.poId IN (
#         SELECT DISTINCT poId 
#         FROM poDetails 
#         WHERE supplierId = :supplierId
#     )
#     """
#     params = {'supplierId': supplier_id}
#     results = db_query_func(query, params)

#     if results and results[0].get('delivered_items') is not None:
#         total = results[0]['delivered_items']
#         try:
#             return int(float(total))
#         except (ValueError, TypeError):
#             logging.error(f"Could not convert delivered_items value '{total}' to int.")
#             return 0
#     else:
#         logging.info(f"No results or NULL sum found for total_delivered, supplier_id: '{supplier_id}'")
#         return 0


# def get_average_delay_for_supplier(supplier_id: str, db_query_func: DbQueryFunc) -> float:
#     logging.info(f"Fetching average delay for supplier_id: '{supplier_id}'")
#     query = """
#      SELECT AVG(DATEDIFF(sh.receivedDate, sh.expectedDate)) AS avg_delay
#     FROM shipmentHeader sh WHERE sh.poId IN (
#         SELECT DISTINCT poId 
#         FROM poDetails 
#         WHERE supplierId = 'SUP130')
#     AND sh.receivedDate IS NOT NULL AND sh.expectedDate IS NOT NULL -- Avoid issues with NULL dates

#     """
#     # Adjust DATEDIFF and JOINs based on your actual DB and schema
#     params = {'supplierId': supplier_id}
#     results = db_query_func(query, params)

#     if results and results[0].get('avg_delay') is not None:
#         delay = results[0]['avg_delay']
#         try:
#             # AVG might return Decimal or float
#             return float(delay)
#         except (ValueError, TypeError):
#             logging.error(f"Could not convert avg_delay value '{delay}' to float.")
#             return 0.0
#     else:
#         logging.info(f"No results or NULL average found for delay, supplier_id: '{supplier_id}'")
#         return 0.0


# def get_quality_metrics_from_db(supplier_id: str, db_query_func: DbQueryFunc) -> Tuple[int, int]:
#     logging.info(f"Fetching quality metrics for supplier_id: '{supplier_id}'")
#     query = """
#     SELECT 
#         SUM(sd.receivedItemQuantity) AS total_received,
#         SUM(sd.damagedItemQuantity) AS total_damaged
#     FROM shipmentDetails sd
#     JOIN shipmentHeader sh ON sd.receiptId = sh.receiptId
#     WHERE sh.poId IN (
#         SELECT DISTINCT poId 
#         FROM poDetails 
#         WHERE supplierId = :supplierId
#     ) -- Simplified JOIN assumption
#     """
#     # Adjust JOINs based on your actual schema
#     # Using COALESCE to ensure 0 is returned instead of NULL if SUM finds no rows
#     params = {'supplierId': supplier_id}
#     results = db_query_func(query, params)

#     # Since COALESCE is used, we expect one row even if no matching shipments
#     if results:
#         total_received = results[0].get('total_received', 0)
#         total_damaged = results[0].get('total_damaged', 0)
#         try:
#             # Ensure results are integers
#             return int(total_received), int(total_damaged)
#         except (ValueError, TypeError) as e:
#              logging.error(f"Could not convert quality metrics to int: {e}. Received: R={total_received}, D={total_damaged}")
#              return 0,0
#     else:
#          # Should not happen with COALESCE but good to have a fallback
#         logging.warning(f"No rows returned for quality metrics query, supplier_id: '{supplier_id}'")
#         return 0, 0


# # --- Main insight generation function now accepts db_query_func ---

# def generate_supplier_insights(supplier_id: str, db_query_func: DbQueryFunc) -> str:
#     """
#     Generates a risk assessment report for a given supplier using the provided db query function.
#     """
#     # Retrieve data using the passed-in db_query_func
#     total_ordered = get_total_ordered(supplier_id, db_query_func)
#     total_delivered = get_total_delivered(supplier_id, db_query_func)
#     avg_delay = get_average_delay_for_supplier(supplier_id, db_query_func)
#     total_received, total_damaged = get_quality_metrics_from_db(supplier_id, db_query_func)

#     # Calculate fill rate.
#     fill_rate, pending_rate = calculate_fill_rate(total_ordered, total_delivered)

#     # Calculate quality metrics.
#     non_defective_rate, defective_rate = calculate_quality_metrics(total_received, total_damaged)

#     # Compute the composite risk score.
#     risk_score = compute_risk_score(fill_rate, avg_delay, defective_rate)

#     # Format the insights string
#     # (Removed the hardcoded "15%" values, using calculated values instead)
#     insights = (
#         f"Supplier Risk Assessment for ID: {supplier_id}\n"
#         f"--------------------------------------------\n"
#         f"- Fill Rate: {fill_rate:.1f}% delivered ({total_delivered}/{total_ordered} items), {pending_rate:.1f}% pending.\n"
#         f"- On-Time Delivery: Average delay of {avg_delay:.1f} days across relevant shipments.\n"
#         f"- Quality: {non_defective_rate:.1f}% non-defective, {defective_rate:.1f}% defective (based on {total_received} received items).\n"
#         f"- Supplier Risk Score: {risk_score} out of 5.\n\n"
#         "Key Insights & Interpretation:\n"
#         f"1) Order Fulfillment: Based on past performance, there's a {pending_rate:.1f}% chance orders may not be fully completed.\n"
#         f"2) Timeliness: Orders have historically been delayed by roughly {avg_delay:.1f} days on average.\n"
#         f"3) Item Quality: Expect around {defective_rate:.1f}% of items received to potentially have quality issues.\n"
#         f"4) Overall Risk: The score of {risk_score} suggests a [TODO: Add interpretation based on score, e.g., Low/Medium/High] risk level.\n"
#        # f"Alert: Supplier not meeting environmental and safety standards." # Keep if this comes from another source
#     )

#     # Add interpretation based on score
#     risk_level = "Low"
#     if risk_score > 3.5:
#         risk_level = "High"
#     elif risk_score > 1.5:
#         risk_level = "Medium"
#     insights = insights.replace("[TODO: Add interpretation based on score, e.g., Low/Medium/High]", risk_level)


#     return insights