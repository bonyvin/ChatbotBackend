
import os
from typing import Optional,Any
from pydantic import BaseModel, Field,field_validator
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# class UserIntent(BaseModel):
#     intent: str | None = Field(None)
class UserIntent(BaseModel):
    intent: Optional[str] = Field(None)

    @field_validator('intent', mode='before')
    @classmethod
    def extract_intent_value(cls, v: Any) -> Any:
        """Extract value from nested dict structure"""
        if v is None:
            return None
        if isinstance(v, dict) and 'value' in v:
            return v['value']
        return v

    class Config:
        populate_by_name = True
        extra = 'ignore'

intent_extractor_llm = ChatOpenAI(
    model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0
)
intent_extractor_llm_structured = intent_extractor_llm.with_structured_output(
    UserIntent, method="function_calling", include_raw=False
)
llm_tool_test = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _to_plain(obj):
    """Return a plain-serializable Python object from dict / pydantic v1/v2 / simple values."""
    if obj is None:
        return None
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
    # fallback to jsonable_encoder
    try:
        return jsonable_encoder(obj)
    except Exception:
        return str(obj)
    
# Helper function to unwrap nested values (reusable across your codebase)
def unwrap_nested_values(data: Any) -> Any:
    """
    Recursively unwrap {value: ..., is_example: ...} structures.
    This can be used as a fallback if validators don't catch everything.
    """
    if data is None:
        return None
    
    if isinstance(data, dict):
        # If it has the nested structure, extract the value
        if 'value' in data and 'is_example' in data:
            return unwrap_nested_values(data['value'])
        
        # Otherwise, unwrap all dict values
        return {key: unwrap_nested_values(val) for key, val in data.items()}
    
    if isinstance(data, list):
        return [unwrap_nested_values(item) for item in data]
    
    return data