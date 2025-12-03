
import os
from pydantic import BaseModel, Field
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class UserIntent(BaseModel):
    intent: str | None = Field(None)

intent_extractor_llm = ChatOpenAI(
    model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.0
)
intent_extractor_llm_structured = intent_extractor_llm.with_structured_output(
    UserIntent, method="function_calling", include_raw=False
)
llm_tool_test = ChatOpenAI(model="gpt-4o-mini", temperature=0)