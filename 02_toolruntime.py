from typing import Literal, Any
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ = load_dotenv()

class CalcInfo(BaseModel):
    """Calculation information."""
    output: int = Field(description="The calculation result")

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4.1", 
)

class Context(BaseModel):
    authority: Literal["admin", "user"]

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def math_add(runtime: ToolRuntime[Context, Any], a: int, b: int) -> int:
    """Add two numbers together."""
    authority = runtime.context.authority
    if authority != "admin":
        raise PermissionError("Only admins can add numbers.")
    return a + b

structured_agent = create_agent(
    model=llm,
    tools=[get_weather, math_add],
    system_prompt="You are a helpful assistant",
    response_format=CalcInfo,
)

response = structured_agent.invoke(
    {"messages": [{"role": "user", "content": "请计算 8234783 + 94123832 = ?"}]},
    config={"configurable": {"thread_id": "1"}},
    context=Context(authority="admin"),
)

for message in response['messages']:
    message.pretty_print()

print(response['messages'][-1].content)