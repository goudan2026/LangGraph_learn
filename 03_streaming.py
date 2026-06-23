from typing import Literal, Any
from pydantic import BaseModel
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4.1", 
)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=llm,
    tools=[get_weather],)

for chunk in agent.stream(
    {
        "messages":[
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ]
    },
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")