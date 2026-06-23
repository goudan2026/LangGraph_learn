import os
import uuid
import sqlite3

from typing import Callable
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call, ModelRequest, ModelResponse, SummarizationMiddleware
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite import SqliteStore



# 加载模型配置
_ = load_dotenv()

# 加载模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
    temperature=0.7,
)

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id

    # Read from Store: get user preferences
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    base = "You are a helpful assistant."

    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."

    return base

store = InMemoryStore()

agent = create_agent(
    model=llm,
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=store,
)

# 预置两条偏好信息
store.put(("preferences",), "user_1", {"communication_style": "Chinese"})
store.put(("preferences",), "user_2", {"communication_style": "Korean"})

# 用户1喜欢中文回复
# result = agent.invoke(
#     {"messages": [
#         {"role": "system", "content": "You are a helpful assistant. Please be extra concise."},
#         {"role": "user", "content": 'What is a "hold short line"?'}
#     ]},
#     context=Context(user_id="user_1"),
# )

# for message in result['messages']:
#     message.pretty_print()

result = agent.invoke(
    {"messages": [
        {"role": "system", "content": "You are a helpful assistant. Please be extra concise."},
        {"role": "user", "content": 'What is a "hold short line"?'}
    ]},
    context=Context(user_id="user_2"),
)

for message in result['messages']:
    message.pretty_print()