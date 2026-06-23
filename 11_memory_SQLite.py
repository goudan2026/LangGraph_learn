

import os
import sqlite3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent

# 加载模型配置
_ = load_dotenv()

# 删除SQLite数据库
if os.path.exists("short-memory.db"):
    os.remove("short-memory.db")

# 加载模型
model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
    temperature=0.7,
)

# 创建sqlite支持的短期记忆
checkpointer = SqliteSaver(
    sqlite3.connect("short-memory.db", check_same_thread=False)
)

# 创建Agent
agent = create_agent(
    model=model,
    checkpointer=checkpointer,
)

# # 告诉智能体我叫 luochang
# result = agent.invoke(
#     {'messages': ['hi! i am luochang']},
#     {"configurable": {"thread_id": "3"}},
# )

# 让智能体回忆我的名字
result = agent.invoke(
    {'messages': ['What is my name?']},
    {"configurable": {"thread_id": "3"}},
)

for message in result['messages']:
    message.pretty_print()