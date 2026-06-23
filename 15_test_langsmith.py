import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 加载模型配置
_ = load_dotenv()

# 配置大模型服务
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
)

# 创建 Agent
agent = create_agent(model=llm)

# langgraph-cli 入口：可以是函数，也可以直接 export agent
def get_app():
    return agent
