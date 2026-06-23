import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# 加载模型配置
# 请事先在 .env 中配置 DASHSCOPE_API_KEY
_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4.1", 
)

agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
)

response = agent.invoke({"messages": "hello, how are you?"})

print(response['messages'][-1].content)