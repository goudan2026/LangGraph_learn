import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient  
from langchain.agents import create_agent
import asyncio

# 加载模型配置
_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
    temperature=0.7,
)

async def mcp_agent():
    # 我们用两种方式启动 MCP Server：stdio 和 streamable_http
    client = MultiServerMCPClient(  
        {
            "calculate": {
                "url": "http://localhost:8001/mcp",
                "transport": "streamable_http",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )
    
    tools = await client.get_tools()
    agent = create_agent(
        llm,
        tools=tools,
    )

    return agent

async def use_mcp(messages):
    agent = await mcp_agent()
    response = await agent.ainvoke(messages)
    return response

# 调用天气 MCP
async def main():
    # 调用天气 MCP
    messages = {"messages": [{"role": "user", "content": "福州天气怎么样？"}]}
    response = await use_mcp(messages)
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())