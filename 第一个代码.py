from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool

model = ChatOpenAI(
    model="gpt-4.1", 
    api_key="sk-ZaU8A0bRPIe1X9tL9Umey3x3yXTfOvo6eJezoEWILeefSzhu",
    base_url="https://az.gptplus5.com/v1",
    )

@tool
def search(query: str) -> str:
    """根据关键词搜索信息。"""
    return f"结果：{query}"

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city}的天气是26度，天气晴朗"

agent = create_agent(
    model=model,
    tools=[search, get_weather],
)

query = "成都今天天气如何？"

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": query
        }
    ]
})

print(result)