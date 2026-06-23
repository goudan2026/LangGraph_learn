"""
随着对话轮次增加，每次请求携带的对话记录也会越来越长，从而导致请求费用上升。
为了控制预算，可以设定在对话轮次超过某个阈值后，切换到低费率模型。
这个功能可以通过自定义中间件实现。
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

# 加载模型配置
_ = load_dotenv()


# 低费率模型
basic_model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
)

# 高费率模型
advanced_model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4.1",
)

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 5:
        # Use a basic model for longer conversations
        model = basic_model
    else:
        model = advanced_model

    request.model = model
    print(f"message_count: {message_count}")
    print(f"model_name: {model.model_name}")

    return handler(request)

agent = create_agent(
    model=advanced_model,  # Default model
    middleware=[dynamic_model_selection]
)


state: MessagesState = {"messages": []}
items = ['汽车', '飞机', '摩托车', '自行车']
for idx, i in enumerate(items):
    print(f"\n=== Round {idx+1} ===")
    state["messages"] += [HumanMessage(content=f"{i}有几个轮子，请简单回答")]
    result = agent.invoke(state)
    state["messages"] = result["messages"]
    print(f'content: {result["messages"][-1].content}')