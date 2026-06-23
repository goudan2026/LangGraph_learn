"""
智能体的上下文存在长度限制。一旦超过限制，就需要对上下文进行压缩。在众多方法中，截断是最简单粗暴、易于实现的方法。消息截断功能可以通过 @before_model 装饰器实现。
"""
import os
from dotenv import load_dotenv
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any
from langchain_openai import ChatOpenAI

_ = load_dotenv()

@before_model
def trim_without_first_message(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *messages[-2:]
        ]
    }

basic_model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
)

agent = create_agent(
    basic_model,
    middleware=[trim_without_first_message],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}


def agent_invoke(agent):
    agent.invoke({"messages": "hi, my name is bob"}, config)
    agent.invoke({"messages": "write a short poem about cats"}, config)
    agent.invoke({"messages": "now do the same but for dogs"}, config)
    final_response = agent.invoke({"messages": "what's my name?"}, config)

    final_response["messages"][-1].pretty_print()

agent_invoke(agent)