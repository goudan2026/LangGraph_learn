"""
在 LangGraph 中，护栏可以通过中间件轻松实现。下面我们实现一个简单的护栏：若用户的最新消息中包含某些敏感词，智能体将拒绝回答。
"""

from typing import Any

from langchain.agents.middleware import before_agent, AgentState
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
import os
from dotenv import load_dotenv

_ = load_dotenv()

banned_keywords = ["hack", "exploit", "malware"]

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Deterministic guardrail: Block requests containing banned keywords."""
    # Get the first user message
    if not state["messages"]:
        return None
    last_message = state["messages"][-1]
    if last_message.type != "human":
        return None

    content = last_message.content.lower()

    # Check for banned keywords
    for keyword in banned_keywords:
        if keyword in content:
            # Block execution before any processing
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                }],
                "jump_to": "end"
            }
    return None

basic_model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
)


agent = create_agent(
    model=basic_model,
    middleware=[content_filter],
)

# This request will be blocked before any processing
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})


for message in result["messages"]:
    message.pretty_print()