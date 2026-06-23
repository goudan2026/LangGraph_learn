"""
下面的例子来源于生活。我们经常把报错复制给大模型，让它帮忙 debug。但报错中可能包含个人隐私信息。针对这种情况，采用以下两种方法进行处置：

拒绝回答问题

屏蔽隐私信息
"""

from textwrap import dedent
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from langchain.agents.middleware import before_agent, AgentState
from langgraph.runtime import Runtime
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from typing import Any

_ = load_dotenv()

# 可信任的模型，一般是本地模型，为了方便，这里依然使用qwen
basic_model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
)

class PiiCheck(BaseModel):
    """Structured output indicating whether text contains PII."""
    is_pii: bool = Field(description="Whether the text contains PII")

def message_with_pii(pii_middleware):
    agent = create_agent(
        model=basic_model,
        middleware=[pii_middleware],
    )

    result = agent.invoke({
        "messages":[{
            "role":"user",
            "content": dedent(
                """
                File "/home/luochang/proj/agent.py", line 53, in my_agent
                    agent = create_react_agent(
                            ^^^^^^^^^^^^^^^^^^^
                File "/home/luochang/miniconda3/lib/python3.12/site-packages/typing_extensions.py", line 2950, in wrapper
                    return arg(*args, **kwargs)
                        ^^^^^^^^^^^^^^^^^^^^
                File "/home/luochang/miniconda3/lib/python3.12/site-packages/langgraph/prebuilt/chat_agent_executor.py", line 566, in create_react_agent
                    model = cast(BaseChatModel, model).bind_tools(
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                AttributeError: 'RunnableLambda' object has no attribute 'bind_tools'
    
                ---
    
                为啥报错
                """                
            ).strip(),
        }],
    })

    return result

@before_agent(can_jump_to=["end"])

def content_blocker(state: AgentState,  runtime: Runtime) -> dict[str, Any] | None:
    """Deterministic guardrail: Block requests containing banned keywords."""
    # Get the first user message
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if last_message.type != "human":
        return None

    content = last_message.content.lower()
    prompt = (
        "你是一个隐私保护助手。请识别下面文本中涉及个人可识别信息（PII），"
        "例如：姓名、身份证号、护照号、电话号码、邮箱、住址、银行卡号、社交账号、车牌等。"
        "特别注意，若代码、文件路径中包含用户名，也应被视为敏感信息。"
        "若包含敏感信息，请返回{\"is_pii\": True}，否则返回{\"is_pii\": False}。"
        "请严格以 json 格式返回，并且只输出 json。文本如下：\n\n" + content
    )

    pii_agent = basic_model.with_structured_output(PiiCheck)
    result = pii_agent.invoke(prompt)

    if result.is_pii is True:
        # Block execution before any processing
        return {
            "messages": [{
                "role": "assistant",
                "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
            }],
            "jump_to": "end"
        }
    else:
        print("No PII found")

    return None

result = message_with_pii(pii_middleware=content_blocker)

for message in result["messages"]:
    message.pretty_print()