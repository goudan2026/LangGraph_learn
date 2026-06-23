import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.tools import tool
from langgraph_supervisor import create_supervisor

# 加载模型配置
_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
    temperature=0.7,
)

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

@tool
def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b

# 创建subagent1：用于计算两数相加
subagent1 = create_agent(
    model=llm,
    tools=[add],
    name="subagent-1",
)

@tool(
    "subagent-1",
    description="可以准确地计算两数相加"
)
def call_subagent1(query: str):
    result = subagent1.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# 创建subagent2：用于计算两数相乘
subagent2 = create_agent(
    model=llm,
    tools=[multiply],
    name="subagent-2",
)
@tool(
    "subagent-2",
    description="可以准确地计算两数相乘"
)
def call_subagent2(query: str):
    result = subagent2.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# 创建supervisor agent
supervisor_agent = create_agent(
    model=llm,
    tools=[call_subagent1, call_subagent2, divide],
    name="supervisor-agent",
    system_prompt="提示：如遇两数相减仍可用两数相加工具实现，只需将一个数加上另一个数的负数",
)

result = supervisor_agent.invoke({
    "messages": [{"role": "user", "content": "计算 38462 + 378 / 49 * 83723 - 123 的结果"}]}
)


subagent3 = create_agent(
    model=llm,
    tools=[divide],
    name="subagent-3",
)

supervisor_graph = create_supervisor(
    [subagent1, subagent2, subagent3],
    model=llm,
    prompt="提示：如遇两数相减仍可用两数相加工具实现，只需将一个数加上另一个数的负数"
)

supervisor_app = supervisor_graph.compile()


result = supervisor_app.invoke({
    "messages": [{"role": "user", "content": "计算 38462 + 378 / 49 * 83723 - 123 的结果"}]}
)

for message in result["messages"]:
    message.pretty_print()