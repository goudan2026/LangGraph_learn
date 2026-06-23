import os
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 加载模型配置
_ = load_dotenv()

# 配置大模型服务
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
)


# 工具函数
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters."""
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("height_m and weight_kg must be greater than 0.")
    return weight_kg / (height_m ** 2)

# 创建带工具调用的Agent
tool_agent = create_agent(
    model=llm,
    tools=[get_weather, add_numbers, calculate_bmi],
    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on = {
                #无需人工审批
                "get_weather": False,
                #需要审批，且允许approve,edit,reject三种操作
                "add_numbers": True,
                #需要审批，且允许approve,reject两种操作
                "calculate_bmi": {"allowed_decisions": ["approve", "reject"]}
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer = InMemorySaver(),
    system_prompt="You are a helpful assistant.",
    )

# 运行Agent
config = {'configurable': {'thread_id': str(uuid.uuid4())}}
result = tool_agent.invoke(
    {"messages": [{
        "role": "user",
        "content": "我身高180cm，体重180斤，我的BMI是多少"
        # "content": "what is the weather in sf"
    }]},
    config=config,
)

# result['messages'][-1].content
inter = result.get('__interrupt__')
print(inter)

# Resume with approval decision
result = tool_agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}  # or "edit", "reject"
    ), 
    config=config
)

print(result['messages'][-1].content)