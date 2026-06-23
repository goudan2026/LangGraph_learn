import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

# 加载模型配置
_ = load_dotenv()

# 加载模型
model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
    temperature=0.7,
)

# 创建助手节点
def assistant(state: MessagesState):
    return {'messages': [model.invoke(state['messages'])]}


# 创建短期记忆
checkpointer = InMemorySaver()

# 创建图
builder = StateGraph(MessagesState)


# 添加节点
builder.add_node('assistant', assistant)


# 添加边
builder.add_edge(START, 'assistant')
builder.add_edge('assistant', END)


graph = builder.compile(checkpointer=checkpointer)

# 告诉智能体我叫 luochang
result = graph.invoke(
    {'messages': ['hi! i am luochang']},
    {"configurable": {"thread_id": "1"}},
)

for message in result['messages']:
    message.pretty_print()