import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

# 关键节点：ToolNode
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig

_ = load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4.1",
    temperature=0.7,
)

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

#创建工具节点
tools = [get_weather]
tool_node = ToolNode(tools)

#创建带工具的LLM节点
def assistant(state: MessagesState, config: RunnableConfig):
    system_prompt = "You are a helpful assistant that can get weather information."
    all_messages = [SystemMessage(system_prompt), *state["messages"]]
    model = llm.bind_tools(tools)
    return {"messages": [model.invoke(all_messages)]}


# 创建条件边
def should_continue(state: MessagesState, config: RunnableConfig):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return 'continue'
    # 否则，结束
    return 'end'

# 创建状态图
builder = StateGraph(MessagesState)

builder.add_node('assistant', assistant)
builder.add_node('tool', tool_node)

builder.add_edge(START, 'assistant')

builder.add_conditional_edges('assistant', should_continue,

    {
        'continue': 'tool',
        'end': END,
    },
)

builder.add_edge('tool', 'assistant')

my_graph = builder.compile(name="my_graph")

# 渲染状态图（保存 PNG，和教程里那种节点图一样）
# png_path = os.path.join(os.path.dirname(__file__), "my_graph.png")
# with open(png_path, "wb") as f:
#     f.write(my_graph.get_graph().draw_mermaid_png())
# print(f"状态图已保存: {png_path}")
# 调用图
response = my_graph.invoke({'messages': [HumanMessage(content='上海天气怎么样？')]})
for message in response['messages']:
    message.pretty_print()