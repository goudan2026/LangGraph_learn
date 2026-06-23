import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1024

# 加载模型配置
_ = load_dotenv()


# 用于获取text embedding的接口
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)

# 加载模型
model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    model="gpt-4o-mini",
    temperature=0.7,
)

# embedding生成函数
def embed(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIM,
    )

    return [item.embedding for item in response.data]

# 测试能否正常生成text embedding
texts = [
    "LangGraph的中间件非常强大",
    "LangGraph的MCP也很好用",
]
vectors = embed(texts)

# InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
store = InMemoryStore(index={"embed": embed, "dims": EMBED_DIM})
# 添加两条用户数据
namespace = ("users", )
key = "user_1"
store.put(
    namespace,
    key,
    {
        "rules": [
            "User likes short, direct language",
            "User only speaks English & python",
        ],
        "rule_id": "3",
    },
)

store.put( 
    ("users",),  # Namespace to group related data together (users namespace for user data)
    "user_2",  # Key within the namespace (user ID as key)
    {
        "name": "John Smith",
        "language": "English",
    }  # Data to store for the given user
)

# get the "memory" by ID
item = store.get(namespace, "a-memory") 

# search for "memories" within this namespace, filtering on content equivalence, sorted by vector similarity
items = store.search( 
    namespace, filter={"rule_id": "3"}, query="language preferences"
)
