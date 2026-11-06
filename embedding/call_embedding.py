import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv, find_dotenv

from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from llm.call_llm import parse_llm_api_key

_ = load_dotenv(find_dotenv())

def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    if embedding == 'm3e':
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    if embedding_key == None:
        embedding_key = parse_llm_api_key(embedding)
    if embedding == "openai":
        # 中转 API：向量模型同样走 OPENAI_API_BASE，模型名可用 OPENAI_EMBEDDING_MODEL 配置
        return OpenAIEmbeddings(openai_api_key=embedding_key,
                                openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"))
    elif embedding == "zhipuai":
        return ZhipuAIEmbeddings(zhipuai_api_key=embedding_key)
    else:
        raise ValueError(f"embedding {embedding} not support ")
