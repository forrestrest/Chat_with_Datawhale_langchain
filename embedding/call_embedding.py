import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv, find_dotenv

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from embedding.api_embedding import ApiEmbeddings

_ = load_dotenv(find_dotenv())

def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    # embedding_key 参数为兼容旧调用保留，当前两种方式均不使用
    if embedding == 'm3e':
        # 方式一：本地推理（首次运行自动从 HuggingFace 下载 moka-ai/m3e-base）
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    elif embedding == 'api':
        # 方式二：远程自部署 Embedding 服务（OpenAI 兼容接口，如 BGE-M3）
        # 地址/模型名/鉴权在 .env 中配置：EMBEDDING_API_BASE / EMBEDDING_MODEL / EMBEDDING_API_KEY
        return ApiEmbeddings()
    else:
        raise ValueError(f"embedding {embedding} not support ")
