import os
from typing import List
import requests
from langchain.embeddings.base import Embeddings


class ApiEmbeddings(Embeddings):
    """
    调用自部署 Embedding 服务（OpenAI 兼容接口 /embeddings 端点）。
    适用 TEI / Xinference / Ollama / vLLM 等方式部署的 BGE-M3 等模型。

    通过环境变量（.env）配置：
    - EMBEDDING_API_BASE：服务地址（含 /v1），如 http://192.168.1.10:9997/v1
    - EMBEDDING_MODEL：模型名，如 bge-m3
    - EMBEDDING_API_KEY：可选鉴权 Token，服务端无鉴权可留空
    """
    def __init__(self, api_base: str = None, model: str = None,
                 api_key: str = None, batch_size: int = 32, timeout: int = 60):
        self.api_base = (api_base or os.getenv("EMBEDDING_API_BASE", "")).rstrip("/")
        self.model = model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")
        self.batch_size = batch_size
        self.timeout = timeout
        if not self.api_base:
            raise ValueError(
                "未配置 EMBEDDING_API_BASE（自部署 Embedding 服务地址），请在 .env 中设置，"
                "例如 EMBEDDING_API_BASE = \"http://服务器IP:端口/v1\""
            )

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """单次请求：调用服务端 /embeddings 端点"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.api_base}/embeddings",
            headers=headers,
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        # 按 index 排序，保证向量与输入文本顺序一一对应
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """建库时批量向量化（自动按 batch_size 分批，避免单次请求过大）"""
        result = []
        for i in range(0, len(texts), self.batch_size):
            result.extend(self._embed_batch(texts[i:i + self.batch_size]))
        return result

    def embed_query(self, text: str) -> List[float]:
        """提问时把用户问题转成向量"""
        return self._embed_batch([text])[0]
