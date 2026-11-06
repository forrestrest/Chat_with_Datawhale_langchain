#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Desc    : FastAPI 服务：通过 HTTP 接口调用知识库问答链（不带历史）
           启动：python api.py  或  uvicorn api:app --reload
'''

from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys

# 导入功能模块目录（锚定到项目根）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qa_chain.QA_chain_self import QA_chain_self
from database.create_db import DEFAULT_DB_PATH, DEFAULT_PERSIST_PATH

app = FastAPI()  # 创建 api 对象

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
有用的回答:"""


# 定义一个数据模型，用于接收 POST 请求中的数据
class Item(BaseModel):
    prompt: str                      # 用户 prompt
    model: str = "glm-5.3-flash"     # 使用的模型名（OpenAI 兼容通道）
    temperature: float = 0.1         # 温度系数
    api_key: str = None              # 大模型 API Key（不传则从 .env 读取）
    db_path: str = DEFAULT_PERSIST_PATH   # 向量库持久化目录（默认项目根 vector_db/chroma）
    file_path: str = DEFAULT_DB_PATH      # 知识库源文件目录（向量库为空时用于建库）
    prompt_template: str = template       # 提示词模板
    input_variables: list = ["context", "question"]
    embedding: str = "m3e"           # Embedding：m3e 本地 / api 自部署服务
    embedding_key: str = None        # Embedding 服务鉴权 Key（可选）
    top_k: int = 5                   # 检索返回条数


@app.post("/")
async def get_response(item: Item):
    # API 为即时服务，仅支持不带历史的检索问答链
    chain = QA_chain_self(model=item.model, temperature=item.temperature, top_k=item.top_k,
                          file_path=item.file_path, persist_path=item.db_path,
                          api_key=item.api_key, embedding=item.embedding,
                          template=item.prompt_template, embedding_key=item.embedding_key)
    return chain.answer(question=item.prompt)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
