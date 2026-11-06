from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import sys
sys.path.append("../")
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb


class Chat_QA_chain_self:
    """
    带历史记录的检索问答链

    - model：调用的模型名称（OpenAI 兼容通道）
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前 k 个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - api_key：大模型 API Key（不传则从 .env 读取）
    - embeddings：使用的 embedding 模型（m3e / api）
    - embedding_key：embedding 服务的鉴权 Key（api 方式且服务端有鉴权时使用）
    """
    def __init__(self, model: str, temperature: float = 0.0, top_k: int = 4,
                 chat_history: list = [], file_path: str = None, persist_path: str = None,
                 api_key: str = None, embedding: str = "m3e", embedding_key: str = None):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        self.file_path = file_path
        self.persist_path = persist_path
        self.api_key = api_key
        self.embedding = embedding
        self.embedding_key = embedding_key

        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding, self.embedding_key)

    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()

    def change_history_length(self, history_len: int = 1):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len：控制保留的最近 history_len 次对话
        - chat_history：当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

    def answer(self, question: str = None, temperature=None, top_k=4):
        """
        核心方法，调用问答链
        - question：用户提问
        """
        if len(question) == 0:
            return "", self.chat_history

        if temperature == None:
            temperature = self.temperature

        llm = model_to_llm(self.model, temperature, self.api_key)

        retriever = self.vectordb.as_retriever(search_type="similarity",
                                               search_kwargs={'k': top_k})  # 默认 similarity，k=4

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever
        )

        # 注意：带历史的链每问一个问题会调用两次大模型（改写问题 + 生成回答），
        # 如中转有请求频率限制需留意
        result = qa({"question": question, "chat_history": self.chat_history})
        answer = result['answer']
        self.chat_history.append((question, answer))  # 更新历史记录

        return self.chat_history  # 返回更新后的历史记录
