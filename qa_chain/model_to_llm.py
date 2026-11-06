import sys
sys.path.append("../llm")
import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from llm.call_llm import parse_llm_api_key

_ = load_dotenv(find_dotenv())


def model_to_llm(model: str = None, temperature: float = 0.0, api_key: str = None):
    """
    将模型名转换为 LangChain 的 LLM 实例。
    所有模型统一走 OpenAI 兼容通道（官方或中转 API），
    模型是否可用由 API 端校验，无需在此维护名单。

    参数:
        model: 模型名（从界面下拉框或 /models 接口获取）
        temperature: 温度系数
        api_key: 不传则从 .env 的 OPENAI_API_KEY 读取
    """
    if api_key == None:
        api_key = parse_llm_api_key("openai")
    return ChatOpenAI(model_name=model,
                      temperature=temperature,
                      openai_api_key=api_key,
                      openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
