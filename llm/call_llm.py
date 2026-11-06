#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Desc   : 大模型调用统一入口。
          所有模型（gpt / glm / deepseek 等）通过 OpenAI 兼容通道调用，
          支持官方 API 或各类中转 API（one-api / new-api 等）。
'''

import os
import requests
import openai
from dotenv import load_dotenv, find_dotenv

# ===== 中转 API（OpenAI 兼容通道）配置 =====
# 提前加载 .env，保证 OPENAI_API_BASE 等配置在 import 时即可用
_ = load_dotenv(find_dotenv())

# 默认模型名单：启动时会尝试从 /models 端点实时拉取并覆盖此名单
OPENAI_MODEL_LIST = [
    "glm-5.3", "glm-5.3-flash", "glm-5.3-flashx",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra",
    "gpt-6-astra", "gpt-6-luna", "gpt-6-sol", "gpt-6.1-sol",
    "mimo-v2.6-flash", "mimo-v2.6-pro",
]


def parse_llm_api_key(model: str = "openai", env_file: dict = None):
    """从环境变量解析 API Key"""
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    return env_file["OPENAI_API_KEY"]


def get_supported_models(api_key: str = None):
    """
    从 OpenAI 兼容接口（官方或中转）的 /models 端点实时拉取当前 Key 可用的模型列表。
    拉取失败（网络异常/端点不支持）时返回 None，调用方可回退到 OPENAI_MODEL_LIST。
    """
    if api_key == None:
        api_key = parse_llm_api_key("openai")
    base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
    try:
        resp = requests.get(f"{base}/models",
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=10)
        resp.raise_for_status()
        models = sorted(m["id"] for m in resp.json().get("data", []) if m.get("id"))
        return models if models else None
    except Exception as e:
        print(f"[get_supported_models] 拉取模型列表失败，将回退默认名单: {e}")
        return None


def get_completion(prompt: str, model: str, temperature: float = 0.1,
                   api_key: str = None, max_tokens: int = 2048):
    """
    调用大模型获取回复（统一走 OpenAI 兼容通道）。

    参数:
        prompt: 输入提示
        model: 模型名（可从中转平台的模型列表中选择）
        temperature: 温度系数，控制生成随机性
        api_key: 不传则从 .env 的 OPENAI_API_KEY 读取
        max_tokens: 回复最大长度
    返回:
        模型回复，字符串
    """
    if api_key == None:
        api_key = parse_llm_api_key("openai")
    openai.api_key = api_key
    # 中转 API：读取 .env 中的 OPENAI_API_BASE，未配置则默认官方地址
    openai.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]
