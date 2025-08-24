# services/openai_client.py
import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 可换成你代理支持的

# 统一创建客户端（在这里设置超时）
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, timeout=30.0)

def chat_completion(messages, temperature=0.2, max_tokens=600):
    """
    非流式：返回完整文本
    """
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = resp.choices[0].message
    return {
        "id": resp.id,
        "content": msg.content,
        "usage": getattr(resp, "usage", None)
    }

def chat_stream(messages, temperature=0.2, max_tokens=600):
    """
    流式：逐块 yield 文本（SSE/NDJSON 外层由路由处理）
    """
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if not delta or delta.content is None:
            continue
        yield delta.content
