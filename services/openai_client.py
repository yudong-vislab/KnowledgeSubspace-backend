# services/openai_client.py
import os
import json
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量中读取配置，提供默认值作为后备
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# 统一的请求头
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

def chat_completion(messages, temperature=0.2, max_tokens=600):
    """
    非流式：返回完整文本
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        
        return {
            "id": data.get("id"),
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage")
        }
    except requests.RequestException as e:
        raise Exception(f"请求OpenAI API失败: {str(e)}")

def chat_stream(messages, temperature=0.2, max_tokens=600):
    """
    流式：逐块 yield 文本（SSE/NDJSON 外层由路由处理）
    """
    url = f"{OPENAI_BASE_URL}/chat/completions/v1"
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }
    
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=30.0) as response:
            response.raise_for_status()  # 检查请求是否成功
            
            for chunk in response.iter_lines():
                if chunk:
                    # 移除 'data: ' 前缀
                    chunk_str = chunk.decode('utf-8')
                    if chunk_str.startswith('data: '):
                        chunk_str = chunk_str[6:]
                    # 跳过结束标记
                    if chunk_str == '[DONE]':
                        continue
                    try:
                        # 解析JSON
                        data = json.loads(chunk_str)
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue
    except requests.RequestException as e:
        raise Exception(f"请求OpenAI API失败: {str(e)}")
