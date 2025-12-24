import json
import requests
from urllib import error

RETRY_STATUS = {429, 500, 502, 503, 504}

def call_chat_completion(base_url, api_key, model, messages, *, temperature=0.7, max_tokens=None, timeout=120):
    """
    通用 OpenAI-Compatible Chat API 调用函数。
    支持本地 LM Studio / OpenAI / DeepSeek / Claude 等。
    """

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        # 与原 urllib 兼容：抛出 error.URLError 或 error.HTTPError
        if hasattr(e, "response") and e.response is not None:
            raise error.HTTPError(endpoint, e.response.status_code, str(e), headers, None)
        else:
            raise error.URLError(str(e))
