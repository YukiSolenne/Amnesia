import json
from urllib import request, error

RETRY_STATUS = {429, 500, 502, 503, 504}

def call_chat_completion(base_url, api_key, model, messages, *, temperature=0.7, max_tokens=None, timeout=120):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(endpoint, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)
