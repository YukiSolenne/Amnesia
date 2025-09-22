from urllib import error

from scripts.openai_client import call_chat_completion

BASE_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
MODEL = "qwen-7b-chat"

messages = [{"role": "user", "content": "你好"}]

def call():
    return call_chat_completion(BASE_URL, API_KEY, MODEL, messages, temperature=0.7, max_tokens=64, timeout=60)

if __name__ == "__main__":
    try:
        response = call()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTPError {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc.reason}") from exc
    print(response["choices"][0]["message"]["content"])
