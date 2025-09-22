# test_ping.py
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
r = client.chat.completions.create(
    model="qwen-7b-chat",
    messages=[{"role":"user","content":"你好"}],
    max_tokens=64,   # 有些服务端没这个会翻车
    temperature=0.7,
)
print(r.choices[0].message.content)
