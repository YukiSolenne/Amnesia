# 调本地模型 → 产出 reply+draft → 写入库

import os, json, time, uuid, re
from dotenv import load_dotenv
from openai import OpenAI
from openai import InternalServerError, APIConnectionError, RateLimitError

# 1) 读取环境变量
load_dotenv()
BASE = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:1234/v1")
KEY  = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL= os.getenv("MODEL_NAME", "qwen-7b-chat")

client = OpenAI(base_url=BASE, api_key=KEY)

# 2) 读取系统提示和光谱
with open(os.path.join("prompts","system_librarian.txt"), "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
with open("emotion_schema.json","r",encoding="utf-8") as f:
    EMO = json.load(f)

CENSOR = ["悲伤","难过","抑郁","委屈","孤独","失落","焦虑","紧张","恐惧",
          "害怕","羞耻","内疚","愤怒","生气","烦躁","厌恶","厌倦","沮丧",
          "绝望","快乐","高兴","开心","幸福","兴奋","满足","惊喜","安心",
          "平静","宁静","感动","想念","思念","怀念"]

def aphasia_guard(text:str)->str:
    for w in CENSOR:
        text = re.sub(w, "∎", text)
    return text

def save_card(raw_text:str, draft:dict):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data","cards.jsonl")
    card = {
        "id": str(uuid.uuid4()),
        "created_at": int(time.time()*1000),
        "raw_text": raw_text,
        "summary": draft.get("summary",""),
        "keywords": draft.get("keywords",[]),
        "spectrum": draft.get("spectrum",{}),
        "metaphor_domain": draft.get("metaphor_domain",""),
        "metaphor_seed": draft.get("metaphor_seed",0)
    }
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(card, ensure_ascii=False)+"\n")
    print("✔ 已封存到 data/cards.jsonl\n")

def main():
    user_input = input("写下一段要封存的记忆：\n> ").strip()

    # 3) 调本地模型（单轮对话）
    msg = [
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": f"user_input: {user_input}\ncontext_cards: []"}
    ]
    # 加入重试，以缓解本地代理偶发 5xx/连接抖动
    delay = 1.0
    last_err = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=msg,
                temperature=0.7,
                timeout=120,
            )
            break
        except (InternalServerError, APIConnectionError, RateLimitError) as e:
            last_err = e
            if attempt == 4:
                raise
            time.sleep(delay)
            delay *= 2
    content = resp.choices[0].message.content

    # 4) 粗暴解析：尝试提取 JSON（模型会输出含 reply 和 draft 的文本）
    # 约定：模型只输出一个对象，包含 reply 和 draft
    try:
        data = json.loads(content)
    except:
        # 如果模型前后带了说明文字，尝试抓取第一个 {...}
        m = re.search(r"\{[\s\S]*\}", content)
        assert m, "模型没有返回 JSON，可以再试一次或降低温度"
        data = json.loads(m.group(0))

    reply = aphasia_guard(data.get("reply",""))
    draft = data.get("draft", {})

    print("\n—— 馆员的回复 ——")
    print(reply)

    # 5) 落库
    save_card(user_input, draft)

if __name__ == "__main__":
    main()
