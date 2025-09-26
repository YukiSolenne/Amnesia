# 调本地模型，产出 reply+draft 并写入库

import os, json, time, uuid, re, ast
from dotenv import load_dotenv
from urllib import error
from openai_client import call_chat_completion, RETRY_STATUS

# 1) 读取环境变量
load_dotenv()
BASE = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:1234/v1")
KEY  = os.getenv("OPENAI_API_KEY", "lm-studio")
MODEL= os.getenv("MODEL_NAME", "deepseek-r1-distill-qwen-14b")

# 2) 读取系统提示和光谱
with open(os.path.join("prompts","system_librarian.txt"), "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
with open("emotion_schema.json","r",encoding="utf-8") as f:
    EMO = json.load(f)

CENSOR = ["悲伤","难过","抑郁","委屈","孤独","失落","焦虑","紧张","恐惧",
          "害怕","羞愧","内疚","愤怒","生气","烦躁","厌恶","厌倦","沮丧",
          "绝望","快乐","高兴","开心","幸福","兴奋","满足","惊喜","安心",
          "平静","宁静","感动","想念","思念","怀念"]

CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def _extract_braced_candidates(text: str):
    candidates = []
    stack = []
    start = None
    string_char = None
    escape = False
    for idx, ch in enumerate(text):
        if string_char:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == string_char:
                string_char = None
            continue
        if ch in ('"', "'"):
            string_char = ch
            continue
        if ch == '{':
            if not stack:
                start = idx
            stack.append('{')
        elif ch == '}' and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:idx + 1])
                start = None
    return candidates


def _try_parse_dict(candidate: str):
    candidate = candidate.strip()
    if not (candidate.startswith('{') and candidate.endswith('}')):
        return None
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    try:
        data = ast.literal_eval(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


def parse_model_output(content: str) -> dict:
    raw = content.strip()
    if raw:
        data = _try_parse_dict(raw)
        if data is not None:
            return data
    for candidate in _extract_braced_candidates(content):
        data = _try_parse_dict(candidate)
        if data is not None:
            return data
    for block in CODE_BLOCK_PATTERN.findall(content):
        block = block.strip()
        data = _try_parse_dict(block)
        if data is not None:
            return data
        match = re.search(r"return\s+({[\s\S]+?})", block)
        if match:
            data = _try_parse_dict(match.group(1))
            if data is not None:
                return data
    raise ValueError('模型回复无法解析为 JSON。请调整提示词或降低温度')

def aphasia_guard(text:str)->str:
    for w in CENSOR:
        text = re.sub(w, "*", text)
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
        "thinking": draft.get("thinking", ""),
        "metaphor_domain": draft.get("metaphor_domain",""),
        "metaphor_seed": draft.get("metaphor_seed",0)
    }
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(card, ensure_ascii=False)+"\n")
    print("记忆已封存到 data/cards.jsonl\n")

def main():
    user_input = input("写下一段要封存的记忆：\n> ").strip()

    # 3) 调本地模型（单轮对话）
    msg = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"user_input: {user_input}\nemotion_schema: {json.dumps(EMO, ensure_ascii=False)}"}
]

    # 加入重试，以缓解本地代理偶发 5xx/连接抖动
    delay = 1.0
    last_err = None
    resp = None
    for attempt in range(5):
        try:
            resp = call_chat_completion(BASE, KEY, MODEL, msg, temperature=0.7, timeout=300)
            break
        except error.HTTPError as e:
            last_err = e
            if e.code not in RETRY_STATUS or attempt == 4:
                raise
        except error.URLError as e:
            last_err = e
            if attempt == 4:
                raise
        time.sleep(delay)
        delay *= 2
    if resp is None:
        raise last_err
    content = resp["choices"][0]["message"]["content"]

    # 4) 粗暴解析：兼容模型带格式说明或代码块的输出
    data = parse_model_output(content)

    reply = aphasia_guard(data.get("reply",""))
    draft = data.get("draft", {})

    print("\n——馆员的回复——")
    print(reply)

    # 5) 落库
    save_card(user_input, draft)

if __name__ == "__main__":
    main()

