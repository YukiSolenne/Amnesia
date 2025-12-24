# 调本地模型，产出 reply+draft 并写入库
import os, json, time, uuid, re, ast
from urllib import error
from pathlib import Path
import sys

# Ensure project root on sys.path so `config` and `scripts` are importable
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.openai_client import call_chat_completion, RETRY_STATUS
from config.model_config import get_model_config

# 读取模型配置
cfg = get_model_config()
BASE = cfg["base_url"]
KEY = cfg["api_key"]
MODEL = cfg["name"]

# 读取系统提示与情感光谱
with open(os.path.join("prompts", "system_librarian.txt"), "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
with open("emotion_schema.json", "r", encoding="utf-8") as f:
    EMO = json.load(f)

CENSOR = ["悲伤","难过","抑郁","委屈","孤独","失落","焦虑","紧张","恐惧",
          "害怕","羞愧","内疚","愤怒","生气","烦躁","厌恶","厌倦","沮丧",
          "绝望","快乐","高兴","开心","幸福","兴奋","满足","惊喜","安心",
          "平静","宁静","感动","想念","思念","怀念"]

CODE_BLOCK_PATTERN = re.compile(r"```(?:\w+)?\s*\n(.*?)```", re.DOTALL)


def _extract_braced_candidates(text: str):
    candidates, stack = [], []
    start = string_char = None
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


def aphasia_guard(text: str) -> str:
    for w in CENSOR:
        text = re.sub(w, "*", text)
    return text


def save_card(raw_text: str, draft: dict, verbose: bool = True):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", "cards.jsonl")
    card = {
        "id": str(uuid.uuid4()),
        "created_at": int(time.time() * 1000),
        "raw_text": raw_text,
        "summary": draft.get("summary", ""),
        "keywords": draft.get("keywords", []),
        "spectrum": draft.get("spectrum", {}),
        "thinking": draft.get("thinking", ""),
        "metaphor_domain": draft.get("metaphor_domain", ""),
        "metaphor_seed": draft.get("metaphor_seed", 0)
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(card, ensure_ascii=False) + "\n")
    if verbose:
        print("记忆已封存到 data/cards.jsonl\n")


def process_single_text(user_input: str, verbose: bool = True):
    """处理单条文本并保存"""
    if not user_input or not user_input.strip():
        if verbose:
            print("跳过空文本")
        return False
    
    user_input = user_input.strip()
    
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"user_input: {user_input}\nemotion_schema: {json.dumps(EMO, ensure_ascii=False)}"}
    ]

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

    data = parse_model_output(content)
    draft = data.get("draft", {}) if isinstance(data, dict) else {}
    reply_raw = data.get("reply") or draft.get("reply", "")
    reply = aphasia_guard(reply_raw)

    if verbose:
        print("\n——馆员的回复——")
        print(reply)

    save_card(user_input, draft, verbose=verbose)
    return True


def read_csv_file(filepath: str):
    """尝试多种编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                lines = [line.strip() for line in f if line.strip()]
                return lines
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件时出错 (编码: {enc}): {e}")
            continue
    raise ValueError(f"无法使用常见编码读取文件: {filepath}")


def main():
    # 批量处理 raw.csv 中的文本
    csv_path = os.path.join("raw.csv")
    
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    print(f"开始读取文件: {csv_path}")
    try:
        texts = read_csv_file(csv_path)
        print(f"成功读取 {len(texts)} 条文本\n")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    total = len(texts)
    success_count = 0
    fail_count = 0
    
    for idx, text in enumerate(texts, 1):
        print(f"\n[{idx}/{total}] 处理文本: {text[:50]}...")
        try:
            process_single_text(text, verbose=False)
            success_count += 1
            print(f"✓ 成功处理 ({success_count}/{total})")
        except Exception as e:
            fail_count += 1
            print(f"✗ 处理失败: {e} ({fail_count}/{total})")
            # 继续处理下一条
            continue
    
    print(f"\n\n处理完成！")
    print(f"成功: {success_count}/{total}")
    print(f"失败: {fail_count}/{total}")


if __name__ == "__main__":
    main()

