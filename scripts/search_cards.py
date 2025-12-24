import json, os, sys

def load_cards():
    path = os.path.join("data","cards.jsonl")
    if not os.path.exists(path): return []
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def search(keyword=None, tone=None):
    cards = load_cards()
    out = []
    for c in cards:
        text = (c.get("raw_text","") + " " + c.get("summary","")).lower()
        ok1 = True if not keyword else (keyword.lower() in text)
        tones = c.get("spectrum",{}).get("tones",[])
        ok2 = True if not tone else (tone in tones)
        if ok1 and ok2:
            out.append(c)
    return out

if __name__ == "__main__":
    keyword = sys.argv[1] if len(sys.argv)>1 else None
    tone    = sys.argv[2] if len(sys.argv)>2 else None
    res = search(keyword, tone)
    for r in res:
        print(f"[{r['id']}] {r['summary']}  tones={r.get('spectrum',{}).get('tones',[])}")
