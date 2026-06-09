"""產生「有教材根據」的測試資料集。

方法:從每章簡報挑出內容最豐富的片段,請 LLM 依該片段生成一題問答(問題+簡潔答案),
中英文交錯。輸出兩個檔:
  - data/測試資料集.csv  (題目, 參考答案, 來源)  → 供自我評估/調教
  - data/測試題目.csv    (題目)                  → 供 batch.py 當輸入

用法: python src/make_testset.py
"""
import csv
import json
import re
from collections import defaultdict

import config
import ollama_client as oc

PER_SOURCE = 3          # 每章取幾個片段出題
MIN_LEN = 120           # 片段至少多長才拿來出題

GEN_PROMPT = """你是出題老師。以下是一段 NLP 課程簡報的內容片段。
請根據「這段內容」設計一個學生可能會問的問題,並給出簡潔、正確的參考答案。

要求:
- 問題與答案都必須能從片段內容得到,不要超出片段。
- 答案要簡短(一句或重點即可),不要冗長。
- 使用語言:{lang}。
- 只輸出 JSON,格式:{{"question": "...", "answer": "..."}}

片段內容:
{chunk}"""


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def main() -> None:
    err = oc.check_health()
    if err:
        raise SystemExit(err)

    chunks = json.load(open(config.CHUNKS_PATH, encoding="utf-8"))
    by_src = defaultdict(list)
    for c in chunks:
        if len(c["text"]) >= MIN_LEN:
            by_src[c["source"]].append(c)

    rows = []
    i = 0
    for src in sorted(by_src):
        picked = sorted(by_src[src], key=lambda c: len(c["text"]), reverse=True)[:PER_SOURCE]
        for c in picked:
            lang = "繁體中文" if i % 2 == 0 else "English"
            i += 1
            prompt = GEN_PROMPT.format(lang=lang, chunk=c["text"][:1200])
            try:
                out = oc.chat([{"role": "user", "content": prompt}])
                qa = parse_json(out)
                q, a = qa.get("question", "").strip(), qa.get("answer", "").strip()
                if q and a:
                    rows.append((q, a, f"{src} p{c['page']}"))
                    print(f"[{len(rows):02d}] ({lang}) {q}\n      → {a}")
            except Exception as e:
                print(f"  跳過 {src} p{c['page']}: {e}")

    # 寫出含參考答案的完整版
    config.INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    full = config.ROOT / "data" / "測試資料集.csv"
    with open(full, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["題目", "參考答案", "來源"])
        w.writerows(rows)

    # 寫出只有題目的輸入版(給 batch.py)
    qonly = config.ROOT / "data" / "測試題目.csv"
    with open(qonly, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        for q, _, _ in rows:
            w.writerow([q])

    print(f"\n完成!共 {len(rows)} 題")
    print(f"  含參考答案:{full}")
    print(f"  只有題目  :{qonly}")


if __name__ == "__main__":
    main()
