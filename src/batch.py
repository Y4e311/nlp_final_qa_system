"""批次評分用:讀入題目 CSV → 逐題作答 → 輸出「題目+答案」CSV。

用法:
    python src/batch.py 輸入.csv 輸出.csv
    python src/batch.py NLP期末專題_測資範例.csv result.csv

輸入格式:第一欄為題目(可有或沒有標題列,程式會自動判斷)。
輸出格式:A 欄題目、B 欄答案,utf-8-sig 編碼(Excel 可正常顯示中文)。
"""
import sys
import time

import pandas as pd

import config
import embedder
import ollama_client as oc
import rag

HEADER_WORDS = {"題目", "問題", "question", "questions", "q", "input"}


def read_questions(path: str) -> list:
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    col = df.iloc[:, 0].tolist()
    # 若第一列看起來像標題就略過
    if col and col[0].strip().lower() in HEADER_WORDS:
        col = col[1:]
    return [q.strip() for q in col if q and q.strip()]


def main() -> None:
    if len(sys.argv) < 3:
        print("用法: python src/batch.py 輸入.csv 輸出.csv")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    err = embedder.check() or oc.check_health()
    if err:
        raise SystemExit(err)

    questions = read_questions(in_path)
    print(f"共 {len(questions)} 題,使用模型 {config.LLM_MODEL}\n")

    rows, total = [], 0.0
    for i, q in enumerate(questions, start=1):
        t0 = time.time()
        ans, _ = rag.answer(q)
        dt = time.time() - t0
        total += dt
        rows.append({"題目": q, "答案": ans})
        print(f"[{i}/{len(questions)}] ({dt:.1f}s) {q}\n    → {ans}\n")

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False, header=False, encoding="utf-8-sig")
    print(f"已輸出 {out_path}(共 {len(rows)} 題,平均 {total/max(len(rows),1):.1f}s/題)")


if __name__ == "__main__":
    main()
