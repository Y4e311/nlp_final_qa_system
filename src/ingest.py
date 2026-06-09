"""建立知識庫索引:課程 PDF → 擷取文字 → 切塊 → 向量化 → 存檔。

用法:
    python src/ingest.py
重新建立(課程內容有更動時)再跑一次即可。
"""
import json
import re
from typing import List

import fitz  # PyMuPDF
import numpy as np

import config
import embedder


def clean(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> List[str]:
    """以字元為單位的滑動視窗切塊,盡量在換行/句號處斷開以保留語意。"""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # 往回找一個自然的斷點
            window = text[start:end]
            for sep in ("\n\n", "\n", "。", ". ", "!", "?", "?", ";"):
                pos = window.rfind(sep)
                if pos > size * 0.5:
                    end = start + pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_chunks() -> List[dict]:
    pdfs = sorted(config.SLIDES_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"在 {config.SLIDES_DIR} 找不到任何 PDF")
    records: List[dict] = []
    cid = 0
    for pdf in pdfs:
        doc = fitz.open(pdf)
        for page_no, page in enumerate(doc, start=1):
            page_text = clean(page.get_text("text"))
            if len(page_text) < 10:  # 跳過幾乎沒文字的純圖頁
                continue
            for piece in split_text(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                records.append(
                    {"id": cid, "source": pdf.name, "page": page_no, "text": piece}
                )
                cid += 1
        doc.close()
        print(f"  ✓ {pdf.name}")

    # 補充知識(.md/.txt):自行整理的課程資訊
    if config.EXTRA_DIR.exists():
        for extra in sorted(list(config.EXTRA_DIR.glob("*.md")) + list(config.EXTRA_DIR.glob("*.txt"))):
            text = clean(extra.read_text(encoding="utf-8"))
            for piece in split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                records.append({"id": cid, "source": extra.name, "page": 1, "text": piece})
                cid += 1
            print(f"  ✓ {extra.name}(補充)")
    return records


def main() -> None:
    print(f"[1/3] 擷取並切塊 PDF(來源:{config.SLIDES_DIR})")
    records = extract_chunks()
    print(f"      共 {len(records)} 個片段")

    label = config.LOCAL_EMBED_MODEL if config.EMBED_BACKEND == "local" else config.EMBED_MODEL
    print(f"[2/3] 產生向量(backend={config.EMBED_BACKEND}, model={label};首次下載模型會稍久)")
    err = embedder.check()
    if err:
        raise SystemExit(err)
    embeddings = embedder.embed([r["text"] for r in records])
    print(f"      向量矩陣 shape = {embeddings.shape}")

    print(f"[3/3] 寫入索引到 {config.INDEX_DIR}")
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMB_PATH, embeddings)
    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("完成!現在可以執行 streamlit run src/app.py 或 python src/batch.py")


if __name__ == "__main__":
    main()
