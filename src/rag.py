"""RAG 核心:檢索相關片段 + 組 prompt + 呼叫 LLM。網頁與批次共用這支。"""
import json
from functools import lru_cache
from typing import Iterator, List, Tuple

import numpy as np

import config
import embedder
import ollama_client as oc

SYSTEM_PROMPT = """你是一個「自然語言處理(NLP)課程」的問答助理。
你只能根據下方提供的「課程內容片段」回答使用者問題。

規則:
1. 只根據片段內容作答,不要編造或補充課程以外的知識。
2. 若片段中找不到答案,直接回答「根據課程內容找不到相關資訊」。
3. 答案務必簡潔、直接:能用一句話或重點就好,不要重複題目、不要多餘客套或冗長說明。
4. 用與問題相同的語言回答:中文問題一律用「繁體中文」(台灣用語)回答,英文問題用英文回答。"""


@lru_cache(maxsize=1)
def _load_index() -> Tuple[np.ndarray, List[dict]]:
    if not config.EMB_PATH.exists() or not config.CHUNKS_PATH.exists():
        raise SystemExit("找不到索引,請先執行: python src/ingest.py")
    emb = np.load(config.EMB_PATH)
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    return emb, chunks


def retrieve(query: str, k: int = None) -> List[dict]:
    """回傳最相關的前 k 個片段,每個附帶 score。"""
    k = k or config.TOP_K
    emb, chunks = _load_index()
    qv = embedder.embed_one(query)        # 已正規化
    scores = emb @ qv                     # 餘弦相似度
    top = np.argsort(-scores)[:k]
    results = []
    for i in top:
        item = dict(chunks[int(i)])
        item["score"] = float(scores[int(i)])
        results.append(item)
    return results


def build_messages(query: str, contexts: List[dict]) -> List[dict]:
    blocks = []
    for n, c in enumerate(contexts, start=1):
        blocks.append(f"[片段 {n} | 來源:{c['source']} 第{c['page']}頁]\n{c['text']}")
    context_text = "\n\n".join(blocks)
    user = f"課程內容片段:\n{context_text}\n\n問題:{query}\n\n請依規則作答。"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer(query: str, k: int = None) -> Tuple[str, List[dict]]:
    """一次取得完整答案 + 使用到的來源片段(批次用)。"""
    contexts = retrieve(query, k)
    reply = oc.chat(build_messages(query, contexts))
    return reply, contexts


def answer_stream(query: str, k: int = None) -> Tuple[Iterator[str], List[dict]]:
    """串流答案 + 來源片段(網頁用)。"""
    contexts = retrieve(query, k)
    return oc.chat_stream(build_messages(query, contexts)), contexts
