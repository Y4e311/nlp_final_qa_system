"""統一的 embedding 介面,可切換本機(sentence-transformers)或遠端(Ollama)。

預設 backend="local" 使用 BAAI/bge-m3:多語(中英)檢索品質遠優於用 LLM 充當 embedding。
"""
from typing import List

import numpy as np

import config

_model = None


def _get_local():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.LOCAL_EMBED_MODEL)
    return _model


def embed(texts: List[str]) -> np.ndarray:
    """回傳已 L2 正規化的向量矩陣 (N, D);之後用內積即為餘弦相似度。"""
    if config.EMBED_BACKEND == "local":
        model = _get_local()
        arr = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
        )
        return arr.astype("float32")
    # 遠端 Ollama backend
    import ollama_client as oc

    return oc.embed(texts)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]


def check() -> str:
    """確認 embedding backend 可用。沒問題回傳空字串。"""
    try:
        v = embed_one("健康檢查 health check")
        if v is None or len(v) == 0:
            return "embedding 回傳空向量"
        return ""
    except Exception as e:
        if config.EMBED_BACKEND == "local":
            return (
                f"本機 embedding 模型 '{config.LOCAL_EMBED_MODEL}' 載入失敗:{e}\n"
                f"請確認已安裝 sentence-transformers(pip install -r requirements.txt)。"
            )
        return f"遠端 embedding 失敗:{e}"
