"""與本機 Ollama 溝通的薄封裝:embedding 與 chat。"""
from typing import Iterator, List

import numpy as np
import requests

import config


def _headers() -> dict:
    """遠端伺服器若設了 API key,帶上 Bearer 驗證標頭。"""
    if config.OLLAMA_API_KEY:
        return {"Authorization": f"Bearer {config.OLLAMA_API_KEY}"}
    return {}


def embed(texts: List[str], batch_size: int = 16) -> np.ndarray:
    """把一批文字轉成正規化後的向量矩陣 (N, D)，方便用內積算餘弦相似度。"""
    vectors: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/embed",
            json={"model": config.EMBED_MODEL, "input": batch},
            headers=_headers(),
            timeout=300,
        )
        resp.raise_for_status()
        vectors.extend(resp.json()["embeddings"])
    arr = np.asarray(vectors, dtype=np.float32)
    # L2 正規化：之後做矩陣乘法就等於餘弦相似度
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_one(text: str) -> np.ndarray:
    """單句版本，回傳一維向量。"""
    return embed([text])[0]


def chat(messages: List[dict]) -> str:
    """一次性取得完整回覆(批次處理用)。"""
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": config.TEMPERATURE},
        },
        headers=_headers(),
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def chat_stream(messages: List[dict]) -> Iterator[str]:
    """串流逐字回覆(網頁介面用,體感較快)。"""
    import json

    with requests.post(
        f"{config.OLLAMA_HOST}/api/chat",
        json={
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": True,
            "options": {"temperature": config.TEMPERATURE},
        },
        headers=_headers(),
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break


def check_health() -> str:
    """確認可連到 Ollama 伺服器、且生成模型存在。沒問題回傳空字串。"""
    try:
        resp = requests.get(
            f"{config.OLLAMA_HOST}/api/tags", headers=_headers(), timeout=10
        )
        resp.raise_for_status()
    except Exception as e:
        return f"無法連線到 Ollama 伺服器 ({config.OLLAMA_HOST})。詳細:{e}"
    installed = {m["name"] for m in resp.json().get("models", [])}
    installed |= {n.split(":")[0] for n in installed}  # 同時容許不帶 tag 的比對
    if config.LLM_MODEL not in installed and config.LLM_MODEL.split(":")[0] not in installed:
        return f"伺服器上找不到生成模型 '{config.LLM_MODEL}'。"
    return ""
