"""集中管理所有設定。可用環境變數或 .env 覆寫,不改程式碼也能調參數。"""
import os
from pathlib import Path

# ---- 路徑 ----
ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """極簡 .env 載入器(無額外套件):把專案根目錄的 .env 寫進環境變數。"""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv()
SLIDES_DIR = ROOT / "上課簡報"          # 課程簡報 PDF 來源
EXTRA_DIR = ROOT / "補充資料"           # 自行整理的補充知識(.md/.txt)
INDEX_DIR = ROOT / "data" / "index"     # 索引(向量 + 片段)輸出位置
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMB_PATH = INDEX_DIR / "embeddings.npy"

# ---- Ollama(指向遠端伺服器,負責「生成」)----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")        # 遠端伺服器需要時填(放 .env)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")        # 生成模型

# ---- Embedding(負責「檢索」)----
# backend = "local" 用本機 sentence-transformers(品質佳,推薦);"ollama" 走遠端 /api/embed
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "local")
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-m3")  # 多語檢索模型
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen2.5:7b")    # 僅 backend="ollama" 時使用

# ---- 切塊 (chunking) ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # 每塊最大字元數
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # 相鄰塊重疊字元數

# ---- 檢索 / 生成 ----
TOP_K = int(os.getenv("TOP_K", "5"))                    # 取最相關的前 K 個片段
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))    # 越低越穩定、不發散
