# NLP 課程問答系統(RAG)

以 **RAG(檢索增強生成)** 架構打造的問答系統,能回答「自然語言處理」課程教材相關的中、英文問題。
知識庫來自 `上課簡報/` 內的課程 PDF 與 `補充資料/` 的課程資訊。

## 架構

```
13 份課程 PDF ──擷取文字──▶ 切塊(chunk) ──embedding(bge-m3)──▶ 向量索引(numpy)
                                                                       │
使用者問題 ──embedding──▶ 餘弦相似度檢索 Top-K 片段 ───────────────────┤
                                                                       ▼
                          組成 prompt(系統規則 + 片段 + 問題)──▶ LLM(qwen2.5)──▶ 簡潔答案
```

| 元件 | 技術 |
|---|---|
| PDF 文字擷取 | PyMuPDF |
| Embedding | 本機 sentence-transformers `bge-m3`(多語) |
| 向量檢索 | numpy 餘弦相似度 |
| 生成模型 | Ollama `qwen2.5:7b`(可改 `3b`) |
| 介面 | Streamlit 聊天網頁 + CSV 批次腳本 |

## 環境需求與安裝

### 1. 安裝 Python(3.11 或 3.12)
到 <https://www.python.org/downloads/> 下載安裝,安裝時**務必勾選「Add Python to PATH」**。

### 2. 設定連線(.env)
答案「生成」連線到一台 **Ollama 伺服器**(遠端,或本機自架皆可);「檢索向量」則在本機以 sentence-transformers 計算。複製範本並填入連線資訊:
```bash
copy .env.example .env      # macOS/Linux: cp .env.example .env
```
編輯 `.env`:
```
OLLAMA_HOST=https://你的-ollama-伺服器網址
OLLAMA_API_KEY=你的金鑰          # 伺服器需驗證時填,本機自架可留空
LLM_MODEL=qwen2.5:7b
EMBED_BACKEND=local              # 本機用 bge-m3 做檢索(預設,推薦)
LOCAL_EMBED_MODEL=BAAI/bge-m3
```
> ⚠️ **`.env` 已列入 `.gitignore`,API key 不會被上傳到 GitHub。** 切勿把金鑰寫進程式碼或 commit。
>
> 若改用本機 Ollama 做生成:從 <https://ollama.com/download> 安裝,執行 `ollama pull qwen2.5:7b`,並把 `OLLAMA_HOST` 設為 `http://localhost:11434`、`OLLAMA_API_KEY` 留空。

### 3. 安裝 Python 套件
```bash
pip install -r requirements.txt
```
(建議先建虛擬環境:`python -m venv .venv` 後啟用再安裝。)

## 使用步驟

### 步驟 1：建立索引(只需做一次,教材更動時重跑)
```bash
python src/ingest.py
```

### 步驟 2A：開啟網頁聊天介面(互動 demo)
```bash
streamlit run src/app.py
```
瀏覽器開啟後即可輸入問題,並可展開「參考來源」檢視檢索到的簡報頁。

### 步驟 2B：批次處理測資(評分用,CSV 進/出)
```bash
python src/batch.py 輸入題目.csv 輸出結果.csv
# 例:
python src/batch.py NLP期末專題_測資範例.csv result.csv
```
- **輸入**:第一欄為題目(有無標題列皆可)。
- **輸出**:A 欄題目、B 欄答案,`utf-8-sig` 編碼(Excel 直接開不亂碼)。

## 資料集與評估
- **知識庫**:`上課簡報/` 的 12 份課程簡報 PDF,經 `ingest.py` 切成 656 個片段建立向量索引。
- **測試資料集**:`src/make_testset.py` 從每章挑出內容最豐富的片段,請 LLM 生成「有教材根據」的中、英文問答,產出:
  - `data/測試資料集.csv`(題目、參考答案、來源)
  - `data/測試題目.csv`(僅題目,供 `batch.py` 當輸入)
  - `data/測試結果.csv`(本系統實跑輸出,供比對)
  ```bash
  python src/make_testset.py     # 重新產生測試資料集
  ```

## 參數調整
所有參數集中在 `src/config.py`,也可用環境變數覆寫,例如:
```bash
# Windows PowerShell
$env:LLM_MODEL="qwen2.5:3b"; $env:TOP_K="6"; python src/batch.py in.csv out.csv
```
| 變數 | 預設 | 說明 |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:7b` | 生成模型 |
| `EMBED_MODEL` | `bge-m3` | 向量模型 |
| `TOP_K` | `5` | 檢索片段數 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | 切塊大小與重疊 |
| `TEMPERATURE` | `0.2` | 生成隨機度(越低越穩定) |

## 專案結構
```
.
├── 上課簡報/                  # 課程 PDF(知識庫來源)
├── src/
│   ├── config.py             # 設定
│   ├── ollama_client.py      # Ollama embedding / chat 封裝
│   ├── ingest.py             # 建索引:PDF → 切塊 → 向量
│   ├── rag.py                # RAG 核心:檢索 + 生成
│   ├── app.py                # Streamlit 網頁介面
│   └── batch.py              # CSV 批次處理
├── data/index/               # 產生的索引(已 gitignore)
├── requirements.txt
└── README.md
```
