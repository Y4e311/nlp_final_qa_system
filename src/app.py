"""Streamlit 網頁聊天介面。

用法:
    streamlit run src/app.py
"""
import streamlit as st

import config
import embedder
import ollama_client as oc
import rag

st.set_page_config(page_title="NLP 課程問答系統", page_icon="📚")
st.title("📚 NLP 課程問答系統 (RAG)")
_emb = config.LOCAL_EMBED_MODEL if config.EMBED_BACKEND == "local" else config.EMBED_MODEL
st.caption(f"知識庫:課程簡報 PDF　|　生成:{config.LLM_MODEL}　|　檢索:{_emb}")

# 啟動前健檢
health = embedder.check() or oc.check_health()
if health:
    st.error(health)
    st.stop()

with st.sidebar:
    st.header("設定")
    top_k = st.slider("檢索片段數 (Top-K)", 1, 10, config.TOP_K)
    st.markdown(
        "**使用說明**\n\n"
        "輸入與 NLP 課程相關的問題(中、英文皆可),系統會從課程簡報檢索相關內容後作答。\n\n"
        "範例:\n"
        "- 什麼是 Time-homogeneous Markov process?\n"
        "- What is TF-IDF?\n"
        "- 幾月幾號是期中考?"
    )
    if st.button("🗑️ 清除對話"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 參考來源"):
                for c in msg["sources"]:
                    st.markdown(
                        f"**{c['source']} 第{c['page']}頁** (相似度 {c['score']:.2f})\n\n"
                        f"> {c['text'][:300]}{'…' if len(c['text']) > 300 else ''}"
                    )

# 接收輸入
if prompt := st.chat_input("請輸入問題…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream, contexts = rag.answer_stream(prompt, k=top_k)
        full = st.write_stream(stream)
        with st.expander("📄 參考來源"):
            for c in contexts:
                st.markdown(
                    f"**{c['source']} 第{c['page']}頁** (相似度 {c['score']:.2f})\n\n"
                    f"> {c['text'][:300]}{'…' if len(c['text']) > 300 else ''}"
                )
    st.session_state.messages.append(
        {"role": "assistant", "content": full, "sources": contexts}
    )
