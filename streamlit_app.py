"""Browser interface for the local grounded document Q&A pipeline."""

from pathlib import Path

import streamlit as st

from rag_bot import _default_persist_dir, answer_question, ingest


st.set_page_config(page_title="Document Q&A", layout="centered")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');
    :root { --ink: #202925; --muted: #68736d; --line: #dce3de; --green: #176b52; }
    .stApp { background: #f7f9f6; color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #edf2ed; border-right: 1px solid var(--line); }
    h1, h2, h3 { font-family: 'Newsreader', Georgia, serif !important; color: var(--ink); }
    p, label, input, button, textarea { font-family: 'DM Sans', 'Segoe UI', sans-serif; }
    .eyebrow { color: var(--green); font: 700 11px 'DM Sans', sans-serif; letter-spacing: 1.4px; }
    .stChatMessage { border: 1px solid var(--line); border-radius: 8px; }
    [data-testid="stChatInput"] { border-color: #b7c7bd; }
    [data-testid="stChatInput"]:focus-within { border-color: var(--green); box-shadow: 0 0 0 1px var(--green); }
    [data-testid="stBaseButton-primary"] { background: var(--green); border-color: var(--green); }
    [data-testid="stBaseButton-primary"]:hover { background: #10543f; border-color: #10543f; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

persist_dir = _default_persist_dir()
data_dir = Path("data")

with st.sidebar:
    st.markdown('<div class="eyebrow">YOUR KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    st.header("Documents")
    st.caption(f"Source folder: `{data_dir}`")
    st.caption(f"Local index: `{persist_dir}`")
    if st.button("Index documents", type="primary", use_container_width=True):
        try:
            with st.spinner("Reading documents and building the search index..."):
                chunk_count = ingest(data_dir, persist_dir)
            st.success(f"Indexed {chunk_count:,} passages.")
        except Exception as error:
            st.error(str(error))

    st.divider()
    top_k = st.slider("Sources per answer", min_value=1, max_value=10, value=4)
    st.caption("Answers use retrieved passages and include source locations.")

st.markdown('<div class="eyebrow">LOCAL DOCUMENT SEARCH · GEMINI + CHROMA</div>', unsafe_allow_html=True)
st.title("Ask your library")
st.write("Get answers grounded in your PDFs and text files, with citations back to the source.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources")
            for source in message["sources"]:
                st.caption(f"[{source['reference']}] {source['source']} · {source['location']}")

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents..."):
                result = answer_question(question, persist_dir, top_k)
            st.markdown(result["answer"])
            st.caption("Sources")
            for source in result["sources"]:
                st.caption(f"[{source['reference']}] {source['source']} · {source['location']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
            )
        except Exception as error:
            message = f"I couldn't answer that question: {error}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})