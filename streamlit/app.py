"""
RAG Chatbot — Streamlit UI  (static two-column layout, no toggleable sidebar)
Requires: streamlit, langchain, langchain-community, langchain-google-genai,
          langchain-huggingface, faiss-cpu, python-dotenv
"""

import os, tempfile
from pathlib import Path

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Tokens ──────────────────────────────────────────────── */
:root {
  --bg:       #0a0b1e;
  --panel:    #10112a;
  --surface:  #16183a;
  --surface2: #1e2040;
  --border:   #2a2d5a;
  --primary:  #6366f1;
  --indigo2:  #4f46e5;
  --violet:   #7c3aed;
  --emerald:  #10b981;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --label:    #94a3b8;
}

/* ── Kill Streamlit chrome & sidebar completely ───────────── */
#MainMenu, footer, header { visibility: hidden !important; }
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── Full-bleed background ───────────────────────────────── */
.stApp, .stApp > div { background: var(--bg) !important; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}

/* ── Typography ──────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  color: var(--text) !important;
}

/* ── Left panel ──────────────────────────────────────────── */
.left-panel {
  background: var(--panel);
  border-right: 1px solid var(--border);
  height: 100vh;
  padding: 20px 18px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 0;
}

/* Panel header gradient card */
.vault-header {
  background: linear-gradient(135deg, var(--indigo2), var(--violet));
  border-radius: 12px;
  padding: 16px 18px;
  margin-bottom: 20px;
}
.vault-title { font-size: 17px; font-weight: 800; color: #fff; letter-spacing: -.3px; }
.vault-sub   { font-size: 11px; color: #c7d2fe; margin-top: 3px; }

/* Section headers inside panel */
.panel-section {
  font-size: 10px; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: var(--primary);
  margin: 16px 0 8px;
}

/* File cards */
.file-card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--primary);
  border-radius: 8px;
  padding: 9px 13px;
  margin-bottom: 7px;
  font-size: 12px;
}
.file-card-name { color: #a5b4fc; font-weight: 600; word-break: break-all; }
.file-card-meta { color: var(--muted); font-size: 11px; margin-top: 2px; }

/* Badges */
.badge {
  display: inline-block; border-radius: 999px;
  padding: 3px 11px; font-size: 11px; font-weight: 700;
}
.badge-green { background:#052e16; color:#4ade80; border:1px solid #166534; }

/* Hint text */
.hint-text {
  color: #475569; font-size: 12px; line-height: 1.7; margin-top: 10px;
}

/* ── Right chat panel ────────────────────────────────────── */
.right-panel {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--bg);
  padding: 0;
}

/* Chat header bar */
.chat-header {
  border-bottom: 1px solid var(--border);
  padding: 18px 28px 14px;
  display: flex; align-items: center; gap: 12px;
  flex-shrink: 0;
}
.chat-title {
  font-size: 20px; font-weight: 800;
  background: linear-gradient(135deg, #a5b4fc, #e879f9);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -.4px;
}
.status-pill {
  margin-left: auto; display: flex; align-items: center; gap: 6px;
  font-size: 12px; color: var(--muted);
}
.dot-on  { width:8px; height:8px; border-radius:50%; background:var(--emerald); box-shadow:0 0 6px var(--emerald); }
.dot-off { width:8px; height:8px; border-radius:50%; background:var(--muted); }

/* Scrollable messages area */
.messages-area {
  flex: 1; overflow-y: auto; padding: 20px 28px 12px;
}

/* User bubble */
.msg-user {
  display: flex; justify-content: flex-end; margin: 8px 0;
}
.msg-user-inner {
  background: linear-gradient(135deg, var(--indigo2), var(--violet));
  color: #fff;
  border-radius: 18px 18px 4px 18px;
  padding: 11px 17px;
  max-width: 68%;
  box-shadow: 0 4px 16px rgba(99,102,241,.3);
  font-size: 14px; line-height: 1.6;
}

/* Bot bubble */
.msg-bot {
  display: flex; justify-content: flex-start; margin: 8px 0;
}
.msg-bot-inner {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 18px 18px 18px 4px;
  padding: 13px 17px;
  max-width: 78%;
  box-shadow: 0 2px 8px rgba(0,0,0,.35);
  font-size: 14px; line-height: 1.7;
}

/* Role micro-labels */
.rlabel {
  font-size: 10px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; margin-bottom: 5px; opacity: .6;
}
.rlabel-u { color: #c7d2fe; }
.rlabel-b { color: var(--primary); }

/* Source pills */
.src-row { margin-top: 9px; padding-top: 8px; border-top: 1px solid var(--border); }
.src-lbl  { font-size: 10px; color: var(--muted); margin-bottom: 4px; letter-spacing:.08em; text-transform:uppercase; }
.src-pill {
  display: inline-block;
  background: #1a1b38; border: 1px solid var(--primary);
  color: #a5b4fc; border-radius: 999px;
  padding: 2px 10px; margin: 2px; font-size: 11px;
}

/* Welcome screen */
.welcome {
  text-align: center; padding: 70px 20px 40px; color: var(--muted);
}
.welcome-icon  { font-size: 52px; margin-bottom: 14px; }
.welcome-title { font-size: 22px; font-weight: 700; color: var(--label); margin-bottom: 8px; }
.welcome-body  { font-size: 13px; line-height: 1.75; max-width: 360px; margin: 0 auto; }

/* ── Input strip at bottom ───────────────────────────────── */
.input-strip {
  border-top: 1px solid var(--border);
  padding: 14px 28px;
  flex-shrink: 0;
  background: var(--bg);
}

/* Override Streamlit widget colours inside columns */
.stTextInput > div > div > input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-size: 14px !important;
}
.stTextInput > div > div > input:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,.18) !important;
}

.stButton > button {
  background: linear-gradient(135deg, var(--indigo2), var(--violet)) !important;
  color: #fff !important; border: none !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
  height: 42px !important;
}
.stButton > button:hover { opacity: .87 !important; }

/* Expander */
details > summary {
  background: var(--surface2) !important;
  border-radius: 8px !important;
  color: var(--label) !important;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 13px;
  list-style: none;
}

/* File uploader drop zone */
[data-testid="stFileUploaderDropzone"] {
  background: var(--surface2) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--primary) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)


# ── Lazy RAG imports ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _import_rag():
    from rag import DocumentProcessor, SessionMemory, RAGEngine
    return DocumentProcessor, SessionMemory, RAGEngine


# ── Session state ──────────────────────────────────────────────────────────
def _init():
    defaults = {
        "processor":   None,
        "memory":      None,
        "engine":      None,
        "messages":    [],
        "indexed":     False,
        "file_names":  [],
        "chunk_count": 0,
        "session_dir": tempfile.mkdtemp(prefix="rag_"),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Two-column layout ──────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 3], gap="small")

# ══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL
# ══════════════════════════════════════════════════════════════════════════════
with left_col:
    st.markdown("""
    <div class='vault-header'>
      <div class='vault-title'>🔮 RAG Chatbot</div>
      <div class='vault-sub'>Gemini · LangChain · FAISS</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Settings ─────────────────────────────────────────────────────────────
    with st.expander("⚙️  Settings"):
        temperature = st.slider("Temperature",      0.0, 1.0, 0.3, 0.05)
        retriever_k = st.slider("Top-K Retrieval",  1,   10,  4)
        memory_win  = st.slider("Memory (turns)",   1,   20,  5)
        chunk_size  = st.slider("Chunk Size",       200, 3000, 1000, 100)

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown("<div class='panel-section'>Documents</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "files", type=["pdf","txt","docx","md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    btn_l, btn_r = st.columns(2)
    with btn_l:
        index_btn = st.button("⚡ Index",    use_container_width=True)
    with btn_r:
        clear_btn = st.button("🗑 Clear All", use_container_width=True)

    # ── Index action ──────────────────────────────────────────────────────────
    if index_btn:
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            try:
                DocumentProcessor, SessionMemory, RAGEngine = _import_rag()
                session_dir = st.session_state["session_dir"]

                for uf in uploaded_files:
                    with open(os.path.join(session_dir, uf.name), "wb") as fh:
                        fh.write(uf.getbuffer())

                with st.spinner("Building knowledge base…"):
                    proc = DocumentProcessor(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_size // 5,
                        retriever_k=retriever_k,
                    )
                    for f in Path(session_dir).iterdir():
                        if f.is_file() and f.suffix.lower() in DocumentProcessor.LOADER_MAP:
                            proc.load_file(str(f))
                    proc.build_vector_store()
                    mem = SessionMemory(window_k=memory_win)
                    eng = RAGEngine(proc.get_retriever(), mem, temperature=temperature)

                st.session_state.update({
                    "processor":   proc,
                    "memory":      mem,
                    "engine":      eng,
                    "indexed":     True,
                    "file_names":  proc.loaded_files,
                    "chunk_count": len(proc.chunks),
                    "messages": [{
                        "role": "assistant",
                        "content": (
                            f"Knowledge base ready — **{len(proc.loaded_files)} file(s)** "
                            f"indexed, **{len(proc.chunks)} chunks** created.\n\n"
                            "Ask me anything about your documents."
                        ),
                        "sources": [],
                    }],
                })
                st.rerun()
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    # ── Clear action ──────────────────────────────────────────────────────────
    if clear_btn:
        for f in Path(st.session_state["session_dir"]).iterdir():
            try: f.unlink()
            except: pass
        st.session_state.update({
            "processor": None, "memory": None, "engine": None,
            "messages": [], "indexed": False,
            "file_names": [], "chunk_count": 0,
        })
        st.rerun()

    # ── Knowledge vault cards ─────────────────────────────────────────────────
    if st.session_state["indexed"]:
        st.markdown("<div class='panel-section'>Knowledge Vault</div>", unsafe_allow_html=True)
        st.markdown(
            f"<span class='badge badge-green'>"
            f"● {len(st.session_state['file_names'])} files &nbsp;·&nbsp; "
            f"{st.session_state['chunk_count']} chunks</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        for fname in st.session_state["file_names"]:
            ext = Path(fname).suffix.upper().lstrip(".")
            st.markdown(
                f"<div class='file-card'>"
                f"<div class='file-card-name'>📄 {fname}</div>"
                f"<div class='file-card-meta'>{ext} document</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("💬 New Chat", use_container_width=True):
            if st.session_state["engine"]:
                st.session_state["engine"].memory.clear()
            st.session_state["messages"] = []
            st.rerun()
    else:
        st.markdown(
            "<div class='hint-text'>Upload PDF, TXT, DOCX or MD files "
            "then click <b>⚡ Index</b> to build the knowledge base.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with right_col:
    is_ready = st.session_state["indexed"]

    # Header
    dot = "<span class='dot-on'></span>" if is_ready else "<span class='dot-off'></span>"
    status_txt = "Ready" if is_ready else "No documents indexed"
    st.markdown(
        f"<div class='chat-header'>"
        f"<div class='chat-title'>Document Q&amp;A</div>"
        f"<div class='status-pill'>{dot}{status_txt}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Messages ──────────────────────────────────────────────────────────────
    messages = st.session_state.get("messages", [])

    if not messages:
        st.markdown(
            "<div class='welcome'>"
            "<div class='welcome-icon'>🔮</div>"
            "<div class='welcome-title'>Your AI Research Assistant</div>"
            "<div class='welcome-body'>"
            "Upload documents in the left panel, click <b>⚡ Index</b>, "
            "then ask questions in natural language. "
            "Every answer cites which file it came from."
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in messages:
            role    = msg["role"]
            content = msg["content"]
            sources = msg.get("sources") or []

            if role == "user":
                st.markdown(
                    f"<div class='msg-user'><div class='msg-user-inner'>"
                    f"<div class='rlabel rlabel-u'>You</div>{content}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                src_html = ""
                if sources:
                    pills = "".join(f"<span class='src-pill'>📄 {s}</span>" for s in sources)
                    src_html = f"<div class='src-row'><div class='src-lbl'>Sources</div>{pills}</div>"

                # Open bubble
                st.markdown(
                    f"<div class='msg-bot'><div class='msg-bot-inner'>"
                    f"<div class='rlabel rlabel-b'>Assistant</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(content)
                if src_html:
                    st.markdown(src_html, unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)

    # ── Input bar ─────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    inp_c, btn_c = st.columns([9, 1])
    with inp_c:
        question = st.text_input(
            "q", placeholder="Ask a question about your documents…",
            label_visibility="collapsed",
            key="q_input",
            disabled=not is_ready,
        )
    with btn_c:
        send_btn = st.button("Send", use_container_width=True, disabled=not is_ready)

    # ── Handle send ───────────────────────────────────────────────────────────
    if send_btn and question and is_ready:
        engine = st.session_state["engine"]
        st.session_state["messages"].append({"role": "user", "content": question, "sources": []})

        with st.spinner("Thinking…"):
            try:
                result  = engine.ask(question)
                answer  = result["answer"]
                sources = result.get("sources") or []
            except Exception as exc:
                answer  = f"⚠️ Error: {exc}"
                sources = []

        st.session_state["messages"].append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()