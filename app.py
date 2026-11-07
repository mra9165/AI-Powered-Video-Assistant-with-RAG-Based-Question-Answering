import os
import re
import tempfile

import streamlit as st

from main import run_pipeline
from core.rag_engine import ask_question

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
)

# ───────────────────────── Theme: cinematic video studio ─────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton, .stTabs {
    font-family: 'Poppins', sans-serif;
}

/* Background: deep midnight with neon glow, like a video editor / theatre screen */
.stApp {
    background:
        radial-gradient(circle at 12% 8%,  rgba(0, 229, 255, 0.16), transparent 38%),
        radial-gradient(circle at 88% 22%, rgba(255, 45, 149, 0.16), transparent 40%),
        radial-gradient(circle at 50% 100%, rgba(124, 77, 255, 0.20), transparent 45%),
        linear-gradient(160deg, #05060f 0%, #0b0f24 55%, #0a0618 100%);
    background-attachment: fixed;
    color: #e8ecff;
}

/* Film-strip edge on the left and right of the page */
.stApp::before, .stApp::after {
    content: "";
    position: fixed;
    top: 0; bottom: 0;
    width: 14px;
    z-index: 0;
    background: repeating-linear-gradient(
        to bottom,
        rgba(255,255,255,0.10) 0 14px,
        transparent 14px 28px
    );
    opacity: 0.35;
}
.stApp::before { left: 0; }
.stApp::after  { right: 0; }

header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1150px; }

/* Hero */
.hero {
    text-align: center;
    padding: 1.6rem 1rem 1.2rem;
}
.hero .play {
    display: inline-flex; align-items: center; justify-content: center;
    width: 74px; height: 74px; border-radius: 50%;
    background: linear-gradient(135deg, #00e5ff, #7c4dff 55%, #ff2d95);
    box-shadow: 0 0 34px rgba(0,229,255,0.55), 0 0 60px rgba(255,45,149,0.35);
    font-size: 2rem; color: #fff; margin-bottom: 0.8rem;
}
.hero h1 {
    font-size: 2.6rem; font-weight: 700; margin: 0.2rem 0;
    background: linear-gradient(90deg, #00e5ff, #a78bfa, #ff2d95);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #9aa4d6; font-size: 1.02rem; margin: 0; }

/* Glass cards */
.card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
}
.title-card {
    border-left: 4px solid #00e5ff;
    margin-bottom: 1rem;
}
.title-card h2 { margin: 0; font-size: 1.5rem; color: #fff; }
.title-card span { color: #7c8bd1; font-size: 0.85rem; letter-spacing: .12em; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.06) !important;
    color: #e8ecff !important;
    border: 1px solid rgba(0,229,255,0.35) !important;
    border-radius: 12px !important;
}
.stTextInput input:focus {
    border-color: #00e5ff !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.25) !important;
}
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.04);
    border: 1px dashed rgba(124,77,255,0.6);
    border-radius: 14px;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #00b8d4, #7c4dff 60%, #ff2d95);
    color: #fff; border: 0; border-radius: 12px;
    padding: 0.65rem 1.4rem; font-weight: 600;
    box-shadow: 0 0 20px rgba(124,77,255,0.45);
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 30px rgba(0,229,255,0.6);
    color: #fff;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px 12px 0 0;
    color: #9aa4d6; padding: 0.6rem 1.1rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.14) !important;
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff;
}

/* Chat */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
}
[data-testid="stChatInput"] textarea { color: #e8ecff !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(8, 10, 28, 0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.small-muted { color: #7c8bd1; font-size: 0.85rem; }
hr { border-color: rgba(255,255,255,0.10); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────────────────────── Helpers ─────────────────────────
def split_title_summary(summary_text: str):
    """summarize() returns 'Title: ...\\n\\nSummary:\\n...' -> (title, body)."""
    m = re.search(r"Title:\s*(.+?)\s*\n+\s*Summary:\s*(.*)", summary_text, flags=re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "Meeting Summary", summary_text.strip()


def save_upload(uploaded) -> str:
    suffix = os.path.splitext(uploaded.name)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return tmp.name


def build_report(result: dict) -> str:
    title, body = split_title_summary(result["summary"])
    return (
        f"# {title}\n\n"
        f"## Summary\n{body}\n\n"
        f"## Action Items\n{result['action_items']}\n\n"
        f"## Key Decisions\n{result['key_decisions']}\n\n"
        f"## Open Questions\n{result['open_questions']}\n\n"
        f"## Full Transcript\n{result['transcript']}\n"
    )


# ───────────────────────── Session state ─────────────────────────
st.session_state.setdefault("result", None)
st.session_state.setdefault("messages", [])

# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    st.markdown("### 🎞️ Studio Settings")
    language = st.radio(
        "Spoken language",
        options=["english", "hinglish"],
        format_func=lambda x: "English (Whisper, local)" if x == "english"
        else "Hinglish (Sarvam, auto-translate)",
    )
    st.markdown("---")
    st.markdown(
        "<div class='small-muted'>Pipeline:<br>"
        "🎧 Download / convert audio<br>"
        "📝 Speech to text<br>"
        "🧠 Summary, actions, decisions, questions<br>"
        "💬 Chat with your video</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.result and st.button("🔄 Start a new video"):
        st.session_state.result = None
        st.session_state.messages = []
        st.rerun()

# ───────────────────────── Hero ─────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="play">▶</div>
        <h1>AI Video Assistant</h1>
        <p>Turn any video or meeting recording into a summary, action items and a chat you can ask anything.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────── Input section ─────────────────────────
if st.session_state.result is None:
    with st.container():
        tab_url, tab_file = st.tabs(["🔗 YouTube link", "📁 Upload file"])

        source = None
        with tab_url:
            url = st.text_input("Paste a YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            if url.strip():
                source = url.strip()
        with tab_file:
            uploaded = st.file_uploader(
                "Upload an audio or video file",
                type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "mov", "aac", "ogg"],
            )

        go = st.button("🚀 Analyze video", use_container_width=True)

    if go:
        if uploaded is not None:
            source = save_upload(uploaded)
        if not source:
            st.warning("Please paste a YouTube link or upload a file first.")
        else:
            with st.status("Processing your video...", expanded=True) as status:
                st.write("🎧 Preparing audio, transcribing, then analyzing. "
                         "Longer videos take a few minutes.")
                try:
                    st.session_state.result = run_pipeline(source, language)
                    st.session_state.messages = []
                    status.update(label="Done! ✅", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Something went wrong", state="error")
                    st.error(f"{type(e).__name__}: {e}")
            if st.session_state.result:
                st.rerun()

# ───────────────────────── Results ─────────────────────────
result = st.session_state.result
if result:
    title, summary_body = split_title_summary(result["summary"])

    st.markdown(
        f"<div class='card title-card'><span>MEETING TITLE</span><h2>{title}</h2></div>",
        unsafe_allow_html=True,
    )

    t_sum, t_act, t_dec, t_q, t_chat, t_tr = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Decisions", "❓ Open Questions",
         "💬 Chat", "📝 Transcript"]
    )

    with t_sum:
        st.markdown(f"<div class='card'>{''}</div>", unsafe_allow_html=True) if False else None
        st.markdown(summary_body)

    with t_act:
        st.markdown(result["action_items"])

    with t_dec:
        st.markdown(result["key_decisions"])

    with t_q:
        st.markdown(result["open_questions"])

    with t_chat:
        st.caption("Ask anything about the video. Answers come from the transcript.")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        question = st.chat_input("Ask a question about this video...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as e:
                        answer = f"Sorry, something went wrong: {e}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

    with t_tr:
        st.text_area("Full transcript", result["transcript"], height=380)

    st.markdown("---")
    st.download_button(
        "⬇️ Download full report (.md)",
        data=build_report(result),
        file_name="video_report.md",
        mime="text/markdown",
        use_container_width=True,
    )