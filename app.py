"""
app.py  ─  CHI Research Assistant  |  Streamlit Frontend v4
============================================================
Frontend ONLY – wraps 9_rag_pipeline.py.

v4 changes:
  - Chip keys use index (not label) – fixes Mobile UI + AI crash
  - UI redesigned: editorial / academic aesthetic, less "AI chatbot" look
  - query_planner.py already has KEY_ALIASES normalization fix
"""

import sys, os, importlib, importlib.util
from pathlib import Path
from collections import Counter

import streamlit as st
import plotly.graph_objects as go
import google.generativeai as genai

# ── path + dotenv ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()

# ── load backend via importlib (file starts with digit) ───────────────────────
_spec = importlib.util.spec_from_file_location("rag_pipeline", ROOT / "9_rag_pipeline.py")
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CHIResearchAssistant = _mod.CHIResearchAssistant
RAGConfig            = _mod.RAGConfig
RAGResponse          = _mod.RAGResponse

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CHI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS  — editorial / academic aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── base ─────────────────────────────────────────────────────────────── */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .block-container { background: #111118 !important; }

.stApp, .stApp *, p, span, li, label,
.stMarkdown, .stMarkdown *,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] * {
    font-family: 'Inter', -apple-system, sans-serif;
    color: #d4d4d8;
}

.main .block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 900px !important;
}

/* ── sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div { background: #0e0e14 !important; border-right: 1px solid #1e1e2a !important; }
[data-testid="stSidebar"] * { color: #71717a !important; }
[data-testid="stSidebar"] input { color: #d4d4d8 !important; background: #18181f !important; border-radius: 6px !important; }

/* ── hide chrome ─────────────────────────────────────────────────────────── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stHeader"], header { visibility: hidden !important; height: 0 !important; }
.stDeployButton { display: none !important; }

/* ── text input ──────────────────────────────────────────────────────────── */
.stTextInput input,
[data-testid="stTextInput"] input,
[data-baseweb="input"] input,
input[type="text"] {
    background: #18181f !important;
    border: 1px solid #27272e !important;
    border-radius: 8px !important;
    color: #f4f4f5 !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.75rem 1rem !important;
    caret-color: #818cf8 !important;
    transition: border-color 0.15s !important;
}
.stTextInput input:focus, [data-baseweb="input"] input:focus {
    border-color: #4f46e5 !important;
    background: #1a1a24 !important;
    box-shadow: none !important;
    outline: none !important;
}
.stTextInput input::placeholder { color: #3f3f46 !important; opacity: 1 !important; }
[data-baseweb="input"], .stTextInput > div > div { background: transparent !important; border: none !important; }

/* ── form ────────────────────────────────────────────────────────────────── */
[data-testid="stForm"],
[data-testid="stForm"] > div { background: transparent !important; border: none !important; padding: 0 !important; }

/* ── form submit button ──────────────────────────────────────────────────── */
[data-testid="stFormSubmitButton"] > button,
.stFormSubmitButton > button {
    background: #4f46e5 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    padding: 0.75rem 1rem !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stFormSubmitButton"] > button:hover,
.stFormSubmitButton > button:hover { background: #4338ca !important; }

/* ── chip / regular buttons ──────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #27272e !important;
    border-radius: 6px !important;
    color: #71717a !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.28rem 0.8rem !important;
    transition: all 0.12s !important;
    box-shadow: none !important;
}
.stButton > button:hover {
    background: #1e1e2a !important;
    border-color: #4f46e5 !important;
    color: #a5b4fc !important;
}

/* ── expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #16161f !important;
    border: 1px solid #1e1e2a !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p { color: #52525b !important; font-size: 0.84rem !important; font-weight: 500 !important; }

/* ── sidebar inputs ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] .stTextInput input { font-size: 0.8rem !important; }
[data-baseweb="select"] div { background: #18181f !important; border-color: #27272e !important; }
[data-baseweb="select"] span { color: #d4d4d8 !important; }
[data-baseweb="popover"] { background: #18181f !important; }

/* ── plotly ──────────────────────────────────────────────────────────────── */
.stPlotlyChart { background: transparent !important; }

/* ── scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #111118; }
::-webkit-scrollbar-thumb { background: #27272e; border-radius: 99px; }

/* ─── component classes ─────────────────────────────────────────────────── */

/* Hero */
.chi-wordmark {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: #4f46e5; margin-bottom: 1.8rem;
}
.chi-title {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 700; letter-spacing: -0.02em; line-height: 1.15;
    color: #f4f4f5; margin-bottom: 0.5rem;
}
.chi-subtitle {
    font-size: 0.92rem; color: #52525b; line-height: 1.6;
    max-width: 480px; margin: 0 auto;
}

/* Answer section */
.answer-section {
    border-left: 3px solid #4f46e5;
    padding: 0 0 0 1.4rem;
    margin-bottom: 1.6rem;
}
.answer-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #4f46e5; margin-bottom: 0.7rem;
}

/* Paper card — bibliography style */
.paper-entry {
    display: flex; gap: 1rem; padding: 0.9rem 0;
    border-bottom: 1px solid #1a1a24;
}
.paper-entry:last-child { border-bottom: none; }
.paper-rank-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #3f3f46; padding-top: 2px;
    min-width: 20px;
}
.paper-body { flex: 1; }
.paper-title-txt {
    font-size: 0.9rem; font-weight: 500; color: #e4e4e7;
    line-height: 1.45; margin-bottom: 0.35rem;
}
.paper-meta-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
.pmeta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #52525b;
}
.pmeta-year { color: #818cf8; }
.pmeta-score { color: #34d399; }
.score-bar { height: 2px; background: #1e1e2a; border-radius: 99px; margin-top: 0.45rem; }
.score-fill { height: 100%; border-radius: 99px; background: #4f46e5; }

/* Section label */
.sec-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #3f3f46; margin: 1.8rem 0 0.8rem;
    display: flex; align-items: center; gap: 0.8rem;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: #1e1e2a; }

/* Query chip */
.qchip {
    display: inline-block; background: #18181f;
    border: 1px solid #27272e; border-radius: 4px;
    padding: 2px 8px; font-size: 0.74rem; color: #71717a;
    margin: 2px 3px 2px 0; font-family: 'JetBrains Mono', monospace;
}

/* Stat grid */
.stat-box {
    background: #16161f; border: 1px solid #1e1e2a;
    border-radius: 6px; padding: 0.7rem; text-align: center;
}
.stat-num { font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #f4f4f5; }
.stat-txt { font-size: 0.62rem; color: #3f3f46; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px; }

/* Status pill */
.status-pill { display: flex; align-items: center; gap: 7px; padding: 0.25rem 0; font-size: 0.78rem; }
.sdot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.sdot-g { background: #22c55e; }
.sdot-r { background: #ef4444; }
.sdot-a { background: #f59e0b; }

/* Quota warning */
.quota-pill {
    background: #1a140a; border: 1px solid #422006;
    border-radius: 6px; padding: 0.4rem 0.7rem;
    font-size: 0.73rem; color: #fb923c; margin-top: 0.5rem;
}

/* Success bar */
.success-bar {
    display: flex; align-items: center; gap: 8px;
    padding: 0.45rem 0.8rem;
    background: #0d1a12; border: 1px solid #14532d;
    border-radius: 6px; margin-bottom: 1.4rem;
    font-size: 0.78rem; color: #4ade80; font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
YEAR_COLORS = {2021: "#818cf8", 2023: "#22d3ee", 2024: "#4ade80"}
DEFAULT_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite"
]
# Key: use index, NOT label — labels with spaces/+ break Streamlit button keys
CHIPS = [
    ("Trust in AI",        "How do users develop trust in AI-assisted decision making systems?"),
    ("Mobile UI + AI",     "What AI-enhanced mobile UI techniques improve user experience?"),
    ("Accessibility",      "What accessibility solutions have been proposed for users with disabilities?"),
    ("Human-AI collab",    "How do humans and AI collaborate in creative and cognitive tasks?"),
    ("AI papers CHI 2024", "What AI-related papers were published in CHI 2024?"),
]

# ══════════════════════════════════════════════════════════════════════════════
def _init_state():
    if "api_keys" not in st.session_state:
        keys = []
        for i in range(1, 10):
            k = os.getenv(f"GOOGLE_API_KEY{i}")
            if k:
                keys.append(k.strip())
        if not keys:
            k = os.getenv("GOOGLE_API_KEY", "")
            if k:
                keys.append(k.strip())
        st.session_state.api_keys = keys
    for k, v in {
        "models":        list(DEFAULT_MODELS),
        "cur_model_idx": 0,
        "cur_key_idx":   0,
        "response":      None,
        "run_error":     None,
        "history":       [],
        "quota_events":  [],
        "auto_run":      "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_assistant():
    try:
        init_key   = st.session_state.api_keys[0] if st.session_state.api_keys else ""
        init_model = st.session_state.models[0]   if st.session_state.models   else "gemini-2.5-flash"
        os.environ["GOOGLE_API_KEY2"] = init_key
        cfg = RAGConfig(model_name=init_model, planner_model=init_model, max_output_tokens=8192)
        return CHIResearchAssistant(cfg), None
    except Exception as e:
        return None, str(e)

def _current_model():
    idx = min(st.session_state.cur_model_idx, len(st.session_state.models) - 1)
    return st.session_state.models[idx]

def _current_key():
    idx = min(st.session_state.cur_key_idx, len(st.session_state.api_keys) - 1)
    return st.session_state.api_keys[idx] if st.session_state.api_keys else ""

def _apply_config(assistant):
    model = _current_model()
    key   = _current_key()
    assistant.config.model_name    = model
    assistant.config.planner_model = model
    if key:
        genai.configure(api_key=key)

def _is_quota(e):
    s = str(e).lower()
    return "429" in s or "quota" in s or "rate limit" in s or "resource_exhausted" in s

def run_with_fallback(assistant, query):
    models = st.session_state.models
    keys   = st.session_state.api_keys
    if not models: return None, "No models configured."
    if not keys:   return None, "No API keys configured."
    for ki, key in enumerate(keys):
        for mi, model in enumerate(models):
            try:
                assistant.config.model_name    = model
                assistant.config.planner_model = model
                genai.configure(api_key=key)
                resp = assistant.ask(query)
                st.session_state.cur_model_idx = mi
                st.session_state.cur_key_idx   = ki
                return resp, None
            except Exception as e:
                if _is_quota(e):
                    st.session_state.quota_events.append(f"429 on {model} / key ...{key[-5:]}")
                    continue
                return None, str(e)
    return None, "All model/key combinations hit quota. Add more in ⚙️ Settings."

# ══════════════════════════════════════════════════════════════════════════════
def _year_chart(papers):
    if not papers: return None
    counts = Counter(p.year for p in papers)
    years  = sorted(counts)
    fig = go.Figure(go.Bar(
        x=[str(y) for y in years],
        y=[counts[y] for y in years],
        marker=dict(color=["#4f46e5","#0e7490","#166534"][:len(years)], line=dict(width=0), opacity=0.9),
        text=[counts[y] for y in years], textposition="outside",
        textfont=dict(color="#3f3f46", size=10, family="JetBrains Mono"),
        hovertemplate="<b>CHI %{x}</b><br>%{y} retrieved<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Distribution by year", font=dict(color="#3f3f46",size=11,family="Inter"), x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0,r=0,t=32,b=0), height=170,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#3f3f46",size=10,family="JetBrains Mono"), tickcolor="rgba(0,0,0,0)"),
        yaxis=dict(showgrid=True, gridcolor="#1a1a24", zeroline=False, tickfont=dict(color="#3f3f46",size=9), tickcolor="rgba(0,0,0,0)"),
        bargap=0.4, showlegend=False,
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def _sidebar(assistant, load_err):
    with st.sidebar:
        st.markdown("""
        <div style='padding:1rem 0 0.6rem;'>
          <div style='font-size:0.65rem;font-weight:700;letter-spacing:0.16em;
                      text-transform:uppercase;color:#4f46e5;margin-bottom:4px;'>
            CHI Research
          </div>
          <div style='font-size:1.1rem;font-weight:700;color:#d4d4d8;letter-spacing:-0.01em;'>
            RAG Assistant
          </div>
          <div style='font-size:0.68rem;color:#3f3f46;margin-top:2px;'>v3.0 — NTNU Demo</div>
        </div>
        <hr style='border:none;border-top:1px solid #1e1e2a;margin:0.5rem 0;'>
        """, unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.63rem;font-weight:700;letter-spacing:0.12em;"
                    "text-transform:uppercase;color:#3f3f46;margin-bottom:0.5rem;'>"
                    "Dataset</p>", unsafe_allow_html=True)
        for lbl, val in [
            ("Papers", "2,635"), ("Chunks", "75,817"),
            ("Years", "2021 · 2023 · 2024"), ("Embedding", "BGE-large"),
            ("LLM", _current_model()),
        ]:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:0.22rem 0;
                        border-bottom:1px solid #161620;'>
              <span style='font-size:0.75rem;color:#3f3f46;'>{lbl}</span>
              <span style='font-size:0.74rem;color:#52525b;font-family:JetBrains Mono,monospace;
                           max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                           text-align:right;'>{val}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border:none;border-top:1px solid #1e1e2a;margin:0.7rem 0;'>",
                    unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.63rem;font-weight:700;letter-spacing:0.12em;"
                    "text-transform:uppercase;color:#3f3f46;margin-bottom:0.4rem;'>"
                    "Status</p>", unsafe_allow_html=True)

        if load_err:
            rows = [("sdot-r","Error loading backend")]
        elif assistant:
            rows = [("sdot-g","Retriever ready"),("sdot-g","Gemini connected"),("sdot-g","FAISS loaded")]
        else:
            rows = [("sdot-a","Initializing…")]
        for cls, lbl in rows:
            st.markdown(f"<div class='status-pill'><span class='sdot {cls}'></span>"
                        f"<span style='color:#52525b;'>{lbl}</span></div>",
                        unsafe_allow_html=True)

        if st.session_state.quota_events:
            n = len(st.session_state.quota_events)
            st.markdown(f"<div class='quota-pill'>⚡ {n} auto-switch(es) — now on "
                        f"<b>{_current_model()}</b></div>", unsafe_allow_html=True)

        st.markdown("<hr style='border:none;border-top:1px solid #1e1e2a;margin:0.7rem 0;'>",
                    unsafe_allow_html=True)

        with st.expander("⚙️ Settings", expanded=False):
            st.markdown("<p style='font-size:0.68rem;font-weight:600;color:#52525b;margin-bottom:0.35rem;'>"
                        "Models (priority order)</p>", unsafe_allow_html=True)
            for i, m in enumerate(st.session_state.models):
                c1, c2 = st.columns([4,1])
                with c1:
                    nm = st.text_input(f"m{i}", value=m, key=f"mi_{i}", label_visibility="collapsed")
                    if nm != m: st.session_state.models[i] = nm
                with c2:
                    if st.button("×", key=f"dm_{i}") and len(st.session_state.models) > 1:
                        st.session_state.models.pop(i)
                        st.session_state.cur_model_idx = 0; st.rerun()
            new_m = st.text_input("add_m", value="", key="add_mi", placeholder="+ model name",
                                   label_visibility="collapsed")
            if st.button("Add model", key="add_m_btn") and new_m.strip():
                st.session_state.models.append(new_m.strip()); st.rerun()

            if st.session_state.models:
                sel = st.selectbox("Active model", options=st.session_state.models,
                                   index=min(st.session_state.cur_model_idx, len(st.session_state.models)-1),
                                   key="msel")
                st.session_state.cur_model_idx = st.session_state.models.index(sel)
                if assistant: _apply_config(assistant)

            st.markdown("<hr style='border:none;border-top:1px solid #1e1e2a;margin:0.5rem 0;'>",
                        unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.68rem;font-weight:600;color:#52525b;margin-bottom:0.35rem;'>"
                        "API Keys</p>", unsafe_allow_html=True)
            for i, k in enumerate(st.session_state.api_keys):
                masked = k[:6]+"…"+k[-4:]
                c1, c2 = st.columns([4,1])
                with c1:
                    active = " ✓" if i == st.session_state.cur_key_idx else ""
                    st.markdown(f"<div style='font-size:0.72rem;color:#52525b;padding:0.2rem 0;"
                                f"font-family:JetBrains Mono,monospace;'>{masked}{active}</div>",
                                unsafe_allow_html=True)
                with c2:
                    if st.button("×", key=f"dk_{i}") and len(st.session_state.api_keys) > 1:
                        st.session_state.api_keys.pop(i)
                        st.session_state.cur_key_idx = 0; st.rerun()
            new_k = st.text_input("add_k", value="", key="add_ki", type="password",
                                   placeholder="+ paste API key", label_visibility="collapsed")
            if st.button("Add key", key="add_k_btn") and new_k.strip():
                st.session_state.api_keys.append(new_k.strip()); st.rerun()
            if st.button("Reset quota counters", key="rqc"):
                st.session_state.quota_events = []
                st.session_state.cur_model_idx = 0
                st.session_state.cur_key_idx   = 0; st.rerun()

        if st.session_state.history:
            st.markdown("<hr style='border:none;border-top:1px solid #1e1e2a;margin:0.7rem 0;'>",
                        unsafe_allow_html=True)
            st.markdown("<p style='font-size:0.63rem;font-weight:700;letter-spacing:0.12em;"
                        "text-transform:uppercase;color:#3f3f46;margin-bottom:0.4rem;'>"
                        "History</p>", unsafe_allow_html=True)
            for q, _ in list(reversed(st.session_state.history))[:5]:
                short = (q[:34]+"…") if len(q) > 34 else q
                st.markdown(f"<div style='font-size:0.73rem;color:#3f3f46;padding:0.18rem 0 0.18rem 7px;"
                             f"border-left:2px solid #27272e;margin-bottom:3px;'>{short}</div>",
                             unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
def _hero():
    st.markdown("""
    <div style='padding:2.4rem 0 1.6rem;border-bottom:1px solid #1a1a24;margin-bottom:1.8rem;'>
      <div class='chi-wordmark'>ACM CHI · Human-Computer Interaction</div>
      <div class='chi-title'>Research Assistant</div>
      <div style='margin-top:0.4rem;font-size:0.88rem;color:#52525b;'>
        Search across 2,635 CHI papers from 2021, 2023 and 2024 using
        BAAI/bge-large-en-v1.5 embeddings and Gemini.
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH — Enter works via st.form; chips use index key (fix for +/space in label)
# ══════════════════════════════════════════════════════════════════════════════
def _render_search():
    if st.session_state.auto_run:
        st.session_state["_q"] = st.session_state.auto_run

    with st.form(key="sf", clear_on_submit=False):
        c1, c2 = st.columns([6, 1])
        with c1:
            st.text_input("q", key="_q", placeholder="Ask a research question…",
                          label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button("Search →", use_container_width=True)

    q = (st.session_state.get("_q") or "").strip()
    if submitted and q:
        return q
    if st.session_state.auto_run:
        val = st.session_state.auto_run
        st.session_state.auto_run = ""
        return val.strip()
    return ""


def _chips():
    st.markdown(
        "<div style='margin:0.6rem 0 0.3rem;font-size:0.7rem;color:#3f3f46;'>"
        "Examples:</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(len(CHIPS))
    for i, (col, (lbl, full_q)) in enumerate(zip(cols, CHIPS)):
        with col:
            # ← FIX: use index i as key, NOT lbl (which has spaces and + chars)
            if st.button(lbl, key=f"chip_{i}"):
                st.session_state.auto_run = full_q
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def _results(resp: RAGResponse):
    plan = resp.plan

    # success bar
    st.markdown(f"""
    <div class='success-bar'>
      <span>✓</span>
      <span>Answer from <strong style='color:#22c55e;'>{len(resp.papers)}</strong> papers
        — <em style='color:#166534;font-style:italic;'>{plan.topic}</em>
      </span>
      <span style='margin-left:auto;font-family:JetBrains Mono,monospace;
                   font-size:0.7rem;color:#166534;'>{_current_model()}</span>
    </div>""", unsafe_allow_html=True)

    # ── Answer ─────────────────────────────────────────────────────────────
    st.markdown("<div class='answer-label'>Answer</div>", unsafe_allow_html=True)
    # Render inside a left-bordered container via a container hack:
    # We use st.container + CSS class via markdown border-left
    st.markdown("""
    <div style='border-left:3px solid #4f46e5;padding-left:1.2rem;margin-bottom:1.4rem;'>
    </div>""", unsafe_allow_html=True)
    # Actual answer rendered as native Streamlit markdown — no HTML div wrapping
    # to prevent any overflow/truncation. This is the fix for cut-off answers.
    left, _ = st.columns([20, 1])
    with left:
        st.markdown(resp.answer)

    # ── Sources ──────────────────────────────────────────────────────────────
    st.markdown("<div class='sec-label'>Sources</div>", unsafe_allow_html=True)

    max_s = max((p.score for p in resp.papers), default=1.0)
    for i, paper in enumerate(resp.papers, 1):
        pct  = int(paper.score / max(max_s, 1e-9) * 100)
        yclr = YEAR_COLORS.get(paper.year, "#818cf8")
        sec  = (paper.section[:28]+"…") if len(paper.section) > 28 else paper.section
        st.markdown(f"""
        <div class='paper-entry'>
          <div class='paper-rank-num'>{i:02d}</div>
          <div class='paper-body'>
            <div class='paper-title-txt'>{paper.title}</div>
            <div class='paper-meta-row'>
              <span class='pmeta pmeta-year' style='color:{yclr};'>{paper.year}</span>
              <span class='pmeta' style='color:#27272e;'>·</span>
              <span class='pmeta pmeta-score'>{paper.score:.3f}</span>
              <span class='pmeta' style='color:#27272e;'>·</span>
              <span class='pmeta'>{sec}</span>
            </div>
            <div class='score-bar'><div class='score-fill' style='width:{pct}%'></div></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # chart
    fig = _year_chart(resp.papers)
    if fig:
        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Retrieval process ───────────────────────────────────────────────────
    with st.expander("View retrieval process", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='stat-box' style='text-align:left;padding:0.8rem;'>
              <div style='font-size:0.6rem;color:#3f3f46;font-weight:700;letter-spacing:0.1em;
                          text-transform:uppercase;margin-bottom:0.3rem;'>Topic</div>
              <div style='font-size:0.85rem;font-weight:500;color:#a5b4fc;'>{plan.topic}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            yval = str(plan.year) if plan.year else "—"
            yclr = YEAR_COLORS.get(plan.year, "#3f3f46") if plan.year else "#3f3f46"
            st.markdown(f"""
            <div class='stat-box' style='text-align:left;padding:0.8rem;'>
              <div style='font-size:0.6rem;color:#3f3f46;font-weight:700;letter-spacing:0.1em;
                          text-transform:uppercase;margin-bottom:0.3rem;'>Year filter</div>
              <div style='font-size:0.85rem;font-weight:500;color:{yclr};'>{yval}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:0.72rem;font-weight:600;color:#3f3f46;margin-bottom:0.3rem;'>"
                    "Generated queries</p>", unsafe_allow_html=True)
        for i, q in enumerate(plan.queries, 1):
            st.markdown(f"<span class='qchip'>Q{i}: {q}</span>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:0.9rem;'></div>", unsafe_allow_html=True)
        sc = st.columns(4)
        for col, (lbl, val) in zip(sc, [
            ("Merged",    resp.total_raw_papers),
            ("Deduped",   resp.after_dedup),
            ("Yr filter", resp.after_filter),
            ("Final",     len(resp.papers)),
        ]):
            with col:
                st.markdown(f"""
                <div class='stat-box'>
                  <div class='stat-num'>{val}</div>
                  <div class='stat-txt'>{lbl}</div>
                </div>""", unsafe_allow_html=True)

        fi = "✓" if resp.filter_applied else ("⚠ fallback" if resp.filter_fallback else "—")
        ft = (f"Applied (year={plan.year})" if resp.filter_applied
              else ("Too few results after filter" if resp.filter_fallback else "Not requested"))
        st.markdown(f"""
        <div style='margin-top:0.8rem;padding:0.55rem 0.8rem;background:#16161f;
                    border:1px solid #1e1e2a;border-radius:6px;
                    font-size:0.74rem;color:#3f3f46;font-family:JetBrains Mono,monospace;'>
          year_filter={fi} &nbsp;|&nbsp; {ft}
          &nbsp;·&nbsp; model={resp.model}
          &nbsp;·&nbsp; ~{resp.context_length:,} tokens
        </div>""", unsafe_allow_html=True)

        if st.session_state.quota_events:
            st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
            for ev in st.session_state.quota_events[-4:]:
                st.markdown(f"<div class='qchip' style='color:#fb923c;border-color:#422006;'>⚡ {ev}</div>",
                            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
def _show_error(msg):
    st.markdown(f"""
    <div style='background:#18100e;border:1px solid #3b1a14;border-radius:8px;
                padding:1.2rem 1.4rem;margin-top:0.8rem;'>
      <div style='font-size:0.88rem;font-weight:600;color:#f87171;margin-bottom:0.3rem;'>Error</div>
      <div style='font-size:0.8rem;color:#71717a;font-family:JetBrains Mono,monospace;
                  white-space:pre-wrap;word-break:break-all;'>{msg}</div>
      <div style='font-size:0.73rem;color:#3f3f46;margin-top:0.6rem;'>
        429 quota error? Open <b>⚙️ Settings</b> in the sidebar to add more API keys or models.
      </div>
    </div>""", unsafe_allow_html=True)

def _empty_state():
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;
                border:1px solid #1a1a24;border-radius:8px;
                margin-top:0.5rem;'>
      <div style='font-size:0.75rem;font-weight:600;letter-spacing:0.12em;
                  text-transform:uppercase;color:#27272e;margin-bottom:0.6rem;'>
        Ready
      </div>
      <div style='font-size:0.85rem;color:#27272e;'>
        Type a question and press <code style='color:#4f46e5;background:#1a1a2e;
        padding:1px 5px;border-radius:3px;'>Enter</code> — or use the examples above.
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    assistant, load_err = _load_assistant()
    _sidebar(assistant, load_err)
    _hero()

    query = _render_search()
    _chips()
    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

    if load_err:
        _show_error(f"Backend failed to load:\n{load_err}")
        return

    if query:
        if assistant: _apply_config(assistant)
        loading = st.empty()
        with loading:
            st.markdown("""
            <div style='text-align:center;padding:2.5rem;'>
              <div style='font-size:0.9rem;color:#52525b;margin-bottom:0.3rem;'>Searching…</div>
              <div style='font-size:0.75rem;color:#27272e;font-family:JetBrains Mono,monospace;'>
                query planner → FAISS retrieval → answer generation
              </div>
            </div>""", unsafe_allow_html=True)
        resp, err = run_with_fallback(assistant, query)
        loading.empty()
        if err:
            st.session_state.response  = None
            st.session_state.run_error = err
        else:
            st.session_state.response  = resp
            st.session_state.run_error = None
            st.session_state.history.append((query, resp))

    if st.session_state.run_error:
        _show_error(st.session_state.run_error)
    elif st.session_state.response:
        _results(st.session_state.response)
    else:
        _empty_state()

if __name__ == "__main__":
    main()
