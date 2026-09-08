import html
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from backend.tools.question_tools import answer_question as run_question
from backend.tools.question_tools import infer_visualization_column


st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(circle at 12% 0%, #173b3b 0, transparent 32rem), radial-gradient(circle at 95% 24%, #2b2930 0, transparent 26rem), #0b151b; color: #edf4f2; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1120px; padding: 3.2rem 2rem 4.5rem; }
    .app-header { display: flex; justify-content: space-between; align-items: end; margin-bottom: 2.8rem; }
    .eyebrow { color: #70e0c6; font: 700 0.7rem 'Space Mono', monospace; letter-spacing: 0.14em; text-transform: uppercase; }
    .app-title { color: #f8fbfa; font: 700 2.55rem/1.08 'DM Sans', sans-serif; letter-spacing: -0.02em; margin: 0.45rem 0 0; }
    .app-subtitle { color: #a8bdbc; font: 400 0.98rem/1.5 'DM Sans', sans-serif; margin: 0.7rem 0 0; max-width: 34rem; }
    .status { background: rgba(28, 87, 76, 0.58); border: 1px solid #398675; border-radius: 999px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18); color: #8df0d6; font: 600 0.76rem 'DM Sans', sans-serif; padding: 0.48rem 0.82rem; }
    .input-panel, .answer-panel { background: rgba(18, 36, 44, 0.88); border: 1px solid #2c4b52; border-radius: 12px; box-shadow: 0 18px 50px rgba(0, 0, 0, 0.16); padding: 1.5rem; }
    .panel-title { color: #f8fbfa; font: 700 1.05rem 'DM Sans', sans-serif; margin: 0 0 0.3rem; }
    .panel-copy { color: #96acab; font: 400 0.86rem/1.45 'DM Sans', sans-serif; margin: 0 0 1.1rem; }
    div[data-testid="stFileUploader"] section { background: rgba(10, 27, 34, 0.82); border: 1px dashed #5b8e8d; border-radius: 9px; padding: 1.1rem; transition: border-color 160ms ease, background 160ms ease; }
    div[data-testid="stFileUploader"] section:hover { background: rgba(18, 47, 49, 0.82); border-color: #70e0c6; }
    div[data-testid="stFileUploader"] small { display: none; }
    div[data-testid="stFileUploaderDropzoneInstructions"] span { color: #c7d5d3; font-family: 'DM Sans', sans-serif; }
    div[data-testid="stTextArea"] textarea { background: rgba(10, 27, 34, 0.82); border: 1px solid #31515a; border-radius: 9px; color: #edf4f2; font-family: 'DM Sans', sans-serif; padding: 0.8rem; }
    div[data-testid="stTextArea"] textarea:focus { border-color: #70e0c6; box-shadow: 0 0 0 1px #70e0c6; }
    div[data-testid="stTextArea"] label, div[data-testid="stFileUploader"] label { display: none; }
    .stButton > button { background: #70e0c6; border: 0; border-radius: 7px; box-shadow: 0 8px 20px rgba(44, 168, 143, 0.18); color: #08201c; font: 700 0.9rem 'DM Sans', sans-serif; margin-top: 1rem; padding: 0.7rem 1.25rem; transition: transform 160ms ease, background 160ms ease, box-shadow 160ms ease; }
    .stButton > button:hover { background: #a0f1de; box-shadow: 0 11px 24px rgba(44, 168, 143, 0.28); color: #08201c; transform: translateY(-1px); }
    .section-heading { color: #f8fbfa; font: 700 1.15rem 'DM Sans', sans-serif; margin: 2rem 0 0.8rem; }
    .metric-card, .driver-card { background: rgba(18, 36, 44, 0.88); border: 1px solid #2c4b52; border-radius: 10px; box-shadow: 0 12px 30px rgba(0, 0, 0, 0.12); padding: 1rem; }
    .metric-label { color: #8fa5a8; font: 500 0.75rem 'DM Sans', sans-serif; text-transform: uppercase; }
    .metric-value { color: #f8fbfa; font: 700 1.45rem 'Space Mono', monospace; margin-top: 0.35rem; }
    .driver-rank { color: #62d3bd; font: 700 0.75rem 'Space Mono', monospace; }
    .driver-name { color: #f8fbfa; font: 600 1rem 'DM Sans', sans-serif; margin-top: 0.35rem; }
    .answer-panel { border-left: 3px solid #f3bd62; }
    .answer-label { color: #f3bd62; font: 700 0.75rem 'Space Mono', monospace; text-transform: uppercase; }
    .answer-text { color: #d5e2df; font: 400 1rem/1.65 'DM Sans', sans-serif; margin: 0.6rem 0 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode this CSV. Please save it as UTF-8 or Latin-1 and upload it again.")
    return pd.read_excel(uploaded_file)


def render_visualization(df: pd.DataFrame, question: str):
    st.markdown('<div class="section-heading">Dataset visualization</div>', unsafe_allow_html=True)
    available_columns = [
        column for column in df.columns
        if not (str(column).lower().endswith("id") or str(column).lower() in {"customerid", "customer id"})
    ]
    question_lower = question.lower()
    requested_column = infer_visualization_column(df, question)
    if requested_column is not None and requested_column not in available_columns:
        available_columns.insert(0, requested_column)
    default_index = available_columns.index(requested_column) if requested_column in available_columns else 0
    selected_column = st.selectbox(
        "Chart column",
        available_columns,
        index=default_index,
        key="visualization_column",
    )
    series = df[selected_column].dropna()
    figure, axis = plt.subplots(figsize=(10, 4.5))
    figure.patch.set_facecolor("#14252d")
    axis.set_facecolor("#14252d")
    if pd.api.types.is_numeric_dtype(series):
        axis.hist(series, bins=min(30, max(8, series.nunique())), color="#62d3bd", edgecolor="#0d1820")
        axis.set_ylabel("Frequency", color="#b7c9c7")
    else:
        counts = series.astype(str).value_counts().head(12).sort_values()
        axis.barh(counts.index, counts.values, color="#f3bd62")
        axis.set_ylabel("")
        axis.set_xlabel("Count", color="#b7c9c7")
    axis.set_title(f"Distribution of {selected_column}", color="#f8fbfa", loc="left", pad=14)
    axis.tick_params(colors="#b7c9c7")
    for spine in axis.spines.values():
        spine.set_color("#31515a")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def answer_question(df: pd.DataFrame, question: str, drivers: list[str]) -> str:
    """Run the shared question router used by the agent graph."""
    return run_question(df, question)["answer"]


def is_visualization_request(question: str) -> bool:
    """Identify common requests for a chart, including requests without a column."""
    return bool(
        re.search(
            r"\b(chart|charts|graph|graphs|plot|plots|visual|visualize|visualization|"
            r"histogram|distribution)\b",
            question.casefold(),
        )
    )


def render_analysis(df: pd.DataFrame, question: str):
    rows, columns = df.shape
    missing_percent = df.isnull().sum().sum() / max(rows * columns, 1) * 100
    insight = answer_question(df, question, [])

    st.markdown('<div class="section-heading">Dataset overview</div>', unsafe_allow_html=True)
    metric_columns = st.columns(3)
    metrics = [("Rows", f"{rows:,}"), ("Columns", f"{columns}"), ("Missing values", f"{missing_percent:.1f}%")]
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">AI answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-panel"><div class="answer-label">Question answered</div><div class="answer-text">{html.escape(insight)}</div></div>', unsafe_allow_html=True)
    if is_visualization_request(question):
        render_visualization(df, question)


def main():
    if "analysis_requested" not in st.session_state:
        st.session_state.analysis_requested = False

    st.markdown('<div class="app-header"><div><div class="eyebrow">AI DATA ANALYST</div><div class="app-title">Ask your data anything.</div><div class="app-subtitle">Upload a dataset and get a clear, focused answer in seconds.</div></div><div class="status">● Ready</div></div>', unsafe_allow_html=True)
    with st.form("analysis_form", clear_on_submit=False):
        st.markdown('<div class="input-panel"><div class="panel-title">Start an analysis</div><div class="panel-copy">Bring a CSV or Excel file, then tell the analyst what you want to know.</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Dataset",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            key="dataset_uploader",
        )
        question = st.text_area(
            "Question",
            placeholder="Example: What are the most common values in each category?",
            height=90,
            label_visibility="collapsed",
            key="analysis_question",
        )
        analyze = st.form_submit_button("Analyze dataset  →")
        st.markdown('</div>', unsafe_allow_html=True)

    if analyze:
        st.session_state.analysis_requested = True
        st.session_state.question = question

    if st.session_state.analysis_requested:
        try:
            if uploaded_file is None:
                st.error("Upload a CSV or Excel dataset before starting the analysis.")
                return
            df = read_uploaded_dataset(uploaded_file)
            render_analysis(df, st.session_state.get("question", ""))
        except Exception as exc:
            st.error(f"Could not analyze dataset: {exc}")


if __name__ == "__main__":
    main()
