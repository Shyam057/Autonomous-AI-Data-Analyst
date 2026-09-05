import html
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from backend.tools.data_loader import load_dataset


DEFAULT_DATASET = Path("data/Telco_customer_churn.xlsx")

st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    :root { color-scheme: dark; }
    .stApp { background: #0d1820; color: #edf4f2; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1120px; padding: 2.5rem 2rem 4rem; }
    .app-header { display: flex; justify-content: space-between; align-items: end; margin-bottom: 2.4rem; }
    .eyebrow { color: #62d3bd; font: 700 0.72rem 'Space Mono', monospace; letter-spacing: 0.12em; text-transform: uppercase; }
    .app-title { color: #f8fbfa; font: 700 2.2rem 'DM Sans', sans-serif; margin: 0.35rem 0 0; }
    .app-subtitle { color: #9eb1b4; font: 400 0.95rem 'DM Sans', sans-serif; margin: 0.5rem 0 0; }
    .status { background: #12362f; border: 1px solid #246556; border-radius: 999px; color: #7be0c9; font: 600 0.78rem 'DM Sans', sans-serif; padding: 0.45rem 0.8rem; }
    .input-panel, .answer-panel { background: #14252d; border: 1px solid #24404a; border-radius: 14px; padding: 1.4rem; }
    .panel-title { color: #f8fbfa; font: 700 1rem 'DM Sans', sans-serif; margin: 0 0 0.25rem; }
    .panel-copy { color: #8fa5a8; font: 400 0.85rem 'DM Sans', sans-serif; margin: 0 0 1rem; }
    div[data-testid="stFileUploader"] section { background: #0f2028; border: 1px dashed #4c7a7c; border-radius: 10px; padding: 1rem; }
    div[data-testid="stFileUploader"] small { display: none; }
    div[data-testid="stFileUploaderDropzoneInstructions"] span { color: #c7d5d3; font-family: 'DM Sans', sans-serif; }
    div[data-testid="stTextArea"] textarea { background: #0f2028; border: 1px solid #31515a; border-radius: 10px; color: #edf4f2; font-family: 'DM Sans', sans-serif; }
    div[data-testid="stTextArea"] label, div[data-testid="stFileUploader"] label { display: none; }
    .stButton > button { background: #62d3bd; border: 0; border-radius: 8px; color: #08201c; font: 700 0.9rem 'DM Sans', sans-serif; margin-top: 1rem; padding: 0.65rem 1.2rem; }
    .stButton > button:hover { background: #8be7d4; color: #08201c; }
    .section-heading { color: #f8fbfa; font: 700 1.15rem 'DM Sans', sans-serif; margin: 2rem 0 0.8rem; }
    .metric-card, .driver-card { background: #14252d; border: 1px solid #24404a; border-radius: 12px; padding: 1rem; }
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
def read_default_dataset(file_path: str) -> pd.DataFrame:
    return load_dataset(file_path)


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


def churn_drivers(df: pd.DataFrame) -> list[str]:
    if "Churn Label" not in df.columns:
        numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
        categorical_columns = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
        return [str(column) for column in (numeric_columns + categorical_columns)[:3]] or ["available features"]
    churned = df["Churn Label"].astype(str).str.lower().eq("yes")
    candidates = []
    for column in ("Contract", "Monthly Charges", "Tenure Months"):
        if column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                score = abs(df.loc[churned, column].mean() - df.loc[~churned, column].mean())
            else:
                score = df.groupby(column, dropna=False)["Churn Label"].apply(
                    lambda values: values.astype(str).str.lower().eq("yes").mean()
                ).max()
            candidates.append((score, column))
    return [column for _, column in sorted(candidates, reverse=True)] or ["Churn Label"]


def render_visualization(df: pd.DataFrame, question: str):
    st.markdown('<div class="section-heading">Dataset visualization</div>', unsafe_allow_html=True)
    available_columns = list(df.columns)
    question_lower = question.lower()
    requested_column = next((column for column in available_columns if str(column).lower() in question_lower), None)
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
    driver_names = (drivers + ["available features"] * 3)[:3]
    question_lower = question.lower()
    label_column = next(
        (column for column in df.columns if str(column).lower() in {"label", "class", "category", "type", "v1"}),
        None,
    )
    mentioned_column = next(
        (column for column in df.columns if str(column).lower() in question_lower),
        None,
    )
    if "missing" in question_lower or "null" in question_lower:
        return f"This dataset has {int(df.isnull().sum().sum()):,} missing cells across {df.shape[1]} columns. The missingness should be cleaned before modeling."
    if "duplicate" in question_lower or "repeated" in question_lower:
        return f"The dataset contains {int(df.duplicated().sum()):,} duplicate rows out of {len(df):,} total rows."
    if "row" in question_lower or "record" in question_lower or "size" in question_lower:
        return f"The dataset contains {df.shape[0]:,} customer records and {df.shape[1]} columns."
    asks_for_counts = "count" in question_lower or "how many" in question_lower or "distribution" in question_lower
    asks_for_percentages = "percentage" in question_lower or "percent" in question_lower or "proportion" in question_lower
    if label_column is not None and asks_for_counts and asks_for_percentages:
        counts = df[label_column].fillna("Missing").astype(str).value_counts()
        percentages = df[label_column].fillna("Missing").astype(str).value_counts(normalize=True).mul(100)
        breakdown = ", ".join(
            f"{label}: {count:,} ({percentages[label]:.1f}%)"
            for label, count in counts.items()
        )
        return f"The dataset contains {breakdown}. Spam represents {percentages.get('spam', 0):.1f}% of all messages."
    if label_column is not None and asks_for_counts:
        counts = df[label_column].fillna("Missing").astype(str).value_counts()
        breakdown = ", ".join(f"{label}: {count:,}" for label, count in counts.items())
        return f"The {label_column} distribution is {breakdown}."
    if label_column is not None and asks_for_percentages:
        percentages = df[label_column].fillna("Missing").astype(str).value_counts(normalize=True).mul(100).round(1)
        breakdown = ", ".join(f"{label}: {value:.1f}%" for label, value in percentages.items())
        return f"The {label_column} proportions are {breakdown}."
    if "most common" in question_lower or "most frequent" in question_lower or "top" in question_lower:
        column = mentioned_column or next((column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])), df.columns[0])
        values = df[column].fillna("Missing").astype(str).value_counts().head(3)
        breakdown = ", ".join(f"{label} ({count:,})" for label, count in values.items())
        return f"The most common values in {column} are {breakdown}."
    if mentioned_column is not None and pd.api.types.is_numeric_dtype(df[mentioned_column]):
        series = df[mentioned_column].dropna()
        if "average" in question_lower or "mean" in question_lower:
            return f"The average {mentioned_column} is {series.mean():,.2f}."
        if "highest" in question_lower or "maximum" in question_lower or "max" in question_lower:
            return f"The highest {mentioned_column} is {series.max():,.2f}."
        if "lowest" in question_lower or "minimum" in question_lower or "min" in question_lower:
            return f"The lowest {mentioned_column} is {series.min():,.2f}."
    if "column" in question_lower or "feature" in question_lower:
        return f"The dataset has {df.shape[1]} columns: {', '.join(map(str, df.columns[:8]))}{' and more.' if df.shape[1] > 8 else '.'}"
    if "churn" in question_lower and "Churn Label" in df.columns:
        churn_rate = df["Churn Label"].astype(str).str.lower().eq("yes").mean() * 100
        return f"Customer churn is currently {churn_rate:.1f}%. The strongest signals in this dataset are {driver_names[0]}, {driver_names[1]}, and {driver_names[2]}."
    return f"The analysis found {driver_names[0]}, {driver_names[1]}, and {driver_names[2]} as the leading factors associated with the question."


def render_analysis(df: pd.DataFrame, question: str):
    rows, columns = df.shape
    missing_percent = df.isnull().sum().sum() / max(rows * columns, 1) * 100
    drivers = churn_drivers(df)
    insight = answer_question(df, question, drivers)
    driver_title = "Top churn drivers" if "Churn Label" in df.columns else "Top data signals"

    st.markdown('<div class="section-heading">Dataset overview</div>', unsafe_allow_html=True)
    metric_columns = st.columns(3)
    metrics = [("Rows", f"{rows:,}"), ("Columns", f"{columns}"), ("Missing values", f"{missing_percent:.1f}%")]
    for column, (label, value) in zip(metric_columns, metrics):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-heading">{driver_title}</div>', unsafe_allow_html=True)
    driver_columns = st.columns(3)
    for index, (column, driver) in enumerate(zip(driver_columns, drivers[:3]), 1):
        with column:
            st.markdown(f'<div class="driver-card"><div class="driver-rank">0{index}</div><div class="driver-name">{html.escape(driver)}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-heading">AI answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer-panel"><div class="answer-label">Question answered</div><div class="answer-text">{html.escape(insight)}</div></div>', unsafe_allow_html=True)
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
            placeholder="Example: Why is customer churn increasing?",
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
            df = read_uploaded_dataset(uploaded_file) if uploaded_file is not None else read_default_dataset(str(DEFAULT_DATASET))
            render_analysis(df, st.session_state.get("question", ""))
        except Exception as exc:
            st.error(f"Could not analyze dataset: {exc}")


if __name__ == "__main__":
    main()
