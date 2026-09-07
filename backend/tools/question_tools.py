"""Question-answering tools used by the data analyst agent.

Each tool accepts a DataFrame and returns a human-readable answer. The router
at the bottom chooses the most specific tool for a user's question.
"""

from collections import Counter
import re
from typing import Callable

import pandas as pd


Tool = Callable[[pd.DataFrame, str], str]


def _question_words(question: str) -> str:
    return question.strip().lower()


def _find_column(df: pd.DataFrame, question: str):
    """Find a column named in the question, ignoring letter case."""
    question_lower = _question_words(question)
    columns = sorted(df.columns, key=lambda column: len(str(column)), reverse=True)
    return next((column for column in columns if str(column).lower() in question_lower), None)


def _find_label_column(df: pd.DataFrame):
    common_names = {"label", "class", "category", "target", "type", "churn label"}
    for column in df.columns:
        if str(column).lower() in common_names:
            return column

    for column in df.columns:
        values = set(df[column].dropna().astype(str).str.lower().unique())
        if {"spam", "ham"}.issubset(values):
            return column
    return None


def _find_requested_value(df: pd.DataFrame, question: str):
    """Find a categorical value explicitly mentioned in the question."""
    question_lower = _question_words(question)
    label_column = _find_label_column(df)
    if label_column is not None:
        columns = [label_column]
    else:
        columns = [
            column for column in df.columns
            if df[column].nunique(dropna=True) <= 20
        ]
    for column in columns:
        if not (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or pd.api.types.is_categorical_dtype(df[column])
        ):
            continue
        values = sorted(df[column].dropna().astype(str).unique(), key=len, reverse=True)
        for value in values:
            if value.lower() in question_lower:
                return column, value
    return None, None


def _question_measure(question: str, value: str) -> str:
    text = _question_words(question)
    if "message" in text:
        return "messages"
    if "customer" in text:
        return "customers"
    return f"'{value}' values"


def dataset_overview(df: pd.DataFrame, question: str = "") -> str:
    if "customer" in _question_words(question):
        return f"There are {len(df):,} customers in the dataset."
    return f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns."


def missing_values(df: pd.DataFrame, question: str = "") -> str:
    missing_cells = int(df.isna().sum().sum())
    columns_with_missing = int(df.isna().any().sum())
    return (
        f"The dataset has {missing_cells:,} missing cells across "
        f"{columns_with_missing:,} columns."
    )


def duplicate_rows(df: pd.DataFrame, question: str = "") -> str:
    duplicates = int(df.duplicated().sum())
    return f"The dataset contains {duplicates:,} duplicate rows out of {len(df):,} total rows."


def column_list(df: pd.DataFrame, question: str = "") -> str:
    names = ", ".join(map(str, df.columns[:10]))
    suffix = " and more" if len(df.columns) > 10 else ""
    return f"The dataset has {len(df.columns):,} columns: {names}{suffix}."


def numeric_summary(df: pd.DataFrame, question: str) -> str:
    column = _find_column(df, question)
    if column is None:
        return "Please include the name of a numeric column, such as Monthly Charges."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"{column} is not a numeric column, so an average or range cannot be calculated."

    values = df[column].dropna()
    if values.empty:
        return f"{column} has no numeric values to summarize."

    question_lower = _question_words(question)
    if "average" in question_lower or "mean" in question_lower:
        return f"The average {column} is {values.mean():,.2f}."
    if "highest" in question_lower or "maximum" in question_lower or "max" in question_lower:
        return f"The highest {column} is {values.max():,.2f}."
    if "lowest" in question_lower or "minimum" in question_lower or "min" in question_lower:
        return f"The lowest {column} is {values.min():,.2f}."
    return (
        f"{column} ranges from {values.min():,.2f} to {values.max():,.2f}, "
        f"with an average of {values.mean():,.2f}."
    )


def value_distribution(df: pd.DataFrame, question: str) -> str:
    column = _find_column(df, question)
    if column is None:
        return "Please include the name of the column whose values you want to count."

    values = df[column].fillna("Missing").astype(str).value_counts().head(5)
    breakdown = ", ".join(f"{value}: {count:,}" for value, count in values.items())
    return f"The most common values in {column} are {breakdown}."


def categorical_value_analysis(df: pd.DataFrame, question: str) -> str:
    """Count or calculate the percentage of a named categorical value."""
    column, value = _find_requested_value(df, question)
    if column is None:
        return "Please include the categorical value you want to count."

    values = df[column].fillna("Missing").astype(str)
    matching = values.str.casefold().eq(value.casefold())
    count = int(matching.sum())
    total = len(values)
    percentage = count / total * 100 if total else 0
    measure = _question_measure(question, value)
    if "percent" in _question_words(question) or "percentage" in _question_words(question) or "proportion" in _question_words(question):
        return f"{value} represents {percentage:.1f}% of the {measure}."
    return f"There are {count:,} {value} {measure}."


def infer_visualization_column(df: pd.DataFrame, question: str):
    """Resolve a requested column or a categorical value to its source column."""
    column = _find_column(df, question)
    if column is not None:
        return column
    column, _ = _find_requested_value(df, question)
    return column


def visualization_request(df: pd.DataFrame, question: str) -> str:
    column = infer_visualization_column(df, question)
    if column is None:
        return "Please include the column to visualize."
    if pd.api.types.is_numeric_dtype(df[column]):
        return f"A numerical histogram is appropriate for {column}."
    return f"A categorical bar chart is appropriate for {column}."


def label_distribution(df: pd.DataFrame, question: str) -> str:
    label_column = _find_label_column(df)
    if label_column is None:
        return "I could not find a label column to calculate a class distribution."

    values = df[label_column].fillna("Missing").astype(str)
    counts = values.value_counts()
    percentages = values.value_counts(normalize=True).mul(100)
    breakdown = ", ".join(
        f"{label}: {count:,} ({percentages[label]:.1f}%)"
        for label, count in counts.items()
    )
    return f"The {label_column} distribution is {breakdown}."


def churn_summary(df: pd.DataFrame, question: str) -> str:
    if "Churn Label" not in df.columns:
        return "This dataset does not contain a Churn Label column."

    churn_rate = df["Churn Label"].astype(str).str.lower().eq("yes").mean() * 100
    return f"Customer churn is {churn_rate:.1f}% of all records."


def unsupported_time_analysis(df: pd.DataFrame, question: str = "") -> str:
    """Explain why a trend over time cannot be inferred from this dataset."""
    time_columns = [
        column for column in df.columns
        if any(token in str(column).lower() for token in ("date", "time", "year"))
    ]
    if time_columns:
        return "This dataset contains time-related columns, but a time trend requires further analysis."
    if "Churn Label" not in df.columns:
        return "This dataset has no time-based variable, so a change over time cannot be determined."
    churn_rate = df["Churn Label"].astype(str).str.lower().eq("yes").mean() * 100
    return (
        f"This dataset shows an overall churn rate of {churn_rate:.1f}%, but it does not contain "
        "a time-based variable, so I cannot determine whether churn is increasing over time. "
        "I can analyze which customer characteristics are associated with churn."
    )


def churn_by_column(df: pd.DataFrame, column_name: str) -> str:
    """Compare churn rates for each value in a categorical column."""
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist."
    if "Churn Label" not in df.columns:
        return "This dataset does not contain a Churn Label column."

    churned = df["Churn Label"].astype(str).str.lower().eq("yes")
    rates = churned.groupby(df[column_name].fillna("Missing")).mean().mul(100).sort_values(ascending=False)
    if rates.empty:
        return f"There are no values available in {column_name} to compare."
    highest = rates.index[0]
    breakdown = ", ".join(f"{value}: {rate:.1f}%" for value, rate in rates.items())
    return f"{highest} has the highest churn at {rates.iloc[0]:.1f}%. Rates by {column_name}: {breakdown}."


def churn_by_category_question(df: pd.DataFrame, question: str) -> str:
    column = _find_column(df, question)
    if column is None:
        return "Please include the categorical column to compare with churn."
    return churn_by_column(df, column)


def frequent_words(df: pd.DataFrame, question: str) -> str:
    label_column = _find_label_column(df)
    text_column = next(
        (
            column for column in df.columns
            if column != label_column and pd.api.types.is_string_dtype(df[column])
        ),
        None,
    )
    if text_column is None:
        return "I could not find a text column for a word-frequency analysis."

    rows = df
    if label_column is not None:
        spam_rows = df[label_column].astype(str).str.lower().eq("spam")
        if spam_rows.any():
            rows = df.loc[spam_rows]

    stop_words = {
        "the", "and", "for", "you", "your", "this", "that", "with", "have", "are",
        "was", "but", "not", "from", "will", "has", "can", "all", "our", "they",
        "what", "just", "now", "www", "com",
    }
    words = Counter()
    for message in rows[text_column].dropna().astype(str):
        words.update(
            word for word in re.findall(r"[a-zA-Z]{3,}", message.lower())
            if word not in stop_words
        )

    if not words:
        return "I could not find enough text to calculate frequent words."
    common_words = ", ".join(f"{word} ({count})" for word, count in words.most_common(10))
    return f"The most frequent words in {text_column} are: {common_words}."


TOOLS: dict[str, Tool] = {
    "missing_values": missing_values,
    "duplicate_rows": duplicate_rows,
    "frequent_words": frequent_words,
    "label_distribution": label_distribution,
    "value_distribution": value_distribution,
    "categorical_value_analysis": categorical_value_analysis,
    "visualization_request": visualization_request,
    "numeric_summary": numeric_summary,
    "churn_summary": churn_summary,
    "column_list": column_list,
    "dataset_overview": dataset_overview,
    "unsupported_time_analysis": unsupported_time_analysis,
    "churn_by_category": churn_by_category_question,
}


def choose_tool(df: pd.DataFrame, question: str) -> str:
    """Return the tool name that best matches the question."""
    text = _question_words(question)
    column = _find_column(df, question)

    if "missing" in text or "null" in text:
        return "missing_values"
    if "duplicate" in text or "repeated" in text:
        return "duplicate_rows"
    if any(word in text for word in ("row", "record", "size", "shape", "customer", "message")) and _find_requested_value(df, question)[0] is None and not any(
        word in text for word in ("churn", "contract", "average", "mean")
    ):
        return "dataset_overview"
    if "churn" in text and any(word in text for word in ("increasing", "increase", "rising", "trend", "over time")):
        return "unsupported_time_analysis"
    if any(word in text for word in ("chart", "graph", "plot", "visual", "visualization", "histogram", "distribution")):
        return "visualization_request"
    if ("word" in text or "words" in text) and ("spam" in text or _find_label_column(df)):
        return "frequent_words"
    if ("count" in text or "how many" in text or "percentage" in text or "proportion" in text) and _find_requested_value(df, question)[0] is not None:
        return "categorical_value_analysis"
    if ("count" in text or "percentage" in text or "proportion" in text) and _find_label_column(df):
        return "label_distribution"
    if "churn" in text and any(word in text for word in ("highest", "lowest", "by", "which")) and column is not None:
        return "churn_by_category"
    if "most common" in text or "most frequent" in text or "top" in text:
        return "value_distribution"
    if column is not None and pd.api.types.is_numeric_dtype(df[column]) and any(
        word in text for word in ("average", "mean", "highest", "maximum", "max", "lowest", "minimum", "min")
    ):
        return "numeric_summary"
    if "churn" in text:
        return "churn_summary"
    if "column" in text or "feature" in text:
        return "column_list"
    return "dataset_overview"


def answer_question(df: pd.DataFrame, question: str) -> dict[str, str]:
    """Choose and run one tool, returning both its name and its answer."""
    tool_name = choose_tool(df, question)
    return {"tool_name": tool_name, "answer": TOOLS[tool_name](df, question)}
