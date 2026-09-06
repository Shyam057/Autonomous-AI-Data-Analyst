"""LangChain tools exposed to the autonomous analyst.

These adapters keep the tool schemas clear to the LLM while reusing the
existing pandas analysis functions.
"""

from langchain_core.tools import tool

try:
    from backend.tools.data_loader import load_dataset
    from backend.tools.data_profile import get_dataset_profile
    from backend.tools.question_tools import (
        churn_summary,
        churn_by_column,
        unsupported_time_analysis,
        categorical_value_analysis,
        infer_visualization_column,
        column_list,
        dataset_overview,
        duplicate_rows,
        label_distribution,
        missing_values,
        numeric_summary,
        value_distribution,
    )
    from backend.tools.statistics import get_column_statistics
    from backend.tools.visualization import visualize_column
except ModuleNotFoundError:
    from tools.data_loader import load_dataset
    from tools.data_profile import get_dataset_profile
    from tools.question_tools import (
        churn_summary,
        churn_by_column,
        unsupported_time_analysis,
        categorical_value_analysis,
        infer_visualization_column,
        column_list,
        dataset_overview,
        duplicate_rows,
        label_distribution,
        missing_values,
        numeric_summary,
        value_distribution,
    )
    from tools.statistics import get_column_statistics
    from tools.visualization import visualize_column


@tool
def dataset_profile(file_path: str) -> str:
    """Inspect dataset shape, columns, data types, and missing-value counts."""
    df = load_dataset(file_path)
    profile = get_dataset_profile(df)
    return (
        f"Rows: {profile['number_of_rows']:,}; "
        f"columns: {profile['number_of_columns']:,}; "
        f"column names: {', '.join(map(str, profile['columns']))}; "
        f"missing values: {profile['missing_values']}"
    )


@tool
def dataset_overview_tool(file_path: str, question: str = "") -> str:
    """Answer how many rows and columns are in the dataset."""
    return dataset_overview(load_dataset(file_path), question)


@tool
def column_statistics(file_path: str, column_name: str) -> str:
    """Calculate statistics for one named numeric or categorical column."""
    result = get_column_statistics(load_dataset(file_path), column_name)
    return result if isinstance(result, str) else str(result)


@tool
def numeric_column_question(file_path: str, question: str) -> str:
    """Answer average, minimum, maximum, or range questions for a numeric column."""
    return numeric_summary(load_dataset(file_path), question)


@tool
def categorical_distribution(file_path: str, question: str) -> str:
    """Count the most common values in a named categorical column."""
    return value_distribution(load_dataset(file_path), question)


@tool
def categorical_value_count(file_path: str, question: str) -> str:
    """Count or calculate the percentage of a named categorical value."""
    return categorical_value_analysis(load_dataset(file_path), question)


@tool
def visualize_dataset_column(file_path: str, question: str) -> str:
    """Create a bar chart for categorical data or histogram for numeric data."""
    df = load_dataset(file_path)
    return visualize_column(df, question_column(df, question))


def question_column(df, question: str):
    return infer_visualization_column(df, question) or ""


@tool
def churn_distribution(file_path: str, question: str = "") -> str:
    """Calculate churn or target-label counts and percentages."""
    df = load_dataset(file_path)
    if any(word in question.lower() for word in ("distribution", "percentage", "count")):
        return label_distribution(df, question)
    return churn_summary(df, question)


@tool
def churn_by_category(file_path: str, column_name: str) -> str:
    """Find which category in a column has the highest customer churn rate."""
    return churn_by_column(load_dataset(file_path), column_name)


@tool
def time_trend_limitation(file_path: str, question: str) -> str:
    """Check whether the dataset can support a churn trend over time."""
    return unsupported_time_analysis(load_dataset(file_path), question)


@tool
def data_quality(file_path: str, question: str) -> str:
    """Answer missing-value or duplicate-row questions."""
    df = load_dataset(file_path)
    if "duplicate" in question.lower() or "repeated" in question.lower():
        return duplicate_rows(df)
    return missing_values(df)


@tool
def list_columns(file_path: str, question: str = "") -> str:
    """List dataset columns when the user asks about features or columns."""
    return column_list(load_dataset(file_path))


TOOLS = [
    dataset_profile,
    dataset_overview_tool,
    column_statistics,
    numeric_column_question,
    categorical_distribution,
    categorical_value_count,
    visualize_dataset_column,
    churn_distribution,
    churn_by_category,
    time_trend_limitation,
    data_quality,
    list_columns,
]