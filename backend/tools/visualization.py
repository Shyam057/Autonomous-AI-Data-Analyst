import os

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


def plot_categorical_distribution(df, column_name):
    """
    Plot the distribution of a categorical column using a bar chart.
    """
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist."

    counts = df[column_name].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.title(f"Distribution of {column_name}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

    return f"Created bar chart for {column_name}."


def plot_numerical_distribution(df, column_name):
    """
    Plot the distribution of a numerical column using a histogram.
    """
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist."

    plt.figure(figsize=(8, 5))
    plt.hist(df[column_name].dropna(), bins=30)
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column_name}")
    plt.tight_layout()
    plt.show()

    return f"Created histogram for {column_name}."


def visualize_column(df, column_name):
    """
    Automatically choose a suitable visualization
    based on the column data type.
    """
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist."

    if pd.api.types.is_numeric_dtype(df[column_name]):
        return plot_numerical_distribution(df, column_name)

    return plot_categorical_distribution(df, column_name)
