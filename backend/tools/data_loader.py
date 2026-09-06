from pathlib import Path

import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV or Excel file based on its file extension."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)

    raise ValueError("Unsupported dataset format. Use a CSV or Excel file.")
