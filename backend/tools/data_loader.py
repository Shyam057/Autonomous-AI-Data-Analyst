import pandas as pd

def load_dataset(file_path: str):
    df = pd.read_excel(file_path)
    return df
