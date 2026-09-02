import pandas as pd

def load_dataset(file_path:str):
    df=pd.read_excel("Autonomous-AI-Data-Aanalyst/data/Telco_customer_churn.xlsx")
    return df