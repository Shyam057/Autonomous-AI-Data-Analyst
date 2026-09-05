try:
    from backend.tools.data_loader import load_dataset
except ModuleNotFoundError:
    from tools.data_loader import load_dataset

def main():

    file_path = "data/Telco_customer_churn.xlsx"

    df = load_dataset(file_path)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
