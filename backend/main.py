import argparse

try:
    from backend.tools.data_loader import load_dataset
except ModuleNotFoundError:
    from tools.data_loader import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Inspect an uploaded CSV or Excel dataset.")
    parser.add_argument("file_path", help="Path to a CSV, XLS, or XLSX dataset")
    args = parser.parse_args()

    df = load_dataset(args.file_path)

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
