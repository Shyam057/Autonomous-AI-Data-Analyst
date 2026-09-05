def get_dataset_profile(df):

    profile = {
        "number_of_rows": len(df),
        "number_of_columns": len(df.columns),
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }

    return profile
