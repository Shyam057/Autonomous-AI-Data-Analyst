def get_dataset_profile(df):

    profile={
        "num_of_rows":len(df),
        "num_of_cols":len(df.cols),
        "cols":list(df.cols),
        "data_types":df.dtypes.astype(str).to_dict(),
        "missing_values":df.isnull().sum().to_dict(),
    }

    return profile