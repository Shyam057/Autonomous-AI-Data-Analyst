def get_column_statistics(df,column_name):
    if column_name not in df.columns:
        return f"Column '{column_name}' does not exist."

    column=df[column_name]
    if column.dtype == "object":
        return {
            "column":column_name,
            "type":"categorical",
            "unique_values":column.nunique(),
            "most_common": column.value_counts().head(10).to_dict(),
        }
    return{
        "column": column_name,
        "type":"numerical",
        "count": int(column.count()),
        "mean":float(column.mean()),
        "minimum":float(column.min()),
        "maximum":float(column.max()),
        "median":float(column.median()),
    }