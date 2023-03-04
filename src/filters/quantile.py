import pandas as pd


def filter_df_by_quantile(df: pd.DataFrame, quantile: float, columns: list = None):
    if columns is None:
        return df
    for column in columns:
        mean = df[column].mean()
        quan = df[column].quantile(q=quantile)
        df = df[(df[column] <= quan) & (df[column] >= 2*mean-quan)]

    return df
