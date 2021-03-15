import pandas as pd
import numpy as np
import datetime 


def calculating_zscore(df, cols):
    """
    This function gets a Python Pandas dataframe and calculating z score for column list and creating new column to show outlier and non-outlier values as categorical. 
    :param df: Dataframe to be analyze
    :param cols: The column list for calculating zscore.
    :return: Returning Python Pandas dataframe.
    """
    try:
        df_dummy = df.copy()
        for col in cols:
            col_zscore = col + "_zscore"
            df_dummy[col_zscore] = (df_dummy[col] - df_dummy[col].mean()) / df_dummy[
                col
            ].std(ddof=0)
            
            col_zscore_outlier = col_zscore + "_outlier"
        
            df_dummy[col_zscore_outlier] = np.where(
        (
            (df_dummy[col_zscore] > 3)
            | (df_dummy[col_zscore] < -3)
        ),
        "outlier",
        "non-outlier",
    )


        return df_dummy

    except Exception as e:
        print("Error at df_first_look function: ", str(e))
        return df



