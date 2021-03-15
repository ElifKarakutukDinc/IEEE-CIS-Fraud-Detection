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

def creating_date_columns(df, date_column, START_DATE):
    """
    This function gets a Python Pandas dataframe and converting time delta date_column to date and creating new columns as date, weekdays, hours and days. 
    :param df: Dataframe to be analyze
    :param date_column: The column is main date column in dataframe that is time delta. 
    :return: Returning Python Pandas dataframe.
    """
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    df["Date"] = df[date_column].apply(
        lambda x: (startdate + datetime.timedelta(seconds=x))
    )

    df["Weekdays"] = df["Date"].dt.dayofweek
    df["Hours"] = df["Date"].dt.hour
    df["Days"] = df["Date"].dt.day
