import pandas as pd
import re



def df_numeric_column_filler_with_aggregated_data(
    df, group_list, column_to_be_filled, aggregation_method="median"
):
    """
    This function gets a Python Pandas dataframe and filling missing values in the column of the dataframe.
    :param df: Dataframe to be analyze
    :param group_list: The list of columns
    :param column_to_be_filled: The column's NaN values are filled.
    :param aggregation_method: Aggregation method to be used while filling. Default value is 'median'.
    :return: Dataframe
    """
    try:
        df_dummy = df.groupby(group_list)[column_to_be_filled].transform(
            aggregation_method
        )  

        df.loc[
            df[column_to_be_filled].isnull(), column_to_be_filled
        ] = df_dummy  # to fill the NaN rows by df_dummy.

        return df

    except Exception as e:
        print("Error at df_first_look function: ", str(e))
        return df

    
def missing_data_finder(df):
    """
    This function gets a Python Pandas dataframe and finding missing values and showing these percentages in the column of the dataframe 
    :param df: Dataframe to be analyze     
    :return: This function doesn't return anything.  
    
    """
    df_missing = df.isnull().sum().reset_index().rename(columns={'index': 'column_name', 0: 'missing_row_count'}).copy()
    df_missing_rows = df_missing[df_missing['missing_row_count'] > 0].sort_values(by='missing_row_count',ascending=False)
    df_missing_rows['missing_row_percent'] = (df_missing_rows['missing_row_count'] / df.shape[0]).round(4)
    return df_missing_rows