import pandas as pd
import io
import numpy as np

def calculate_date_zscore(df, title, date):
    """
    Calculate the z-score of the data for a given date based on the
    distribution of other data points for the same month, after filtering
    by title, month, and year.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data. It must include columns 'title',
        'month', 'year', 'date', and 'data'.
    title : str
        The title to filter the DataFrame.
    date : int
        The date for which to calculate the z-score.

    Returns
    -------
    float
        The z-score of the summed data for the given date.
    """
    month = date.to_period('M')
    filtered_df = df[(df['title'] == title) & (df['month'] == month)]
    date_data_sum = (filtered_df[filtered_df['date'] == date]['daily_count']
                     .sum())
    rest_of_month_data = (filtered_df[filtered_df['date'] != date]
                          ['daily_count'])
    z_score = ((date_data_sum - rest_of_month_data.mean()) /
               rest_of_month_data.std())
    
    return z_score


def df_info_to_dataframe(df):
    """
    Convert the output of DataFrame.info() to a DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame for which to obtain info.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the information from df.info().
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    lines = info_str.split('\n')
    data = []
    
    for line in lines[5:-2]:
        parts = line.split()
        if len(parts) > 3:
            col_name = parts[1]
            col_type = parts[-1]
            non_null_count = parts[-3] + " " + parts[-2]
            data.append([col_name, non_null_count, col_type])
    
    info_df = pd.DataFrame(data, columns=['Column', 'Non-Null Count', 'Dtype'])
    return info_df