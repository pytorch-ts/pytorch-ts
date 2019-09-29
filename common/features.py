"""
time series features
"""
import pandas as pd
import numpy as np
from chinese_calendar import is_workday, is_holiday


def last_year_value(df, date_col_name, value_col_name):
    last_year_values = []
    for date in df[date_col_name]:
        last_year = date - pd.Timedelta(days=1 * 365)
        value = df.loc[df[date_col_name] == last_year][value_col_name]
        if not value.empty:
            last_year_values.append(value.values[0])
        else:
            last_year_values.append(0)
    return last_year_values


def is_workdays(dates):
    return dates.apply(lambda date: is_workday(date))


def is_holidays(dates):
    return dates.apply(lambda date: is_holiday(date))
