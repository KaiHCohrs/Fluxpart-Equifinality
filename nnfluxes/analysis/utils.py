import os

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def unfold_time(data):
    if 'DateTime' not in data.keys():
        data['DateTime'] = data.index
    data["Date"] = pd.to_datetime(data['DateTime']).dt.date
    data["Time"] = pd.to_datetime(data['DateTime']).dt.time
    data["Month"] = pd.to_datetime(data['Date']).dt.month
    data["dom"] = pd.to_datetime(data['DateTime']).dt.day
    data["Year"] = pd.to_datetime(data['Date']).dt.year
    data["doy"] = pd.to_datetime(data['DateTime']).dt.dayofyear
    data["tom"] = time_of_month(data)
    return data
    
def time_of_month(data):
    day_of_month = pd.to_datetime(data['DateTime']).dt.day
    hour_of_day = pd.to_datetime(data['DateTime']).dt.hour
    minute_of_hour = pd.to_datetime(data['DateTime']).dt.minute
    return (day_of_month-1) * 24 * 2 + hour_of_day * 2 + minute_of_hour //30