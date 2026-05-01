import numpy as np
from datetime import datetime, timedelta
from data_loader import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

def k2c(arr): return arr  # already °C

def next_date_str(date_str):
    return (datetime.strptime(date_str,"%Y-%m-%d")+timedelta(days=1)).strftime("%Y-%m-%d")

def get_trend(data):
    return [float(np.nanmean(arr)) for arr in data.values()]  # no -273.15

def classify_hotspot(temp_c):
    if temp_c >= 55: return "🔴 Extreme", "#e63946"
    if temp_c >= 45: return "🟠 Severe",  "#f4a261"
    if temp_c >= 35: return "🟡 High",    "#e9c46a"
    return "🟢 Normal", "#2ec4b6"

def in_tile(lat, lon):
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX