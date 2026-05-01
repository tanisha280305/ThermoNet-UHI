import os, re, glob
import numpy as np
import pandas as pd
import streamlit as st

SAMPLE  = 150
LAT_MAX, LAT_MIN = 30.0, 20.0
LON_MIN, LON_MAX = 72.0, 84.0
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_MOD11A1_2022_2025", "data")

def extract_date(path):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

@st.cache_data(show_spinner="📡 Loading MODIS satellite data...")
def load_all(files):
    out = {}
    for f in sorted(files):
        df = pd.read_csv(f).dropna(subset=["lst_k"])
        df["lst_c"] = df["lst_k"] - 273.15
        df["row"] = ((LAT_MAX - df["lat"]) / (LAT_MAX - LAT_MIN) * SAMPLE).astype(int).clip(0, SAMPLE-1)
        df["col"] = ((df["lon"] - LON_MIN) / (LON_MAX - LON_MIN) * SAMPLE).astype(int).clip(0, SAMPLE-1)
        grid = np.full((SAMPLE, SAMPLE), np.nan, dtype=np.float32)
        # vectorised — no iterrows
        grid[df["row"].values, df["col"].values] = df["lst_c"].values
        out[extract_date(f)] = grid
    return out

def auto_load():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:  # fallback
        files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MOD11A1*.csv")))
    return load_all(tuple(files)) if files else {}

def pixel_to_latlon(row, col):
    lat = LAT_MAX - (row / SAMPLE) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (col / SAMPLE) * (LON_MAX - LON_MIN)
    return round(lat, 4), round(lon, 4)

def latlon_to_pixel(lat, lon):
    row = int((LAT_MAX - lat) / (LAT_MAX - LAT_MIN) * SAMPLE)
    col = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * SAMPLE)
    return max(0, min(row, SAMPLE-1)), max(0, min(col, SAMPLE-1))

def in_tile(lat, lon):
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


# Add to data_loader.py:
URBAN_CENTERS = [
    (28.67, 77.43),  # Delhi
    (19.07, 72.87),  # Mumbai
    (22.57, 88.36),  # Kolkata
]
URBAN_RADIUS_DEG = 0.5

def get_urban_mask():
    """Returns 150x150 boolean grid — True = urban pixel"""
    lats = np.linspace(LAT_MAX, LAT_MIN, SAMPLE)
    lons = np.linspace(LON_MIN, LON_MAX, SAMPLE)
    mask = np.zeros((SAMPLE, SAMPLE), dtype=bool)
    for clat, clon in URBAN_CENTERS:
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                if ((lat-clat)**2 + (lon-clon)**2) < URBAN_RADIUS_DEG**2:
                    mask[i, j] = True
    return mask

def compute_uhi_intensity(grid, urban_mask):
    """UHI Intensity = mean(urban LST) - mean(rural LST)"""
    urban_t = grid[urban_mask & ~np.isnan(grid)]
    rural_t = grid[~urban_mask & ~np.isnan(grid)]
    if urban_t.size == 0 or rural_t.size == 0:
        return np.nan
    return float(urban_t.mean() - rural_t.mean())

def get_urban_mask():
    lats = np.linspace(LAT_MAX, LAT_MIN, SAMPLE)[:, None]   # (150,1)
    lons = np.linspace(LON_MIN, LON_MAX, SAMPLE)[None, :]   # (1,150)
    mask = np.zeros((SAMPLE, SAMPLE), dtype=bool)
    for clat, clon in URBAN_CENTERS:
        mask |= ((lats - clat)**2 + (lons - clon)**2) < URBAN_RADIUS_DEG**2
    return mask