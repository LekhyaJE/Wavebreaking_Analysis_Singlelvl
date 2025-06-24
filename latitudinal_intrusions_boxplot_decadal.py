# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 19:51:49 2025

@author: lenovo
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import label
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# === Load and Preprocess Data ===
strurl = r"D:\IITB_Project\Data\rwb_greaterthan_1lvl.nc"
demo_data = xr.open_dataset(strurl)
demo_data = demo_data['overturning_event']
flag_data = demo_data.sel(lat=slice(40, 0), lon=slice(55, 105))

# Ensure binary flag (1 for event, 0 otherwise)
flag_data = xr.where(flag_data > 0, 1, 0)

latitudes = flag_data['lat']
longitudes = flag_data['lon']
times = flag_data.time.values

# Create 2D latitude array
lat2d = flag_data['lat'].broadcast_like(flag_data.isel(time=0)).values

# === Process Time Steps to Find Lowest Intrusions ===
lowest_latitudes = []

for i, time in enumerate(times):
    binary_data = flag_data.isel(time=i).values.astype(bool)
    labeled_array, num_features = label(binary_data)
    
    lowest_lat = None
    for label_id in range(1, num_features + 1):
        mask = labeled_array == label_id
        streamer_latitudes = lat2d[mask]
        if streamer_latitudes.size > 0:
            min_lat = np.nanmin(streamer_latitudes)
            if lowest_lat is None or min_lat < lowest_lat:
                lowest_lat = min_lat

    if lowest_lat is None:
        lowest_lat = np.nan
    lowest_latitudes.append((time, lowest_lat))

# === Create DataFrame ===
df = pd.DataFrame(lowest_latitudes, columns=['time', 'lowest_latitude'])
df['time'] = pd.to_datetime(df['time'])

# === Assign Decades ===
def map_to_decade(date):
    year = date.year
    if 1980 <= year < 1990:
        return "1980–1990"
    elif 1990 <= year < 2000:
        return "1990–2000"
    elif 2000 <= year < 2010:
        return "2000–2010"
    elif 2010 <= year <= 2019:
        return "2010–2019"
    else:
        return np.nan

df["decade"] = df["time"].apply(map_to_decade)
df = df.dropna(subset=["decade"])

# === Group and Clean Data ===
decadal_grouped = df.groupby("decade")["lowest_latitude"].apply(list).reset_index()
decadal_grouped["lowest_latitude"] = decadal_grouped["lowest_latitude"].apply(lambda x: [val for val in x if not np.isnan(val)])
decadal_grouped = decadal_grouped.explode("lowest_latitude")
decadal_grouped["lowest_latitude"] = pd.to_numeric(decadal_grouped["lowest_latitude"], errors="coerce")

# Define correct order
decade_order = ["1980–1990", "1990–2000", "2000–2010", "2010–2019"]

# === Plot ===
plt.figure(figsize=(10, 6))

sns.boxplot(
    x="decade",
    y="lowest_latitude",
    data=decadal_grouped,
    order=decade_order,
    color="lightblue",
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 8},
    boxprops=dict(edgecolor="black"),
    medianprops=dict(color="black", linewidth=2),
    whiskerprops=dict(color="black", linestyle="-"),
    capprops=dict(color="black"),
    flierprops=dict(marker="o", color="green", alpha=0.5)
)

plt.ylabel("Latitudes (°N)", fontsize=16, weight='bold')
plt.xlabel("Decade", fontsize=16, weight='bold')
plt.title("Decadal Variation of RWB Latitudinal Intrusions (1980–2019)", fontsize=18, weight='bold')
plt.xticks(fontsize=14, weight='bold')
plt.yticks(fontsize=14, weight='bold')

# === Custom Legend ===
legend_patches = [
    mpatches.Patch(facecolor="lightblue", edgecolor="black", label="Interquartile Range (IQR)"),
    mlines.Line2D([], [], color="black", linewidth=2, label="Median"),
    mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, label="Mean"),
    mlines.Line2D([], [], color="green", marker="o", linestyle="None", markersize=8, alpha=0.5, label="Outliers")
]

plt.legend(
    handles=legend_patches,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=12,
    borderaxespad=0
)

plt.tight_layout()
plt.savefig(r"D:\IITB_Project\Plots\Multiple_level_analysis\latitudinal_intrusions_decadal.png", dpi=300)
plt.show()
