# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 13:27:24 2025

@author: lenovo

This code is for Latitudinal intrusion of RWB events conditioned over ENSO.
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import label
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ---------------------------------------------
# STEP 1: Load RWB dataset and calculate daily lowest latitude
# ---------------------------------------------

strurl = r"D:\IITB_Project\Data\rwb_greaterthan_1lvl.nc"
demo_data = xr.open_dataset(strurl)['overturning_event']
flag_data = demo_data.sel(lat=slice(40, 0), lon=slice(55, 105))
flag_data = xr.where(flag_data > 0, 1, 0)

latitudes = flag_data['lat']
longitudes = flag_data['lon']
times = flag_data.time.values

lat2d = flag_data['lat'].broadcast_like(flag_data.isel(time=0)).values

daily_counts = []
lowest_latitudes = []

for i, time in enumerate(times):
    binary_data = flag_data.isel(time=i).values.astype(bool)
    labeled_array, num_features = label(binary_data)
    
    daily_counts.append((time, num_features))
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
    
    lowest_latitudes.append((pd.to_datetime(str(time)), lowest_lat))

df = pd.DataFrame(lowest_latitudes, columns=['time', 'lowest_latitude'])

# ---------------------------------------------
# STEP 2: Load MEI index and classify ENSO phases
# ---------------------------------------------

# Load MEI index (ensure your CSV has 'time' in YYYY-MM-DD and 'mei' columns)
df_enso = pd.read_excel(r"D:\ENSO_MEI.xlsx")
df_enso=pd.DataFrame(df_enso)
df_long=df_enso.melt(id_vars='YEAR',var_name="MONTH",value_name="ENSO_Index")

season_to_month =  {
    'DJ': '-01', 'JF': '-02', 'FM': '-03', 'MA': '-04', 'AM': '-05', 'MJ': '-06',
    'JJ': '-07', 'JA': '-08', 'AS': '-09', 'SO': '-10', 'ON': '-11', 'ND': '-12'
}

# Safely convert and concatenate YEAR and mapped MONTH to 'YYYY-MM' format
df_long['date_str'] = df_long['YEAR'].astype(str) + '-' + df_long['MONTH'].map(season_to_month)

# Now convert to datetime and optionally to period
df_long['MONTH'] = pd.to_datetime(df_long['date_str']).dt.to_period('M')
df_long.drop(columns='date_str', inplace=True)  # Clean up



# ---------------------------------------------
# STEP 3: Merge ENSO and RWB latitudinal data
# ---------------------------------------------

# Step 1: Ensure df['time'] is datetime
df['time'] = pd.to_datetime(df['time'])

# Step 2: Create a 'month' column in df to match df_long's period
df['month'] = df['time'].dt.to_period('M')  # This gives YYYY-MM as a Period

# Step 4: Merge the ENSO index into the daily dataframe
df = df.merge(df_long[['MONTH', 'ENSO_Index']], left_on='month', right_on='MONTH', how='left')

# Step 5: Clean up (drop redundant column)
df.drop(columns=['month', 'MONTH'], inplace=True)

# Step 1: Classify ENSO phase
def classify_enso(index):
    if index >= 1.0:
        return 'El Niño'
    elif index <= -1.0:
        return 'La Niña'
    else:
        return 'Neutral'

df['ENSO_Phase'] = df['ENSO_Index'].apply(classify_enso)


# ---------------------------------------------
# STEP 4: Plot ENSO-conditioned latitudinal boxplot
# ---------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(x="ENSO_Phase", 
            y="lowest_latitude", 
            data=df, 
            palette={"El Niño": "red", "La Niña": "blue", "Neutral": "gray"},
            showmeans=True,
            meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":8},
            boxprops=dict(edgecolor="black"),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(marker="o", color="green", alpha=0.5)
            )

plt.title("Latitudinal Extent of RWB Events by ENSO Phase (1980–2019)", fontsize=18, weight='bold')
plt.xlabel("ENSO Phase", fontsize=16, weight='bold')
plt.ylabel("Latitude (N)", fontsize=16, weight='bold')
#plt.ylim(5, 45)  # expand y-axis to stretch vertically
plt.xticks(fontsize=14, weight='bold')
plt.yticks(fontsize=14, weight='bold')

# Legend
legend_patches = [
    mpatches.Patch(facecolor="gray", edgecolor="black", label="IQR"),
    mlines.Line2D([], [], color="black", linewidth=2, label="Median"),
    mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, label="Mean"),
    mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, alpha=0.5, label="Outliers")
]

plt.legend(handles=legend_patches,
           loc="center left", 
           bbox_to_anchor=(1.02, 0.5),
           fontsize=12,
           borderaxespad=0)

plt.tight_layout()
plt.savefig(r"D:\IITB_Project\Plots\Multiple_level_analysis\rwb_enso_latitudinal_boxplot.png", dpi=300)
plt.show()
