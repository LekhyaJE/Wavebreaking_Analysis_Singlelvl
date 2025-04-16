# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:14:29 2025

@author: lenovo
"""
# Test code for 5 years of merged overturning data.
import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import label
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Load dataset
strurl = r"D:\IITB_Project\Data\merged_daily_overturning_350K.nc"
demo_data = xr.open_dataset(strurl)
flag_data = demo_data['overturning_event']
flag_data = flag_data.sel(latitude=slice(40,0), longitude=slice(55, 105))
# Ensure flag is binary (1 for streamer occurrence, 0 otherwise)
flag_data = xr.where(flag_data > 0, 1, 0)


latitudes = flag_data['latitude']  # Assuming latitude is a coordinate
longitudes = flag_data['longitude']  # Assuming latitude is a coordinate
# Initialize lists to store counts and lowest latitude per timestep
daily_counts = []
lowest_latitudes = []
times = flag_data.floor.values  # Assuming 'time' is a coordinate

# Loop through each time step
for i, time in enumerate(times):   
    # Extract binary data for the current time step
    binary_data = flag_data.isel(floor=i).values  # Shape: (lat, lon)

    # Apply connected component labeling
    labeled_array, num_features = label(binary_data)

    # Store daily count
    daily_counts.append((time, num_features))

    # Find the lowest latitude of the most equatorward streamer
    lowest_lat = None
   
    for label_id in range(1, num_features + 1):
        # Create a mask ensuring correct dimensions
        streamer_mask = xr.DataArray(
            labeled_array == label_id, 
            dims=("latitude", "longitude"),  # Ensure correct shape
            coords={"lat":latitudes, "lon": longitudes}  # Assign coordinates
        )
        # Solution 1: Using .where()
        streamer_latitudes = xr.DataArray(latitudes, dims=("latitude",)).where(streamer_mask.any(dim="longitude"), drop=True)

        # Extract actual latitudes instead of binary data
        # streamer_latitudes = xr.DataArray(latitudes[:, np.newaxis], dims=("lat", "lon")).where(streamer_mask, drop=True)
        # streamer_latitudes = latitudes[streamer_mask.values].flatten()
        #print(f"Label {label_id}: Streamer latitudes shape {streamer_latitudes.shape}, values: {streamer_latitudes}")

        # Ensure we have valid latitude values
        if streamer_latitudes.size > 0 and not np.isnan(streamer_latitudes.values).all():
            min_lat = np.nanmin(streamer_latitudes.values)  # Use np.nanmin to ignore NaNs
            if lowest_lat is None or min_lat < lowest_lat:
                lowest_lat = min_lat
        
    # Convert None to NaN to avoid issues in DataFrame
    if lowest_lat is None:
        lowest_lat = np.nan
    
    lowest_latitudes.append((time, lowest_lat))
    

# Convert to DataFrame for monthly statistics
df = pd.DataFrame(lowest_latitudes, columns=['time', 'lowest_latitude'])
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.to_period('M')
# Convert 'month' to month name (removes year)
df["month"] = df["month"].dt.strftime("%B")  # Converts '2001-01' → 'January'


# Group by month and collect all latitude values in a list
monthly_grouped = df.groupby("month")["lowest_latitude"].apply(list).reset_index()

# Define the correct month order
month_order = ["January", "February", "March", "April", "May", "June", 
               "July", "August", "September", "October", "November", "December"]


# Sort DataFrame based on month order
monthly_grouped["month"] = pd.Categorical(monthly_grouped["month"], categories=month_order, ordered=True)
monthly_grouped = monthly_grouped.sort_values("month").reset_index(drop=True)


# Apply function to remove NaN values from lists
monthly_grouped["lowest_latitude"] = monthly_grouped["lowest_latitude"].apply(lambda x: [val for val in x if not np.isnan(val)])

monthly_grouped = monthly_grouped.explode("lowest_latitude")
monthly_grouped["lowest_latitude"] = pd.to_numeric(monthly_grouped["lowest_latitude"], errors="coerce")


# Plot monthly statistics
plt.figure(figsize=(12, 6))
# Convert Period to string for plotting

sns.boxplot(x="month", 
            y="lowest_latitude", 
            data=monthly_grouped, 
            order=month_order,   
            color="lightblue",  # Set all boxes to light blue
            showmeans=True,  # Show mean line
            meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black", "markersize":8},
            boxprops=dict(edgecolor="black"),
            medianprops=dict(color="black", linewidth=2),  # Median line
            whiskerprops=dict(color="black", linestyle="-"),  # Whiskers
            capprops=dict(color="black"),  # Caps
            flierprops=dict(marker="o", color="green", alpha=0.5) 
            )
                 
plt.ylabel("Latitudes(N)", fontsize=20, weight='bold')
plt.title("RWB event latitudinal extent (1980-2019)",fontsize=20, weight='bold')
plt.xticks(fontsize=15, weight='bold', rotation=90)
plt.yticks(fontsize=15, weight='bold')

# Create custom legend
legend_patches = [
    mpatches.Patch(facecolor="lightblue", edgecolor="black", label="Interquartile Range (IQR)"),
    mlines.Line2D([], [], color="black", linewidth=2, label="Median"),  # Median Line
    mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, label="Mean"),  # Mean Marker
    mlines.Line2D([], [], color="black", marker="o", linestyle="None", markersize=8, alpha=0.5, label="Outliers")  # Outliers
]

# Add legend
plt.legend(handles=legend_patches, loc="lower right", fontsize=12)
plt.show()


