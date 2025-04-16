# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 18:14:45 2025

@author: lenovo
"""
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

#Loading ENSO indices for
df_enso = pd.read_excel("D:\ENSO_MEI.xlsx")

# Load dataset
strurl = r"D:\IITB_Project\Data\overturning_lifecycles_2PVU.nc"
demo_data = xr.open_dataset(strurl)
start_time = demo_data['start']
end_time = demo_data['end']

# Extract time data
times = demo_data['end'].values

# Convert list to DataFrame and extract year-month
wb_df = pd.DataFrame({'event_time': pd.to_datetime(times)})
wb_df['year_month'] = wb_df['event_time'].dt.strftime('%Y-%m')


# Count occurrences of each year-month
event_counts = wb_df['year_month'].value_counts().reset_index()
event_counts.columns = ['year_month', 'num_events']

# Sort event_counts chronologically
event_counts['year_month'] = pd.to_datetime(event_counts['year_month'])
event_counts = event_counts.sort_values(by='year_month')
event_counts['month']=event_counts['year_month'] 
event_counts = event_counts.drop(['year_month'], axis=1) 
event_counts['month'] = event_counts['month'].dt.strftime('%Y-%m')

df_enso=pd.DataFrame(df_enso)
df_long=df_enso.melt(id_vars='YEAR',var_name="MONTH",value_name="ENSO_Index")

# Create correct timestamps
monthly_mapping = {
    'DJ': '-01', 'JF': '-02', 'FM': '-03', 'MA': '-04', 'AM': '-05', 'MJ': '-06',
    'JJ': '-07', 'JA': '-08', 'AS': '-09', 'SO': '-10', 'ON': '-11', 'ND': '-12'
}
#creating a new column with pandas datetime64 ns
# ENSO Indices
df_long['month'] = pd.to_datetime(df_long['YEAR'].astype(str) 
                                      + df_long['MONTH'].map(monthly_mapping)).dt.to_period('M')
df_long = df_long.drop(["YEAR","MONTH"], axis=1)
# Convert 'year_month' in both DataFrames to the same string format YYYY-MM
df_long['month'] = df_long['month'].dt.strftime('%Y-%m')
event_counts['month'] = pd.to_datetime(event_counts['month']).dt.strftime('%Y-%m')



#Make a new  merged dataset with ENSO indices and corresponding no of events.
#This is used for plotting the histogram
# Ensure event_counts has all year_month entries from enso_df

# Merge ENSO data with event counts (left join to retain all ENSO months)
final_df = pd.merge(df_long, event_counts, on='month', how='left')

# # Fill missing num_events with 0 where no events occurred
final_df['num_events'] = final_df['num_events'].fillna(0).astype(int)

df_filtered = final_df[final_df['num_events'] > 0]

######################## PLOTTING ################################
# Define bin edges
x_bins = np.linspace(-3, 3, 7)  # ENSO Index bins
y_bins = np.arange(1, final_df['num_events'].max() + 2, 1)  # Event counts

# 2D histogram
heatmap_data, x_edges, y_edges = np.histogram2d(
    df_filtered['ENSO_Index'],
    df_filtered['num_events'],
    bins=[x_bins, y_bins]
)
heatmap_data = heatmap_data.T  # Transpose for heatmap


# Bin centers for correct ticks
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

# Define discrete color levels
levels = np.arange(0, np.max(heatmap_data)+5, 10)  # e.g., [0, 5, 10, 15, ...]
#norm = BoundaryNorm(boundaries=levels, ncolors=len(levels)-1)
# Format labels for annotation
annotations = np.where(heatmap_data == 0, "", heatmap_data.astype(int).astype(str))
# Plot
plt.figure(figsize=(12, 7))
ax = sns.heatmap(
    heatmap_data,
    cmap='YlOrRd',
    annot=annotations,
    fmt='', 
    xticklabels=np.round(x_bins, 1),
    yticklabels=y_bins,
    cbar_kws={'label': 'Number of Months', 'ticks': levels},
    linewidths=0.5,
    #norm=norm
    square=True,
    annot_kws={'fontsize': 12, 
               'weight': 'bold',
               'color': 'black'}  # Customize as needed
)

# Labels
plt.xlabel('ENSO Index', fontsize=12, weight='bold')
plt.ylabel('Number of RWB Events', fontsize=12, weight='bold')
plt.title('Heatmap of RWB Events vs ENSO Index (1980–2019)', fontsize=14, weight='bold')

# Format
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()