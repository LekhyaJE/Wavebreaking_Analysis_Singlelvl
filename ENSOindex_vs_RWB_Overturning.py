# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:29:47 2025

@author: lenovo
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import label
import matplotlib.pyplot as plt
import calendar
import seaborn as sns

#Loading ENSO indices for
df_enso = pd.read_excel("D:\ENSO_MEI.xlsx")

# Load dataset
strurl = r"D:\IITB_Project\Data\overturning_lifecycles_2PVU.nc"
demo_data = xr.open_dataset(strurl)
start_time = demo_data['start']
end_time = demo_data['end']

# Extract time data
times = demo_data['start'].values

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

# Fill missing num_events with 0 where no events occurred
final_df['num_events'] = final_df['num_events'].fillna(0).astype(int)


# # Plot the histogram
# plt.figure(figsize=(14,16))

# # Create a JointGrid
# g = sns.JointGrid(
#     data=final_df, 
#     x="ENSO_Index", 
#     y="num_events", 
#     marginal_ticks=False
# )

# # Add the contour fill plot in the center
# g.plot_joint(sns.kdeplot, cmap="Greens", fill=True, levels=50)

# # Add histograms on the margins
# g.plot_marginals(sns.kdeplot, color="black", fill=True, alpha=1, clip_on=True)

# # Adjust y-axis limits
# g.ax_joint.set_ylim(0, final_df["num_events"].max())

# # Set axis labels and title
# g.set_axis_labels("ENSO Index", "Number of Events", fontsize=14, weight='bold')
# g.fig.suptitle("RWB events wrt ENSO Japan (1980-2019)", fontsize=15, weight='bold', y=1.05)

# # Adjust tick parameters
# g.ax_joint.tick_params(axis='x', labelsize=14, width=2)  
# g.ax_joint.tick_params(axis='y', labelsize=14, width=2)

# # **Set x-ticks from -3 to 3 with an interval of 1**
# g.ax_joint.set_xticks(np.arange(-3, 4, 1))

# # **Insert vertical lines at x = 0, 1, -1**
# for x in [0, 1, -1]:
#     g.ax_joint.axvline(x=x, color='red', linestyle='--', linewidth=2, alpha=0.8)

# # Show the plot
# plt.show()

# HEAT MAP
plt.figure(figsize=(10, 6))

# Bin the data
x_bins = np.linspace(-3, 3, 15)  # ENSO Index bins
y_bins = np.arange(0, final_df['num_events'].max() + 1, 1)  # Event count bins

# Create a 2D histogram of the data
heatmap_data, x_edges, y_edges = np.histogram2d(
    final_df['ENSO_Index'],
    final_df['num_events'],
    bins=[x_bins, y_bins]
)

# Transpose because imshow expects [rows, columns] = [y, x]
heatmap_data = heatmap_data.T

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(
    heatmap_data,
    cmap='YlGnBu',
    xticklabels=np.round(x_bins, 1),
    yticklabels=y_bins,
    cbar_kws={'label': 'Number of Months'},
)

plt.xlabel('ENSO Index')
plt.ylabel('Number of RWB Events')
plt.title('Heatmap of RWB Events vs ENSO Index (1980–2019)', fontsize=14, weight='bold')

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()











