# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 16:47:48 2024

@author: lenovo
"""

"""
This is a Visualization code
The following code generates GIF for Rossby wave Movement on world map over an year
Input: PV_{year}.nc file

Can see for specific seasons eg, summer monsoon, winter etc comment out one to get the other

"""

#import libraries

import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import wavebreaking as wb
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import MaxNLocator
import imageio

#reading file
strurl = r"D:\Monsoon\ERA5_potential_vorticity_1980.nc"
demo_data = xr.open_dataset(strurl)
strurl=str(strurl)
year = strurl[36:40]


PV_df = demo_data['pv']


# Resample to daily frequency and calculate the mean
daily_avg = PV_df.resample(valid_time="1D").mean() #to reduce time steps I'm obtaining a daily mean


# smoothed = wb.calculate_smoothed_field(data=daily_avg,
#                                        passes=5,
#                                        weights=np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]]), # optional
#                                        mode="wrap") # optional
##Why we doing smoothing? n##########


# Extract data
data_array = daily_avg  # Assuming time is the first dimension
latitudes = data_array.latitude.values
longitudes = data_array.longitude.values

  
ds_coarse = data_array * 1e6
   
############ RWB Indices ##########################


# calculate streamers
streamers = wb.calculate_streamers(data=ds_coarse,
                                   contour_levels=[-1, 1],
                                   #contours=contours, #optional
                                   geo_dis=800, # optional
                                   cont_dis=1200, # optional
                                   #intensity=mflux, # optional
                                   periodic_add=120) # optional  
                                 

############# Event Classification #############################
stratospheric = streamers[streamers.mean_var >= 1]
tropospheric = streamers[streamers.mean_var < 1]

# transform to xarray.DataArray
strato_flag_array = wb.to_xarray(data=ds_coarse,events=stratospheric)
tropo_flag_array = wb.to_xarray(data=ds_coarse,events=tropospheric)




#########################################################
########### Plotting ####################################
#########################################################

# Specify the directory to save the GIF
save_directory = r'C:\Users\lenovo\Downloads\IITB_Project\Plots'  # Change this to your desired path

global_min = -12
global_max = 12
# Define colorbar custom label positions and labels
custom_label_positions = np.linspace(global_min, global_max, num=12)  # 12 evenly spaced positions
custom_labels = [int(label) for label in custom_label_positions]  # Ensure integer labels


##################### GIF FOR MAY TO OCTOBER ##########################
# Create GIF directly from the generated maps
with imageio.get_writer(f'{save_directory}/{year}.gif', mode='I', duration=0.5) as writer:

    for i in range(123):
# initialize figure
        fig, ax = plt.subplots(1, 1, figsize=(13, 8),gridspec_kw={'wspace': 0.001},facecolor='w', edgecolor='k',subplot_kw={'projection':
                                    ccrs.PlateCarree()})

# Add map features
        ax.coastlines(color="dimgrey", linewidth=0.8)
        ax.gridlines(draw_labels=True, color='black', linestyle=':', linewidth=2)

        first_timestep = ds_coarse[i,:,:] # 2D array at first time step 00UTC
# Apply Gaussian smoothing (adjust sigma for different smoothing levels)
        smoothed_data = gaussian_filter(first_timestep, sigma=1.5)  # Increase sigma for stronger smoothing

# Define latitude and longitude
        latitudes = first_timestep['latitude']
        longitudes = first_timestep['longitude']
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
        contour_levels = [-1,1]

# Plot the box for region of interest (60° to 100° longitude and 5° to 40° latitude)
        box_lon = [60, 100, 100, 60, 60]  # Longitudes of the box corners
        box_lat = [5, 5, 40, 40, 5]       # Latitudes of the box corners
        ax.plot(box_lon, box_lat, color='red', linewidth=2, linestyle='-')


# Define contour levels
        # Define contour levels
        filled_contour_levels = np.linspace(-12, 12, 20)  # For filled contours
        contour_levels = [-1, 1]  # For the line contours
    
# Plot filled contour for the rest of the values
        filled_contours = ax.contourf(
        lon_grid, lat_grid,first_timestep ,
        levels=filled_contour_levels,
        cmap="RdBu_r",  # Color map for the filled contours
        extend='both',  # Extend the colorbar at both ends
        transform=ccrs.PlateCarree()
        )
    
#Plot Line contour    
        contours = ax.contour(lon_grid, lat_grid, smoothed_data,
        levels=contour_levels,
        colors="black", 
        linewidths=2,
        linestyles='-',
            transform=ccrs.PlateCarree()  # Ensure correct projection
            )

 # Customize ticks and spines
       # Customize plot ticks and make them bold
        ax.tick_params(
           axis="both", which="both", top=False, right=False, 
           labelsize=14, width=2, length=8, direction='in', labelcolor='black'
       )
       
       # Set bold font weight for tick labels on x and y axes
        for tick in ax.xaxis.get_ticklabels():
           tick.set_fontweight('bold')
        for tick in ax.yaxis.get_ticklabels():
               tick.set_fontweight('bold')
               
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)     
            
            
 # Add a colorbar with fixed ticks
    # Add a consistent colorbar with custom labels
        cbar = plt.colorbar(filled_contours, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
        cbar.ax.set_yticks(custom_label_positions)  # Set positions for the ticks
        cbar.ax.set_yticklabels(custom_labels, fontsize=14)  # Set custom labels
        cbar.ax.tick_params(labelsize=14, width=2)  # Make colorbar ticks bold and set width
        # Make colorbar tick labels bold
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontweight('bold')
            
            
        cbar.set_label('Potential Vorticity [PVU]', fontsize=16,labelpad=15)  # Set title, font size, and padding
        cbar.ax.yaxis.label.set_rotation(270)          
            
            
########### Plotting streamer patches ############################

# Plot the strartosphere streamers as patches (yellow)
        strato_streamer=strato_flag_array[i,:,:]
# Plot the flag data (streamers) on the map
        strato_streamer.where(strato_streamer > 0).plot.contourf(
        ax=ax,
        colors=["white", "gold"],  # Set color for streamers
        levels=[0, 0.5],  # Ensure you get correct levels to display streamers
        transform=ccrs.PlateCarree(),  # Ensure you are using the correct coordinate reference system
        add_colorbar=False  # Don't add a separate colorbar for streamers
        )

        tropo_streamer=tropo_flag_array[i,:,:]
# Plot the flag data (streamers) on the map
        tropo_streamer.where(tropo_streamer > 0).plot.contourf(
        ax=ax,
        colors=["white", "lime"],  # Set color for streamers
        levels=[0, 0.5],  # Ensure you get correct levels to display streamers
        transform=ccrs.PlateCarree(),  # Ensure you are using the correct coordinate reference system
        add_colorbar=False  # Don't add a separate colorbar for streamers
        )
 # Add a title for the plot
        ax.set_title(f"{year} Timestep {i + 1}", fontweight='bold')

        # Convert the current plot to an image and append it to the GIF
        plt.draw()  # Make sure the plot is rendered
        img = np.array(fig.canvas.renderer.buffer_rgba())  # Capture the figure as an image
        writer.append_data(img)  # Append the image to the GIF

        # Close the figure to avoid memory leaks
        plt.close(fig)
















