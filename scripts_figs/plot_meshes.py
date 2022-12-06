import os
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import xsphere  # required for xarray 'sphere' accessor 

# Plotting options
import matplotlib
matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'

##-----------------------------------------------------------------------------.
# Define directories
data_dir = "/data/weather_prediction/data"
figs_dir = "/data/weather_prediction/figs"
##-----------------------------------------------------------------------------.
# Define samplings 
sampling_names = [ 
    # 400 km 
    'Healpix_400km', 
    'Icosahedral_400km',
    'Cubed_400km',
    'O24',
    'Equiangular_400km',

]
# Define samplings title 
samplings_labels_dict = {
    # 400 km 
    'Healpix_400km': 'Healpix', 
    'Icosahedral_400km': 'Icosahedral' ,
    'Cubed_400km' : 'Cubed Sphere',
    'O24' : 'Reduced Gaussian Grid',
    'Equiangular_400km' : 'Equiangular Grid',
}

# Define reference and projection CRS  
crs_ref = ccrs.Geodetic()
crs_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0) # from the North Pole 

##----------------------------------------------------------------------------.
### Plot meshes
fig, axes = plt.subplots(2, 3, subplot_kw=dict(projection=crs_proj), figsize=(12,10))
for ax, sampling_name in zip(axes.flat, sampling_names):
    # Load netCDF4 Datasets
    data_sampling_dir = os.path.join(data_dir, sampling_name)
    ds = readDatasets(data_dir=data_sampling_dir, feature_type='static')
    ds = ds.sphere.add_SphericalVoronoiMesh(x='lon',y='lat')
    # Plot mesh   
    ds.sphere.plot_mesh(ax = ax, 
                        transform = crs_ref,
                        add_background = True, 
                        antialiaseds = True,
                        facecolors = 'none',
                        edgecolors = "black",
                        linewidths = 0.5,
                        alpha = 0.8)
    ax.set_title(samplings_labels_dict[sampling_name])
fig.tight_layout() 
fig.savefig(os.path.join(figs_dir, "Meshes.png"))

##----------------------------------------------------------------------------.