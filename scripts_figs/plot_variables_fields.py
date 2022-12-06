import os
import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xsphere  # required for xarray 'sphere' accessor


# Plotting options
import matplotlib

matplotlib.use("cairo")  # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white"  # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = "none"

##-----------------------------------------------------------------------------.
# Define directories
data_dir = "/data/weather_prediction/data"
figs_dir = "/data/weather_prediction/figs"

##-----------------------------------------------------------------------------.
# Define samplings
sampling_names = [
    # 400 km
    "Healpix_400km",
    "Icosahedral_400km",
    "Cubed_400km",
    "O24",
    "Equiangular_400km",
]
# Define samplings title
samplings_labels_dict = {
    # 400 km
    "Healpix_400km": "Healpix",
    "Icosahedral_400km": "Icosahedral",
    "Cubed_400km": "Cubed Sphere",
    "O24": "Reduced Gaussian Grid",
    "Equiangular_400km": "Equiangular Grid",
}

# Define reference and projection CRS
crs_ref = ccrs.Geodetic()
crs_proj = ccrs.Orthographic(
    central_longitude=0.0, central_latitude=90.0
)  # from the North Pole

##-----------------------------------------------------------------------------.
### Plot model variables
antialiased = False
edgecolors = None

fig, axes = plt.subplots(2, 3, subplot_kw=dict(projection=crs_proj), figsize=(12, 10))
for ax, sampling_name in zip(axes.flat, sampling_names):
    # --------------------------------------------------------------------------.
    ## Load netCDF4 Datasets
    data_sampling_dir = os.path.join(data_dir, sampling_name)
    # - Dynamic data (i.e. pressure and surface levels variables)
    ds_dynamic = readDatasets(data_dir=data_sampling_dir, feature_type="dynamic")
    ds_dynamic = ds_dynamic.drop(["level"])
    ds_dynamic = ds_dynamic.isel(time=0).compute()
    # - Boundary conditions data (i.e. TOA)
    ds_bc = readDatasets(data_dir=data_sampling_dir, feature_type="bc")
    ds_bc = ds_bc.isel(time=24 * 172 + 12 - 7).compute()  # summer solstice
    # - Static features
    ds_static = readDatasets(data_dir=data_sampling_dir, feature_type="static")
    # --------------------------------------------------------------------------.
    # Add mesh
    ds_dynamic = ds_dynamic.sphere.add_SphericalVoronoiMesh(x="lon", y="lat")
    ds_bc = ds_bc.sphere.add_SphericalVoronoiMesh(x="lon", y="lat")
    ds_static = ds_static.sphere.add_SphericalVoronoiMesh(x="lon", y="lat")
    # --------------------------------------------------------------------------.
    # Add plots
    if sampling_name == "Healpix_400km":
        # fig, ax = plt.subplots(1,1, subplot_kw=dict(projection=crs_proj), figsize=(12,10))
        xsphere._plot(
            ds_dynamic["z500"],
            ax=ax,
            edgecolors=edgecolors,
            antialiased=antialiased,
            robust=True,
            cmap=plt.get_cmap("BuPu"),
            add_colorbar=False,
        )
        ax.set_title("Z500")
        ax.coastlines(alpha=0.2)

    if sampling_name == "Icosahedral_400km":
        # fig, ax = plt.subplots(1,1, subplot_kw=dict(projection=crs_proj), figsize=(12,10))
        xsphere._plot(
            ds_dynamic["t850"],
            ax=ax,
            edgecolors=edgecolors,
            antialiased=antialiased,
            robust=True,
            cmap=plt.get_cmap("RdYlBu_r"),
            add_colorbar=False,
        )
        ax.set_title("T500")
        ax.coastlines(alpha=0.2)

    if sampling_name == "Cubed_400km":
        # fig, ax = plt.subplots(1,1, subplot_kw=dict(projection=crs_proj), figsize=(12,10))
        xsphere._plot(
            ds_bc["tisr"],
            ax=ax,
            edgecolors=edgecolors,
            antialiased=antialiased,
            robust=True,
            cmap=plt.get_cmap("RdYlBu_r"),
            add_colorbar=False,
        )
        ax.set_title("TOA Radiation")
        ax.coastlines(alpha=0.2)

    if sampling_name == "O24":
        # fig, ax = plt.subplots(1,1, subplot_kw=dict(projection=crs_proj), figsize=(12,10))
        xsphere._plot(
            ds_static["orog"],  # np.log(ds_static['orog'])
            ax=ax,
            edgecolors=edgecolors,
            antialiased=antialiased,
            robust=True,
            cmap=plt.get_cmap("gist_earth"),
            add_colorbar=False,
        )
        ax.set_title("Topography")
        ax.coastlines(alpha=0.2)

    if sampling_name == "Equiangular_400km":
        # fig, ax = plt.subplots(1,1, subplot_kw=dict(projection=crs_proj), figsize=(12,10))
        xsphere._plot(
            ds_static["slt"],
            ax=ax,
            edgecolors=edgecolors,
            antialiased=antialiased,
            robust=True,
            cmap=plt.get_cmap("Paired"),
            add_colorbar=False,
        )
        ax.set_title("Soil type")
        ax.coastlines(alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(figs_dir, "Model_variables.png"))

# 'q500'  plt.get_cmap('GnBu')
