import matplotlib.pyplot as plt
import cartopy.crs as ccrs

bc_scaler = LoadScaler(
    os.path.join(data_sampling_dir, "Scalers", "GlobalMinMaxScaler_bc.nc")
)
bc_scaler = LoadScaler(
    os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_bc.nc")
)
data_bc = data_bc.sphere.add_SphericalVoronoiMesh(x="lon", y="lat")
a = data_bc.isel(time=24 * 6).compute()

a = bc_scaler.transform(a, variable_dim="feature").compute()

crs_proj = ccrs.PlateCarree()

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
a.squeeze().sphere.plot(ax=ax, add_colorbar=True, cmap=plt.get_cmap("Spectral_r"))
plt.show()

plt.hist(a.values)
plt.show()

# MinMax bc.... there is a strange max ...

### EDA of dynamic variables
dynamic_scaler = LoadScaler(
    os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_dynamic.nc")
)
dynamic_scaler = LoadScaler(
    os.path.join(data_sampling_dir, "Scalers", "GlobalMinMaxScaler_dynamic.nc")
)

data_dynamic = data_dynamic.sphere.add_SphericalVoronoiMesh(x="lon", y="lat")

for i in range(9):
    a = data_dynamic.isel(time=341872, feature=slice(i, i + 1)).compute()
    a = dynamic_scaler.transform(a, variable_dim="feature").compute()

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs_proj))
    a.squeeze().sphere.plot(ax=ax, add_colorbar=True, cmap=plt.get_cmap("Spectral_r"))
    plt.show()

    plt.hist(a.values)
    plt.show()
