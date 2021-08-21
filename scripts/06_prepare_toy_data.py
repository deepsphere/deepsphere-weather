
import os
import sys
sys.path.append('../')
import shutil 
import xarray as xr

##-----------------------------------------------------------------------------.
# Define directories
base_data_dir = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES"
base_toy_data_dir = "/ltenas3/DeepSphere/data/toy_data/ERA5_HRES"

# Define spherical samplings
spherical_samplings = [ 
    # 400 km 
    # 'Healpix_400km', 
    'Icosahedral_400km',
    'O24',
    'Equiangular_400km',
    'Equiangular_400km_tropics',
    'Cubed_400km',
    # # 100 km 
    # 'Healpix_100km'
] 

# Define time period for toy data
start_time = '2010-01-01T00:00:00'
end_time = '2010-12-31T23:00:00'

##-----------------------------------------------------------------------------.
# Prepare Toy Data
spherical_samplings = ['Healpix_400km']
sampling_name = spherical_samplings[0]

for sampling_name in spherical_samplings: 

    full_data_dir = os.path.join(base_data_dir, sampling_name)
    toy_data_dir = os.path.join(base_toy_data_dir, sampling_name)

    # Define paths 
    src_dynamic_fpath = os.path.join(full_data_dir, "Data","dynamic", "time_chunked", "dynamic.zarr")
    dst_dynamic_fpath = os.path.join(toy_data_dir, "Data","dynamic", "time_chunked", "dynamic.zarr")
    src_bc_fpath = os.path.join(full_data_dir, "Data","bc", "time_chunked", "bc.zarr")
    dst_bc_fpath = os.path.join(toy_data_dir, "Data","bc", "time_chunked", "bc.zarr")
    src_static_fpath = os.path.join(full_data_dir, "Data", "static.zarr")
    dst_static_fpath = os.path.join(toy_data_dir, "Data", "static.zarr")

    # Load dataset 
    da_dynamic = xr.open_zarr(src_dynamic_fpath) 
    da_bc = xr.open_zarr(src_bc_fpath) 

    # Subset over time 
    da_dynamic = da_dynamic.sel(time=slice(start_time,end_time))
    da_bc = da_bc.sel(time=slice(start_time,end_time))

    # Save Toy Data to Zarr 
    da_dynamic['data'].encoding.pop("chunks")
    da_bc['data'].encoding.pop("chunks")
    da_dynamic['feature'] = da_dynamic['feature'].astype(str)
    da_bc['feature'] = da_bc['feature'].astype(str)

    da_dynamic.to_zarr(dst_dynamic_fpath) 
    da_bc.to_zarr(dst_bc_fpath)     
    shutil.copytree(src_static_fpath, dst_static_fpath)  

    # Copy all scalers  
    src_scaler_fpath = os.path.join(full_data_dir, "Scalers")
    dst_scaler_fpath = os.path.join(toy_data_dir, "Scalers")
    shutil.copytree(src_scaler_fpath, dst_scaler_fpath)  

    # Copy all climatology

    # Copy all benchmarks