import os
import sys
sys.path.append('../')
import os
os.chdir('/home/ghiggi/Projects/deepsphere-weather')
import time
import xarray as xr

# from multiprocessing import Pool
from functools import partial
from dask.distributed import Client, LocalCluster
 

from modules.xscaler import GlobalStandardScaler   
from modules.xscaler import GlobalMinMaxScaler
from modules.xscaler import AnomalyScaler
from modules.xscaler import Climatology

base_data_dir = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES/"
sampling_name = 'O24'
data_dir = os.path.join(base_data_dir, sampling_name)
print("Data directory:", data_dir)
##------------------------------------------------------------------------.
### Load data 
# - Define path for dynamic data (i.e. pressure and surface levels variables)
dynamic_fpath = os.path.join(data_dir, "Data","dynamic", "space_chunked", "dynamic.zarr")
# dynamic_fpath = os.path.join(data_dir, "Data","dynamic", "time_chunked", "dynamic.zarr")
# - Define path for boundary conditions data (i.e. TOA)
bc_fpath = os.path.join(data_dir, "Data","bc", "space_chunked", "bc.zarr")
# bc_fpath = os.path.join(data_dir, "Data","bc", "time_chunked", "bc.zarr")

##------------------------------------------------------------------------------.
### Benchmark reaading data
# --> Dask rechunking helps a lot for performance !!!

# - Define dask chunks 
node_dask_chunks = 200
n_feature = 2
n_node = node_dask_chunks*30

dask_chunks = {'feature': 1, # each single feature 
                'time': -1,   # all across time
                'node': node_dask_chunks,  # 2.73 MB per disk chunk --> 1092 MB each chunk 
                }
# - Without dask rechunking 
da_dynamic = xr.open_zarr(dynamic_fpath)['data']
da_dynamic = da_dynamic.isel(node=slice(0,n_node)).isel(feature=slice(0,5))
da_orig = da_dynamic
t_i = time.time()
b = da_orig.compute()
print('- Oring Elapsed time: {:.2f}s'.format((time.time() - t_i)))
del da_dynamic
del b

# - With dask rechunking 
da_dynamic = xr.open_zarr(dynamic_fpath)['data']
da_dynamic = da_dynamic.isel(node=slice(0,n_node)).isel(feature=slice(0,n_feature))
da_rechunked = da_dynamic.chunk(dask_chunks)
t_i = time.time()
a = da_rechunked.compute()
print('- Rechunked Elapsed time: {:.2f}s'.format((time.time() - t_i)))
del da_dynamic
del a 

da_rechunked
da_orig

##------------------------------------------------------------------------------.
# Benchmark to_dataset() use 
node_dask_chunks = 200
n_feature = 8
n_node = node_dask_chunks*30

dask_chunks = {'feature': 1, # each single feature 
               'time': -1,   # all across time
               'node': node_dask_chunks,  # 2.73 MB per disk chunk --> 1092 MB each chunk 
               }

da_dynamic = xr.open_zarr(dynamic_fpath)['data']
# da_dynamic = da_dynamic.isel(node=slice(0,n_node)).isel(feature=slice(0,n_feature))
da_rechunked = da_dynamic.chunk(dask_chunks)
t_i = time.time()
b = da_rechunked.max('feature').compute()   # 380 s , 155s
print('- DataArray Elapsed time: {:.2f}s'.format((time.time() - t_i)))
del da_dynamic
del da_rechunked
del b 

da_dynamic = xr.open_zarr(dynamic_fpath)['data']
# da_dynamic = da_dynamic.isel(node=slice(0,n_node)).isel(feature=slice(0,n_feature))
da_rechunked = da_dynamic.chunk(dask_chunks)
ds_rechunked = da_rechunked.to_dataset('feature')
t_i = time.time()
a = ds_rechunked.max().compute()  # 77 s, 83 s
print('- Dataset Elapsed time: {:.2f}s'.format((time.time() - t_i)))
del da_dynamic
del da_rechunked
del a 

##------------------------------------------------------------------------------.
## Graph viz  
da_dynamic = xr.open_zarr(dynamic_fpath)['data']
da_dynamic = da_dynamic.isel(node=slice(0,10)).isel(feature=slice(0,5))
da_dynamic = da_dynamic.compute()
dask_chunks = {'feature': 1, # each single feature 
                'time': -1,   # all across time
                'node': 2,  # 2.73 MB per disk chunk --> 1092 MB each chunk 
               }
da_rechunked = da_dynamic.chunk(dask_chunks)
 
 
da_rechunked.data.visualize()
ds = da_rechunked.to_dataset('feature')
ds.mean().data.visualize()
