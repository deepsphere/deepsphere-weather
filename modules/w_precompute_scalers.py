"""
Created on Fri Feb 12 21:44:06 2021

@author: ghiggi
"""
import os
import sys
sys.path.append('../')

from modules.xscaler import GlobalStandardScaler  # TemporalStandardScaler
from modules.my_io import readDatasets   

# ##############################
#### Precompute the scalers ####
# ##############################
base_data_dir = "/data/weather_prediction/data"

sampling_name_list = ['Healpix_400km','Equiangular_400km','Equiangular_400km_tropics',
                      'Icosahedral_400km','O24','Cubed_400km']

for sampling_name in sampling_name_list:
    data_dir = os.path.join(base_data_dir, sampling_name)
    print(data_dir)
    # - Dynamic data (i.e. pressure and surface levels variables)
    ds_dynamic = readDatasets(data_dir=data_dir, feature_type='dynamic')
    # - Boundary conditions data (i.e. TOA)
    ds_bc = readDatasets(data_dir=data_dir, feature_type='bc')
    # - Static features
    ds_static = readDatasets(data_dir=data_dir, feature_type='static')
    
    ds_dynamic = ds_dynamic.drop(["level","lat","lon"])
    ds_bc = ds_bc.drop(["lat","lon"])
    ds_static = ds_static.drop(["lat","lon"])
  
    ##------------------------------------------------------------------------.
    #### Define scalers and fit 
    dynamic_scaler = GlobalStandardScaler(data=ds_dynamic)
    dynamic_scaler.fit()
    dynamic_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
     
    bc_scaler = GlobalStandardScaler(data=ds_bc)
    bc_scaler.fit()
    bc_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
    
    static_scaler = GlobalStandardScaler(data=ds_static)
    static_scaler.fit()
    static_scaler.save(os.path.join(data_dir, "Scalers", "GlobalStandardScaler_static.nc"))

##----------------------------------------------------------------------------.
## If you prefer to pre-scale the data  
# ds_dynamic = dynamic_scaler.transform(ds_dynamic).compute()  
# ds_bc = bc_scaler.transform(ds_bc).compute()  
# ds_static = static_scaler.transform(ds_static).compute()    
# scaler = None # if you prescale, then provide scaler=None to AR_training