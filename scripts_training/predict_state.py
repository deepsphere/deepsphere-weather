"""
Created on Thu Aug 19 13:12:02 2021

@author: ghiggi
"""
import os
# os.chdir('/home/ghiggi/Projects/deepsphere-weather')
import sys
sys.path.append('../')
import shutil
import argparse
import dask
import numpy as np
import xarray as xr
 
## DeepSphere-Weather
from modules.utils_config import read_config_file
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_ar_settings
from modules.utils_config import get_dataloader_settings
from modules.utils_config import get_pytorch_model
from modules.utils_config import set_pytorch_settings
from modules.utils_config import load_pretrained_model
from modules.utils_io import get_ar_model_diminfo
from modules.predictions_autoregressive import AutoregressivePredictions

## Project specific functions
import modules.my_models_graph as my_architectures

## Side-project utils (maybe migrating to separate packages in future)
from modules.xscaler import LoadScaler
from modules.xscaler import SequentialScaler
 
# 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# data_dir = "/data/weather_prediction/data"
# exp_dir = "/data/weather_prediction/experiments_GG"
# model_name = "RNN-UNetSpherical-healpix-16-k20-MaxAreaPooling-float32-AR6-LinearStep_weight_corrected"
# model_dir = os.path.join(exp_dir, model_name)
# ar_iterations = 20 
# forecast_reference_times = ['2013-12-31T18:00']
# batch_size = 32
# ar_blocks = 1000
# dst_dirpath = None
# zarr_fname = "pred.zarr"
# force_zarr = False # force to write a new zarr if already existing
# force_gpu = True

def main(data_dir, 
         model_dir, 
         forecast_reference_times, # ['2013-12-31T18:00']
         ar_iterations = 20, 
         batch_size = 32, 
         ar_blocks = 1000, 
         dst_dirpath = None,   
         zarr_fname = "pred.zarr", 
         force_zarr = False, # force to write a new zarr if already existing
         force_gpu = True):
    #-------------------------------------------------------------------------.
    # Check model_dir exist 
    if not os.path.exists(model_dir):
        raise ValueError("The 'model_dir' {!r} does not exist!".format(model_dir))
    # Check dst_dir 
    if dst_dirpath is None:
        dst_dirpath = os.path.join(model_dir, "model_predictions/space_chunked")
    #-------------------------------------------------------------------------.
    # Define forecast_reference_times
    forecast_reference_times = np.array(forecast_reference_times, dtype='M8[m]')
    #-------------------------------------------------------------------------.
    # Define forecast_zarr_fpath
    forecast_zarr_fpath = os.path.join(dst_dirpath, zarr_fname)
    if os.path.exists(forecast_zarr_fpath):
        if not force_zarr: 
            raise ValueError("The zarr store {!r} already exist.".format(forecast_zarr_fpath))
        else:
            shutil.rmtree(forecast_zarr_fpath)
    #-------------------------------------------------------------------------.
    # Read config file 
    cfg_path = os.path.join(model_dir, 'config.json')
    cfg = read_config_file(fpath=cfg_path)
    # Some special options to adjust for prediction
    cfg['dataloader_settings']["autotune_num_workers"] = False
    if force_gpu:
        cfg['training_settings']['gpu_training'] = True  # to run prediction in GPU if possible 
    ##------------------------------------------------------------------------.
    ### Retrieve experiment-specific configuration settings   
    model_settings = get_model_settings(cfg)   
    ar_settings = get_ar_settings(cfg)
    training_settings = get_training_settings(cfg) 
    dataloader_settings = get_dataloader_settings(cfg) 
    
    ##------------------------------------------------------------------------.
    #### Load Datasets
    # - Retrieve directory with required data 
    data_sampling_dir = os.path.join(data_dir, cfg['model_settings']["sampling_name"])
    da_dynamic = xr.open_zarr(os.path.join(data_sampling_dir, "Data","dynamic", "time_chunked", "dynamic.zarr"))["data"]
    da_bc = xr.open_zarr(os.path.join(data_sampling_dir, "Data","bc", "time_chunked", "bc.zarr"))["data"]
    ds_static = xr.open_zarr(os.path.join(data_sampling_dir, "Data", "static.zarr")) 
    
    # - Align Datasets (currently required)
    # ds_dynamic, ds_bc = xr.align(ds_dynamic, ds_bc) 
    # - Select dynamic features 
    da_dynamic = da_dynamic.sel(feature=["z500", "t850"])
 
    ##------------------------------------------------------------------------.
    # - Prepare static data 
    # - Keep land-surface mask as it is 
    # - Keep sin of latitude and remove longitude information 
    ds_static = ds_static.drop(["sin_longitude","cos_longitude"])
    # - Scale orography between 0 and 1 (is already left 0 bounded)
    ds_static['orog'] = ds_static['orog']/ds_static['orog'].max()
    # - One Hot Encode soil type 
    # ds_slt_OHE = xscaler.OneHotEnconding(ds_static['slt'])
    # ds_static = xr.merge([ds_static, ds_slt_OHE])
    # ds_static = ds_static.drop('slt')
    # - Convert to DataArray 
    da_static = ds_static.to_array(dim="feature")  
    
    ##------------------------------------------------------------------------.
    #### Define scaler to apply on the fly within DataLoader 
    # - Load scalers
    dynamic_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_dynamic.nc"))
    bc_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_bc.nc"))
    static_scaler = LoadScaler(os.path.join(data_sampling_dir, "Scalers", "GlobalStandardScaler_static.nc"))
    # # - Create single scaler 
    scaler = SequentialScaler(dynamic_scaler, bc_scaler, static_scaler)

    ##------------------------------------------------------------------------.
    ### Define pyTorch settings 
    device = set_pytorch_settings(training_settings)

    ##------------------------------------------------------------------------.
    ## Retrieve dimension info of input-output Torch Tensors
    dim_info = get_ar_model_diminfo(ar_settings=ar_settings,
                                    da_dynamic=da_dynamic, 
                                    da_static=da_static, 
                                    da_bc=da_bc)
    # Check that dim_info match between training and now 
    assert model_settings['dim_info'] == dim_info

    ##------------------------------------------------------------------------.
    ### Define the model architecture  
    model = get_pytorch_model(module = my_architectures,
                              model_settings = model_settings)            

    ###-----------------------------------------------------------------------.
    ## Load a pre-trained model  
    load_pretrained_model(model = model, 
                          model_dir = model_dir)
    
    ###-----------------------------------------------------------------------.
    ### Transfer model to the device (i.e. GPU)
    model = model.to(device)

    ##------------------------------------------------------------------------.
    # Run predictions 
    dask.config.set(scheduler='synchronous')
    ds_forecasts = AutoregressivePredictions( model = model, 
                                              # Data
                                              da_dynamic = da_dynamic,
                                              da_static = da_static,              
                                              da_bc = da_bc, 
                                              scaler_transform = scaler,
                                              scaler_inverse = scaler,
                                              # Dataloader options
                                              device = device,
                                              batch_size = batch_size,  # number of forecasts per batch
                                              num_workers = dataloader_settings['num_workers'], 
                                              prefetch_factor = dataloader_settings['prefetch_factor'], 
                                              prefetch_in_gpu = dataloader_settings['prefetch_in_gpu'],  
                                              pin_memory = dataloader_settings['pin_memory'],
                                              asyncronous_gpu_transfer = dataloader_settings['asyncronous_gpu_transfer'],
                                              # Autoregressive settings
                                              input_k = ar_settings['input_k'], 
                                              output_k = ar_settings['output_k'], 
                                              forecast_cycle = ar_settings['forecast_cycle'],                         
                                              stack_most_recent_prediction = ar_settings['stack_most_recent_prediction'], 
                                              # Prediction options 
                                              forecast_reference_times = forecast_reference_times, 
                                              ar_blocks = ar_blocks,
                                              ar_iterations = ar_iterations,  # How many time to autoregressive iterate
                                              # Save options 
                                              zarr_fpath = forecast_zarr_fpath,  # None --> do not write to disk
                                              rounding = 2,             # Default None. Accept also a dictionary 
                                              compressor = "auto",      # Accept also a dictionary per variable
                                              chunks = "auto")
    print(ds_forecasts)
    print("========================================================================================")
    print("- Job done ;) ")

if __name__ == '__main__':
    default_data_dir = "/ltenas3/DeepSphere/data/preprocessed/ERA5_HRES"
    parser = argparse.ArgumentParser(description='Launch weather predictions.')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--forecast_reference_times',
                        nargs="*", type=str)
    parser.add_argument('--data_dir', type=str, default=default_data_dir)
    parser.add_argument('--ar_iterations', type=int, default= 500)
    parser.add_argument('--ar_blocks', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dst_dirpath', type=str)
    parser.add_argument('--zarr_fname', type=str, default='pred.zarr')             
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--force_zarr', type=str, default='False') 
    parser.add_argument('--force_gpu', type=str, default='True')                
    
    args = parser.parse_args()    
    
    print('dst_dirpath:', args.dst_dirpath)
    print(args.forecast_reference_times)
    print(type(args.forecast_reference_times))
    
    if args.force_zarr == 'True':
        force_zarr = True
    else: 
        force_zarr = False
    if args.force_gpu == 'True':
        force_gpu = True
    else: 
        force_gpu = False
    
    os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda  
        
    main(data_dir = args.data_dir, 
         model_dir = args.model_dir,
         ar_iterations =  args.ar_iterations, 
         forecast_reference_times =  args.forecast_reference_times, 
         batch_size =  args.ar_iterations, 
         ar_blocks =  args.ar_iterations, 
         dst_dirpath = args.dst_dirpath,   
         zarr_fname = args.zarr_fname,
         force_zarr = force_zarr,
         force_gpu = force_gpu,)
 
 
    
   

 

 