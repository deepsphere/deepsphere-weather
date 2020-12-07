import argparse
import sys
import os
import glob
import json
from datetime import datetime
import time

import xarray as xr
import matplotlib.pyplot as plt
import torch
import healpy as hp
import numpy as np

from modules.full_pipeline import load_data_split, WeatherBenchDatasetXarrayHealpixTemp, create_iterative_predictions_healpix_temp, compute_errors
from modules.architectures import UNetSpherical
from modules.test import compute_rmse, compute_weighted_rmse
from modules.plotting import plot_rmses, plot_general_skills, plot_benchmark, plot_skillmaps, plot_benchmark_simple
from modules.data import pix2ang

def generate_file_name(path, tag, desc, epoch):
    return "{}{}_{}_epoch_{}.nc".format(path, tag, desc, epoch)

def main(cfg):
    net_params = {}
    net_params["sampling"] = cfg['model_parameters'].get("sampling", None)
    net_params["knn"] = cfg['model_parameters'].get("knn", None)
    net_params["conv_type"] = cfg['model_parameters'].get("conv_type", None)
    net_params["pool_method"] = cfg['model_parameters'].get("pool_method", None)
    net_params["ratio"] = cfg['model_parameters'].get("ratio", None)
    net_params["periodic"] = cfg['model_parameters'].get("periodic", None)

    description = "{}_{}_{}_{}_{}_{}".format(*net_params.values())

    datadir = cfg['directories']['datadir']
    input_dir = datadir + cfg['directories']['input_dir']
    result_path = cfg['directories']['save_dir']
    model_path = result_path + cfg['directories']['model_save_path']
    prediction_path = result_path + cfg['directories']['pred_save_path']

    chunk_size = cfg['training_constants']['chunk_size']
    train_years = cfg['training_constants']['train_years']
    val_years = cfg['training_constants']['val_years']
    test_years = cfg['training_constants']['test_years']
    nodes = cfg['training_constants']['nodes']
    max_lead_time = cfg['training_constants']['max_lead_time']
    nb_timesteps = cfg['training_constants']['nb_timesteps']
    len_sqce = cfg['model_parameters']['len_sqce']
    delta_t = cfg['model_parameters']['delta_t']
    in_features = cfg['model_parameters']['in_features']
    out_features = cfg['model_parameters']['out_features']
    resolution = cfg["model_parameters"]["resolution"]

    metrics_path = result_path + 'metrics/'
    figures_path = result_path + 'figures/'
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    obs = xr.open_mfdataset(prediction_path + 'observations.nc', combine='by_coords', chunks={'time':chunk_size})
    rmses_weyn = xr.open_dataset(datadir + 'metrics/rmses_weyn.nc')

    constants = xr.open_dataset(f'{input_dir}constants/constants_5.625deg_standardized.nc')
    orog = constants['orog']
    lsm = constants['lsm']
    lats = constants['lat2d']
    slt = constants['slt']
    num_constants = len([orog, lats, lsm, slt])

    train_mean_ = xr.open_mfdataset(f'{input_dir}mean_train_features_dynamic.nc')
    train_std_ = xr.open_mfdataset(f'{input_dir}std_train_features_dynamic.nc')

    _, _, ds_test = load_data_split(input_dir, train_years, val_years, test_years, chunk_size)

    # Testing data
    testing_ds = WeatherBenchDatasetXarrayHealpixTemp(ds=ds_test, out_features=out_features,
                                                    len_sqce=len_sqce, delta_t=delta_t, years=test_years, 
                                                    nodes=nodes, nb_timesteps=nb_timesteps, 
                                                    mean=train_mean_, std=train_std_, 
                                                    max_lead_time=max_lead_time)

    constants_tensor = torch.tensor(xr.merge([orog, lats, lsm, slt], compat='override').to_array().values, dtype=torch.float)
    # standardize 
    constants_tensor_mean = torch.mean(constants_tensor, dim=1, keepdim=True)
    constants_tensor_std = torch.std(constants_tensor, dim=1, keepdim=True)
    constants_tensor = (constants_tensor - constants_tensor_mean) / (constants_tensor_std + 1e-6)

    model = UNetSpherical(N=nodes, in_channels=in_features * len_sqce, out_channels=out_features, kernel_size=3, **net_params)
    extract_epoch = lambda x: int(x.split('_')[-1][:-3]) # filename e.g. XXXX_epoch_1.h5
    saved_models = glob.glob(model_path + '*.h5')
    saved_models.sort(key=extract_epoch, reverse=True)
    model.load_state_dict(torch.load(saved_models[0]), strict=False) # Use latest state dict

    if torch.cuda.is_available():
        device = 'cuda: 0'
        model = model.to(device)
    else:
        device = 'cpu'

    epoch = extract_epoch(saved_models[0])
    pred_filename = generate_file_name(prediction_path, 'pred', description, epoch)
    rmse_filename = generate_file_name(metrics_path, 'rmse', description, epoch)

    model.eval()
    with torch.set_grad_enabled(False):
        pred, lead_times, times = create_iterative_predictions_healpix_temp(model, device, testing_ds, constants_tensor.transpose(1,0))

    das = []
    for ind, var in enumerate(['z', 't']):       
        curr = xr.DataArray(pred[:, :, :, ind], dims=['lead_time', 'time', 'node'], coords={'lead_time': lead_times, 'time': times[:pred.shape[1]], 'node': np.arange(nodes)}, name=var)
        das.append(curr)

    if net_params['sampling'] == 'equiangular':
        out_lat, out_lon = pix2ang(nodes)
    elif net_params['sampling'] == 'healpix':
        nside = int(np.sqrt(nodes/12))
        out_lat, out_lon = hp.pix2ang(nside, np.arange(nodes), lonlat=True)
        
    pred_merged = xr.merge(das)
    pred_merged = pred_merged.assign_coords({'lat': out_lat, 'lon': out_lon})

    pred_merged.to_netcdf(pred_filename)

    # select observations
    obs_curr = obs.isel(time=slice(6, pred_merged.time.shape[0] + 6))

    # compute RMSE
    rmse = None
    if net_params['sampling'] == 'equiangular':
        rmse = compute_weighted_rmse(pred_merged, obs_curr)
    elif net_params['sampling'] == 'healpix':
        rmse = compute_rmse(pred_merged, obs_curr)
    rmse.to_netcdf(rmse_filename)
        
        
    # plot RMSE
    print('Z500 - 0:', rmse.z.values[0])
    print('T850 - 0:', rmse.t.values[0])

    plot_rmses(rmse, rmses_weyn.rename({'z500':'z', 't850':'t'}).isel(lead_time=list(range(20))), lead_time=6)

    t = time.time()
    corr_map, rbias_map, rsd_map, rmse_map, obs_rmse, rmse_map_norm = compute_errors(pred_merged, obs_curr)
    print(time.time() - t)

    rmse_spherical = xr.load_dataset(rmse_filename)
    rbias_spherical = rbias_map.mean('node').compute()
    rsd_spherical = rsd_map.mean('node').compute()
    corr_spherical = corr_map.mean('node').compute()

    rbias_spherical.to_netcdf(generate_file_name(metrics_path, 'rbias', description, epoch))
    rsd_spherical.to_netcdf(generate_file_name(metrics_path, 'rsd', description, epoch))
    corr_spherical.to_netcdf(generate_file_name(metrics_path, 'corr', description, epoch))

    plot_benchmark_simple(rmse_spherical, description, lead_times, input_dir=datadir, output_dir=figures_path, title=False)
    plot_general_skills(rmse_map_norm, corr_map, rbias_map, rsd_map, description, lead_times, output_dir=figures_path, title=False)
    plot_skillmaps(rmse_map_norm, rsd_map, rbias_map, corr_map, description, lead_times, resolution, output_dir=figures_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training weather prediction model')
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--cuda', type=str, default=0)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    main(config)