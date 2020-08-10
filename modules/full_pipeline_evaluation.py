import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))

import xarray as xr
import numpy as np
import time
import os
import pandas as pd
#import yaml
import json
import matplotlib.pyplot as plt
from matplotlib import cm, colors


import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from modules.utils import init_device
from modules.healpix_models import UNetSphericalHealpix, UNetSphericalTempHealpix, Conv1dAuto, \
UNetSphericalHealpixResidual, _compute_laplacian_healpix, ConvBlock, ConvCheb, UNetSphericalHealpixResidual_2, \
UNetSphericalHealpixDeep
from modules.test import compute_rmse_healpix
from modules.plotting import plot_rmses
from modules.full_pipeline import load_data_split, WeatherBenchDatasetXarrayHealpixTemp, \
                                  train_model_2steps, create_iterative_predictions_healpix_temp, \
                                  compute_errors, plot_climatology
from modules.plotting import plot_general_skills, plot_benchmark, plot_skillmaps, plot_benchmark_simple
from modules.data import hp_to_equiangular


def main():

    print('Reading confing file and set up folders...')

    #with open("../modules/config_train.yml", "r") as ymlfile:
    #    cfg = yaml.full_load(ymlfile)

    with open("../modules/config_train.json") as json_data_file:
        cfg = json.load(json_data_file)


    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu = [0]
    num_workers = 10
    pin_memory = True

    datadir = cfg['directories']['datadir']
    input_dir = datadir + cfg['directories']['input_dir']
    model_save_path = datadir + cfg['directories']['model_save_path']
    pred_save_path = datadir + cfg['directories']['pred_save_path']
    metrics_path = datadir + cfg['directories']['metrics_path']


    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    if not os.path.isdir(pred_save_path):
        os.mkdir(pred_save_path)


    chunk_size = cfg['training_constants']['chunk_size']

    train_years = (cfg['training_constants']['train_years'][0], cfg['training_constants']['train_years'][1])
    val_years = (cfg['training_constants']['val_years'][0], cfg['training_constants']['val_years'][1])
    test_years = (cfg['training_constants']['test_years'][0], cfg['training_constants']['test_years'][1])

    # training parameters
    nodes = cfg['training_constants']['nodes']
    max_lead_time = cfg['training_constants']['max_lead_time']
    nb_timesteps = cfg['training_constants']['nb_timesteps']
    epochs = cfg['training_constants']['nb_epochs']
    learning_rate = cfg['training_constants']['learning_rate']
    batch_size = cfg['training_constants']['batch_size']

    # model parameters
    len_sqce = cfg['model_parameters']['len_sqce']
    delta_t = cfg['model_parameters']['delta_t']
    in_features = cfg['model_parameters']['in_features']
    out_features = cfg['model_parameters']['out_features']
    architecture_name = cfg['model_parameters']['architecture_name']
    resolution = cfg['model_parameters']['resolution']
    lead_time = delta_t
    kernel_size_pooling = cfg['model_parameters']['kernel_size_pooling']

    description = "all_const_len{}_delta_{}_architecture_".format(len_sqce, delta_t) + architecture_name
    model_filename = model_save_path + description + ".h5"
    pred_filename = pred_save_path +  description + ".nc"
    rmse_filename = datadir + 'metrics/rmse_' + description + '.nc'

    ##############################################

    print('Load data files...')
    obs = xr.open_mfdataset(pred_save_path + cfg['directories']['obs_file_name'], combine='by_coords', chunks={'time':chunk_size})
    rmses_weyn = xr.open_dataset(datadir + cfg['directories']['rmse_weyn_name'])

    ct_filename = cfg['directories']['constants']
    constants = xr.open_dataset(f'{input_dir}' + ct_filename)

    orog = constants['orog']
    lsm = constants['lsm']
    lats = constants['lat2d']
    slt = constants['slt']

    num_constants = len([orog, lats, lsm, slt])

    train_mean_ = xr.open_mfdataset(f'{input_dir}' + cfg['directories']['train_mean_file'])
    train_std_ = xr.open_mfdataset(f'{input_dir}' + cfg['directories']['train_std_file'])

    # load data and provide train, test and validation datasets
    ds_train, ds_valid, ds_test = load_data_split(input_dir, train_years, val_years, test_years, chunk_size)

    training_ds = WeatherBenchDatasetXarrayHealpixTemp(ds=ds_train, out_features=out_features, delta_t=delta_t,
                                                       len_sqce=len_sqce, max_lead_time=max_lead_time,
                                                       years=train_years, nodes=nodes, nb_timesteps=nb_timesteps,
                                                       mean=train_mean_, std=train_std_, load=False)

    validation_ds = WeatherBenchDatasetXarrayHealpixTemp(ds=ds_valid, out_features=out_features, delta_t=delta_t,
                                                         len_sqce=len_sqce, max_lead_time=max_lead_time,
                                                         years=val_years, nodes=nodes, nb_timesteps=nb_timesteps,
                                                         mean=train_mean_, std=train_std_, load=False)

    ##############################################

    print('Define model...')
    spherical_unet = UNetSphericalHealpix(N=nodes, in_channels=in_features*len_sqce, out_channels=out_features, \
                                              kernel_size=3)
    spherical_unet, device = init_device(spherical_unet, gpu=gpu)

    constants_tensor = torch.tensor(xr.merge([orog, lats, lsm, slt], compat='override').to_array().values, \
                                dtype=torch.float)

    # standardize
    constants_tensor = (constants_tensor - torch.mean(constants_tensor, dim=1).view(-1,1).expand(num_constants, nodes)) / \
                        torch.std(constants_tensor, dim=1).view(-1,1).expand(num_constants, nodes)


    print('Train model...')
    # Train model
    torch.cuda.empty_cache()
    train_loss, val_loss, train_loss_it, times_it = train_model_2steps(spherical_unet, device, training_ds, constants_tensor.transpose(1,0), \
                                              batch_size=batch_size, epochs=epochs, \
                                               lr=learning_rate, validation_ds=validation_ds)

    # save model
    torch.save(spherical_unet.state_dict(), model_filename)

    figures_path = datadir + 'figures/' + description + '/'
    if not os.path.isdir(figures_path):
        os.mkdir(figures_path)

    # Show training losses
    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(figures_path + 'MSE_train_val.png')
    plt.show()

    #return train_loss, val_loss, train_loss_it, times_it
    del training_ds, validation_ds
    torch.cuda.empty_cache()

    ##############################################

    print('Load test data and generate predictions...')
    # Testing data
    testing_ds = WeatherBenchDatasetXarrayHealpixTemp(ds=ds_test, out_features=out_features,
                                                      len_sqce=len_sqce, delta_t=delta_t, years=test_years,
                                                      nodes=nodes, nb_timesteps=nb_timesteps,
                                                      mean=train_mean_, std=train_std_,
                                                      max_lead_time=max_lead_time)

    predictions, lead_times, times, nodes, out_lat, out_lon = \
    create_iterative_predictions_healpix_temp(spherical_unet, device, testing_ds, constants_tensor.transpose(1,0))

    # generate xarray ds and save predictions file
    das = [];
    lev_idx = 0
    for var in ['z', 't']:
        das.append(xr.DataArray(
            predictions[:, :, :, lev_idx],
            dims=['lead_time', 'time', 'node'],
            coords={'lead_time': lead_times, 'time': times[:predictions.shape[1]], 'node': np.arange(nodes)},
            name=var
        ))
        lev_idx += 1

    prediction_ds = xr.merge(das)
    prediction_ds = prediction_ds.assign_coords({'lat': out_lat, 'lon': out_lon})
    prediction_ds.to_netcdf(pred_filename)

    # Compute and save RMSE
    rmse = compute_rmse_healpix(prediction_ds, obs).load()
    rmse.to_netcdf(rmse_filename)

    # Show RMSE
    print('Z500 - 0:', rmse.z.values[0])
    print('T850 - 0:', rmse.t.values[0])

    plot_rmses(rmse, rmses_weyn.rename({'z500':'z', 't850':'t'}).isel(lead_time=list(range(20))), lead_time=6)

    #del spherical_unet
    del prediction_ds, rmse
    torch.cuda.empty_cache()


    ##############################################

    print('Generate plots for evaluation...')


    start_time = len_sqce * lead_time - delta_t
    end_time = (delta_t-lead_time) if (delta_t-lead_time) > 0 else None

    # Data
    lead_times = np.arange(lead_time, max_lead_time+lead_time, lead_time)

    pred = xr.open_mfdataset(pred_save_path  + description + '.nc', combine='by_coords',  chunks={'time':chunk_size})
    obs = obs.isel(time=slice(delta_t,pred.time.shape[0]+delta_t))

    print('Compute errors...')
    corr_map, rbias_map, rsd_map, rmse_map, obs_rmse, rmse_map_norm = compute_errors(pred, obs)

    rmse_spherical = xr.load_dataset(metrics_path + 'rmse_' + description + '.nc')
    rbias_spherical = rbias_map.mean('node').compute()
    rsd_spherical = rsd_map.mean('node').compute()
    corr_spherical = corr_map.mean('node').compute()

    rbias_spherical.to_netcdf(metrics_path + 'rbias_' + description + '.nc')
    rsd_spherical.to_netcdf(metrics_path + 'rsd_' + description + '.nc')
    corr_spherical.to_netcdf(metrics_path + 'corr_' + description + '.nc')

    print('Generate plots to evaluate model performance...')
    plot_benchmark_simple(rmse_spherical, description, lead_times,
                   input_dir=metrics_path, output_dir=figures_path, title=False)

    plot_general_skills(rmse_map_norm, corr_map, rbias_map, rsd_map, description, lead_times,
                        output_dir=figures_path, title=False)

    plot_skillmaps(rmse_map_norm, rsd_map, rbias_map, corr_map, description, lead_times, resolution,
                   output_dir=figures_path)


    monthly_mean = pred.groupby('time.month').mean().compute()
    monthly_mean_obs = obs.groupby('time.month').mean().compute()

    lead_idx = 19

    # Computations
    monthly_mean.isel(lead_time=lead_idx)
    monthly_mean_eq = []
    for month in range(12):
        monthly_mean_eq.append(hp_to_equiangular(monthly_mean.isel(lead_time=lead_idx, month=month),
                                         resolution))
    monthly_mean_eq = xr.concat(monthly_mean_eq, pd.Index(np.arange(1, 13, 1), name='month'))
    monthly_lat_eq = monthly_mean_eq.mean('lon')

    monthly_mean_obs.isel(lead_time=lead_idx)
    monthly_mean_eq_obs = []
    for month in range(12):
        monthly_mean_eq_obs.append(hp_to_equiangular(monthly_mean_obs.isel(lead_time=lead_idx, month=month),
                                         resolution))
    monthly_mean_eq_obs = xr.concat(monthly_mean_eq_obs, pd.Index(np.arange(1, 13, 1), name='month'))
    monthly_lat_eq_obs = monthly_mean_eq_obs.mean('lon')

    pred_z = np.rot90(monthly_lat_eq.z.values, 3)
    pred_t = np.rot90(monthly_lat_eq.t.values, 3)
    obs_z = np.rot90(monthly_lat_eq_obs.z.values, 3)
    obs_t = np.rot90(monthly_lat_eq_obs.t.values, 3)

    diff_z = pred_z / obs_z
    diff_t = pred_t / obs_t

    # Labels and limits
    ticks = np.linspace(0, 31, 7).astype(int)
    lat_labels = np.linspace(-90, 90, 7).astype(int)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    vmin_z = min(np.min(monthly_lat_eq.z).values.flatten()[0], np.min(monthly_lat_eq_obs.z).values.flatten()[0])
    vmax_z = max(np.max(monthly_lat_eq.z).values.flatten()[0], np.max(monthly_lat_eq_obs.z).values.flatten()[0])
    vmin_t = min(np.min(monthly_lat_eq.t).values.flatten()[0], np.min(monthly_lat_eq_obs.t).values.flatten()[0])
    vmax_t = max(np.max(monthly_lat_eq.t).values.flatten()[0], np.max(monthly_lat_eq_obs.t).values.flatten()[0])

    delta = min((np.min(diff_z)-1), (1-np.max(diff_z)), (np.min(diff_t)-1), (1-np.max(diff_t)))

    vmin_sd = 1 - delta
    vmax_sd = 1 + delta

    predictions_vals = {'pred_z': pred_z, 'pred_t': pred_t,'obs_z': obs_z,'obs_t': obs_t}
    val_limits = {'vmin_z':vmin_z, 'vmax_z':vmax_z, 'vmin_t':vmin_t, \
                  'vmax_t':vmax_t, 'vmin_sd':vmin_sd, 'vmax_sd':vmax_sd}

    figname = figures_path + description + '_hovmoller'
    plot_climatology(figname, predictions_vals, val_limits, ticks, lat_labels, month_labels)

    print('DONE!')
    print(description)


if __name__=="__main__":
    main()