import xarray as xr
import numpy as np
import time
import os
import random
import matplotlib.pyplot as plt


import torch
from torch import nn, optim

from modules.plotting import plot_rmses
from modules.utils import init_device
from modules.architectures import UNetSphericalHealpixResidualShort3LevelsOnlyEncoder
from modules.test import compute_rmse_healpix
from modules.full_pipeline import load_data_split, WeatherBenchDatasetXarrayHealpixTemp, \
    train_model_2steps, create_iterative_predictions_healpix_temp, \
    compute_errors, plot_climatology, WeatherBenchDatasetXarrayHealpixTempMultiple


import warnings
warnings.filterwarnings("ignore")

def main():
    def update_w(w):
        for i in range(1,8):
            w[8 - i] += w[8 - i -1]*0.4
            w[8 - i - 1] *= 0.8
        w = np.array(w)/sum(w)
        return w


    def train_model_multiple_steps(model, weights_loss, device, training_ds, len_output, constants, batch_size, \
                                   epochs, lr, validation_ds, model_filename):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7, weight_decay=0, amsgrad=False)

        train_losses = []
        val_losses = []
        n_samples = training_ds.n_samples
        n_samples_val = validation_ds.n_samples
        num_nodes = training_ds.nodes
        num_constants = constants.shape[1]
        out_features = training_ds.out_features
        num_in_features = training_ds.in_features + num_constants

        constants_expanded = constants.expand(batch_size, num_nodes, num_constants)
        constants1 = constants_expanded.to(device)
        idxs_val = validation_ds.idxs

        # initially give full weight to first output
        #weights_loss = [1] + [0] * (len_output - 1)
        # latest prediction included in the loss
        required_output = np.max(np.where(weights_loss > 0)) + 1

        weight_variations = [(weights_loss, 0, 0)]
        count_upd = 0
        train_loss_steps = {}
        for step_ahead in range(len_output):
            train_loss_steps['t{}'.format(step_ahead)] = []

        test_loss_steps = {}
        for step_ahead in range(len_output):
            test_loss_steps['t{}'.format(step_ahead)] = []

        threshold = 1e-4

        for epoch in range(epochs):

            print('\rEpoch : {}'.format(epoch), end="")

            time1 = time.time()

            val_loss = 0
            train_loss = 0

            model.train()

            random.shuffle(training_ds.idxs)
            idxs = training_ds.idxs

            batch_idx = 0
            train_loss_it = []
            times_it = []
            t0 = time.time()
            for i in range(0, n_samples - batch_size, batch_size):
                i_next = min(i + batch_size, n_samples)

                if len(idxs[i:i_next]) < batch_size:
                    constants_expanded = constants.expand(len(idxs[i:i_next]), num_nodes, num_constants)
                    constants1 = constants_expanded.to(device)

                batch, labels = training_ds[idxs[i:i_next]]

                # Transfer to GPU
                batch_size = batch[0].shape[0] // 2

                batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                    constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(device)

                #  generate predictions multiple steps ahead sequentially
                output = model(batch1)
                label1 = labels[0].to(device)
                l0 = criterion(output, label1[batch_size:, :, :out_features])
                loss_ahead = weights_loss[0] * l0
                train_loss_steps['t0'].append(l0.item())

                for step_ahead in range(1, required_output):
                    # input t-2
                    inp_t2 = batch1[:, :, num_in_features:]

                    #  toa at t-1
                    toa_delta = labels[step_ahead][:batch_size, :, -1].view(-1, num_nodes, 1).to(device)
                    batch1 = torch.cat((inp_t2, output, toa_delta, constants1), dim=2)

                    output = model(batch1)
                    label1 = labels[step_ahead].to(device)
                    l0 = criterion(output, label1[batch_size:, :, :out_features])
                    loss_ahead += weights_loss[step_ahead] * l0
                    train_loss_steps['t{}'.format(step_ahead)].append(l0.item())

                optimizer.zero_grad()
                loss_ahead.backward()
                optimizer.step()

                train_loss += loss_ahead.item() * batch_size
                train_loss_it.append(train_loss / (batch_size * (batch_idx + 1)))
                if len(train_loss_it) > 5:
                    if (np.std(train_loss_it[-10:]) < threshold) and count_upd > 2e2:
                        weights_loss = update_w(weights_loss)
                        required_output = np.max(np.where(weights_loss > 0)) + 1
                        count_upd = 0
                        # print('New weights ', weights_loss, ' Epoch {} Iter {}'.format(epoch, i))
                        weight_variations.append((weights_loss, epoch, len(train_loss_steps['t0'])))
                        threshold /= 10
                    else:
                        count_upd += 1

                times_it.append(time.time() - t0)
                t0 = time.time()

                if batch_idx % 50 == 0:
                    print('\rBatch idx: {}; Loss: {:.3f} - Other {:.5f} - {}' \
                          .format(batch_idx, train_loss / (batch_size * (batch_idx + 1)), np.std(train_loss_it[-10:]),
                                  count_upd),
                          end="")
                batch_idx += 1

            train_loss = train_loss / n_samples
            train_losses.append(train_loss)

            model.eval()

            constants1 = constants_expanded.to(device)
            with torch.set_grad_enabled(False):
                index = 0

                for i in range(0, n_samples_val - batch_size, batch_size):
                    i_next = min(i + batch_size, n_samples_val)

                    if len(idxs_val[i:i_next]) < batch_size:
                        constants_expanded = constants.expand(len(idxs_val[i:i_next]), num_nodes, num_constants)
                        constants1 = constants_expanded.to(device)

                    batch, labels = training_ds[idxs[i:i_next]]

                    # Transfer to GPU
                    batch_size = batch[0].shape[0] // 2
                    batch1 = torch.cat((batch[0][:batch_size, :, :], \
                                        constants_expanded, batch[0][batch_size:, :, :], constants_expanded), dim=2).to(
                        device)

                    #  generate predictions multiple steps ahead sequentially
                    output = model(batch1)
                    label1 = labels[0].to(device)
                    l0 = criterion(output, label1[batch_size:, :, :out_features]).item()
                    loss_ahead = weights_loss[0] * l0
                    test_loss_steps['t0'].append(l0)

                    for step_ahead in range(1, required_output):
                        # input t-2
                        inp_t2 = batch1[:, :, num_in_features:]

                        #  toa at t-1
                        toa_delta = labels[step_ahead][:batch_size, :, -1].view(-1, num_nodes, 1).to(device)
                        batch1 = torch.cat((inp_t2, output, toa_delta, constants1), dim=2)

                        output = model(batch1)
                        label1 = labels[step_ahead].to(device)
                        l0 = criterion(output, label1[batch_size:, :, :out_features]).item()
                        loss_ahead += weights_loss[step_ahead] * l0
                        test_loss_steps['t{}'.format(step_ahead)].append(l0)

                    val_loss += loss_ahead * batch_size
                    index = index + batch_size

            val_loss = val_loss / n_samples_val
            val_losses.append(val_loss)

            time2 = time.time()

            # Print stuff
            print('Epoch: {e:3d}/{n_e:3d}  - loss: {l:.3f}  - val_loss: {v_l:.5f}  - time: {t:2f}'
                  .format(e=epoch + 1, n_e=epochs, l=train_loss, v_l=val_loss, t=time2 - time1))

            torch.save(model.state_dict(), model_filename[:-3] + '_epoch{}'.format(epoch) + '.h5')

        return train_losses, val_losses, train_loss_it, times_it, train_loss_steps, test_loss_steps, weight_variations, weights_loss

    # Define paths
    datadir = "../data/healpix/"
    input_dir = datadir + "5.625deg_nearest/"
    model_save_path = datadir + "models/"
    pred_save_path = datadir + "predictions/"
    prediction_path = '../data/healpix/predictions/'

    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    if not os.path.isdir(pred_save_path):
        os.mkdir(pred_save_path)

    # Define constants
    chunk_size = 521

    train_years = ('1990', '2012')#('1979', '2012')
    val_years = ('2013', '2016')
    test_years = ('2017', '2018')

    nodes = 12*16*16
    max_lead_time = 5*24
    nb_timesteps = 2

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    gpu = [0]
    num_workers = 10
    pin_memory = True

    batch_size = 10
    epochs = 10
    learning_rate = 8e-3

    len_sqce = 2
    # define time resolution
    delta_t = 6

    # predict 5days data
    max_lead_time = 5*24
    in_features = 7
    out_features = 2



    obs = xr.open_mfdataset(pred_save_path + 'observations_nearest.nc', combine='by_coords', chunks={'time':chunk_size})
    rmses_weyn = xr.open_dataset(datadir + 'metrics/rmses_weyn.nc')

    # get training, validation and test data
    ds_train, ds_valid, ds_test = load_data_split(input_dir, train_years, val_years, test_years, chunk_size)

    constants = xr.open_dataset(f'{input_dir}constants/constants_5.625deg_standardized.nc')

    orog = constants['orog']
    lsm = constants['lsm']
    lats = constants['lat2d']
    slt = constants['slt']

    num_constants = len([orog, lats, lsm, slt])

    train_mean_ = xr.open_mfdataset(f'{input_dir}mean_train_features_dynamic.nc')
    train_std_ = xr.open_mfdataset(f'{input_dir}std_train_features_dynamic.nc')

    # define model name
    architecture_name = "loss_v0_8steps_variation0_residual_only_enc_l3_per_epoch"
    description = "all_const_len{}_delta_{}_architecture_".format(len_sqce, delta_t) + architecture_name

    model_filename = model_save_path + description + ".h5"
    pred_filename = pred_save_path +  description + ".nc"
    rmse_filename = datadir + 'metrics/rmse_' + description + '.nc'
    metrics_path = '../data/healpix/metrics/'
    figures_path = '../data/healpix/figures/' + description + '/'

    if not os.path.isdir(figures_path):
        os.mkdir(figures_path)

    # generate dataloaders
    training_ds = WeatherBenchDatasetXarrayHealpixTempMultiple(ds=ds_train, out_features=out_features, delta_t=delta_t,
                                                       len_sqce_input=len_sqce, len_sqce_output=8, max_lead_time=max_lead_time,
                                                       years=train_years, nodes=nodes, nb_timesteps=nb_timesteps,
                                                       mean=train_mean_, std=train_std_)

    validation_ds = WeatherBenchDatasetXarrayHealpixTempMultiple(ds=ds_valid, out_features=out_features, delta_t=delta_t,
                                                         len_sqce_input=len_sqce, len_sqce_output=8, max_lead_time=max_lead_time,
                                                         years=val_years, nodes=nodes, nb_timesteps=nb_timesteps,
                                                         mean=train_mean_, std=train_std_)

    # generate model
    spherical_unet = UNetSphericalHealpixResidualShort3LevelsOnlyEncoder(N=nodes, in_channels=in_features*len_sqce, out_channels=out_features, kernel_size=3)
    spherical_unet, device = init_device(spherical_unet, gpu=gpu)

    iterations_per_epoch = int(np.ceil(training_ds.n_samples / batch_size))

    constants_tensor = torch.tensor(xr.merge([orog, lats, lsm, slt], compat='override').to_array().values, \
                                dtype=torch.float)

    # standardize
    constants_tensor = (constants_tensor - torch.mean(constants_tensor, dim=1).view(-1,1).expand(4, 3072)) / \
                        torch.std(constants_tensor, dim=1).view(-1,1).expand(4, 3072)




    w = [1] + [0] * 7
    #w = [1] * 8
    w = np.array(w)
    w = w / sum(w)
    f, ax = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
    ax = ax.flatten()

    x_vals = list(range(8))
    ax[0].scatter(x_vals, w)
    ax[0].set_title('Initial weights')

    for i in range(15):
        w = update_w(w)
        ax[i + 1].scatter(x_vals, w)
        ax[i + 1].set_title('Update ' + str(i))

    plt.xlabel('Time step ahead')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.savefig(figures_path + 'weight_updates.png')
    plt.show()

    # Testing data
    testing_ds = WeatherBenchDatasetXarrayHealpixTemp(ds=ds_test, out_features=out_features,
                                                      len_sqce=len_sqce, delta_t=delta_t, years=test_years,
                                                      nodes=nodes, nb_timesteps=nb_timesteps,
                                                      mean=train_mean_, std=train_std_,
                                                      max_lead_time=max_lead_time)




    for ep in range(epochs):

        print('Starting epoch {}'.format(ep + 1))

        spherical_unet.train()

        train_losses, val_losses, train_loss_it, times_it, train_loss_steps, test_loss_steps, weight_variations, w = \
            train_model_multiple_steps(spherical_unet, w, device, training_ds, 8, constants_tensor.transpose(1, 0), \
                                       batch_size=batch_size, epochs=1, lr=learning_rate, validation_ds=validation_ds, \
                                       model_filename=model_filename)

        # save model
        torch.save(spherical_unet.state_dict(), model_filename[:-3] + '_epoch_{}'.format(ep) + '.h5')

        torch.cuda.empty_cache()

        print('Generating predictions...')
        predictions, lead_times, times, nodes, out_lat, out_lon = \
            create_iterative_predictions_healpix_temp(spherical_unet, device, testing_ds, constants_tensor.transpose(1, 0))

        das = []
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

        prediction_ds.to_netcdf(pred_filename[:-3] + '_epoch_{}'.format(ep) + '.nc')

        print('Computing error...')
        rmse = compute_rmse_healpix(prediction_ds, obs).load()
        rmse.to_netcdf(rmse_filename[:-3] + '_epoch_{}'.format(ep) + '.nc')

        print('Z500 - 0:', rmse.z.values[0])
        print('T850 - 0:', rmse.t.values[0])

        print('Z500 - 120h:', rmse.z.values[-1])
        print('T850 - 120h:', rmse.t.values[-1])

        print('Current state of weights: ', w)

        plot_rmses(rmse, rmses_weyn.rename({'z500': 'z', 't850': 't'}).isel(lead_time=list(range(20))), lead_time=6)

        del prediction_ds, rmse

if __name__=="__main__":
    main()