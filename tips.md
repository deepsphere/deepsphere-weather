## How to mount from remote server into local machine

#### For macOS users
Download "FUSE for macOS" and SSHFS from the [FUSE for macOS website](https://osxfuse.github.io/)

In your local terminal, create a directory where you want to mount the files
```
mkidr my_project
```

Then, create the access
```
sshfs username@lts2gdk0.epfl.ch: path/to/folder my_project/
```

## xArray 
When standardizing data based on monthly/weekly anomalies: 

First, data to be standardized needs to be grouped based on the temporal dim we are interested in: 

```ds.groupby({time.month}) or ds.groupby({time.week})```

Then, if done in a single step (ds - ds_mean)/ds_std it won't work since it will create an extra dimension. 
The only way is: 

```
ds = ds.groupby({time.month}) - ds_mean.to_array(dim='level)
ds = ds.groupby({time.month}) / ds_std.to_array(dim='level)
ds.compute()
```

## Download data from ERA5 and rmse metrics for benchmarking

Follow the instructions from [WeatherBench](https://github.com/pangeo-data/WeatherBench)

Data used is ERA5 @ 5.625deg

The rmse values used for comparison can be found [here](https://dataserv.ub.tum.de/index.php/s/m1524895?path=%2Fbaselines)
You only need the file ```rmse.pkl```


## Model comparison

The file ```data/healpix/metrics/final_models_rmse.pkl``` contains the rmse of the following models:


```varying (ep 7)``` = 'all_const_len2_delta_6_architecture_loss_v0_8steps_increas_reinitialize_residual_l3_per_epoch_epoch_7'

```varying (ep 11)``` = 'all_const_len2_delta_6_architecture_loss_v0_8steps_increas_reinitialize_residual_l3_per_epoch_epoch_10'

```varying - long``` = 'all_const_len2_delta_6_architecture_loss_v0_8steps_increas_reinitialize_residual_l3_long_connections_per_epoch_epoch_5'

```constant decr``` = 'all_const_len2_delta_6_architecture_loss_v0_8steps_constant_decresing_l3_per_epoch_epoch_6'

```constant incr``` = 'all_const_len2_delta_6_architecture_loss_v0_8steps_constant_increasing_l3_per_epoch_epoch_5'

```constant incr long``` = 'spherical_unet_3days_direct_forecast_epoch6'

```direct-6h``` = 'all_const_len2_delta_6_architecture_loss_v0_8steps_constant_increasing_static_l3_long_connections_per_epoch_epoch_7'

*varying* --> weights are updated based on the learning curve of the model during training. Initially full weight is given 
for the first prediction (6h). The evolution of the weight is given by the function 

```python
    def update_w(w):
        """
        Update array of weights for the loss function
        :param w: array of weights from earlier step [0] to latest one [-1]
        :return: array of weights modified
        """
        for i in range(1, len(w)):
            len_w = len(w)
            w[len_w - i] += w[len_w - i -1]*0.4
            w[len_w - i - 1] *= 0.8
        w = np.array(w)/sum(w)
        
        return w

```

*constant* --> weights are not updated. Increasing stands for weights distributed in an increasing fashion (more weight is
attributed to further prediction steps in time).  Decreasing acts in the opposite way. 

*long* --> implies the usage of the architecture names ``` UNetSphericalHealpixResidualLongConnections ``` . If not indicated,
the architecture used was ``` UNetSphericalHealpixResidualShort3LevelsOnlyEncoder ```

*direct* --> only the first prediction step is taken into account for the loss for training. 
