## Modules

```AR_Scheduler.py.py``` 

* Autoregressive (AR) weights scheduler to adapt loss weight at each AR iteration during training. 

``` dataloader_autoregressive.py``` 

* pyTorch Autoregressive (AR) DataLoader for nowcasting / forecasting problems.
* It allow for multiprocess-based prefetching in CPU and GPU, with asynchronous GPU data trasfer.
* It expects xarray DataArrays in memory or lazy-loaded from a zarr store. 

``` early_stopping.py.py``` 

* pyTorch utils for early stopping training and controlling AR weights updates. 


``` layers.py```

* pyTorch layers to perform convolutions and pooling operations on spherical unstructured grids.

``` models.py```

* pyTorch architecture general structures

``` predictions_autoregressive.py```

* pyTorch code to generate autoregressive (AR) predictions 

``` remap.py```

* Functions to remap between spherical unstructured grids. 
* It requires CDO > 1.9.8  

``` training_autoregressive.py```

* pyTorch functions for training autoregressive (AR) models using recurrent or AR strategies.

``` xscalers.py```

* Implements MinMaxScaler and StandardScaler à la scikit-learn for multidimensional xarray tensors.
* Implements Climatology, Anomaly, Trend scalers

``` xsphere.py```

* Implement the ```sphere``` accessor to xarray for plotting spherical unstructured grids with FacetGrid capability.  

