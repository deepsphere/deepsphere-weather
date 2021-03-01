## Modules

* ```AR_Scheduler.py.py``` 

Autoregressive (AR) weights scheduler to adapt loss weight at each AR iteration during training. 

* ``` dataloader_autoregressive.py``` 

Autoregressive (AR) pyTorch DataLoader for nowcasting / forecasting problems.
It allow for multiprocess-based prefetching in CPU and GPU, with asynchronous trasfer.
It expects xarray DataArrays in memory or lazy-loaded from a zarr store. 

* ``` early_stopping.py.py``` 

pyTorch utils for early stopping training and controlling AR weight updates. 


* ``` layers.py```

pyTorch layers to perform convolutions and pooling operations on spherical unstructured grids.




Allows to to train and test a model  
with a loss function that includes multiple steps that can be defined by the user. It saves the model after every epoch
but does not generate the predictions (to save time since it can be done in parallel using the notebook 
```generate_evaluate_predictions.ipynb ```). The parameters are defined inside the main function, although it can be 
adapted to use a config file as in ```full_pipeline_evalution.py```

It is important to remark that the update function that takes care of the weight's update is defined on top
of the file and should be adapted to the number of lead steps taken into account in the loss function.

* ```architecture.py```

Contains pytorch models used for both ``` full_pipeline_multiple_steps.py``` and ``` full_pipeline_evaluation.py``` 
Previous architectures used can be found in the folder ``` modules/old_architectures/```

* ``` plotting.py```

Contains different functions to generate evaluation plots. 
