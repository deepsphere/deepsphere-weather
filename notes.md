# Notes on everyday work. 

### Tuesday 11/08

* Evaluate results on x2 number of channels architecture

* Train model with only **12 years of training data** setting as end point a 
validation error lower than 0.4. 
Goal: explore required time for training and effect on results. Figures stored in
`data/healpix/figures/spherical_unet_deeper_original_1990_error`
    - Results: errors increases for geopotential while keeps similar for temperature. 
    - Required 18 epochs for error under target: 
    slower than training 7 epochs with full dataset
    - Strong effect on bias and variability of results
    
* Train model with only **20 years of training data** setting as end point a 
validation error lower than 0.4. 
Goal: explore required time for training and effect on results. Figures stored in
`data/healpix/figures/spherical_unet_deeper_original_1990_error`
    - Results: errors increases for geopotential while keeps similar for temperature. 
    - Required 10 epochs for error under target: around 7h of training
    - Strong effect on bias and variability of results

* Try deeper architecture. Add extra ConvBlock. 
Explore effects on results and trade-off between training time and loss achieved.
    - Architecture code in file `modules/other_architecture.py`
    - Plots in `data/healpix/figures/try_deeper_extra_block/...png`
    - Training performed with full training data; batch size = 50; epochs = 7
    - Results: 
        - RMSE: Z500 - 0: 86.2445
        - RMSE: T850 - 0: 0.7149
        
    - Bias strongly increased; results seem to be worse than originally for further in 
    time steps.

* Read paper: [STConvS2S](https://arxiv.org/abs/1912.00134):
    - Propose using architecture with 3D convolutions split into two phases: 
    an encoder phase where 2D convs are performed to learn spatial features and
    a decoder phase where 1D convs are performed to learn temporal features
    - In both steps, the size of the input and output are conserved across the convolved
    dimension by using appropriate padding.
    - Convolutions are performed using 3D convolutions where the kernel has been modified
    to act only in the desired dimensions. 
    

* TODO - TO complete: Try masking the loss on +- 10º latitude. 
    - Done in notebook `notebooks/train_last_model_l2_masked_loss.ipynb`
    - Results stored in `data/healpix/figures/masked_loss/*.png`
    - Results strongly hurt around equatorial line (as excepted) for Z500 but not for
    T850 which results completely counterintuitive. 
    - Does not reduce training time
   
* TODO - TO complete: Try standardizing data based on monthly - weekly averages. 
    - Standardized data based on month mean and std from training set (1979 to 2012).
    Data is saved in a file called `data/healpix/5.625_nearest/monthly_standardized_data.nc`
    