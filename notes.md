# Notes on everyday work. 

### Week 10/08

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
    - Required 10 epochs for error under target: around 7h of training
    - Strong effect on bias and variability of results
    

* Try masking the loss on +- 10º latitude. 
    - Done in notebook `notebooks/train_last_model_l2_masked_loss.ipynb`
    - Results stored in `data/healpix/figures/masked_loss/*.png`
    - Results strongly hurt around equatorial line (as excepted) for Z500 but not for
    T850 which results completely counterintuitive. 
    - Does not reduce training time
   
* Try standardizing data based on monthly - weekly averages. 
    - Standardized data based on month mean and std from training set (1979 to 2012).
    Data is saved in a file called `data/healpix/5.625_nearest/monthly_standardized_data.nc`
    - Using previously standardized data doesn't work. 
    - Tried standardizing during loading
    - Values may be too small since the output of the network is direcly nan (although there are no nans in the input)
    - Tried different initializations of the weights but didn't work either
    - Removing the batch normalization allowed to obtain non-nan results at some point but for specific inputs;
    didn't solve the problem in general
    - Hypothesis: the problem may come from the different values in the same batch since they are standardized using 
    different mean and std values 
    - Still needs to be fixed
 
 #### Other architectures
 
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
    
 * Architecture with x2 channels per block. 
    - Comment on results
    
 * Architecture with one convBlock less per reduction step but with x2 channels on each block 
    - Model name: spherical_unet_deeper_less_blocks_more_channels_1990_
    - Comment on results
    
  * Architecture with one convBlock more per reduction step in the first two steps. 
  Number of channels also modified.
    - Model name: spherical_unet_deeper_more_blocks_less_channels_1990_
    - Comment on results
    
  * Architecture with one convBlock less per reduction step and with x1 channels on each block 
    - Model name: spherical_unet_deeper_less_blocks_same_channels_1990_
    - Comment on results
    
  * Architecture with one convBlock more per reduction step in the first two steps. 
  Number of channels also modified.
    - Model name: spherical_unet_deeper_more_blocks_more_channels_1990_
    - Trained for 10 epochs with data from 1990 to 2012 with single GPU
    - Time per epoch: 4700-4800s
    - Best results so far:
    
          RMSE | Z500  |  T850
          t0   | 68.24 |  0.695
          t120 | 857   |  3.42
    
 
  * Original architecture. Run model until validation error < 0.02
    - Model name: spherical_unet_original_error_stop
        - Batch idx: 1450; Loss: 0.129Epoch:   2  - loss: 0.127  - val_loss: 0.05565  - time: 2190.418368
        - Batch idx: 1450; Loss: 0.048Epoch:   3  - loss: 0.047  - val_loss: 0.05704  - time: 2188.024854
        - Batch idx: 1450; Loss: 0.043Epoch:   4  - loss: 0.043  - val_loss: 0.05149  - time: 2188.846542
        - Batch idx: 1450; Loss: 0.041Epoch:   5  - loss: 0.041  - val_loss: 0.04653  - time: 2185.252418
        - Batch idx: 1450; Loss: 0.039Epoch:   6  - loss: 0.039  - val_loss: 0.04217  - time: 2186.789021
        - Batch idx: 1450; Loss: 0.038Epoch:   7  - loss: 0.037  - val_loss: 0.04876  - time: 2185.333862
        
        - Batch idx: 1450; Loss: 0.031Epoch:  22  - loss: 0.031  - val_loss: 0.03596  - time: 2200.411880
        - Batch idx: 1450; Loss: 0.031Epoch:  23  - loss: 0.031  - val_loss: 0.03651  - time: 2209.285172
        - Batch idx: 1450; Loss: 0.031Epoch:  24  - loss: 0.031  - val_loss: 0.03668  - time: 2193.242775
        - Batch idx: 1450; Loss: 0.031Epoch:  25  - loss: 0.031  - val_loss: 0.03635  - time: 2187.241703
        - Batch idx: 1450; Loss: 0.030Epoch:  26  - loss: 0.030  - val_loss: 0.03615  - time: 2185.419443
        - Batch idx: 1450; Loss: 0.030Epoch:  27  - loss: 0.030  - val_loss: 0.03619  - time: 2181.425468
        - Batch idx: 1450; Loss: 0.030Epoch:  28  - loss: 0.030  - val_loss: 0.03686  - time: 2180.149090

#### Papers

* Read paper: [STConvS2S](https://arxiv.org/abs/1912.00134):
    - Propose using architecture with 3D convolutions split into two phases: 
    an encoder phase where 2D convs are performed to learn spatial features and
    a decoder phase where 1D convs are performed to learn temporal features
    - In both steps, the size of the input and output are conserved across the convolved
    dimension by using appropriate padding.
    - Convolutions are performed using 3D convolutions where the kernel has been modified
    to act only in the desired dimensions.    
    - Predictions not sequentials? 
    - Similar approaches: https://www.nature.com/articles/s41598-020-65070-5
    - Similar approaches: https://towardsdatascience.com/temporal-convolutional-networks-the-next-revolution-for-time-series-8990af826567
    
* Read paper: [Weather and climate forecasting with neural networks](https://doi.org/10.5194/gmd-12-2797-2019):
     - Similar architecture to used by Iciar but with planar data 


### Week 17/08

Explore architectures with resiudal connections.

### Architectures

   - Short term residual connections
        - 1D convolutions in shortcut: 
        ```all_const_len2_delta_6_architecture_spherical_unet_lbmc_residuals_encoder```
            
                RMSE | Z500  (ORIGINAL)      |  T850 (ORIGINAL)
                t0   | 71.35 (77.87)    -8%  |  0.68 (0.72)      -5%
                t120 | 962   (1049.97)  -8%  |  3.47 (3.81)      -9%
            
        - Identity mapping in shortcut: 
        ```all_const_len2_delta_6_architecture_spherical_unet_lbmc_residuals_encoder_identity```
        
                RMSE | Z500   (ORIGINAL)         |  T850 (ORIGINAL)
                t0   | 68.12  (77.87)    -12.5%  |  0.70 (0.72)      -3%
                t120 | 1059   (1049.97)  +1%     |  3.86 (3.81)      +1%