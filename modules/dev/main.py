#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:43:17 2021

@author: ghiggi
"""
import os
import warnings
import random
import json
import torch
from torch import nn, optim
import numpy as np

import modules.architectures as modelArchitectures
from modules.training import create_DataLoaders, train
from modules.loss import plot_weight_variations 

warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------------.
### TODO
# Check if weights_loss is correct (and init not repeated) 
# plot_weight_variations is necessary? Tend to remove it 

### Recommended changes 
# weights_loss    --> loss_temporal_weights
# test_loss_steps --> validation_loss_steps
# initial_weights --> loss_temporal_weights_init ? (Not look at all details right now)
# nodes           --> n_nodes (Is well the number of nodes of the graph? Why a cfg? Infer from data no?)
# load_model --> load_pretrained_model

# cfg['training_constants']['nb_timesteps'] --> Remove from your cfg is outdated!
# load_in_memory --> Should become a cfg params 

# pytorch_deterministic --> Should become a cfg for choosing if _deterministic() to be executed

# The follow will be modified (and are currently synonyms)
#  len_sqce_input <-- len_sqce
#  len_sqce_output <-- num_steps_ahead

##-----------------------------------------------------------------------------.
### Questions
# - out_features:  variable names to be predicted? 
# - nodes:   this is defined by resolution,...? And can be derived/check against data 
# - Description of the follow? :

# datadir = cfg['directories']['datadir']
# savedir = cfg['directories']["save_dir"]
# input_dir = datadir + cfg['directories']['input_dir']
# --> input_dir is provided to data_dir argument BTW ...
    

##-----------------------------------------------------------------------------.
### Improvements / Flexibility
# - Function that create folder name based on cfg params
# - Loss function (aka criterion) should be generalized to cfg setting (maybe also separate loss.py for further options included temporal weights)
# - Optimizers should be generalized by cfg setting 
# - ... Currently training is set to 1 epoch (and loop is performed outside) to save models and reinitialize weights 

# - Thinking about the case where input data not same resolution of output data (i.e. downscaling)
# --> Now we assume same nodes in input and output. We maybe should make it more flexible 
#     to have input and output different number of nodes    

#-----------------------------------------------------------------------------.
#######################
# Pytorch Settings ####
#######################
def _deterministic(seed=100):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

#-----------------------------------------------------------------------------.
##############  
#### Main ####
############## 
def main(config_file, load_model=False):
    """
    General function for training DeepSphere4Earth models.
    
    Parameters
    ----------
    config_file : TYPE
        DESCRIPTION.
    load_model : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    training_info : dict 
        Dictionary with training information.
        Keys:'training_loss', 'validation_loss, 'training_loss_steps', 
        'test_loss_steps','weight_variations','weights_loss','criterion' and 'optimizer' 

    """
    ##------------------------------------------------------------------------.
    # _deterministic() 
    ##------------------------------------------------------------------------.
    #### Read configuration settings 
    with open(config_file) as json_data_file:
        cfg = json.load(json_data_file)
        
    ##------------------------------------------------------------------------.
    ### - Define paths
    datadir = cfg['directories']['datadir']
    savedir = cfg['directories']["save_dir"]
    input_dir = datadir + cfg['directories']['input_dir']
    model_save_path = savedir + cfg['directories']['model_save_path']
    pred_save_path = savedir + cfg['directories']['pred_save_path']

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(pred_save_path, exist_ok=True)

    ##------------------------------------------------------------------------.
    chunk_size = cfg['training_constants']['chunk_size']

    ##------------------------------------------------------------------------.
    ### - Defining time split for training 
    train_years = (cfg['training_constants']['train_years'][0], cfg['training_constants']['train_years'][1])
    val_years = (cfg['training_constants']['val_years'][0], cfg['training_constants']['val_years'][1])

    ##------------------------------------------------------------------------.
    ### - Define training options 
    epochs = cfg['training_constants']['nb_epochs']
    learning_rate = cfg['training_constants']['learning_rate']
    batch_size = cfg['training_constants']['batch_size']

    ##------------------------------------------------------------------------.
    ### - Define model architecture (time)
    max_lead_time = cfg['training_constants']['max_lead_time']
    len_sqce = cfg['model_parameters']['len_sqce']
    delta_t = cfg['model_parameters']['delta_t']
    num_steps_ahead = cfg['model_parameters']['num_steps_ahead']
    
    ##------------------------------------------------------------------------.
    ### - Define model architecture (space + features)
    nodes = cfg['training_constants']['nodes']         
    model = cfg['model_parameters']['model']
    resolution = cfg['model_parameters']["resolution"]   
    in_features = cfg['model_parameters']['in_features']
    out_features = cfg['model_parameters']['out_features']
 
    ##------------------------------------------------------------------------.
    ### - Define model architecture (ensemble)
    # @ Yasser 

    ##------------------------------------------------------------------------.
    ### - Define architecture structure
    net_params = {}
    net_params["sampling"] = cfg['model_parameters'].get("sampling", None)
    net_params["knn"] = cfg['model_parameters'].get("knn", 10)
    net_params["conv_type"] = cfg['model_parameters'].get("conv_type", None)
    net_params["pool_method"] = cfg['model_parameters'].get("pool_method", None)
    net_params["ratio"] = cfg['model_parameters'].get("ratio", None)
    net_params["periodic"] = cfg['model_parameters'].get("periodic", None)
    net_params["comments"] = cfg['model_parameters'].get("comments", None)
    
    ##------------------------------------------------------------------------.
    ### - Create filename describing the model (folder structure in general)
    description = [str(i) for i in net_params.values() if i is not None]
    description = '_'.join(description)
    print(description)
    net_params.pop('comments')

    assert description in savedir

    model_filename = model_save_path + description + ".h5"
    figures_path = savedir + 'figures/'

    os.makedirs(figures_path, exist_ok=True)
    
    ##------------------------------------------------------------------------.
    ### Generate (lazy) DataLoaders for training and validation Dataset 
    # TODO: Provide function to load data     
    # TODO: Update changes of training.py 
    trainingDataLoader, validationDataLoader  = create_DataLoaders( data_dir = input_dir,    # TODO To change in cfg?
                                                                    training_years = train_years,
                                                                    validation_years = val_years,
                                                                    chunk_size = chunk_size, # TODO: This will not be necessary
                                                                    load_in_memory = True,   # TODO: Shoud become a cgf setting
                                                                    # Scalers ### TODO: to be provided when x-scaler.py ready
                                                                    scaler_dynamic = None,  
                                                                    scaler_bc = None,
                                                                    scaler_static = None,
                                                                    # Necessary?
                                                                    out_features=out_features,  
                                                                    nodes=nodes,  
                                                                    # Autoregressive options
                                                                    delta_t=delta_t,
                                                                    len_sqce_input=len_sqce, 
                                                                    len_sqce_output=num_steps_ahead, 
                                                                    max_lead_time=max_lead_time)
                                                                                             
    ##------------------------------------------------------------------------.
    #### Create model architecture 
    print('Define model...')
    print('Model name: ', description)
    modelClass = getattr(modelArchitectures, model)
    spherical_unet = modelClass(resolution, in_channels=in_features * len_sqce, out_channels=out_features, kernel_size=3, **net_params)

    ###-----------------------------------------------------------------------.
    ### Optional loading of a pre-trained model for fine-tuning 
    if load_model:
        state = torch.load(model_filename)
        spherical_unet.load_state_dict(state, strict=False)
        
    ###-----------------------------------------------------------------------.
    ### GPU settings 
    # TODO: Not sure I unterstand this. Someone can add a line of explanation? 
    # - By default is on cpu without cuda? 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        spherical_unet = spherical_unet.to(device)
    else:
        device = 'cpu'
        
    ###-----------------------------------------------------------------------.
    ### - Initialize loss temporal weights
    # TODO: temporal weights  
    # - Is not a unuseful repetition? Already defined in the below loop 
    # - initial_weights --> another name (loss_weights_feature, loss_weight_temporal)
    # - Check how it behave when not iterating ! 
    # - Weight updating scheme?
    # - The follow plot is it necessary? I would removed it personally (or place somewhere else)
    weights_loss = cfg['model_parameters']['initial_weights']
    weights_loss = np.array(weights_loss)
    weights_loss = weights_loss / sum(weights_loss)
     
    plot_weight_variations(weights_loss=weights_loss, figures_path=figures_path) 
    
    ###-----------------------------------------------------------------------.
    ### - Initialize training info 
    training_info = {'training_loss': [], 
                     'validation_loss': [],
                     'training_loss_steps': [], 
                     'test_loss_steps': [],   
                     'weight_variations': [],
                     'weights_loss': [],        
                     'criterion': [],     
                     'optimizer': []
                     }

    ###-----------------------------------------------------------------------.
    ### - Configure loss function 
    criterion = nn.MSELoss()
    
    ###-----------------------------------------------------------------------.
    ### - Configure optimizer 
    optimizer = optim.Adam(spherical_unet.parameters(),    
                           lr=learning_rate, eps=1e-7,    
                           weight_decay=0, amsgrad=False)
    ###-----------------------------------------------------------------------.
    ### - Model training
    for ep in range(epochs):
        ##--------------------------------------------------------------------.
        # - Print info
        print('Starting epoch {}'.format(ep + 1))
        
        ##--------------------------------------------------------------------.
        # - Train model 
        spherical_unet.train() # TODO: what it does? training is below ? better name? 
        
        ##-------------------------------------------------------------------.
        # - Update temporal weights for the loss function 
        # --> TODO: I would move this after train_model_multiple_steps() 
        # --> Why use cfg ? Error ? 
        # - Does not allow to update the loss temporal weights (unless full main() is relaunched ...)
        # - But in training_info some mysterious weights variations maybe do the job 
        # --> w should changed inside train_model no?   
        weights_loss = cfg['model_parameters']['initial_weights'] # To remove ????
        weights_loss = np.array(weights_loss)
        weights_loss = weights_loss / sum(weights_loss)
        
        ##--------------------------------------------------------------------.
        # Train the model 
        training_info = train(model=spherical_unet, 
                              training_info=training_info, 
                              trainingDataLoader=trainingDataLoader, 
                              validationDataLoader=validationDataLoader,
                              # Loss options
                              weights_loss=weights_loss, criterion=criterion,  
                              # Optimizer options
                              optimizer=optimizer,                 
                              # Training options 
                              batch_size=batch_size, epochs=1,   
                              # Device options
                              device=device,
                              # This will change      
                              len_sqce_output=num_steps_ahead)
        
        ##--------------------------------------------------------------------.
        # - Save the model each epoch
        torch.save(spherical_unet.state_dict(), model_filename[:-3] + '_epoch_{}'.format(ep) + '.h5')
    #-------------------------------------------------------------------------.
    # Return training info 
    return training_info
