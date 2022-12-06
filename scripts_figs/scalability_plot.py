import os
import sys
sys.path.append('../')

import torch
import numpy as np
import matplotlib.pyplot as plt 

from xsphere.remapping import SphericalVoronoiMeshArea_from_pygsp

## DeepSphere-Weather
import modules.my_models_graph as my_architectures
from modules.utils_torch import get_time_function
from modules.utils_config import read_config_file
from modules.utils_config import get_model_settings
from modules.utils_config import get_training_settings
from modules.utils_config import get_ar_settings
from modules.utils_config import get_dataloader_settings
from modules.utils_config import set_pytorch_settings
from modules.utils_models import get_pygsp_graph


# Plotting options
import matplotlib
matplotlib.use('cairo') # Cairo
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["savefig.facecolor"] = "white" # (1,1,1,0)
matplotlib.rcParams["savefig.edgecolor"] = 'none'

# CUDA options
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

##-----------------------------------------------------------------------------.
# Define directories
base_dir = "/data/weather_prediction"
figs_dir = os.path.join(base_dir, "figs")
data_dir = os.path.join(base_dir, "data")  

##-----------------------------------------------------------------------------.
### Model config 
cfg_path = '/home/ghiggi/Projects/weather_prediction/configs/UNetSpherical/O24/MaxAreaPool-k20.json'

### Read experiment configuration settings 
cfg = read_config_file(fpath=cfg_path)

# Some special stuff you might want to adjust 
cfg['model_settings']["architecture_name"] = 'UNetSpherical'
cfg['ar_settings']["input_k"] = [-18,-12,-6]
cfg['ar_settings']["output_k"] = [0]
cfg['ar_settings']["ar_iterations"] = 0
 
##-----------------------------------------------------------------------------.
### Retrieve experiment-specific configuration settings   
model_settings = get_model_settings(cfg)   
ar_settings = get_ar_settings(cfg)
training_settings = get_training_settings(cfg) 
dataloader_settings = get_dataloader_settings(cfg) 

device = set_pytorch_settings(training_settings)

## Retrieve time function 
get_time = get_time_function(device)

##-------------------------------------------------------------------------.
resolutions_RGG = [48, 94, 120, 160, 240, 320, 640] 

# Custom options 
batch_size = 1
n_static_variables = 3
n_bc_variables = 1 
n_dynamic_variables = 2
knn = 20 
sampling = 'gauss'
ar_iterations = ar_settings['ar_iterations']
n_repetitions = 10

knn_list = [8, 20, 40, 60]
resolutions = [48, 120, 160, 240, 320, 440, 520, 640]      
              # 024, 048, 060, 080, 0160  # 4 𝑁( 𝑁 + 9)

##-----------------------------------------------------------------------------.
## Initialize dictionaries 
n_nodes_dict = {}  
forward_timing_dict = {}  
forward_memory_dict = {}  
backward_timing_dict = {}  
total_timing_dict = {}  
km_resolution_dict = {}
for k in knn_list:
    n_nodes_dict[k] = {res: None for res in resolutions}  
    km_resolution_dict[k] = {res: None for res in resolutions}  
    forward_timing_dict[k] = {res: None for res in resolutions}
    forward_memory_dict[k] = {res: None for res in resolutions}
    backward_timing_dict[k] = {res: None for res in resolutions}
    total_timing_dict[k] = {res: None for res in resolutions}

##-----------------------------------------------------------------------------.
### Simulate data 
for resolution in resolutions:
    # Define graph (just to get dimensions) 
    graph = get_pygsp_graph(sampling=sampling,
                            resolution=resolution, 
                            knn=knn)   
    mesh_areas = SphericalVoronoiMeshArea_from_pygsp(graph)
    approx_km_resolution = int(np.sqrt(np.median(mesh_areas)))
    # - Define tensor dimensions 
    input_node_dim = graph.n_vertices
    output_node_dim = graph.n_vertices

    input_feature_dim = n_static_variables + n_bc_variables + n_dynamic_variables
    output_feature_dim = n_dynamic_variables
    input_time_dim = len(ar_settings['input_k']) 
    output_time_dim = len(ar_settings['output_k']) 
    # dim_order = ['sample','node','time','feature']
    dim_order = ['sample','time','node','feature']

    dim_input = {}
    dim_input['node'] = input_node_dim
    dim_input['time'] = input_time_dim
    dim_input['feature'] = input_feature_dim
    input_shape = list([dim_input[k] for k in dim_order[1:]])

    dim_output = {}
    dim_output['node'] = output_node_dim
    dim_output['time'] = output_time_dim
    dim_output['feature'] = output_feature_dim
    output_shape = list([dim_output[k] for k in dim_order[1:]])
    # - Generate data 
    X = torch.randn([batch_size] + input_shape, dtype=torch.float32, device=device)
    Y_obs = torch.randn([batch_size] + output_shape, dtype=torch.float32, device=device)   
    ##-------------------------------------------------------------------------.
    # Test impact of various knn
    for knn in knn_list:
        ##---------------------------------------------------------------------.
        ### Define the model architecture   
        # - Define model infos 
        dim_info = {'input_feature_dim': input_feature_dim,
                    'output_feature_dim': output_feature_dim,
                    'input_time_dim': input_time_dim,
                    'output_time_dim': output_time_dim,
                    'input_node_dim': input_node_dim,
                    'output_node_dim': output_node_dim,
                    'dim_order': dim_order,
                    'input_shape': input_shape,
                    'output_shape': output_shape,
        }
        model_settings['dim_info'] = dim_info 
        model_settings['resolution'] = resolution 
        model_settings['knn'] = knn
        model_settings['sampling'] = sampling
        # - Retrieve required model arguments
        model_keys = ['dim_info', 'sampling', 'resolution',
                    'knn', 'kernel_size_conv',
                    'pool_method', 'kernel_size_pooling']
        model_args = {k: model_settings[k] for k in model_keys}
        model_args['numeric_precision'] = training_settings['numeric_precision']
        # - Define DeepSphere model 
        DeepSphereModelClass = getattr(my_architectures, model_settings['architecture_name'])
        model = DeepSphereModelClass(**model_args)        
        model.to(device)
        model.train()
        ##---------------------------------------------------------------------.
        # - Define optimizer and loss 
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.008, eps=1e-7, weight_decay=0, amsgrad=False)
        ##---------------------------------------------------------------------.
        l_forward_time = []  
        l_backward_time = [] 
        l_total_time = []
        l_forward_memory = []
        weight = 1/(ar_iterations+1)
        for _ in range(n_repetitions):
            # - Compute Forward
            t_i = get_time()
            dict_loss = {} 
            for i in range(ar_iterations+1):
                Y_pred = model(X)
                dict_loss[i] = criterion(Y_pred, Y_obs)
            if device.type != 'cpu':
                l_forward_memory.append(torch.cuda.memory_allocated()/1024/1024)
            # - Compute total weighted loss 
            for i, (_ ,loss) in enumerate(dict_loss.items()):
                if i == 0:
                    training_total_loss = weight * loss 
                else: 
                    training_total_loss += weight * loss

            l_forward_time.append(get_time() - t_i)
            #------------------------------------------------------------------.
            # Compute backward
            t_b = get_time()
            optimizer.zero_grad()
            training_total_loss.backward()
            l_backward_time.append(get_time() - t_b)
            l_total_time.append(get_time() - t_i)

        #----------------------------------------------------------------------.   
        # Add information to dictionaries 
        forward_timing_dict[knn][resolution] = l_forward_time
        backward_timing_dict[knn][resolution] = l_backward_time
        total_timing_dict[knn][resolution] = l_total_time
        forward_memory_dict[knn][resolution] = l_forward_memory
        n_nodes_dict[knn][resolution] = input_node_dim    
        km_resolution_dict[knn][resolution] = approx_km_resolution

##-----------------------------------------------------------------------------.
### Create plot
# - Create figure 
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Model scalability on ECMWF Octahedral Reduced Gaussian Grids")

# - Plot time vs nodes
for knn in knn_list:
    n_nodes = list(n_nodes_dict[knn].values())
    km_resolutions = list(km_resolution_dict[knn].values())
    elapsed_times = [np.median(x) for x in total_timing_dict[knn].values()]
    RGG_grids_labels = ['O' + str(int(x/2)) for x in list(forward_timing_dict[knn].keys())]
    ax.plot(n_nodes, elapsed_times, 'o-', markersize=3)
ax.set_xlabel("Number of mesh cells")
ax.set_ylabel("Time [s]")
ax.set_ylim((0,None))
ax.set_title("Computing time of Forward + Backward Pass")
ax.margins(x=0)
for i in range(2, len(elapsed_times)):
    ax.text(n_nodes[i], elapsed_times[i], s=RGG_grids_labels[i],
            verticalalignment='bottom', horizontalalignment='right')
# - Add legend 
legend_labels = ["k = " + str(knn) for knn in knn_list]
ax.legend(legend_labels, loc='upper left',  
          title = "Graph neighbors",
          frameon = True, fancybox=True, framealpha=1, shadow=False)
# - Add km resolution 
grid_resolution_labels = [str(km_res) + " km" for km_res in km_resolutions] 
for i in range(3, len(grid_resolution_labels)):
    if i != len(grid_resolution_labels)-1:
        ax.text(n_nodes[i], 0, s=grid_resolution_labels[i], 
                fontsize = 8,
                verticalalignment='bottom', horizontalalignment='center')
    else: 
        ax.text(n_nodes[i], 0, s=grid_resolution_labels[i], 
                fontsize = 8,
                verticalalignment='bottom', horizontalalignment='right')
    ax.axvline(x=n_nodes[i], linestyle="solid", color="gray", alpha=0.1) 

# - Plot memory vs nodes
for knn in knn_list:
    n_nodes = list(n_nodes_dict[knn].values())
    km_resolutions = list(km_resolution_dict[knn].values())
    RGG_grids_labels = ['O' + str(int(x/2)) for x in list(forward_timing_dict[knn].keys())]
    required_memory = [np.median(x)/1024 for x in forward_memory_dict[knn].values()]
    ax1.plot(n_nodes, required_memory, 'o-', markersize=2)

ax1.set_xlabel("Number of mesh cells")
ax1.set_ylabel("GPU Memory [GB]")
ax1.set_title("GPU Memory requirement")
ax1.set_ylim((0,None))
ax1.margins(x=0)
for i in range(2, len(required_memory)):
    ax1.text(n_nodes[i], required_memory[i], s=RGG_grids_labels[i],
            verticalalignment='bottom', horizontalalignment='right')
# - Add km resolution 
grid_resolution_labels = [str(km_res) + " km" for km_res in km_resolutions] 
for i in range(3, len(grid_resolution_labels)):
    if i != len(grid_resolution_labels)-1:
        ax1.text(n_nodes[i], 0, s=grid_resolution_labels[i], 
                fontsize = 8,
                verticalalignment='bottom', horizontalalignment='center')
    else: 
        ax1.text(n_nodes[i], 0, s=grid_resolution_labels[i], 
                fontsize = 8,
                verticalalignment='bottom', horizontalalignment='right')
    ax1.axvline(x=n_nodes[i], linestyle="solid", color="gray", alpha=0.1) 
fig.savefig(os.path.join(figs_dir, "Scalability_Nodes.png"))

##-----------------------------------------------------------------------------.
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Model scalability on ECMWF Octahedral Reduced Gaussian Grids")

# - Plot time vs nodes
for knn in knn_list:
    n_nodes = list(n_nodes_dict[knn].values())
    km_resolutions = list(km_resolution_dict[knn].values())
    elapsed_times = [np.median(x) for x in total_timing_dict[knn].values()]
    RGG_grids_labels = ['O' + str(int(x/2)) for x in list(forward_timing_dict[knn].keys())]
    ax.plot(km_resolutions, elapsed_times, 'o-', markersize=3)
ax.set_xlabel("Mesh resolution [km]")
ax.set_ylabel("Time [s]")
ax.set_ylim((0,None))
ax.set_title("Computing time of Forward + Backward Pass")
ax.margins(x=0)
for i in range(0, len(elapsed_times)):
    if i != 0:
        ax.text(km_resolutions[i], elapsed_times[i], s=RGG_grids_labels[i],
                verticalalignment='bottom', horizontalalignment='left')
    else: 
        ax.text(km_resolutions[i], elapsed_times[i], s=RGG_grids_labels[i],
                verticalalignment='bottom', horizontalalignment='right')    
# - Add legend 
legend_labels = ["k = " + str(knn) for knn in knn_list]
ax.legend(legend_labels, loc='upper right',  
          title = "Graph neighbors",
          frameon = True, fancybox=True, framealpha=1, shadow=False)

# - Plot memory vs nodes
for knn in knn_list:
    n_nodes = list(n_nodes_dict[knn].values())
    km_resolutions = list(km_resolution_dict[knn].values())
    RGG_grids_labels = ['O' + str(int(x/2)) for x in list(forward_timing_dict[knn].keys())]
    required_memory = [np.median(x)/1024 for x in forward_memory_dict[knn].values()]
    ax1.plot(km_resolutions, required_memory, 'o-', markersize=2)

ax1.set_xlabel("Mesh resolution [km]")
ax1.set_ylabel("GPU Memory [GB]")
ax1.set_title("GPU Memory requirement")
ax1.set_ylim((0,None))
ax1.margins(x=0)
for i in range(0, len(elapsed_times)):
    if i != 0:
        ax1.text(km_resolutions[i], required_memory[i], s=RGG_grids_labels[i],
                verticalalignment='bottom', horizontalalignment='left')
    else: 
        ax1.text(km_resolutions[i], required_memory[i], s=RGG_grids_labels[i],
                verticalalignment='bottom', horizontalalignment='right')  
fig.savefig(os.path.join(figs_dir, "Scalability_Resolution.png"))



# 48GB GPU : NV Quadro RTX8000 (8000 CHF)