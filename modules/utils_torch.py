#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:53:55 2021

@author: ghiggi
"""
import os
import random
import time
import copy 
import torch 
import torch.autograd.profiler as profiler
import numpy as np 
from collections import OrderedDict
from tabulate import tabulate
import pandas as pd 
#-----------------------------------------------------------------------------.
# ####################
#### Timing utils ####
# ####################
def get_synchronized_cuda_time():
    """Get time after CUDA synchronization.""" 
    torch.cuda.synchronize()
    return time.time()

def get_time_function(device):
    """General function returing a time() function.""" 
    if isinstance(device, str): 
        device = torch.device(device)
    if device.type == "cpu":
        return time.time
    else:
        return get_synchronized_cuda_time
#-----------------------------------------------------------------------------.
# ###############
#### Checks  ####
# ###############

def check_device(device):
    """Check torch device validity."""
    if isinstance(device, str): 
        device = torch.device(device)
    if not isinstance(device, torch.device): 
        raise TypeError("Specify device as torch.device or as string : 'cpu','cuda',...")
    if device.type != 'cpu':
        if not torch.cuda.is_available():
            raise ValueError("Not possible to set up a cuda device. CUDA is not available!")
    return device

def check_pin_memory(pin_memory, num_workers, device):
    """Check pin_memory possibility."""
    if not isinstance(pin_memory, bool):
        raise TypeError("'pin_memory' must be either True or False. If num_workers > 0, set to False ;) ")
    # CPU case
    if device.type == 'cpu':
        if pin_memory:
            print("- GPU is not available. 'pin_memory' set to False.")
            pin_memory = False
    # GPU case with multiprocess
    if num_workers > 0 and pin_memory: 
        print("- Pinned memory can't be shared across processes! \n It is not possible to pin tensors into memory in each worker process. \n If num_workers > 0, pin_memory is set to False.")
        pin_memory = False    
    return pin_memory

def check_prefetch_in_gpu(prefetch_in_gpu, num_workers, device):
    """Check prefetch_in_gpu possibility."""
    if not isinstance(prefetch_in_gpu, bool):
        raise TypeError("'prefetch_in_gpu' must be either True or False. If num_workers > 0, set to False ;) ")
    # CPU case
    if device.type == 'cpu':
        if prefetch_in_gpu:
            print("- GPU is not available. 'prefetch_in_gpu' set to False.")
            prefetch_in_gpu = False
    # GPU case with multiprocess
    elif num_workers > 0 and prefetch_in_gpu: 
        print("- Prefetch in GPU with multiprocessing is currently unstable.\n\
            It is generally not recommended to return CUDA tensors within multi-process data loading\
                loading because of many subtleties in using CUDA and sharing CUDA tensors.")
        prefetch_in_gpu = False    
    else: # num_workers = 0 
        prefetch_in_gpu = prefetch_in_gpu
    return prefetch_in_gpu

def check_asyncronous_gpu_transfer(asyncronous_gpu_transfer, device):
    """Check asyncronous_gpu_transfer possibility."""
    if not isinstance(asyncronous_gpu_transfer, bool):
        raise TypeError("'asyncronous_gpu_transfer' must be either True or False.")
    # CPU case
    if device.type == 'cpu':
        if asyncronous_gpu_transfer:
            print("- GPU is not available. 'asyncronous_gpu_transfer' set to False.")
            asyncronous_gpu_transfer = False
    return asyncronous_gpu_transfer

def check_prefetch_factor(prefetch_factor, num_workers):     
    """Check prefetch_factor validity."""
    if not isinstance(prefetch_factor, int):
        raise TypeError("'prefetch_factor' must be positive integer.")
    if prefetch_factor < 0: 
        raise ValueError("'prefetch_factor' must be positive.")
    if num_workers == 0 and prefetch_factor !=2:
        prefetch_factor = 2 # bug in pytorch ... need to set to 2 
    return prefetch_factor

def check_ar_training_strategy(ar_training_strategy):
    """Check AR training strategy validity."""
    if not isinstance(ar_training_strategy, str):
        raise TypeError("'ar_training_strategy' must be a string: 'RNN' or 'AR'.")
    if ar_training_strategy not in ["RNN","AR"]:
        raise ValueError("'ar_training_strategy' must be either 'RNN' or 'AR'.")
    return ar_training_strategy

#-----------------------------------------------------------------------------.
# #######################
#### Pytorch settings ###
# #######################
def get_torch_dtype(numeric_precision): 
    """Provide torch dtype based on numeric precision string."""
    dtypes = {'float64': torch.float64,
              'float32': torch.float32,
              'float16': torch.float16,
              'bfloat16': torch.bfloat16
              }
    return dtypes[numeric_precision]

def get_torch_tensor_type(numeric_precision, device):
    """Provide torch tensor type based on numeric precision string."""
    device = check_device(device)
    if device.type == 'cpu':
        tensor_types = {'float64': torch.DoubleTensor,
                        'float32': torch.FloatTensor,
                        'float16': torch.HalfTensor,
                        'bfloat16': torch.BFloat16Tensor
                        }
    else:
        tensor_types = {'float64': torch.cuda.DoubleTensor,
                        'float32': torch.cuda.FloatTensor,
                        'float16': torch.cuda.HalfTensor,
                        'bfloat16': torch.cuda.BFloat16Tensor
                        }
    return tensor_types[numeric_precision]   

def torch_tensor_type_2_dtype(tensor_type):
    """Conversion from torch.tensortype to torch dtype."""
    if not isinstance(tensor_type, type(torch.FloatTensor)):
        raise TypeError("Expect a torch.tensortype.")
    return tensor_type.dtype

def torch_dtype_2_tensor_type(dtype, device):
    """Conversion from torch.dtype to torch tensor type."""
    device = check_device(device)
    if not isinstance(dtype, type(torch.float32)):
        raise TypeError("Expect a torch.dtype.")
    # Get the string 
    dtype_str = str(dtype)
    # Define dictionary 
    if device.type == 'cpu':
        tensor_types = {'torch.float64': torch.DoubleTensor,
                        'torch.float32': torch.FloatTensor,
                        'torch.float16': torch.HalfTensor,
                        'torch.bfloat16': torch.BFloat16Tensor
                        }
    else:
        tensor_types = {'torch.float64': torch.cuda.DoubleTensor,
                        'torch.float32': torch.cuda.FloatTensor,
                        'torch.float16': torch.cuda.HalfTensor,
                        'torch.bfloat16': torch.cuda.BFloat16Tensor
                        }
    return tensor_types[dtype_str] 

def set_seeds(seed):
    #os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    ## CUDA >10.2 possible addional configs
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # CUBLAS_WORKSPACE_CONFIG =:4096:8 
    # CUBLAS_WORKSPACE_CONFIG =:16:8
    ## To retrieve torch seed 
    # torch.initial_seed()
    return None      

def set_pytorch_deterministic(seed=100):
    """Set seeds for deterministic training with pytorch."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    set_seeds(seed)
    return None
    
 
def set_pytorch_numeric_precision(numeric_precision, device):
    """Set pytorch numeric precision."""
    tensor_type =  get_torch_tensor_type(numeric_precision, device)
    dtype = get_torch_dtype(numeric_precision)
    # torch.set_default_tensor_type(tensor_type) --> This cause bug since pytorch >1.8 with dataloader data shuffling with torch.randperm  
    torch.set_default_dtype(dtype)
    


#----------------------------------------------------------------------------.
def check_models_have_same_weights(model_1, model_2):
    """Check if two models share the same weights."""
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])  
    if models_differ == 0:
        return True
    else:
        print('There are {} differences.'.format(models_differ))
        return False
    
#----------------------------------------------------------------------------.
#############################
#### Summary / Profiling ####
#############################
# TODOs
# - check dtype validity more robust 
# - check input size tuple values are positive integers

### Define hook for shape, parameter, memory and timing information
def _generate_forward_hook(handle, summary_forward, m_key):
    def forward_hook(module, input, output):
        # register_forward_hook
        # - The hook will be called every time after :func:`forward` has computed an output
        # - hook(module, input, output)
        #-----------------------------------------------------------------.
        # Get device 
        device = input[0].device
        
        #-----------------------------------------------------------------.
        # Initialize dicionary 
        summary_forward[m_key] = OrderedDict()
        
        ##----------------------------------------------------------------.
        # Time execution
        get_time = get_time_function(device)
        summary_forward[m_key]['time'] = get_time()
        
        ##----------------------------------------------------------------.
        # Measure memory allocation           
        if device.type != 'cpu':
            summary_forward[m_key]["memory_allocated"] = torch.cuda.memory_allocated()/1000/1000
            summary_forward[m_key]["memory_cached"] = torch.cuda.memory_cached()/1000/1000
           
        ##----------------------------------------------------------------.
        # Determine shape 
        # summary_forward[m_key]["i"] = input
        # summary_forward[m_key]["o"] = output
        batch_size = len(input[0])
        # - Retrieve input shape 
        if len(input) > 1:
            summary_forward[m_key]["input_shape"] = [list(i.size()) if i is not None else None for i in input]
            #summary_forward[m_key]["input_shape"][0] = batch_size
        else:
            summary_forward[m_key]["input_shape"] = list(input[0].size())
            # summary_forward[m_key]["input_shape"][0] = batch_size

        # - Retrieve output shape 
        if isinstance(output, (list, tuple)):
            summary_forward[m_key]["output_shape"] = [list(o.size()) if o is not None else None for o in output]
        else:
            summary_forward[m_key]["output_shape"] = list(output.size())
        ##----------------------------------------------------------------.
        # Determine params 
        params = 0
        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            params += torch.prod(torch.LongTensor(list(module.weight.size())))
            summary_forward[m_key]["trainable"] = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            params += torch.prod(torch.LongTensor(list(module.bias.size())))
        ##----------------------------------------------------------------.
        #  Ensure params will be a int
        if not isinstance(params, int):
            params = params.item()
        summary_forward[m_key]["nb_params"] = params 
        
    return forward_hook

def _generate_forward_pre_hook(handle, summary_pre_forward, m_key):
    def forward_pre_hook(module, input):
        # register_forward_pre_hook
        # - hook is called every time before :func:`forward` is invoked.
        # - hook(module, input)
        #-----------------------------------------------------------------.
        # Get device 
        device = input[0].device
        
        #-----------------------------------------------------------------.
        # Initialize dicionary 
        summary_pre_forward[m_key] = OrderedDict()
        
        ##----------------------------------------------------------------.
        # Time execution
        get_time = get_time_function(device)
        summary_pre_forward[m_key]['time'] = get_time()
        
        ##----------------------------------------------------------------.
        # Measure memory allocation
        if device.type != 'cpu':
            summary_pre_forward[m_key]["memory_allocated"] = torch.cuda.memory_allocated()/1000/1000
            summary_pre_forward[m_key]["memory_cached"] = torch.cuda.memory_cached()/1000/1000
            
    return forward_pre_hook
    
def profile_layers(model, input_size, batch_size=32, dtypes=None, device=torch.device('cpu')):
    """Profile model execution time and memory consumption of each trainable layer.
    
    Originally inspired from https://github.com/sksq96/pytorch-summary and
    https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks post.
    
    Parameters
    ----------
    model : 
       Pytorch model.
    input_size : tuple or list
        A tuple or list of tuples describing input tensors shapes.
    batch_size : int, optional
        The batch size. The default is 32.
    dtypes : list, optional
        A list of dtype of the input tensors. 
        If set to None (and by default), it uses the default torch type. 
    device : str or torch.device, optional
        The torch device to use. The default is torch.device('cpu').
    
    Returns
    -------
    table : str
        A tabulate string summarizing profiling results.
    summary_str : str
        String summarizing model information.
    summary : dict
        Dictionary with profiling information of each model layer.

    """
    ##------------------------------------------------------------------------.
    # Check device 
    device = check_device(device)
    ##------------------------------------------------------------------------.
    # Check input size 
    # - List of tuples (to deal with multiple input tensors)
    if not isinstance(input_size, (tuple, list)):
        raise TypeError("'input_size' must be a tuple or a list of tuples.")
    if isinstance(input_size, tuple):
        input_size = [input_size]
    if isinstance(input_size, list): 
        idx_not_tuple = [not isinstance(t, tuple) for t in input_size]
        if any(idx_not_tuple):
            raise ValueError("input_size must be a list of tuples.")   
    ##------------------------------------------------------------------------.    
    # Check dtypes
    default_dtype = torch.get_default_dtype()
    if dtypes is None: 
        dtypes = default_dtype
    if not isinstance(dtypes, list):
        dtypes = [dtypes]    
    dtypes = [dtype if dtype is not None else default_dtype for dtype in dtypes]
    ##------------------------------------------------------------------------.
    # Check batch_size 
    if batch_size < 2: 
        raise ValueError("'batch_size' must be at least 2 (for batch norm ...).")
    ##------------------------------------------------------------------------.
    # Retrieve timing function (for cpu and CUDA)
    get_time = get_time_function(device)
    ##------------------------------------------------------------------------.
    # Record how many memory is already taken in GPU (model params + ....)
    if device.type != 'cpu':
        torch.cuda.empty_cache()
        already_allocated = torch.cuda.memory_allocated()/1000/1000
    ##------------------------------------------------------------------------.
    # Create the list of input tensors   
    x = [torch.rand(batch_size, *in_size, dtype=dtype, device=device)
            for in_size, dtype in zip(input_size, dtypes)]
    
    ##------------------------------------------------------------------------.
    ### Profile time (and memory if CUDA) of the entire forward pass 
    if device.type != 'cpu':
        batch_memory = torch.cuda.memory_allocated()/1000/1000 - already_allocated
        t_i = get_time()
        output = model(*x)
        forward_time = get_time() - t_i
        forward_memory = torch.cuda.memory_allocated()/1000/1000 - already_allocated - batch_memory
    else:
        t_i = get_time()
        output = model(*x)
        forward_time = get_time() - t_i
    ##------------------------------------------------------------------------.    
    # Free the memory
    del output 
    if device.type != 'cpu':
        torch.cuda.empty_cache()
    ##------------------------------------------------------------------------.
    # Initialize dictionary to store module properties (params, shape, memory, time)
    summary_forward = {}
    summary_pre_forward = {}
    hooks = []
    ##------------------------------------------------------------------------.
    # Register the hook
    # - Applies register_hook recursively to every submodule
    for idx, module in enumerate(model.modules()):
        if not isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
    # for idx, (nm, module) in enumerate(model.named_children()):
    #     if not issubclass(type(module), (torch.nn.Sequential, torch.nn.ModuleList)):
            # Define dictionary key name
            class_name = module.__class__.__name__ # str(module.__class__).split(".")[-1].split("'")[0]
            m_key = "%s-%i" % (class_name, idx)
            h = module.register_forward_pre_hook( _generate_forward_pre_hook(hooks, 
                                                                             summary_pre_forward=summary_pre_forward,
                                                                             m_key=m_key))      
            hooks.append(h)
            h = module.register_forward_hook(_generate_forward_hook(hooks,
                                                                    summary_forward=summary_forward,
                                                                    m_key=m_key))
            hooks.append(h)
   
    ##------------------------------------------------------------------------.
    # Make a forward pass
    # print(x.shape)
    model(*x)
    
    ##------------------------------------------------------------------------.
    # Remove hooks
    for h in hooks:
        h.remove()
        
    ##------------------------------------------------------------------------.
    # Create summary information
    # - Create summary
    summary = summary_forward.copy()   
    # - Retrieve execution time and memory usage of each module 
    for key in summary.keys():
        summary[key]['time'] = summary_forward[key]['time'] - summary_pre_forward[key]['time']  
        summary[key]['time'] = summary[key]['time']
        if device.type != 'cpu':
            summary[key]['delta_memory_allocated'] = summary_forward[key]['memory_allocated'] - summary_pre_forward[key]['memory_allocated']
            summary[key]['delta_memory_cached'] = summary_forward[key]['memory_cached'] - summary_pre_forward[key]['memory_cached']
    
    ##------------------------------------------------------------------------.
    # Conversion to dataframe for tabulate printing
    df = pd.DataFrame.from_dict(summary).transpose()
    #print(tabulate(df, headers='keys', tablefmt="rst"))
    # Set layer as column 
    df.reset_index(level=0, inplace=True) # index --> layer 
    # Remove not trainable layers
    df = df.dropna() # Remove repeated info (i.e. ConvBlock) ...but discard also pooling
    # Reset df index from 0 to ...
    df.reset_index(level=0, drop = True, inplace=True)    
    df['time'] = df['time'].apply(lambda x: round(x, 3))
    # Rename Layers 
    df['index'] = df['index'].apply(lambda x: x.split("-")[0])
    # - Define Column order 
    if device.type == 'cpu':
        col_dict = {'index': 'Layer',
                    'input_shape': 'Input Shape',
                    'output_shape': 'Output Shape',
                    'nb_params': '# Params', 
                    'time': 'Time [s]'
                    }
        df = df[[*list(col_dict.keys())]]
    else:
        df['delta_memory_allocated'] = df['delta_memory_allocated'].apply(lambda x: round(x, 2))
        df['delta_memory_cached'] = df['delta_memory_cached'].apply(lambda x: round(x, 2))
        df['memory_allocated'] = df['memory_allocated'].apply(lambda x: round(x, 2))
        df['memory_cached'] = df['memory_cached'].apply(lambda x: round(x, 2))
        col_dict = {'index': 'Layer',
                    'input_shape': 'Input Shape',
                    'output_shape': 'Output Shape',
                    'nb_params': '# Params', 
                    'time': 'Time [s]',
                    'delta_memory_allocated': 'Mem. alloc. [MB]',
                    'delta_memory_cached': 'Mem. cached [MB]',
                    'memory_allocated': 'Tot. mem. alloc. [MB]',
                    'memory_cached': 'Tot. mem. cached [MB]',
                    }
        df = df[[*list(col_dict.keys())]]
    # Create tabulate 
    table = tabulate(df, headers=list(col_dict.values()), tablefmt="rst")
    ##------------------------------------------------------------------------.
    ### Parameters information
    total_params = 0
    trainable_params = 0
    for layer in summary:
        total_params += summary[layer]["nb_params"]
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
    ##------------------------------------------------------------------------.
    ## Summary string for parameters
    summary_str = ""
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    ##------------------------------------------------------------------------.
    ## Summary string for overall forward pass memory consumption
    if device.type != 'cpu':
        total_size = batch_memory + forward_memory + already_allocated
        summary_str += "================================================================" + "\n"
        summary_str += "Input batch size (MB): %0.2f" % batch_memory + "\n"
        summary_str += "Forward/backward pass size (MB): %0.2f" % forward_memory + "\n"
        summary_str += "** Params size (MB): %0.2f" % already_allocated + "\n"
        summary_str += "** Estimated Total Size (MB): %0.2f" % total_size + "\n"
    ##------------------------------------------------------------------------.
    ### Summary string with forward pass timing information
    summary_str += "================================================================" + "\n"
    summary_str += "** Forward pass (s): %0.2f" % forward_time + "\n"
    summary_str += "================================================================" + "\n"
    
    ##------------------------------------------------------------------------.
    ### Free memory 
    del model 
    del x 
    ##------------------------------------------------------------------------.
    # Return summary
    return table, summary_str, summary

def summarize_model(model, input_size, batch_size=32,
                    dtypes=None, device=torch.device('cpu')):
    """Print a summary of pytorch model structure and memory requirements.
    
    Originally inspired from https://github.com/sksq96/pytorch-summary and
    https://www.sicara.ai/blog/2019-28-10-deep-learning-memory-usage-and-pytorch-optimization-tricks post.
    Profile the model computing execution time and memory consumption of each trainable layer.
    The allocated memory is the memory that is currently used to store Tensors on the GPU.
    The cached memory is the memory that is currently used on the GPU by pytorch (nvidia-smi)
    
    Parameters
    ----------
    model : 
       Pytorch model.
    input_size : tuple or list
        A tuple or list of tuples describing input tensors shapes.
    batch_size : int, optional
        The batch size. The default is 32.
    dtypes : list, optional
        A list of dtype of the input tensors. 
        If set to None (and by default), it uses the default torch type. 
    device : str or torch.device, optional
        The torch device to use. The default is torch.device('cpu').
    """
    tmp_model = copy.deepcopy(model)
    tmp_model.to(device)
    table, summary_str, summary = profile_layers(model=tmp_model,
                                                 input_size=input_size,
                                                 batch_size=batch_size, 
                                                 device=device,
                                                 dtypes=dtypes)
    print(table)
    print(summary_str)
    del tmp_model
    return summary
    
##----------------------------------------------------------------------------.
def profile_model(model, input_size, batch_size, device='cpu', dtypes=None, row_limit=10):
    """Profile time execution and memory of pytorch operations using pytorch profiler.
        
    Parameters
    ----------
    model : 
       Pytorch model.
    input_size : tuple or list
        A tuple or list of tuples describing input tensors shapes.
    batch_size : int, optional
        The batch size. The default is 32.
    dtypes : list, optional
        A list of dtype of the input tensors. 
        If set to None (and by default), it uses the default torch type. 
    device : str or torch.device, optional
        The torch device to use. The default is torch.device('cpu').
    row_limit : TYPE, optional
        Number of pytorch operations to print. The default is 10.

    Returns
    -------
    prof : torch.autograd.profiler.profile
        Torch Profiler object. 
        Self CUDA/CPU time/memory exludes time/memory spent in children operator calls,

    """
    ##------------------------------------------------------------------------.
    # Check device 
    device = check_device(device)
    ##------------------------------------------------------------------------.
    # Check input size 
    # - List of tuples (to deal with multiple input tensors)
    if not isinstance(input_size, (tuple, list)):
        raise TypeError("'input_size' must be a tuple or a list of tuples.")
    if isinstance(input_size, tuple):
        input_size = [input_size]
    if isinstance(input_size, list): 
        idx_not_tuple = [not isinstance(t, tuple) for t in input_size]
        if any(idx_not_tuple):
            raise ValueError("input_size must be a list of tuples.")   
    ##------------------------------------------------------------------------.    
    # Check dtypes
    # - TODO: check dtype validity more robust 
    default_dtype = torch.get_default_dtype()
    if dtypes is None: 
        dtypes = default_dtype
    if not isinstance(dtypes, list):
        dtypes = [dtypes]    
    dtypes = [dtype if dtype is not None else default_dtype for dtype in dtypes]
    ##------------------------------------------------------------------------.
    # Check batch_size 
    if batch_size < 2: 
        raise ValueError("'batch_size' must be at least 2 (for batch norm ...).")
    ##------------------------------------------------------------------------.
    # Create deep copy of the model 
    tmp_model = copy.deepcopy(model)
    tmp_model.to(device)
    ##------------------------------------------------------------------------.
    # Create the list of input tensors   
    x = [torch.rand(batch_size, *in_size, dtype=dtype, device=device)
         for in_size, dtype in zip(input_size, dtypes)]
    ##------------------------------------------------------------------------.
    # Warm up 
    out = tmp_model(*x) 
    del out
    ##------------------------------------------------------------------------.
    ### Profile (time execution and memory)
    # - use_cuda - whether to measure execution time of CUDA kernels.
    # - record_shapes: currently do not allow named Tensors --> Disabled
    with profiler.profile(record_shapes=False, profile_memory=True) as prof:
        with profiler.record_function("model_inference"):
            tmp_model(*x) 
    ##------------------------------------------------------------------------.        
    ### Print 
    # cpu_time, cuda_time 
    # self_cpu_time, self_cuda_time 
    # cpu_time_total, cuda_time_total  
    # cpu_memory_usage, cuda_memory_usage 
    # self_cpu_memory_usage, self_cuda_memory_usage
    if device.type == "cpu":
        sort_by = "self_cpu_memory_usage"
    else: 
        sort_by = "self_cuda_memory_usage" 
    print(prof.key_averages(group_by_input_shape=False).table(sort_by=sort_by, row_limit=row_limit))
    ##------------------------------------------------------------------------.
    # Remove input tensor and model
    del x 
    del tmp_model
    ##------------------------------------------------------------------------.
    # Return prof 
    return prof
