#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 17:53:55 2021

@author: ghiggi
"""
import random
import time
import copy 
import torch 
import torch.autograd.profiler as profiler
import numpy as np 
from collections import OrderedDict
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

def check_prefetch_in_GPU(prefetch_in_GPU, num_workers, device):
    """Check prefetch_in_GPU possibility."""
    if not isinstance(prefetch_in_GPU, bool):
        raise TypeError("'prefetch_in_GPU' must be either True or False. If num_workers > 0, set to False ;) ")
    # CPU case
    if device.type == 'cpu':
        if prefetch_in_GPU:
            print("- GPU is not available. 'prefetch_in_GPU' set to False.")
            prefetch_in_GPU = False
    # GPU case with multiprocess
    elif num_workers > 0 and prefetch_in_GPU: 
        print("- Prefetch in GPU with multiprocessing is currently unstable.\n\
            It is generally not recommended to return CUDA tensors within multi-process data loading\
                loading because of many subtleties in using CUDA and sharing CUDA tensors.")
        prefetch_in_GPU = False    
    else: # num_workers = 0 
        prefetch_in_GPU = prefetch_in_GPU
    return prefetch_in_GPU

def check_asyncronous_GPU_transfer(asyncronous_GPU_transfer, device):
    """Check asyncronous_GPU_transfer possibility."""
    if not isinstance(asyncronous_GPU_transfer, bool):
        raise TypeError("'asyncronous_GPU_transfer' must be either True or False.")
    # CPU case
    if device.type == 'cpu':
        if asyncronous_GPU_transfer:
            print("- GPU is not available. 'asyncronous_GPU_transfer' set to False.")
            asyncronous_GPU_transfer = False
    return asyncronous_GPU_transfer

def check_prefetch_factor(prefetch_factor, num_workers):     
    """Check prefetch_factor validity."""
    if not isinstance(prefetch_factor, int):
        raise TypeError("'prefetch_factor' must be positive integer.")
    if prefetch_factor < 0: 
        raise ValueError("'prefetch_factor' must be positive.")
    if num_workers == 0 and prefetch_factor !=2:
        prefetch_factor = 2 # bug in pytorch ... need to set to 2 
    return prefetch_factor

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

def set_pytorch_deterministic(seed=100):
    """Set seeds for deterministic training with pytorch."""
    # TODO
    # - https://pytorch.org/docs/stable/generated/torch.set_deterministic.html#torch.set_deterministic
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    return          
 
def set_pytorch_numeric_precision(numeric_precision, device):
    """Set pytorch numeric precision."""
    tensor_type =  get_torch_tensor_type(numeric_precision, device)
    dtype = get_torch_dtype(numeric_precision)
    torch.set_default_tensor_type(tensor_type) 
    torch.set_default_dtype(dtype)
 
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
    
#----------------------------------------------------------------------------.
############################
### Summary / Profiling ####
############################
# TODOs
# - check dtype validity more robust 
# - check input size tuple values are positive integers

def summarize_model(model, input_size, batch_size=32, dtypes=None, device=torch.device('cpu')):
    """Print a summary of pytorch model structure and memory requirements.
    
    Originally inspired from https://github.com/sksq96/pytorch-summary
    
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
    result, params_info = summary_string(model=tmp_model,
                                         input_size=input_size,
                                         batch_size=batch_size, 
                                         device=device,
                                         dtypes=dtypes)
    print(result)

def summary_string(model, input_size, batch_size=32, dtypes=None, device=torch.device('cpu')):
    """Create model summary string."""
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
    # Retrieve timing function (for cpu and CUDA)
    get_time = get_time_function(device)
    ##------------------------------------------------------------------------.
    # Initialize summary string 
    summary_str = ''
    ##------------------------------------------------------------------------.
    ### Define hook (nested definition)
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not (isinstance(module, torch.nn.Sequential) and not isinstance(module, torch.nn.ModuleList)):
            hooks.append(module.register_forward_hook(hook))
            
    ##------------------------------------------------------------------------.
    # Record how many memory is already taken in GPU (model params + ....)
    if device.type != 'cpu':
        already_allocated = torch.cuda.memory_allocated()/1000/1000
    ##------------------------------------------------------------------------.
    # Create the list of input tensors   
    x = [torch.rand(batch_size, *in_size, dtype=dtype, device=device)
         for in_size, dtype in zip(input_size, dtypes)]
    
    ##------------------------------------------------------------------------.
    ### Profile memory (if CUDA)
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
    del output 
    ##------------------------------------------------------------------------.
    # Create properties
    summary = OrderedDict()
    hooks = []

    # Register the hook
    model.apply(register_hook)

    # Make a forward pass
    # print(x.shape)
    model(*x)

    # Remove these hooks
    for h in hooks:
        h.remove()
    ##------------------------------------------------------------------------.
    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    ##------------------------------------------------------------------------.
    ### Parameters information
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"
    ##------------------------------------------------------------------------.
    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    ##------------------------------------------------------------------------.
    ### Memory information 
    if device.type != 'cpu':
        total_size = batch_memory + forward_memory + already_allocated
        summary_str += "================================================================" + "\n"
        summary_str += "Input batch size (MB): %0.2f" % batch_memory + "\n"
        summary_str += "Forward/backward pass size (MB): %0.2f" % forward_memory + "\n"
        summary_str += "Params* size (MB): %0.2f" % already_allocated + "\n"
        summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
        summary_str += "----------------------------------------------------------------" + "\n"
    ##------------------------------------------------------------------------.
    ### Timing information
    summary_str += "================================================================" + "\n"
    summary_str += "Forward pass (s): %0.2f" % forward_time + "\n"
    summary_str += "================================================================" + "\n"
    
    ##------------------------------------------------------------------------.
    ### Free memory 
    del model 
    del x 
    ##------------------------------------------------------------------------.
    # Return summary
    return summary_str, (total_params, trainable_params)

##----------------------------------------------------------------------------.
def profile_model(model, input_size, batch_size, device='cpu', dtypes=None, row_limit=10):
    """Profile time execution and memory of pytorch operations.
        
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
