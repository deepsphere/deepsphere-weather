import torch
from modules.architectures import UNetSpherical
import os
import json
from tqdm import trange
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def run(config_path, resolution, nodes, batchsize, iters_num):
    savedir = f'./timing/nodes_{nodes}/'
    os.makedirs(savedir, exist_ok=True)

    with open(config_path) as json_data_file:
        cfg = json.load(json_data_file)

    len_sqce = cfg['model_parameters']['len_sqce']
    in_features = cfg['model_parameters']['in_features']
    out_features = cfg['model_parameters']['out_features']

    net_params = {}
    net_params["sampling"] = cfg['model_parameters'].get("sampling", None)
    net_params["knn"] = cfg['model_parameters'].get("knn", 10)
    net_params["conv_type"] = cfg['model_parameters'].get("conv_type", None)
    net_params["pool_method"] = cfg['model_parameters'].get("pool_method", None)
    net_params["ratio"] = cfg['model_parameters'].get("ratio", None)
    net_params["periodic"] = cfg['model_parameters'].get("periodic", None)
    net_params["comments"] = cfg['model_parameters'].get("comments", None)
    description = [str(i) for i in net_params.values() if i is not None]
    description = '_'.join(description)
    net_params.pop('comments')
    print(description)

    spherical_unet = UNetSpherical(resolution, in_channels=in_features * len_sqce, out_channels=out_features, kernel_size=3, **net_params)
    spherical_unet.to('cuda')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(spherical_unet.parameters(), lr=0.008, eps=1e-7, weight_decay=0, amsgrad=False)


    epoch = iters_num
    times_train_forward = [] #############################
    times_train_backward = [] #############################

    spherical_unet.train()
    for _ in trange(epoch):
        data = torch.randn([batchsize, nodes, 14], dtype=torch.float32, device='cuda')
        label = torch.randn([batchsize, nodes, 2], dtype=torch.float32, device='cuda')

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = spherical_unet(data)
        end.record()
        torch.cuda.synchronize()
        time_used = start.elapsed_time(end) / data.shape[0]
        times_train_forward.append(time_used)

        loss = criterion(output, label)

        optimizer.zero_grad()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()

        time_used = start.elapsed_time(end) / data.shape[0]
        times_train_backward.append(time_used)
    

    spherical_unet.eval()
    times_eval = []
    with torch.set_grad_enabled(False):
        for _ in trange(epoch):
            data = torch.randn([batchsize, nodes, 14], dtype=torch.float32, device='cuda')

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = spherical_unet(data)
            end.record()
            torch.cuda.synchronize()
            time_used = start.elapsed_time(end) / data.shape[0]
            times_eval.append(time_used)
    
    np.save(savedir + 'forward_time.npy', times_train_forward)
    np.save(savedir + 'backward_time.npy', times_train_backward)
    np.save(savedir + 'inference_time.npy', times_eval)

    print('Median of train(forward) {} train(backward) {} inference {}'
                    .format(np.median(times_train_forward), np.median(times_train_backward), np.median(times_eval)))


if __name__ == '__main__':
    all_nodes = [768, 3072, 12288]
    all_res = [8, 16, 32]
    epoch = 1000
    batchsize = 15
    config_path = '/nfs_home/wefeng/GitHub/weather_prediction/configs/healpix_20_graph_max_400km.json'
    for res, node in zip(all_nodes, all_res):
        _ = run(config_path, res, node, batchsize, epoch)
