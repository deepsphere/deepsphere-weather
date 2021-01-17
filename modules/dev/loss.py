#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 23:59:25 2021

@author: ghiggi
"""
import numpy as np 
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F

#############################
### Weight updating rule ####
#############################
# TODO: To be placed elsewhere ... ideally generalized with a setting ... 
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

def plot_weight_variations(weights_loss, figures_path):
    f, ax = plt.subplots(4, 4, figsize=(15, 10), sharex=True, sharey=True)
    ax = ax.flatten()
    x_vals = list(range(len(weights_loss)))
    ax[0].scatter(x_vals, weights_loss)
    ax[0].set_title('Initial weights')

    for i in range(15):
        weights_loss = update_w(weights_loss)
        ax[i + 1].scatter(x_vals, weights_loss)
        ax[i + 1].set_title('Update ' + str(i))

    plt.xlabel('Time step ahead')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.savefig(figures_path + 'weight_updates.pdf')


class WeightedMSELoss(nn.MSELoss):
    def __init__(self, reduction='sum', weights=None, normalized=False):
        super(WeightedMSELoss, self).__init__(reduction='none')
        if not isinstance(reduction, str) or reduction not in ('mean', 'sum', 'none'):
            raise ValueError("{} is not a valid value for reduction".format(reduction))
        self.weighted_mse_reduction = reduction

        if weights is not None:
            self.check_weights(weights)
            if normalized:
                weights /= torch.sum(weights)
        self.weights = weights
    
    def forward(self, pred, label, weights=None):
        mse = super(WeightedMSELoss, self).forward(pred, label)
        if weights is None:
            weights = self.weights
        else:
            self.check_weights(weights)
        
        _, num_nodes, _ = mse.shape
        if weights is None:
            weights = torch.ones((num_nodes), dtype=mse.dtype, device=mse.device)
        if num_nodes != len(weights):
            raise ValueError("The number of weights does not match the the number of pixels. {} != {}"
                                .format(len(weights), num_nodes))
        weights = weights.view(1, -1, 1).to(mse.device)
        if self.weighted_mse_reduction == 'sum':
            return torch.sum(mse * weights)
        elif self.weighted_mse_reduction == 'mean':
            return torch.mean(mse * weights)
        else:
            return mse * weights

    def check_weights(self, weights):
        if not isinstance(weights, torch.Tensor):
            raise TypeError("Weights type is not a torch.Tensor. Got {}".format(type(weights)))
        if len(weights.shape) != 1:
            raise ValueError("Weights is a 1D vector. Got {}".format(weights.shape))
