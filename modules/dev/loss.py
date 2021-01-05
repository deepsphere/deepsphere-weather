#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 23:59:25 2021

@author: ghiggi
"""
import numpy as np 
import matplotlib.pyplot as plt

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