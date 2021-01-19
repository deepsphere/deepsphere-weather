import torch
from torch import nn

class WeightedMSELoss(nn.MSELoss):
    def __init__(self, reduction='mean', weights=None):
        super(WeightedMSELoss, self).__init__(reduction='none')
        if not isinstance(reduction, str) or reduction not in ('mean', 'sum', 'none'):
            raise ValueError("{} is not a valid value for reduction".format(reduction))
        self.weighted_mse_reduction = reduction

        if weights is not None:
            self.check_weights(weights)
        self.weights = weights
    
    def forward(self, pred, label):
        mse = super(WeightedMSELoss, self).forward(pred, label)
        weights = self.weights
        n_batch, num_nodes, n_val = mse.shape
        if weights is None:
            weights = torch.ones((num_nodes), dtype=mse.dtype, device=mse.device)
        if num_nodes != len(weights):
            raise ValueError("The number of weights does not match the the number of pixels. {} != {}"
                                .format(len(weights), num_nodes))
        weights = weights.view(1, -1, 1).to(mse.device)
        weighted_mse = mse * weights
        if self.weighted_mse_reduction == 'sum':
            return torch.sum(weighted_mse) * len(weights)
        elif self.weighted_mse_reduction == 'mean':
            return torch.sum(weighted_mse) / torch.sum(weights) / n_batch / n_val
        else:
            return weighted_mse

    def check_weights(self, weights):
        if not isinstance(weights, torch.Tensor):
            raise TypeError("Weights type is not a torch.Tensor. Got {}".format(type(weights)))
        if len(weights.shape) != 1:
            raise ValueError("Weights is a 1D vector. Got {}".format(weights.shape))
