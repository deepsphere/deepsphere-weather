import torch
from torch import nn
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 

from modules.remap import compute_interpolation_weights

def AreaWeights(graph):
    """Compute area weights."""
    ds = compute_interpolation_weights(graph, graph, method='conservative', normalization='fracarea')  
    src_grid_area = ds.src_grid_area.values
    weights = src_grid_area / np.sum(src_grid_area)
    weights = torch.from_numpy(weights.astype(np.float32))
    return weights

def plot_weights(weights, pygsp_graph, crs_proj=None): 
    """Plot weights on the sphere.
    
    Example:   
        
    from modules.loss import plot_weights 
    crs_proj = ccrs.PlateCarree() 
    crs_proj = ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0)  # from the North Pole 
    weights = AreaWeights(graph=model.graphs[0])
    plot_weights(weights, pygsp_graph=model.graphs[0], crs_proj=crs_proj) 
    
    """
    # Imports
    import modules.xsphere as xsphere 
    # Checks 
    if isinstance(weights, torch.Tensor):
        weights = weights.numpy()
    # Define a default projection 
    if crs_proj is None: 
        crs_proj = ccrs.PlateCarree() 
    # Retrieve lon and lat coordinates
    pygsp_graph.set_coordinates('sphere', dim=2)
    lat = np.rad2deg(pygsp_graph.coords[:,1])
    lon = np.rad2deg(pygsp_graph.coords[:,0])
    # Ensure lon is between -180 and 180
    lon[lon > 180] = lon[lon > 180] - 360
    # Create DataArray
    da = xr.DataArray(data = weights, 
                      dims = ['node'],
                      coords = {'lat': ('node', lat),
                                'lon': ('node', lon)},
                      name = 'area_weights')
    # Add mesh 
    da = da.sphere.add_SphericalVoronoiMesh(x='lon', y='lat')
    # Plot    
    fig, ax = plt.subplots(1,1, subplot_kw=dict(projection=crs_proj))
    xsphere._plot(da, ax=ax,
                  edgecolors=None, 
                  linewidths=0.01,
                  add_colorbar=True)
    ax.coastlines(alpha=0.2)
    plt.show()
    
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
