import numpy as np
from scipy import sparse
import pygsp
from modules import remap
import torch

from typing import Dict

def _build_interpolation_matrix(src_graph, dst_graph):
    """Return the sparse matrix that interpolates between two spherical samplings."""

    ds = remap.compute_interpolation_weights(src_graph, dst_graph, method='conservative', normalization='fracarea') # destarea’

    # Sanity checks.
    np.testing.assert_allclose(ds.src_grid_center_lat, src_graph.signals['lat'])
    np.testing.assert_allclose(ds.src_grid_center_lon, src_graph.signals['lon'])
    np.testing.assert_allclose(ds.dst_grid_center_lat, dst_graph.signals['lat'])
    np.testing.assert_allclose(ds.dst_grid_center_lon, dst_graph.signals['lon'])
    np.testing.assert_allclose(ds.src_grid_frac, 1)
    np.testing.assert_allclose(ds.dst_grid_frac, 1)
    np.testing.assert_allclose(ds.src_grid_imask, 1)
    np.testing.assert_allclose(ds.dst_grid_imask, 1)

    col = ds.src_address
    row = ds.dst_address
    dat = ds.remap_matrix.squeeze()
    # CDO indexing starts at 1
    row = np.array(row) - 1
    col = np.array(col) - 1
    weights = sparse.csr_matrix((dat, (row, col)))
    assert weights.shape == (dst_graph.n_vertices, src_graph.n_vertices)

    # Destination pixels are normalized to 1 (row-sum = 1).
    # Weights represent the fractions of area attributed to source pixels.
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    # Interpolation is conservative: it preserves area.
    np.testing.assert_allclose(weights.T @ ds.dst_grid_area, ds.src_grid_area)

    # Unnormalize.
    weights = weights.multiply(ds.dst_grid_area.values[:, np.newaxis])

    # Another way to assert that the interpolation is conservative.
    np.testing.assert_allclose(np.asarray(weights.sum(1)).squeeze(), ds.dst_grid_area)
    np.testing.assert_allclose(np.asarray(weights.sum(0)).squeeze(), ds.src_grid_area)

    return weights


def build_pooling_matrices(src_graph, dst_graph):
    weights = _build_interpolation_matrix(src_graph, dst_graph)
    pool = weights.multiply(1/weights.sum(1))
    unpool = weights.multiply(1/weights.sum(0)).T
    return pool, unpool


def compute_pooling_healpix(nodes1, nodes2, k=8, **kwargs):
    sd1 = int(np.sqrt(nodes1 / 12))
    sd2 = int(np.sqrt(nodes2 / 12))
    G1 = pygsp.graphs.SphereHealpix(subdivisions=sd1, nest=True, k=k, kernel_width=None)
    G2 = pygsp.graphs.SphereHealpix(subdivisions=sd2, nest=True, k=k, kernel_width=None)
    return build_pooling_matrices(G1, G2)


def compute_pooling_equiangular(nodes1, nodes2, **kwargs):
    G1 = pygsp.graphs.SphereEquiangular(*nodes1)
    G2 = pygsp.graphs.SphereEquiangular(*nodes2)
    return build_pooling_matrices(G1, G2)

def convert_to_torch_sparse(mat: "sparse.coo.coo_matrix"):
    indices = np.empty((2, mat.nnz), dtype=np.int64)
    np.stack((mat.row, mat.col), axis=0, out=indices)
    indices = torch.from_numpy(indices)
    mat = torch.sparse_coo_tensor(indices, mat.data, mat.shape)
    mat = mat.coalesce()
    return mat


class GeneralSpherePoolUnpool(torch.nn.Module):
    def __init__(self, matrices_dict: Dict={}):
        super().__init__()
        self.matrices = matrices_dict
    
    def __setitem__(self, key, value):
        self.matrices[key] = value

    def __getitem__(self, key):
        return self.matrices[key]
    
    def add(self, key, value):
        self.matrices[key] = value
    
    def pop(self, key):
        self.matrices.pop(key)
    
    def forward(self, x):
        n_batch, n_nodes, n_val = x.shape
        matrix = self.matrices[n_nodes]
        new_nodes, _ = matrix.shape
        x = x.permute(1, 2, 0).reshape(n_nodes, n_batch * n_val)
        x = matrix @ x
        x = x.reshape(new_nodes, n_val, n_batch).permute(2, 0, 1)
        return x
    
    # TODO: use persistent buffer
    def cpu(self):
        for k, v in self.matrices.items():
            self.matrices[k] = v.cpu()
        return self
    
    def cuda(self, device=None):
        for k, v in self.matrices.items():
            self.matrices[k] = v.cuda(device)
        return self
    
    def to(self, device):
        for k, v in self.matrices.items():
            self.matrices[k] = v.to(device)
        return self

