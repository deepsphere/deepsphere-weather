#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:46:41 2021

@author: ghiggi
"""
import xarray as xr

##----------------------------------------------------------------------------.
def reformat_pl(ds, var_dict=None, unstack_plev=True, stack_variables=True):
    """Reformat raw pressure level data netCDFs.

    unstack_plev: bool
         If True, it suppress the vertical dimension and it treats every
           <variable>-<pressure level> as a separate feature
         If False, it keeps the vertical (pressure) dimension
    stack_variables: bool
        If True, all Dataset variables are stacked across a new 'feature' dimension
    """
    ##------------------------------------------------------------------------.
    # Check arguments
    if var_dict is not None:
        if not isinstance(var_dict, dict):
            raise TypeError("var_dict must be a dictionary")

    # Drop information related to mesh vertices
    ds = ds.drop_vars(["lon_bnds", "lat_bnds"])
    # Drop lat lon info of nodes
    # ds = ds.drop_vars(['lon', 'lat'])
    # Rename ncells to nodes
    ds = ds.rename({"ncells": "node"})
    # Rename variables
    if var_dict is not None:
        ds = ds.rename(var_dict)

    ##------------------------------------------------------------------------.
    ### - Unstack pressure levels (as separate variables)
    if unstack_plev:
        l_da_vars = []
        for var in ds.keys():
            for i, p_level in enumerate(ds[var]["plev"].values):
                # - Select variable at given pressure level
                tmp_da = ds[var].isel(plev=i)
                # - Update the name to (<var>_<plevel_in_hPa>)
                tmp_da.name = tmp_da.name + str(int(p_level / 100))
                # - Remove plev dimension
                tmp_da = tmp_da.drop_vars("plev")
                # - Append to the list of DataArrays to then be merged
                l_da_vars.append(tmp_da)

        ds = xr.merge(l_da_vars)

    ##------------------------------------------------------------------------.
    ### - Stack variables
    # - new_dim : new dimension name
    # - variable_dim : level name of the new stacked coordinate <new_dim>
    # - name : DataArray name
    if stack_variables:
        da_stacked = ds.to_stacked_array(
            new_dim="feature",
            variable_dim="variable",
            sample_dims=list(ds.dims.keys()),
            name="data",
        )
        # - Remove MultiIndex for compatibility with netCDF and Zarr
        da_stacked = da_stacked.reset_index("feature")
        da_stacked = da_stacked.set_index(feature="variable", append=True)
        da_stacked["feature"] = da_stacked["feature"].astype(str)
        # - Remove attributes from DataArray
        da_stacked.attrs = {}
        # - Reshape to Dataset
        ds = da_stacked.to_dataset()
        # - Reorder dimension as requested by PyTorch input
        ds = ds.transpose("time", "node", ..., "feature")
    else:
        # - Reorder dimension as requested by PyTorch input
        ds = ds.transpose("time", "node", ...)

    ##------------------------------------------------------------------------.
    return ds


def reformat_toa(ds, var_dict=None, stack_variables=True):
    """Reformat raw TOA data netCDFs."""
    ##------------------------------------------------------------------------.
    # Check arguments
    if var_dict is not None:
        if not isinstance(var_dict, dict):
            raise TypeError("var_dict must be a dictionary")
    ##------------------------------------------------------------------------.
    # Drop information related to mesh vertices
    ds = ds.drop_vars(["lon_bnds", "lat_bnds"])
    # Drop lat lon info of nodes
    # ds = ds.drop_vars(['lon', 'lat'])
    # Rename ncells to nodes
    ds = ds.rename({"ncells": "node"})
    # Rename variables
    if var_dict is not None:
        ds = ds.rename(var_dict)

    ##------------------------------------------------------------------------.
    ### - Stack variables
    # - new_dim : new dimension name
    # - variable_dim : level name of the new stacked coordinate <new_dim>
    # - name : DataArray name
    if stack_variables:
        da_stacked = ds.to_stacked_array(
            new_dim="feature",
            variable_dim="variable",
            sample_dims=list(ds.dims.keys()),
            name="data",
        )
        # - Remove MultiIndex for compatibility with netCDF and Zarr
        da_stacked = da_stacked.reset_index("feature")
        da_stacked = da_stacked.set_index(feature="variable", append=True)
        da_stacked["feature"] = da_stacked["feature"].astype(str)
        # - Remove attributes from DataArray
        da_stacked.attrs = {}
        # - Reshape to Dataset
        ds = da_stacked.to_dataset()
        # - Reorder dimension as requested by PyTorch input
        ds = ds.transpose("time", "node", ..., "feature")
    else:
        # - Reorder dimension as requested by PyTorch input
        ds = ds.transpose("time", "node", ...)
    ##------------------------------------------------------------------------.
    return ds
