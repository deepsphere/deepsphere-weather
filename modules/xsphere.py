#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:55:53 2021

@author: ghiggi
"""

import numpy as np
import xarray as xr
import shapely
import matplotlib as mpl 
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from scipy.interpolate import griddata 
from scipy.spatial import SphericalVoronoi
from cartopy.util import add_cyclic_point
from xarray.plot.facetgrid import FacetGrid 
from xarray.plot.utils import _process_cmap_cbar_kwargs
from xarray.plot.utils import import_matplotlib_pyplot
from xarray.plot.utils import get_axis
from xarray.plot.utils import _add_colorbar
from xarray.plot.utils import label_from_attrs
from xarray.plot.plot import pcolormesh as xr_pcolormesh
from xarray.plot.plot import contour as xr_contour
from xarray.plot.plot import contourf as xr_contourf

##----------------------------------------------------------------------------.
# http://xarray.pydata.org/en/stable/generated/xarray.plot.FacetGrid.html
# https://github.com/pydata/xarray/blob/master/xarray/plot/plot.py

# TODO: 
# - add_HealpixMesh, 
# - add_EquiangularMesh,
# - add_GaussianLegendreMesh, 
# - add_CubedMesh
# --> pgysp graph? 

# - reshape equiangular to lat, lon core dim 
# - reshape_equiangular_to_unstructured

# - Check_mesh()  --> Check Polygon mpatches 

## Add from shapefile 
# - da.sphere.add_mesh_from_shp(fpath)  # poly.shp 
# - da.sphere.add_nodes_from_shp(fpath) # point.shp
# - da.sphere.save_mesh_to_shp()
# - da.sphere.save_nodes_to_shp()

## Spherical polygons area computations (now planar assumption)
# - https://github.com/anutkk/sphericalgeometry
# - https://stackoverflow.com/questions/4681737/how-to-calculate-the-area-of-a-polygon-on-the-earths-surface-using-python

#-----------------------------------------------------------------------------.
# ##############################
#### Coordinates conversion ####
# ##############################
# radius = 6371.0e6
def lonlat2xyz(longitude, latitude, radius=1):
    """From 2D geographic coordinates to cartesian geocentric coordinates."""
    lon, lat = np.deg2rad(longitude), np.deg2rad(latitude)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z

def xyz2lonlat(x,y,z, radius=1):
    """From cartesian geocentric coordinates to 2D geographic coordinates."""
    latitude = np.rad2deg(np.arcsin(z / radius))  
    longitude = np.rad2deg(np.arctan2(y, x)) 
    return longitude, latitude 

## Testing 
# x, y, z = xsphere.lonlat2xyz(lon, lat)
# lon1, lat1 = xsphere.xyz2lonlat(x,y,z)   
# np.testing.assert_allclose(lon, lon1) 
# np.testing.assert_allclose(lat, lat1) 

##----------------------------------------------------------------------------.
## Conversion to spherical coordinate is buggy 
# def xyz2sph(x,y,z):
#     """From cartesian geocentric coordinates to spherical polar coordinates."""
#     r = np.sqrt(x**2 + y**2 + z**2)  
#     theta = np.arccos(z/r) 
#     phi = np.arctan(y, x)
#     return theta, phi, r

# def sph2xyz(theta, phi, radius=1):
#     """From spherical polar coordinates to cartesian geocentric coordinates."""
#     x = radius * np.sin(theta) * np.cos(phi)
#     y = radius * np.sin(theta) * np.sin(phi)
#     z = radius * np.cos(theta)
#     return x, y, z

# def lonlat2sph(longitude, latitude, radius=1):
#     """From 2D geographic coordinates to spherical polar coordinates."""
#     x, y, z = lonlat2xyz(longitude=longitude, latitude=latitude, radius=radius)
#     return xyz2sph(x,y,z)
    
# def sph2lonlat(theta, phi, radius=1):
#     """From spherical polar coordinates to 2D geographic coordinates."""
#     x, y, z = sph2xyz(theta=theta, phi=phi, radius=1)
#     return xyz2lonlat(x,y,z)

## Testing 
# x, y, z = xsphere.lonlat2xyz(lon, lat)
# theta, phi, r = xsphere.xyz2sph(x,y,z)
# x1, y1, z1 = xsphere.sph2xyz(theta, phi, r)
# np.testing.assert_allclose(x, x1)
# np.testing.assert_allclose(y, y1)
# np.testing.assert_allclose(z, z1)

#-----------------------------------------------------------------------------.
def get_polygons_2D_coords(lon_bnds, lat_bnds):
    """Create a list of numpy [x y] array polygon vertex coordinates from CDO lon_bnds and lat_bnds matrices.""" 
    # Input: n_polygons x n_vertices 
    # Output: list (for each polygon) of numpy_array [x, y] polygon vertex coordinates
    n_polygons = lon_bnds.shape[0]
    n_vertices = lon_bnds.shape[1]
    list_polygons_xy = list()
    for i in range(n_polygons):
        poly_corners = np.zeros((n_vertices, 2), np.float64)
        poly_corners[:,0] = lon_bnds[i,:]
        poly_corners[:,1] = lat_bnds[i,:]
        list_polygons_xy.append(poly_corners)
    return(list_polygons_xy)

def get_PolygonPatchesList_from_latlon_bnds(lon_bnds, lat_bnds, fill=True):
    """Create a list of Polygon mpatches from CDO lon_bnds and lat_bnds."""
    # Construct list of polygons 
    l_polygons_xy = get_polygons_2D_coords(lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    l_Polygon_patch = [mpatches.Polygon(xy=p, closed=False, fill=fill) for p in l_polygons_xy]
    return l_Polygon_patch   

def get_PolygonPatchesList(l_polygons_xy, fill=True):
    """Create Polygon mpatches from a numpy [x y] array with polygon vertex coordinates."""
    # Construct list of mpatches.Polygon
    l_Polygon_patch = [mpatches.Polygon(xy=p, closed=False, fill=fill) for p in l_polygons_xy]
    return l_Polygon_patch

def SphericalVoronoiMesh(lon, lat):
    """
    Infer the mesh of a spherical sampling from the mesh node centers provided in 2D geographic coordinates.
    
    Parameters
    ----------
    lon : numpy.ndarray
        Array of longitude coordinates (in degree units).
    lat : numpy.ndarray
        Array of latitude coordinates (in degree units).

    Returns
    -------
    list_polygons_lonlat : list
        List of numpy.ndarray with the polygon mesh vertices for each graph node.

    """
    # Convert to geocentric coordinates 
    radius = 6371.0e6 # radius = 1 can also be used
    x, y, z = lonlat2xyz(lon, lat, radius=radius)
    coords = np.column_stack((x,y,z))
    # Apply Spherical Voronoi tesselation
    sv = SphericalVoronoi(coords,
                          radius=radius, 
                          center=[0, 0, 0])
    ##-------------------------------------------------------------------------.
    # SphericalVoronoi object methods 
    # - sv.vertices : Vertex coords
    # - sv.regions : Vertex ID of each polygon
    # - sv.sort_vertices_of_regions() : sort indices of vertices to be clockwise ordered
    # - sv.calculate_areas() : compute the area of the spherical polygons 
    ##-------------------------------------------------------------------------.
    # Sort vertices indexes to be clockwise ordered
    sv.sort_vertices_of_regions()
    # Retrieve area 
    area = sv.calculate_areas() 
    ##-------------------------------------------------------------------------.
    # Retrieve list of polygons coordinates arrays
    list_polygons_lonlat = []
    for region in sv.regions:  
        tmp_xyz = sv.vertices[region]    
        tmp_lon, tmp_lat = xyz2lonlat(tmp_xyz[:,0],tmp_xyz[:,1],tmp_xyz[:,2], radius=radius)    
        list_polygons_lonlat.append(np.column_stack((tmp_lon, tmp_lat)))
    ##-------------------------------------------------------------------------.
    return list_polygons_lonlat, area
    
#-----------------------------------------------------------------------------.
#### Checks 
def check_node_dim(x, node_dim):
    """Check node dimension."""
    if not isinstance(node_dim, str):
        raise TypeError("'node_dim' must be a string specifying the node dimension name.")
    if not isinstance(x, (xr.Dataset, xr.DataArray)):
        raise TypeError("You must provide an xarray Dataset or DataArray.")
    if isinstance(x, xr.Dataset):
        dims = list(x.dims.keys())
    if isinstance(x, xr.DataArray):
        dims = list(x.dims)
    if node_dim not in dims:
        raise ValueError("Specify the 'node_dim'.")
    return node_dim 
        
def check_mesh(mesh):
    """Check the mesh format."""
    return mesh 

def check_mesh_exist(x):
    """Check the mesh is available in the xarray object."""
    coords = list(x.coords.keys())
    if 'mesh' not in coords:
        raise ValueError("No 'mesh' available in the xarray object.")

def check_mesh_area_exist(x, mesh_area_coord):
    """Check the area coordinate is available in the xarray object."""
    coords = list(x.coords.keys())
    if not isinstance(mesh_area_coord, str):
        raise TypeError("'area_coord' must be a string specifying the area coordinate.")
    if mesh_area_coord not in coords:
        raise ValueError("No {} coordinate available in the xarray object.".format(mesh_area_coord))
        
def check_valid_coords(x, coords):
    """Check coordinates validity."""
    if isinstance(coords, str):
        coords=[coords]
    if not isinstance(coords, list):
        raise TypeError("'coords' must be a string or a list of string.")
    valid_coords = list(x.coords.keys())
    not_valid = np.array(coords)[np.isin(coords, valid_coords, invert=True)]
    if len(not_valid) > 0: 
        raise ValueError("{} are not coordinates of the xarray object. Valid coordinates are {}.".format(not_valid,valid_coords))
 
def check_xy(x_obj, x, y):
    """Check validty of x and y coordinates."""
    if not isinstance(x,str):
        raise TypeError("'x' must be a string indicating the longitude coordinate.")
    if not isinstance(y,str):
        raise TypeError("'x' must be a string indicating the latitude coordinate.")  
    # Check x and y are coords of the xarray object  
    check_valid_coords(x_obj, coords=[x,y])

#-----------------------------------------------------------------------------.
#### FacetGrids utils 
def map_dataarray_unstructured(self, func, **kwargs):
    """
    Apply a plotting function to an unstructured grid subset of data.
    
    This is more convenient and less general than ``FacetGrid.map``
    
    Parameters
    ----------
    func : callable
        A plotting function
    kwargs
        Additional keyword arguments to func
    Returns
    -------
    self : FacetGrid object
    """
    if kwargs.get("cbar_ax", None) is not None:
        raise ValueError("cbar_ax not supported by FacetGrid.")
    ##------------------------------------------------------------------------.    
    # Colorbar settings (exploit xarray defaults)
    if func.__name__ == '_contour':
        xr_func = xr_contour
    if func.__name__ == '_contourf':
        xr_func = xr_contourf
    if func.__name__ == '_plot':
        xr_func = xr_pcolormesh
    cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(xr_func, 
                                                         self.data.values,
                                                         **kwargs)
    self._cmap_extend = cmap_params.get("extend")
    ##------------------------------------------------------------------------.
    # Order is important
    func_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in {"cmap", "colors", "cbar_kwargs", "levels"}
    }
    func_kwargs.update(cmap_params)
    func_kwargs.update({"add_colorbar": False, "add_labels": False})
    ##------------------------------------------------------------------------.
    # Plot 
    for d, ax in zip(self.name_dicts.flat, self.axes.flat):
        # None is the sentinel value
        if d is not None:
            subset = self.data.loc[d]
            mappable = func(subset, ax=ax, **func_kwargs, _is_facetgrid=True)
            self._mappables.append(mappable)
    ##------------------------------------------------------------------------.
    xlabel=''
    ylabel=''
    self._finalize_grid(xlabel, ylabel)
    ##------------------------------------------------------------------------.
    # Add colorbars 
    if kwargs.get("add_colorbar", True):
        self.add_colorbar(**cbar_kwargs)
    ##------------------------------------------------------------------------.    
    return self
    
def _easy_facetgrid(
    data,
    plotfunc,
    x=None,
    y=None,
    row=None,
    col=None,
    col_wrap=None,
    sharex=True,
    sharey=True,
    aspect=None,
    size=None,
    subplot_kws=None,
    ax=None,
    figsize=None,
    **kwargs,
):
    """
    Call xarray.plot.FacetGrid from the plotting methods.
    
    kwargs are the arguments to the plotting method.
    """
    if ax is not None:
        raise ValueError("Can't use axes when making faceted plots.")
    if aspect is None:
        aspect = 1
    if size is None:
        size = 3
    elif figsize is not None:
        raise ValueError("cannot provide both `figsize` and `size` arguments")

    g = FacetGrid(
        data=data,
        col=col,
        row=row,
        col_wrap=col_wrap,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        aspect=aspect,
        size=size,
        subplot_kws=subplot_kws,
    )
    # Add map_dataarray_unstructured to FacetGrid 
    g.map_dataarray_unstructured = map_dataarray_unstructured
    # Plot 
    return g.map_dataarray_unstructured(g, plotfunc, **kwargs)

#-----------------------------------------------------------------------------. 
#### Plot methods for unstructured grid                  
def _contour(darray,
             x='lon',
             y='lat',
             transform = None,
             # Facetgrids arguments
            figsize=None,
            size=None,
            aspect=None,
            ax=None,
            row=None,
            col=None,
            col_wrap=None,
            subplot_kws=None,
            # Line option
            plot_type = "contour",
            add_contour = True,
            linewidths = None,
            linestyles = None,
            antialiased = None, 
            # Contour labels 
            add_contour_labels = True, 
            add_contour_labels_interactively = False, 
            contour_labels_colors = "black", 
            contour_labels_fontsize = 'smaller', 
            contour_labels_inline=True,   
            contour_labels_inline_spacing=5, 
            contour_labels_format="%1.3f", 
            # Colors option
            alpha=1,
            colors=None,
            levels=None, 
            cmap=None,
            norm=None,
            center=None,
            vmin=None,
            vmax=None,
            robust=False,
            extend='both',
            # Colorbar options
            add_colorbar=None,
            cbar_ax=None,
            cbar_kwargs=None,
            # Axis options
            add_labels=True,
            **kwargs):
    """
    Contourf plotting method for unstructured mesh.
    
    The DataArray must have the attribute 'nodes'.
        
    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional, unless creating faceted plots.
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots
    add_contour : bool, optional 
        Wheter to add contour lines. The default is True.
        Set to False is useful to plot just contour labels on top of a contourf plot.
    plot_type : str, optional
        Whether to use the "contour" or "tricontour" function.
        The default is "contour".
    linewidths : float, optional
        The line width of the contour lines.
        If a number, all levels will be plotted with this linewidth.
        If a sequence, the levels in ascending order will be plotted with the linewidths in the order specified.
        If None, this falls back to rcParams["lines.linewidth"] (default: 1.5).
        The default is None --> rcParams["lines.linewidth"]
    linestyles : str, optional 
        If linestyles is None, the default is 'solid' unless the lines are monochrome. 
        In that case, negative contours will take their linestyle from 
        rcParams["contour.negative_linestyle"] (default: 'dashed') setting.
    antialiased: bool, optional
        Enable antialiasing, overriding the defaults. 
        The default is taken from rcParams["lines.antialiased"].
    contour_labels_fontsize : string or float, optional
        Size in points or relative size e.g., 'smaller', 'x-large'.    
    contour_labels_colors : color-spec, optional
        If None, the color of each label matches the color of the corresponding contour.
        If one string color, e.g., colors = 'r' or colors = 'red', all contour labels will be plotted in this color.
        If a tuple of matplotlib color args (string, float, rgb, etc),
        different contour labels will be plotted in different colors in the order specified.    
    contour_labels_inline : bool, optional
        If True the underlying contour is removed where the label is placed. Default is True.    
    contour_labels_inline_spacing : float, optional
        Space in pixels to leave on each side of contour label when placing inline. Defaults to 5.
        This spacing will be exact for contour labels at locations where the contour is straight,
        less so for labels on curved contours.    
    contour_labels_fmt : string or dict, optional
        A format string for the label. Default is '%1.3f'
        Alternatively, this can be a dictionary matching contour levels with arbitrary strings 
        to use for each contour level (i.e., fmt[level]=string),
        It can also be any callable, such as a Formatter instance (i.e. `"{:.0f} ".format`), 
        that returns a string when called with a numeric contour level.    
    contour_labels_manual : bool or iterable, optional
        If True, contour labels will be placed manually using mouse clicks. 
        Click the first button near a contour to add a label, 
        click the second button (or potentially both mouse buttons at once) to finish adding labels. 
        The third button can be used to remove the last label added, but only if labels are not inline.
        Alternatively, the keyboard can be used to select label locations
        (enter to end label placement, delete or backspace act like the third mouse button,
         and any other key will select a label location).
        manual can also be an iterable object of x,y tuples.
        Contour labels will be created as if mouse is clicked at each x,y positions.    
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space.
        If not provided, this will be either be ``viridis`` 
        (if the function infers a sequential dataset) or 
        ``RdBu_r`` (if the function infers a diverging dataset).
        Is mutually exclusive with the color argument.
        When `Seaborn` is installed, ``cmap`` may also be a `seaborn`
        color palette. 
        If ``cmap`` is a seaborn color palette, ``levels`` must not be specified.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    norm : ``matplotlib.colors.Normalize`` instance, optional
        If the ``norm`` has vmin or vmax specified, the corresponding kwarg
        must be None.
    colors : discrete colors to plot, optional
        A single color or a list of colors. 
        Is mutually exclusive with cmap argument.
        Specification of ``levels`` argument is not mandatory.
    alpha : float, default: 1
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    extend : {"neither", "both", "min", "max"}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, extend is inferred from vmin, vmax and the data limits.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only used
        for 2D and FacetGrid plots.
    add_colorbar : bool, optional
        Adds colorbar to axis
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar.
    add_labels : bool, optional
        Use xarray metadata to label axes
    **kwargs : optional
        Additional arguments to mpl.collections.PatchCollection
    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped matplotlib
        function returns
    """
    ##------------------------------------------------------------------------.
    # Checks 
    if not isinstance(plot_type, str):
        raise TypeError("'plot_type' must be a string: either 'contour' or 'tricontour'")
    if plot_type not in ['contour','tricontour']:
        raise NotImplementedError("'plot_type' accept only 'contour' or 'tricontour' options.")
    # Check ax
    if ax is None and row is None and col is None: 
        raise ValueError("'ax' must be specified when not plotting a FacetGrids.")
    # Check transform 
    if transform is None:
        transform =ccrs.PlateCarree()
    # Check x and y are coords of the xarray object  
    check_xy(darray, x=x, y=y)
    ##------------------------------------------------------------------------.
    # Handle facetgrids first
    if row or col:
        if subplot_kws is None:
            print("Tip: If you want to plot a map, you need to specify the projection \
                   using the argument subplot_kws={'projection': crs.Robinson()}")
        allargs = locals().copy()
        del allargs["darray"]
        allargs.update(allargs.pop("kwargs"))
        return _easy_facetgrid(data=darray, plotfunc=_contour, **allargs)
    
    ##------------------------------------------------------------------------.
    # Initialize plot
    plt = import_matplotlib_pyplot()
    
    ##------------------------------------------------------------------------.
    # Pass the data as a masked ndarray too
    masked_arr = darray.to_masked_array(copy=False)
    
    ##------------------------------------------------------------------------.
    # Retrieve colormap and colorbar args 
    cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(xr_contour,
                                                         masked_arr.data,
                                                         **locals(),
                                                         _is_facetgrid=kwargs.pop("_is_facetgrid", False))
    ##------------------------------------------------------------------------.
    # If colors == 'a single color', matplotlib draws dashed negative contours.
    # We lose this feature if we pass cmap and not colors
    if isinstance(colors, str):
        cmap_params["cmap"] = None
    ##------------------------------------------------------------------------.
    # Define axis type
    if subplot_kws is None:
        subplot_kws = dict()
        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
    ##------------------------------------------------------------------------.    
    # Retrieve nodes coordinates
    lons = darray[x].values   
    lats = darray[y].values    
    ##------------------------------------------------------------------------.
    # Plot contour 
    if plot_type=="tricontour":
        primitive = ax.tricontour(lons, lats, masked_arr.data,
                                  transform = transform,
                                  # Color options
                                  vmin=cmap_params['vmin'],
                                  vmax=cmap_params['vmax'],
                                  cmap=cmap_params['cmap'],
                                  norm=cmap_params['norm'],
                                  extend=cmap_params["extend"], 
                                  levels=cmap_params["levels"],
                                  colors=colors, 
                                  alpha=alpha,
                                  # Line options
                                  linewidths = linewidths,
                                  linestyles = linestyles,
                                  antialiased = antialiased, 
                                  # Other args 
                                  **kwargs) 
    ##------------------------------------------------------------------------.
    # Plot with contour
    if plot_type=="contour":
        lons_new = np.linspace(-180, 180, 360*2)
        lats_new = np.linspace(-90, 90, 180*2)
        lons_2d, lats_2d = np.meshgrid(lons_new, lats_new)
        data_new = griddata((lons, lats), masked_arr.data, (lons_2d, lats_2d), method='linear')
        # Add a new longitude band at 360. equals to 0. 
        data_new, lons_new = add_cyclic_point(data_new, coord=lons_new)
        # Plot contourf
        primitive = ax.contour(lons_new, lats_new, data_new,
                               transform = transform,
                               # Color options
                               vmin=cmap_params['vmin'],
                               vmax=cmap_params['vmax'],
                               cmap=cmap_params['cmap'],
                               norm=cmap_params['norm'],
                               extend=cmap_params["extend"],
                               levels=cmap_params["levels"],
                               alpha=alpha,           
                               # Line options
                               linewidths = linewidths,
                               linestyles = linestyles,
                               antialiased = antialiased, 
                               **kwargs) 
    # Set global axis 
    ax.set_global()        
    ##------------------------------------------------------------------------.
    # Make the contours line invisible.
    if not add_contour:    
        plt.setp(primitive.collections, visible=False)
    ##------------------------------------------------------------------------.    
    # Add contour labels 
    if add_contour_labels:
        ax.clabel(primitive,
                  colors=contour_labels_colors,
                  fontsize=contour_labels_fontsize,
                  manual=add_contour_labels_interactively,  
                  inline=contour_labels_inline,  
                  inline_spacing=contour_labels_inline_spacing, 
                  fmt=contour_labels_format)   
    
    # Set global 
    ax.set_global()                
    ##------------------------------------------------------------------------.                                                            
    # Add labels 
    if add_labels:
        ax.set_title(darray._title_for_slice())
        
    ##------------------------------------------------------------------------.      
    # Add colorbar
    if add_colorbar:
        if add_labels and "label" not in cbar_kwargs:
            cbar_kwargs["label"] = label_from_attrs(darray)
            cbar = _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
        elif cbar_ax is not None or cbar_kwargs:
            # inform the user about keywords which aren't used
            raise ValueError("cbar_ax and cbar_kwargs can't be used with add_colorbar=False.")
 
    ##------------------------------------------------------------------------. 
    return primitive 

def _contourf(darray,
              x='lon',
              y='lat',
              transform=None,
              # Facetgrids arguments
              figsize=None,
              size=None,
              aspect=None,
              ax=None,
              row=None,
              col=None,
              col_wrap=None,
              subplot_kws=None,
              # Colors option
              plot_type="contourf",
              antialiased=True, 
              alpha=1,
              colors=None,
              levels=None, 
              cmap=None,
              norm=None,
              center=None,
              vmin=None,
              vmax=None,
              robust=False,
              extend='both',
              # Colorbar options
              add_colorbar=None,
              cbar_ax=None,
              cbar_kwargs=None,
              # Axis options
              add_labels=True,
              **kwargs):
    """
    Contourf plotting method for unstructured mesh.
    
    The DataArray must have the attribute 'nodes'.
        
    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional, unless creating faceted plots.
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots
    add_colorbar : bool, optional
        Adds colorbar to axis
    add_labels : bool, optional
        Use xarray metadata to label axes
    antialiased: bool, optional
        Enable antialiasing, overriding the defaults. For filled contours, the default is True.   
    plot_type : str, optional
        Whether to use the "contourf" or "tricontourf" function.
        The default is "contour".
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space.
        If not provided, this will be either be ``viridis`` 
        (if the function infers a sequential dataset) or 
        ``RdBu_r`` (if the function infers a diverging dataset).
        Is mutually exclusive with the color argument.
        When `Seaborn` is installed, ``cmap`` may also be a `seaborn`
        color palette. 
        If ``cmap`` is a seaborn color palette, ``levels`` must not be specified.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    norm : ``matplotlib.colors.Normalize`` instance, optional
        If the ``norm`` has vmin or vmax specified, the corresponding kwarg
        must be None.
    colors : discrete colors to plot, optional
        A single color or a list of colors. 
        Is mutually exclusive with cmap argument.
        Specification of ``levels`` argument is not mandatory.
    alpha : float, default: 1
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    extend : {"neither", "both", "min", "max"}, optional
        Determines the contourf-coloring of values that are outside the levels range and 
        wheter to draw arrows extending the colorbar beyond its limits.
        If 'neither' (the default), values outside the levels range are not colored.
        If 'min', 'max' or 'both', color the values below, above or below and above the levels range.
        Values below min(levels) and above max(levels) are mapped to the under/over 
        values of the Colormap.
        Note that most colormaps do not have dedicated colors for these by default,
        so that the over and under values are the edge values of the colormap. 
        You may want to set these values explicitly using Colormap.set_under and Colormap.set_over.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only used
        for 2D and FacetGrid plots.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar.
    **kwargs : optional
        Additional arguments to mpl.collections.PatchCollection
    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped matplotlib
        function returns
    """
    # Checks plot_type
    if not isinstance(plot_type, str):
        raise TypeError("'plot_type' must be a string: either 'contourf' or 'tricontourf'")
    if plot_type not in ['contourf','tricontourf']:
        raise NotImplementedError("'plot_type' accept only 'contourf' or 'tricontourf' options.")
    # Check ax
    if ax is None and row is None and col is None: 
        raise ValueError("'ax' must be specified when not plotting a FacetGrids.")
    # Check transform 
    if transform is None:
        transform =ccrs.PlateCarree()
    # Check x and y are coords of the xarray object  
    check_xy(darray, x=x, y=y)
    ##------------------------------------------------------------------------.
    # Handle facetgrids first
    if row or col:
        if subplot_kws is None:
            print("Tip: If you want to plot a map, you need to specify the projection \
                   using the argument subplot_kws={'projection': crs.Robinson()}")
        allargs = locals().copy()
        del allargs["darray"]
        allargs.update(allargs.pop("kwargs"))
        return _easy_facetgrid(data=darray, plotfunc=_contourf, **allargs)
    
    ##------------------------------------------------------------------------.
    # Initialize plot
    plt = import_matplotlib_pyplot()
    
    ##------------------------------------------------------------------------.
    # Pass the data as a masked ndarray too
    masked_arr = darray.to_masked_array(copy=False)
    
    ##------------------------------------------------------------------------.
    # Retrieve colormap and colorbar args 
    cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(xr_contourf,
                                                         masked_arr.data,
                                                         **locals(),
                                                         _is_facetgrid=kwargs.pop("_is_facetgrid", False))
    ##------------------------------------------------------------------------.
    # If colors == 'a single color', matplotlib draws dashed negative contours.
    # We lose this feature if we pass cmap and not colors
    if isinstance(colors, str):
        cmap_params["cmap"] = None
        kwargs["colors"] = colors
        
    ##------------------------------------------------------------------------.
    # Define axis type
    if subplot_kws is None:
        subplot_kws = dict()
        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
    ##------------------------------------------------------------------------.    
    # Retrieve nodes coordinates
    lons = darray[x].values   
    lats = darray[y].values 
    # Plot with tricontourf
    if plot_type=="tricontourf":
        primitive = ax.tricontourf(lons, lats, masked_arr.data, 
                                   transform = transform, 
                                   # Color options
                                   vmin=cmap_params['vmin'],
                                   vmax=cmap_params['vmax'],
                                   cmap=cmap_params['cmap'],
                                   norm=cmap_params['norm'],
                                   extend=cmap_params["extend"],
                                   levels=cmap_params["levels"],
                                   alpha=alpha,           
                                   antialiased=antialiased, 
                                   **kwargs) 
    # Plot with contourf
    if plot_type=="contourf":
        lons_new = np.linspace(-180, 180, 360*2)
        lats_new = np.linspace(-90, 90, 180*2)
        lons_2d, lats_2d = np.meshgrid(lons_new, lats_new)
        data_new = griddata((lons, lats), masked_arr.data, (lons_2d, lats_2d), method='linear')
        # Add a new longitude band at 360. equals to 0. 
        data_new, lons_new = add_cyclic_point(data_new, coord=lons_new)
        # Plot contourf
        primitive = ax.contourf(lons_new, lats_new, data_new,
                                transform = transform,
                                # Color options
                                vmin=cmap_params['vmin'],
                                vmax=cmap_params['vmax'],
                                cmap=cmap_params['cmap'],
                                norm=cmap_params['norm'],
                                extend=cmap_params["extend"],
                                levels=cmap_params["levels"],
                                alpha=alpha,           
                                antialiased=antialiased, 
                                **kwargs) 
    # Set global axis 
    ax.set_global()                
    ##------------------------------------------------------------------------.                                                            
    # Add labels 
    if add_labels:
        ax.set_title(darray._title_for_slice())
        
    ##------------------------------------------------------------------------.      
    # Add colorbar
    if add_colorbar:
        if add_labels and "label" not in cbar_kwargs:
            cbar_kwargs["label"] = label_from_attrs(darray)
            cbar = _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
        elif cbar_ax is not None or cbar_kwargs:
            # inform the user about keywords which aren't used
            raise ValueError("cbar_ax and cbar_kwargs can't be used with add_colorbar=False.")
 
    ##------------------------------------------------------------------------. 
    return primitive

def _plot(darray,
        transform=None,
        # Facetgrids arguments
        figsize=None,
        size=None,
        aspect=None,
        ax=None,
        row=None,
        col=None,
        col_wrap=None,
        subplot_kws=None,
        # Polygon border option
        edgecolors="white",
        linewidths=0.1,
        antialiased=True,
        # Colors option    
        colors=None,
        levels=None, 
        cmap=None,
        norm=None,
        center=None,
        vmin=None,
        vmax=None,
        robust=False,
        extend='both',
        # Colorbar options
        add_colorbar=None,
        cbar_ax=None,
        cbar_kwargs=None,
        # Axis options
        add_labels=True,
        **kwargs):
    """
    Plot the unstructured mesh using mpl.PatchCollection.
    
    The DataArray must have the attribute 'mesh' containing a mpl PolyPatch list.
        
    Parameters
    ----------
    darray : DataArray
        Must be 2 dimensional, unless creating faceted plots.
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches.
        Mutually exclusive with ``size`` and ``ax``.
    aspect : scalar, optional
        Aspect ratio of plot, so that ``aspect * size`` gives the width in
        inches. Only used if a ``size`` is provided.
    size : scalar, optional
        If provided, create a new figure for the plot with the given size.
        Height (in inches) of each plot. See also: ``aspect``.
    ax : matplotlib axes object, optional
        Axis on which to plot this figure. By default, use the current axis.
        Mutually exclusive with ``size`` and ``figsize``.
    row : string, optional
        If passed, make row faceted plots on this dimension name
    col : string, optional
        If passed, make column faceted plots on this dimension name
    col_wrap : int, optional
        Use together with ``col`` to wrap faceted plots
    add_colorbar : bool, optional
        Adds colorbar to axis
    add_labels : bool, optional
        Use xarray metadata to label axes
    vmin, vmax : floats, optional
        Values to anchor the colormap, otherwise they are inferred from the
        data and other keyword arguments. When a diverging dataset is inferred,
        setting one of these values will fix the other by symmetry around
        ``center``. Setting both values prevents use of a diverging colormap.
        If discrete levels are provided as an explicit list, both of these
        values are ignored.
    robust : bool, optional
        If True and ``vmin`` or ``vmax`` are absent, the colormap range is
        computed with 2nd and 98th percentiles instead of the extreme values.
    levels : int or list-like object, optional
        Split the colormap (cmap) into discrete color intervals. If an integer
        is provided, "nice" levels are chosen based on the data range: this can
        imply that the final number of levels is not exactly the expected one.
        Setting ``vmin`` and/or ``vmax`` with ``levels=N`` is equivalent to
        setting ``levels=np.linspace(vmin, vmax, N)``.
    cmap : matplotlib colormap name or object, optional
        The mapping from data values to color space.
        If not provided, this will be either be ``viridis`` 
        (if the function infers a sequential dataset) or 
        ``RdBu_r`` (if the function infers a diverging dataset).
        Is mutually exclusive with the color argument.
        When `Seaborn` is installed, ``cmap`` may also be a `seaborn`
        color palette. 
        If ``cmap`` is seaborn color palette, ``levels`` must also be specified.
    norm : ``matplotlib.colors.Normalize`` instance, optional
        If the ``norm`` has vmin or vmax specified, the corresponding kwarg
        must be None.
    colors : discrete colors to plot, optional
        A single color or a list of colors. 
        Is mutually exclusive with cmap argument.
        Specification of ``levels`` argument is mandatory.
    center : float, optional
        The value at which to center the colormap. Passing this value implies
        use of a diverging colormap. Setting it to ``False`` prevents use of a
        diverging colormap.
    extend : {"neither", "both", "min", "max"}, optional
        How to draw arrows extending the colorbar beyond its limits. If not
        provided, extend is inferred from vmin, vmax and the data limits.
    subplot_kws : dict, optional
        Dictionary of keyword arguments for matplotlib subplots. Only used
        for 2D and FacetGrid plots.
    cbar_ax : matplotlib Axes, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar.
    **kwargs : optional
        Additional arguments to mpl.collections.PatchCollection
    Returns
    -------
    artist :
        The same type of primitive artist that the wrapped matplotlib
        function returns
    """
    # Check ax
    if ax is None and row is None and col is None: 
        raise ValueError("'ax' must be specified when not plotting a FacetGrids.")
    # Check transform 
    if transform is None:
        transform = ccrs.Geodetic()
    # Check mesh is available 
    check_mesh_exist(darray)
    ##------------------------------------------------------------------------.
    # Handle facetgrids first
    if row or col:
        if subplot_kws is None:
            print("Tip: If you want to plot a map, you need to specify the projection \
                   using the argument subplot_kws={'projection': crs.Robinson()}")
        allargs = locals().copy()
        del allargs["darray"]
        allargs.update(allargs.pop("kwargs"))
        return _easy_facetgrid(data=darray, plotfunc=_plot, **allargs)
    ##------------------------------------------------------------------------.
    # Initialize plot
    plt = import_matplotlib_pyplot()
    
    ##------------------------------------------------------------------------.
    # Pass the data as a masked ndarray too
    masked_arr = darray.to_masked_array(copy=False)
    
    ##------------------------------------------------------------------------.
    # Retrieve colormap and colorbar args 
    cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(xr_pcolormesh,
                                                         masked_arr.data,
                                                         **locals(),
                                                         _is_facetgrid=kwargs.pop("_is_facetgrid", False))
    ##------------------------------------------------------------------------.
    # Define axis type
    if subplot_kws is None:
        subplot_kws = dict()
        ax = get_axis(figsize, size, aspect, ax, **subplot_kws)
    ##------------------------------------------------------------------------.    
    # Create Polygon Patch Collection
    patch_list = darray['mesh'].values.tolist()  
    Polygon_Collection = mpl.collections.PatchCollection(patch_list,
                                                         transform = transform,
                                                         array = masked_arr,
                                                         # Polygon border                                                        
                                                         edgecolors = edgecolors,
                                                         linewidths = linewidths,
                                                         antialiaseds = antialiased,
                                                         # Color options
                                                         cmap=cmap_params["cmap"],
                                                         clim=(cmap_params["vmin"], cmap_params["vmax"]),
                                                         norm=cmap_params["norm"],
                                                         **kwargs)
    # Plot polygons
    ax.set_global()
    primitive = ax.add_collection(Polygon_Collection) 
                       
    ##------------------------------------------------------------------------.                                                            
    # Add labels 
    if add_labels:
        ax.set_title(darray._title_for_slice())
        
    ##------------------------------------------------------------------------.      
    # Add colorbar
    if add_colorbar:
        if add_labels: 
            if "label" not in cbar_kwargs:
                cbar_kwargs["label"] = label_from_attrs(darray)
            cbar = _add_colorbar(primitive, ax, cbar_ax, cbar_kwargs, cmap_params)
        elif cbar_ax is not None or cbar_kwargs:
            # inform the user about keywords which aren't used
            raise ValueError("cbar_ax and cbar_kwargs can't be used with add_colorbar=False.")
 
    ##------------------------------------------------------------------------. 
    return primitive 

def _plot_mesh(darray,
               ax,
               transform = None,
               add_background = True, 
               antialiaseds = True,
               facecolors = 'none',
               edgecolors = "black",
               linewidths = 0.5,
               alpha = 0.8,
               **kwargs):  
    """Plot the unstructured mesh.
    
    The DataArray must have the coordinate 'mesh' containing a mpl PolyPatch list.
    """
    # Check ax
    if ax is None: 
        raise ValueError("'ax' must be specified when not plotting a FacetGrids.")
    # Check mesh is available 
    check_mesh_exist(darray)   
    # Check transform 
    if transform is None:
        transform = ccrs.Geodetic()
    ##------------------------------------------------------------------------.
    # Retrieve mesh 
    patch_list = darray['mesh'].values.tolist()                                                                                                           
    # Create PatchCollection
    Polygon_Collection = mpl.collections.PatchCollection(patch_list,
                                                         transform = transform,
                                                         antialiaseds = antialiaseds,
                                                         facecolors = facecolors,
                                                         alpha = alpha, 
                                                         edgecolors = edgecolors,
                                                         linewidths = linewidths,
                                                         **kwargs) 
    # Plot the background     
    ax.set_global()
    if add_background:
        ax.stock_img()
    # Plot the mesh   
    primitive = ax.add_collection(Polygon_Collection)
    return primitive

def _plot_mesh_order(darray, ax, 
                     transform=None,
                    # Polygon border option
                    edgecolors="white",
                    linewidths=0.1,
                    antialiased=True,
                    # Colors option    
                    colors=None,
                    levels=None, 
                    cmap=None,
                    norm=None,
                    center=None,
                    vmin=None,
                    vmax=None,
                    robust=False,
                    extend='neither',
                    # Colorbar options
                    add_colorbar=True,
                    cbar_ax=None,
                    cbar_kwargs=None,
                    # Axis options
                    add_labels=True,
                    **kwargs):
    """Plot the unstructured mesh order.
    
    The DataArray must have the coordinate 'mesh' containing a mpl PolyPatch list.
    """
    # Check mesh is available 
    check_mesh_exist(darray)
    da = darray
    # Select 1 index in all dimensions (except node ...)
    dims = list(da.dims)
    node_dim = list(da['mesh'].dims)
    other_dims = np.array(dims)[np.isin(dims, node_dim, invert=True)].tolist()
    for dim in other_dims: 
        da = da.isel({dim: 0})
    # Replace values with node order ...
    da.values = np.array(range(len(da['mesh'].values)))
    # Specify colorbar title
    if cbar_kwargs is None and add_colorbar: 
        cbar_kwargs = {"label": "Mesh order ID"}   
    # Plot mesh order 
    primitive = da.sphere.plot(ax=ax, 
                               transform=transform,
                               # Polygon border option
                               edgecolors=edgecolors,
                               linewidths=linewidths,
                               antialiased=antialiased,
                               # Colors option    
                               colors=colors,
                               levels=levels, 
                               cmap=cmap,
                               norm=norm,
                               center=center,
                               vmin=vmin,
                               vmax=vmax,
                               robust=robust,
                               extend=extend,
                               # Colorbar options
                               add_colorbar=add_colorbar,
                               cbar_ax=cbar_ax,
                               cbar_kwargs=cbar_kwargs,
                               # Axis options
                               add_labels=add_labels,
                               **kwargs)
    return primitive

def _plot_mesh_area(darray, ax, 
                    transform=None,
                    mesh_area_coord='area',
                    # Polygon border option
                    edgecolors="white",
                    linewidths=0.1,
                    antialiased=True,
                    # Colors option    
                    colors=None,
                    levels=None, 
                    cmap=None,
                    norm=None,
                    center=None,
                    vmin=None,
                    vmax=None,
                    robust=False,
                    extend='both',
                    # Colorbar options
                    add_colorbar=True,
                    cbar_ax=None,
                    cbar_kwargs=None,
                    # Axis options
                    add_labels=True,
                    **kwargs):
    """Plot the unstructured mesh area.
    
    The DataArray must have the coordinate 'mesh' containing a mpl PolyPatch list.
    """
    # Check mesh is available 
    check_mesh_area_exist(darray, mesh_area_coord=mesh_area_coord) 
    da = darray
    # Select 1 index in all dimensions (except node ...)
    dims = list(da.dims)
    node_dim = list(da[mesh_area_coord].dims)
    other_dims = np.array(dims)[np.isin(dims, node_dim, invert=True)].tolist()
    for dim in other_dims: 
        da = da.isel({dim: 0})
    # Replace values with node order ...
    da.values = da[mesh_area_coord].values
    # Specify colorbar title
    if cbar_kwargs is None and add_colorbar: 
        cbar_kwargs = {"label": "Area"}    
    # Plot mesh order 
    primitive = da.sphere.plot(ax=ax,
                               transform=transform,
                               # Polygon border option
                               edgecolors=edgecolors,
                               linewidths=linewidths,
                               antialiased=antialiased,
                               # Colors option    
                               colors=colors,
                               levels=levels, 
                               cmap=cmap,
                               norm=norm,
                               center=center,
                               vmin=vmin,
                               vmax=vmax,
                               robust=robust,
                               extend=extend,
                               # Colorbar options
                               add_colorbar=add_colorbar,
                               cbar_ax=cbar_ax,
                               cbar_kwargs=cbar_kwargs,
                               # Axis options
                               add_labels=add_labels,
                               **kwargs)
    return primitive

def _plot_nodes(darray, ax,
                x='lon', y='lat', 
                c="orange", 
                add_background=True,
                **kwargs):
    # Check x and y are coords of the xarray object  
    check_xy(darray, x=x, y=y)
    # Retrieve nodes coordinates
    lons = darray[x].values   
    lats = darray[y].values  
    # Add background
    if add_background:
        ax.stock_img()
    # Plot nodes 
    primitive = ax.scatter(lons, lats, s=0.5, c=c, **kwargs)
    return primitive 

#-----------------------------------------------------------------------------.    
#### Accessors 
@xr.register_dataarray_accessor("sphere")
class SphereDataArrayAccessor:
    """xarray.sphere DataArray accessor."""
    
    def __init__(self, da):
        self.da = da

    def add_nodes(self, lon, lat, node_dim="node"):
        """Add unstructured grid nodes."""
        # Check node_dim  
        node_dim = check_node_dim(self.da, node_dim=node_dim)
        # Ensure lon is between -180 and 180
        lon[lon > 180] = lon[lon > 180] - 360
        # Add nodes coordinates 
        self.da = self.da.assign_coords({'lon':(node_dim, lon)})
        self.da = self.da.assign_coords({'lat':(node_dim, lat)})
        return self.da 
    
    def add_nodes_from_pygsp(self, pygsp_graph, node_dim="node"):
        """Add nodes from the spherical pygsp graph."""
        # Retrieve lon and lat coordinates
        pygsp_graph.set_coordinates('sphere', dim=2)
        lat = np.rad2deg(pygsp_graph.coords[:,1])
        lon = np.rad2deg(pygsp_graph.coords[:,0])
        # Ensure lon is between -180 and 180
        lon[lon > 180] = lon[lon > 180] - 360
        return self.add_nodes(lon=lon, lat=lat, node_dim=node_dim)
        
    def add_mesh(self, mesh, node_dim='node'):
        """Add unstructured grid mesh."""
        # Check node_dim  
        node_dim = check_node_dim(self.da, node_dim=node_dim)
        # Check mesh 
        mesh = check_mesh(mesh)
        # Add mesh as xarray coordinate 
        self.da = self.da.assign_coords({'mesh':(node_dim, mesh)})
        return self.da 
    
    def add_mesh_area(self, area, node_dim='node'):
        """Add unstructured grid mesh area."""
        # Check node_dim  
        node_dim = check_node_dim(self.da, node_dim=node_dim)
        # Add mesh as xarray coordinate 
        self.da = self.da.assign_coords({'area':(node_dim, area)})
        return self.da 
    
    def compute_mesh_area(self):
        """Compute the mesh area."""
        # TODO: Improve for computing on the sphere... now using shapely planar assumption
        # Scipy - Spherical? https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/spatial/_spherical_voronoi.py#L266
        ##---------------------------------------------------------------------.
        # Check the mesh exist 
        check_mesh_exist(self.da)
        # Retrieve mesh 
        mesh = self.da['mesh'].values
        # Retrieve node_dim 
        node_dim = list(self.da['mesh'].dims)[0]
        # Compute area (with shapely ... planar assumption)
        area = [shapely.geometry.Polygon(p.xy).area for p in mesh]
        return self.da.sphere.add_mesh_area(area=area, node_dim=node_dim)
    
    def add_SphericalVoronoiMesh(self, x='lon', y='lat', add_area=True):
        """Infer the mesh using Spherical Voronoi using node coordinates."""
        # Check x and y are coords of the xarray object  
        check_xy(self.da, x=x, y=y)
        # Retrieve node coordinates 
        lon = self.da[x].values 
        lat = self.da[y].values
        node_dim = list(self.da[x].dims)[0]
        # Compute SphericalVoronoi Mesh 
        list_polygons_lonlat, area = SphericalVoronoiMesh(lon=lon, lat=lat)
        mesh = get_PolygonPatchesList(list_polygons_lonlat) 
        # Add mesh
        self.add_mesh(mesh=mesh, node_dim=node_dim) 
        # Add mesh area 
        if add_area:                                                     
            self.add_mesh_area(area=area, node_dim=node_dim) 
        return self.da
        
    def plot(self, *args, **kwargs):
        """Map unstructured grid values."""
        p = _plot(self.da, *args, **kwargs)
        return p 
    
    def contour(self, *args, **kwargs):
        """Contour of unstructured grid values."""
        p = _contour(self.da, *args, **kwargs)
        return p 
    
    def contourf(self, *args, **kwargs):
        """Contourf of unstructured grid values."""
        p = _contourf(self.da, *args, **kwargs)
        return p 
        
    def plot_mesh(self, *args, **kwargs):  
        """Plot the unstructured grid mesh structure."""
        p = _plot_mesh(self.da, *args, **kwargs)
        return p 
    
    def plot_mesh_order(self, *args, **kwargs):  
        """Plot the unstructured grid mesh order."""
        p = _plot_mesh_order(self.da, *args, **kwargs)
        return p 
    
    def plot_mesh_area(self, *args, **kwargs):  
        """Plot the unstructured grid mesh area."""
        p = _plot_mesh_area(self.da, *args, **kwargs)
        return p 
    
    def plot_nodes(self, *args, **kwargs):  
        """Plot the unstructured grid nodes."""
        p = _plot_nodes(self.da, *args, **kwargs)
        return p 
    
@xr.register_dataset_accessor("sphere")
class SphereDatasetAccessor:
    """xarray.sphere Dataset accessor."""
    
    def __init__(self, ds):
        self.ds = ds
    
    def add_nodes(self, lon, lat, node_dim="node"):
        """Add unstructured grid nodes."""
        # Check node_dim  
        node_dim = check_node_dim(self.ds, node_dim=node_dim)
        # Add nodes coordinates 
        self.ds = self.ds.assign_coords({'lon':(node_dim, lon)})
        self.ds = self.ds.assign_coords({'lat':(node_dim, lat)})
        return self.ds 
    
    def add_nodes_from_pygsp(self, pygsp_graph, node_dim="node"):
        """Add nodes from the spherical pygsp graph."""
        # Retrieve lon and lat coordinates
        pygsp_graph.set_coordinates('sphere', dim=2)
        lon = np.rad2deg(pygsp_graph.coords[:,0])
        lat = np.rad2deg(pygsp_graph.coords[:,1])
         # Ensure lon is between -180 and 180
        lon[lon > 180] = lon[lon > 180] - 360
        return self.add_nodes(lon=lon, lat=lat, node_dim=node_dim)  
    
    def add_mesh(self, mesh, node_dim='node'):
        """Add unstructured grid mesh."""
        # Check node_dim  
        node_dim = check_node_dim(self.ds, node_dim=node_dim)
        # Check mesh 
        mesh = check_mesh(mesh)
        # Add mesh as xarray coordinate 
        self.ds = self.ds.assign_coords({'mesh':(node_dim, mesh)})
        return self.ds
    
    def add_mesh_area(self, area, node_dim='node'):
        """Add unstructured grid mesh area."""
        # Check node_dim  
        node_dim = check_node_dim(self.ds, node_dim=node_dim)
        # Add mesh as xarray coordinate 
        self.ds = self.ds.assign_coords({'area':(node_dim, area)})
        return self.ds
   
    def compute_mesh_area(self):
        """Compute the mesh area."""
        # TODO: Improve for computing on the sphere... now using shapely planar assumption
        # Scipy - Spherical? https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/spatial/_spherical_voronoi.py#L266
        ##---------------------------------------------------------------------.
        # Check the mesh exist 
        check_mesh_exist(self.ds)
        # Retrieve mesh 
        mesh = self.ds['mesh'].values
        # Retrieve node_dim 
        node_dim = list(self.ds['mesh'].dims)[0]
        # Compute area (with shapely ... planar assumption)
        area = [shapely.geometry.Polygon(p.xy).area for p in mesh]
        return self.ds.sphere.add_mesh_area(area=area, node_dim=node_dim)

    def add_SphericalVoronoiMesh(self, x='lon', y='lat', add_area=True):
        """Compute the Spherical Voronoi mesh using the node coordinates.""" 
        # Check x and y are coords of the xarray object  
        check_xy(self.ds, x=x, y=y)
        # Retrieve node coordinates
        lon = self.ds[x].values 
        lat = self.ds[y].values
        node_dim = list(self.ds[x].dims)[0]
        # Compute SphericalVoronoi Mesh 
        list_polygons_lonlat, area = SphericalVoronoiMesh(lon=lon, lat=lat)
        mesh = get_PolygonPatchesList(list_polygons_lonlat) 
        # Add mesh
        self.add_mesh(mesh=mesh, node_dim=node_dim) 
        # Add mesh area
        if add_area:                                                     
            self.add_mesh_area(area=area, node_dim=node_dim) 
        return self.ds
    
    def plot(self, col=None, row=None, *args, **kwargs):
        """Map unstructured grid values."""
        ds = self.ds
        # Check number of variable 
        list_vars = list(ds.data_vars.keys())  
        n_vars = len(list_vars)
        # If only 1 variable, treat as DataArray
        if n_vars == 1:
            return ds[list_vars[0]].sphere.plot(col=col, row=row, *args, **kwargs)
        # Otherwise
        if col is not None and row is not None: 
            raise ValueError("When plotting a Dataset, you must specify either 'row' or 'col'.")
        if col is None and row is None: 
            raise NotImplementedError("When 'col' and 'row' are both None.")
        # Squeeze the dataset (to drop dim with 1)
        ds = self.ds.squeeze()
        # Check remaining dimension
        if len(ds.dims) > 2: 
            raise ValueError("There must be just 1 dimension to facet (in addition to the 'node' dimension).")
        # Convert to DataArray
        da = self.ds.to_array()  
        if col is not None:
            p = da.sphere.plot(row="variable",col=col *args, **kwargs)
            return p 
        elif row is not None: 
            p = da.sphere.plot(col="variable",row=row,*args, **kwargs)
            return p 
        else:
            raise NotImplementedError("When 'col' and 'row' are both None (END).")
            
    def contour(self, col=None, row=None, *args, **kwargs):
        """Contour of unstructured grid values."""
        ds = self.ds
        # Check number of variable 
        list_vars = list(ds.data_vars.keys())  
        n_vars = len(list_vars)
        # If only 1 variable, treat as DataArray
        if n_vars == 1:
            return ds[list_vars[0]].sphere.contour(col=col, row=row, *args, **kwargs)
        # Otherwise
        if col is not None and row is not None: 
            raise ValueError("When contourting a Dataset, you must specify either 'row' or 'col'.")
        if col is None and row is None: 
            raise NotImplementedError("When 'col' and 'row' are both None.")
        # Squeeze the dataset (to drop dim with 1)
        ds = self.ds.squeeze()
        # Check remaining dimension
        if len(ds.dims) > 2: 
            raise ValueError("There must be just 1 dimension to facet (in addition to the 'node' dimension).")
        # Convert to DataArray
        da = self.ds.to_array()  
        if col is not None:
            p = da.sphere.contour(row="variable",col=col *args, **kwargs)
            return p 
        elif row is not None: 
            p = da.sphere.contour(col="variable",row=row,*args, **kwargs)
            return p 
        else:
            raise NotImplementedError("When 'col' and 'row' are both None (END).")   

    def contourf(self, col=None, row=None, *args, **kwargs):
        """Contourf of unstructured grid values."""
        ds = self.ds
        # Check number of variable 
        list_vars = list(ds.data_vars.keys())  
        n_vars = len(list_vars)
        # If only 1 variable, treat as DataArray
        if n_vars == 1:
            return ds[list_vars[0]].sphere.contourf(col=col, row=row, *args, **kwargs)
        # Otherwise
        if col is not None and row is not None: 
            raise ValueError("When contourfting a Dataset, you must specify either 'row' or 'col'.")
        if col is None and row is None: 
            raise NotImplementedError("When 'col' and 'row' are both None.")
        # Squeeze the dataset (to drop dim with 1)
        ds = self.ds.squeeze()
        # Check remaining dimension
        if len(ds.dims) > 2: 
            raise ValueError("There must be just 1 dimension to facet (in addition to the 'node' dimension).")
        # Convert to DataArray
        da = self.ds.to_array()  
        if col is not None:
            p = da.sphere.contourf(row="variable",col=col *args, **kwargs)
            return p 
        elif row is not None: 
            p = da.sphere.contourf(col="variable",row=row,*args, **kwargs)
            return p 
        else:
            raise NotImplementedError("When 'col' and 'row' are both None (END).")
            
    def plot_mesh(self, *args, **kwargs):  
        """Plot the unstructured grid mesh structure."""
        da = self.ds[list(self.ds.data_vars.keys())[0]]
        p = _plot_mesh(da, *args, **kwargs)
        return p 
    
    def plot_mesh_order(self, *args, **kwargs):  
        """Plot the unstructured grid mesh order."""
        da = self.ds[list(self.ds.data_vars.keys())[0]]
        p = _plot_mesh_order(da, *args, **kwargs)
        return p 
    
    def plot_mesh_area(self, *args, **kwargs):  
        """Plot the unstructured grid mesh area."""
        da = self.ds[list(self.ds.data_vars.keys())[0]]
        p = _plot_mesh_area(da, *args, **kwargs)
        return p 
    
    def plot_nodes(self, *args, **kwargs):  
        """Plot the unstructured grid nodes."""
        da = self.ds[list(self.ds.data_vars.keys())[0]]
        p = _plot_nodes(da, *args, **kwargs)
        return p
