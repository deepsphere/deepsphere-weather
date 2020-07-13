import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))


import argparse
import xarray as xr
import numpy as np
import os
from pathlib import Path
import healpy as hp

def main(input_dir, output_dir, nside, interpolation_kind):
    """
    Parameters
    ----------
    
    input_dir : str
        Input data directory 
    output_dir : str
        Output directory
    nside : int
        Number of subdivisions of the HEALpix original cell
    interpolation_kind : str
        Interpolation kind for HEALPIX regridding
    """
    
    # New HEALPix grid
    n_pixels = 12*(nside**2)
    hp_lon, hp_lat = hp.pix2ang(nside, np.arange(n_pixels), lonlat=True, nest=True)
    lon_idx = xr.DataArray(hp_lon, dims=["lon"])
    lat_idx = xr.DataArray(hp_lat, dims=["lat"])

    # Interpolate and save all atmospheric fields
    all_files = os.listdir(input_dir)
    all_files = ['toa_incident_solar_radiation']
    for f in all_files:
        # Interpolate
        ds = xr.open_mfdataset(input_dir + f + '/*.nc', combine='by_coords')
        interp_ds = ds.interp(lon=('node', lon_idx), lat=('node', lat_idx),
                              method=interpolation_kind).interpolate_na(dim='node')
        interp_ds = interp_ds.assign_coords(node=np.arange(n_pixels))

        # Create out folder
        out_path =  output_dir + f + "/"
        Path(out_path).mkdir(parents=True, exist_ok=True)

        # Save
        out_filename = f + '_5.625deg.nc'
        interp_ds.to_netcdf(out_path + out_filename)
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, nargs='+', required=True,
                        help="Input data directory")
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory")
    
    parser.add_argument('--nside', type=int, default=16,
                        help="Number of subdivisions of the HEALpix original cell")
    
    parser.add_argument('--interpolation_kind', type=str, default='linear',
                        help="Interpolation kind for HEALPIX regridding")
    
    args = parser.parse_args()
    
    main(input_dir=args.input_dir,
         output_dir=args.output_dir,
         nside=args.nside, 
         interpolation_kind=args.interpolation_kind)