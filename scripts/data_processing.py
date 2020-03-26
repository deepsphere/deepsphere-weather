import argparse
from pathlib import Path

from modules.data import preprocess_equiangular, preprocess_healpix

def main(input_dir, output_dir, train_years, val_years, test_years, samplings, nside, interpolation_kind):
    """
    Parameters
    ----------
    
    input_dir : str
        Input data directory 
    output_dir : str
        Output directory
    train_years : slice(str)
        Years used for training set
    val_years : slice(str)
        Years used for validation set
    test_years : slice(str)
        Years used for testing set
    samplings : list(str)
        List containing the desired sampling method on the sphere. Options are ´equiangular´ and/or ´healpix´
    nside : int
        Number of subdivisions of the HEALpix original cell
    interpolation_kind : str
        Interpolation kind for HEALPIX regridding
    """
    
    if "equiangular" in samplings:
        out_path = out_data + "equiangular/"
        Path(out_path).mkdir(parents=True, exist_ok=True)
        preprocess_equiangular(input_dir, output_dir, train_years, val_years, test_years)
        
    if "healpix" in samplings: 
        out_path = out_data + "healpix/"
        Path(out_path).mkdir(parents=True, exist_ok=True)
        preprocess_healpix(input_dir, output_dir, train_years, val_years, test_years, nside, interpolation_kind)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, nargs='+', required=True,
                        help="Input data directory")
    
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory")
    
    parser.add_argument('--train_years', type=slice, required=True, 
                        help="Training set years")
    
    parser.add_argument('--val_years', type=slice, required=True, 
                        help="Validation set years" )
    
    parser.add_argument('--test_years', type=slice, required=True,
                        help="Desired sampling method on the sphere. Options are ´equiangular´ and/or ´healpix´")
    
    parser.add_argument('--samplings', type=list, required=True, 
                        help="File ending. Default = nc")
    
    parser.add_argument('--nside', type=int, default=16,
                        help="Number of subdivisions of the HEALpix original cell")
    
    parser.add_argument('--interpolation_kind', type=str, default='linear',
                        help="Interpolation kind for HEALPIX regridding")
    
    args = parser.parse_args()

    
    main(input_dir=args.input_dir,
         output_dir=args.output_dir,
         train_years=args.train_years,
         val_years=args.val_years,
         test_years=args.test_years,
         samplings=args.samplings, 
         nside=args.nside, 
         interpolation_kind=args.interpolation_kind)