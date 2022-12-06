



# DeepSphere-Weather - Deep Learning on the sphere for weather/climate applications.

![weather forecast](./figs/Forecast_State_Errors.gif)

https://user-images.githubusercontent.com/19285200/132585484-8202b624-e487-440f-8ed6-7ed182a1f31b.mp4

The code in this repository provides a scalable and flexible framework to apply convolutions on spherical unstructured grids for weather/climate applications.

ATTENTION: The code is subject to changes in the coming weeks / months.

The folder `experiments` (will) provide examples for:
-  Weather forecasting using autoregressive CNN models
-  Weather field downscaling (aka superesolution) [in preparation].
-  Classication of atmospheric features (i.e. tropical cyclone and atmospheric rivers) [in preparation].

The folder `tutorials` (will) provide jupyter notebooks describing various features of DeepSphere-Weather.

The folder `docs` (will) contains slides and notebooks explaining the DeepSphere-Weather concept.

## Installation

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/deepsphere/deepsphere-weather.git
   cd deepSphere-weather
   ```

2. Install manually the following dependencies:
   - Install first pytorch and its extensions on GPU:
      ```sh
      conda install -c conda-forge pytorch-gpu  
      ```
   - If you don't have GPU available install it on CPU:
      ```sh
      conda install -c conda-forge pytorch-cpu  
      ```
   - Install the required packages: 
   ```sh
   conda create --name weather python=3.8
   conda install xarray dask cdo h5py h5netcdf netcdf4 zarr numcodecs rechunker xskillscore
   conda install notebook jupyterlab
   conda install matplotlib-base cartopy pycairo seaborn cycler
   conda install numpy pandas numba scipy bottleneck
   conda install yaml tabulate tqdm deepdiff
   conda install healpy igl shapely      
   pip install git+https://github.com/epfl-lts2/pygsp@sphere-graphs
   pip install torchinfo
   ```
   - Clone the required repository: 
   ```sh
   git clone git@github.com:ghiggi/xverif.git
   git clone git@github.com:ghiggi/xscaler.git
   git clone git@github.com:ghiggi/xsphere.git
   git clone git@github.com:ghiggi/xforecasting.git
   ```
   
2. Alternatively install the dependencies using one of the appropriate below 
   environment.yml files:
   ```sh
   conda env create -f environment_python3.8.5.yml
   conda env create -f environment_python3.9.yml
   ```
   and clone the required repository: 
   ```sh
   git clone git@github.com:ghiggi/xverif.git
   git clone git@github.com:ghiggi/xscaler.git
   git clone git@github.com:ghiggi/xsphere.git
   git clone git@github.com:ghiggi/xforecasting.git
   ```
   
3. Add the PYTHONPATH (i.e. in the .bashrc) for the following packages. Here is an example: 
   ```sh
    export PYTHONPATH="${PYTHONPATH}:/home/ghiggi/Python_Packages/xscaler"
    export PYTHONPATH="${PYTHONPATH}:/home/ghiggi/Python_Packages/xverif"
    export PYTHONPATH="${PYTHONPATH}:/home/ghiggi/Python_Packages/xsphere"
    export PYTHONPATH="${PYTHONPATH}:/home/ghiggi/Python_Packages/xforecasting"  
    export PYTHONPATH="${PYTHONPATH}:/home/ghiggi/Projects/deepsphere-weather"
   ```

## Tutorials

* [`spherical_grids.ipynb`]: get a taste of samplings/pixelizations of the sphere.
* [`interpolation_pooling.ipynb`]: get to understand our generalized pooling based on interpolation between samplings.

[`spherical_grids.ipynb`]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-weather/blob/outputs/tutorials/spherical_grids.ipynb
[`interpolation_pooling.ipynb`]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-weather/blob/outputs/tutorials/interpolation_pooling.ipynb

# Data structure

Here we are going to document the data structure 

The data required for model training and evaluation are stored with the following folder hierarchy.

```
preprocessed
└───ERA5_HRES
|   └─── <sampling>
|         └─── Data
|         └─── Scalers
|         └─── Benchmarks 
|         └─── Climatology 
| 
└───IF5_HRES
|    └─── <sampling>
|         └─── Data
|    
└───IF5_ENS
|   └─── <sampling>
         └─── Data
```

On the LTE servers, data are available in the directory `/ltenas3/data/DeepSphere`

The data are available upon request to the authors.

## Reproducing our results


## Contributors

* [Gionata Ghiggi](https://people.epfl.ch/gionata.ghiggi) [slides](https://presentations.copernicus.org/EGU21/EGU21-2681_presentation.pdf) [video](https://www.youtube.com/watch?v=7D5OjzAdfz8&t=172s&ab_channel=IS-ENES3H2020)
* [Michaël Defferrard](https://deff.ch)
* [Wentao Feng](https://www.linkedin.com/in/wentaofeng) [[code](https://github.com/ownzonefeng/weather_prediction/), [slides](https://infoscience.epfl.ch/record/282285)]
* [Yann Yasser Haddad](https://www.linkedin.com/in/yann-yasser-haddad) [[code](https://github.com/ownzonefeng/weather_prediction), [slides](https://infoscience.epfl.ch/record/282437)]
* [Natalie Bolón Brun](https://www.linkedin.com/in/nataliebolonbrun) [[code](https://github.com/natbolon/weather_prediction)]
* [Icíar Lloréns Jover](https://www.linkedin.com/in/iciar-llorens-jover) [[code](https://github.com/illorens/weather_prediction), [report & slides](https://infoscience.epfl.ch/record/278138)]

## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
