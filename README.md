# Geometric deep learning for medium-range weather prediction

[Icíar Lloréns Jover][illorens], [Michaël Defferrard][mdeff], [Gionata Ghiggi][gg]

[illorens]: https://www.linkedin.com/in/iciar-llorens-jover/
[mdeff]: http://deff.ch
[gg]: https://people.epfl.ch/gionata.ghiggi

The code in this repository provides a framework for a deep learning medium range weather prediction method based on graph spherical convolutions. The results obtained with this code are detailed in the Masters thesis [report and slides][info_link]. 


Ressources:
* **Report and slides**: [Geometric deep learning for medium-range weather prediction][info_link]

[info_link]: https://infoscience.epfl.ch/record/278138/



## Installation

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/illorens/weather_prediction.git
   cd weather_prediction
   ```

2. Install the dependencies.
   ```sh
   conda env create -f environment.yml
   ```
   
   
3. Create the data folders
    ```sh
   mkdir data/equiangular/5.625deg/ data/healpix/5.625deg/
   ```
   
4. Download the WeatherBench data on the ```data/equiangular/5.625deg/``` folder by following instructions on the [WeatherBench][weatherbench_repo] repository.

5. Interpolate the WeatherBench data onto the HEALPix grid. Modify the paremeters in ```scripts/config_data_interpolation.yml``` as desired.
    ```sh 
    python -m scripts.data_iterpolation -c scripts/config_data_interpolation.yml
    ```
    
Attention:

- If deepsphere is not properly installed:
   ```sh
   conda activate weather_modelling
   pip install git+https://github.com/deepsphere/deepsphere-pytorch 
   ```
   
   If an incompatibility with YAML raises, the following command should solve the problem: 
   ```sh
   conda activate weather_modelling
   pip install git+https://github.com/deepsphere/deepsphere-pytorch --ignore-installed PyYAML
   ```

- If it does not find the module ```SphereHealpix``` from pygsp, install the development branch using: 
   ```sh
   conda activate weather_modelling
   pip install git+https://github.com/Droxef/pygsp@new_sphere_graph
   ```

[weatherbench_repo]: https://github.com/pangeo-data/WeatherBench


## Notebooks

The below notebooks contain all experiments used to create our obtained results. 

1. [Effect of static features on predictability.][static_features]
   Shows the effect of the removal of all static features from the model training. The notebook shows the training, results and comparison of the models. 
1. [Effect of dynamic features on predictability][dynamic_features]
   Shows the effect of the addition of one dynamic feature to the model. The notebook shows the training, results and comparison of the models. 
1. [Effect of temporal sequence length and temporal discretization on predictability][temporal]
   We cross-test the effect of different sequence lengths with the effect of different temporal discretizations. The notebook shows the training, results and comparison of the models. 
   
   
The below notebooks show how to evaluate the performance of our models.

1. [Model evaluation][evaluation]
    Allows to evaluate with multiple metrics the performance of a model with respect to true data.
1. [Error video][error_vid]
    Produces a video of the error between predictions and true data.
   
   
[static_features]: https://nbviewer.jupyter.org/github/illorens/weather_prediction/blob/master/notebooks/test_static_features.ipynb

[dynamic_features]: https://nbviewer.jupyter.org/github/illorens/weather_prediction/blob/master/notebooks/test_dynamic_features.ipynb

[temporal]: https://nbviewer.jupyter.org/github/illorens/weather_prediction/blob/master/notebooks/test_temporal_dimension.ipynb

[evaluation]: https://nbviewer.jupyter.org/github/illorens/weather_prediction/blob/master/notebooks/evaluate_model.ipynb

[error_vid]: https://nbviewer.jupyter.org/github/illorens/weather_prediction/blob/master/notebooks/error_video.ipynb


## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).