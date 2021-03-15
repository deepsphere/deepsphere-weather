# DeepSphere-Weather - Deep Learning on the sphere for weather/climate applications.

![Alt Text](https://github.com/deepsphere/deepsphere-weather/blob/main/figs/Forecast_State_Errors.gif)

The code in this repository provides a scalable and flexible framework to apply convolutions on spherical unstructured grids for weather/climate applications.

ATTENTION: The code is subject to changes in the coming weeks / months. 

The folder `experiments` provide examples for:
-  Weather forecasting using autoregressive CNN models
-  Weather field downscaling (aka superesolution) [in preparation].
-  Classication of atmospheric features (i.e. tropical cyclone and atmospheric rivers) [in preparation].

The folder `tutorials` (will) provide jupyter notebooks describing various features of DeepSphere-Earth.

The folder `docs` (will) contains slides and notebooks explaining the DeepSphere-Earth concept. 

## Quick start 

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/deepsphere/deepsphere-Weather.git
   cd DeepSphere-Weather
   ```

2. Install the dependencies.
   ```sh
   conda env create -f environment.yml
   pip install git+https://github.com/epfl-lts2/pygsp@sphere-graphs
   ```
   
3. If you don't have a GPU and you plan to work on CPU, please install the follow: 
   ```sh
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   pip install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
   ```
   
## Contributors

* [Gionata Ghiggi](https://people.epfl.ch/gionata.ghiggi)
* [Michaël Defferrard](https://deff.ch)
* [Wentao Feng](https://www.linkedin.com/in/wentaofeng) [[code](https://github.com/ownzonefeng/weather_prediction/), [slides](https://infoscience.epfl.ch/record/282437)]
* [Yann Yasser Haddad](https://www.linkedin.com/in/yann-yasser-haddad) [[code](https://github.com/ownzonefeng/weather_prediction), [slides](https://infoscience.epfl.ch/record/282285)]
* [Natalie Bolón Brun](https://www.linkedin.com/in/nataliebolonbrun) [[code](https://github.com/natbolon/weather_prediction)]
* [Icíar Lloréns Jover](https://www.linkedin.com/in/iciar-llorens-jover) [[code](https://github.com/illorens/weather_prediction), [report & slides](https://infoscience.epfl.ch/record/278138)]

## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
