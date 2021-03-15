# DeepSphere-Earth - Deep Learning on the sphere 
![Alt Text](https://github.com/ownzonefeng/weather_prediction/blob/NewPipeLine/figs/Forecast_State_Errors.gif)

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
   git clone https://github.com/DeepSphere/DeepSphere-Weather.git
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
- [Gionata Ghiggi][gg]
- [Wentao Feng][wf]
- [Yann Yasser Haddad][yyh]
- [Natalie Bolón Brun][nbolon]
- [Icíar Lloréns Jover][illorens]
- [Michaël Defferrard][mdeff]

[gg]: https://people.epfl.ch/gionata.ghiggi
[wf]: https://github.com/ownzonefeng
[yyh]: https://www.linkedin.com/in/yann-yasser-haddad/?originalSubdomain=ch
[nbolon]: https://www.linkedin.com/in/nataliebolonbrun/
[illorens]: https://www.linkedin.com/in/iciar-llorens-jover/
[lv]: https://www.linkedin.com/in/laure-vancauwenberghe/
[ma]: https://www.linkedin.com/in/michael-allemann/
[mdeff]: http://deff.ch

## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
