# DeepSphere-Earth - Deep Learning on the sphere for weather / climate applications

The code in this repository provides a scalable and flexible framework to apply convolutions on spherical grids.

The folder `experiments` provide examples for:
-  weather forecasting using autoregressive CNN models  
-  weather field downscaling (aka superesolution)f
-  classication of atmospheric features (i.e. tropical cyclone and atmospheric rivers).

The folder `tutorials` provide jupyter notebooks describing various features of DeepSphere-Earth.

The folder `docs` contains slides and notebooks explaining the DeepSphere-Earth concept. 

## Quick start 

TODO : UPDATE 

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/natbolon/weather_prediction.git
   cd weather_prediction
   ```

2. Install the dependencies.
   ```sh
   conda env create -f environment.yml
   ``
   
## Contributors
[Gionata Ghiggi][gg]
[Wentao Feng][wf]
[Yann Yasser Haddad][yyh]
[Natalie Bolón Brun][nbolon]
[Icíar Lloréns Jover][illorens]
[Michaël Defferrard][mdeff]

[gg]: https://people.epfl.ch/gionata.ghiggi
[wf]: https://github.com/ownzonefeng
[yyh]: https://www.linkedin.com/in/yann-yasser-haddad/?originalSubdomain=ch
[nbolon]: https://www.linkedin.com/in/nataliebolonbrun/
[illorens]: https://www.linkedin.com/in/iciar-llorens-jover/
[mdeff]: http://deff.ch

## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
