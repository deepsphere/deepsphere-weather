# Geometric deep learning for medium-range weather prediction

[Icíar Lloréns Jover][illorens], [Michaël Defferrard][mdeff], [Gionata Ghiggi][gg], [Natalie Bolón Brun][nbolon]

[illorens]: https://www.linkedin.com/in/iciar-llorens-jover/
[mdeff]: http://deff.ch
[gg]: https://people.epfl.ch/gionata.ghiggi
[nbolon]: https://www.linkedin.com/in/nataliebolonbrun/

The code in this repository provides a framework for a deep learning medium range weather prediction method based on graph spherical convolutions. 

[June 2020]: The results obtained with this code are detailed in the Masters thesis [report and slides][info_link].

[September 2020]: Results have been improved from the initial basis thanks to:
  * Introduction of residual connections in the architecture 
  * Inclusion of further consecutive steps in the loss with different weighting schemes to reduce the loss at long term predictions
  
 Model | Z500 (6h) | t850 (6h) | Z500 (120h) | t850 (120h)
------------ | ------------- | ------------- | ------------- | -------------
Weyn et al | 103.17 | 1.0380 | 611.33 | 2.957
Iciar June 2020 | 67.46 | 0.7172 | 861.7 | 3.432
Ours Sep 2020 | 61.58 | 0.7110 | 680.024 | 2.901
  
  * Results can be checked at [Plot Results][plots]
  




Ressources:
* **Report and slides**: [Geometric deep learning for medium-range weather prediction][info_link]

[info_link]: https://infoscience.epfl.ch/record/278138/



## Installation

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/natbolon/weather_prediction.git
   cd weather_prediction
   ```

2. Install the dependencies.
   ```sh
   conda env create -f weather.yml
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
   
- If an incompatibility with YAML raises, the following command should solve the problem: 
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

## Reproduce latest results

* Train model: 

The model listed as "Ours 2020" is trained using the module ```full_pipeline_multiple_steps.py```. An example of how to use it can be found on the notebook ```Restarting_weights_per_epoch.ipynb```. 

The config file to be used is ```configs/config_residual_multiple_steps.json```. You may want to modify the model name and/or the data paths if the data has been relocated. 

* Evaluate model: 

You can generate the model predictions using the notebook ```generate_evaluate_predictions.ipynb```. The parameters to be modified are:
 - model name (third cell)
 - epochs to be evaluated (you can define a range or a single one)
 
In order to evaluate the performance of the model, you only need to run up to "Generate plots for evaluation". This sencond part will generate the skill and climatology plots (you may be interested in generate them for a single epoch usually, not all of them)

* Compare models: 

In order to compare the performance of different models, or the same model at different epochs or simply a model against different baselines, you can use the notebook  ```plot_results.ipynb```. Depending on the purpose of the comparison, you may want to run a different section of the notebook. An explanation of each section and its use case can be found under the heading of the notebook. 


## Modules

* ```full_pipeline_evaluation.py``` 

Allows to train, test, generate predictions and evaluate them for a model trained 
with a loss function that includes 2 steps. All parameters, except GPU configuration, are defined in a config
file such as the ones stored on the folder ```configs/``` .

To use the mail notification at the end of the process, you need to provide a ```confMail.json ``` file which 
must have the following structure:

```
{
  "password": "yourMailPassword",
  "sender": "yourMail"
}
```
**Attention:** If you are using gmail and have activated a two-step verification process, you need to get permission
to the application and generate a new password. Details on how to generate the password can be found 
[here](https://towardsdatascience.com/automate-sending-emails-with-gmail-in-python-449cc0c3c317)


* ``` full_pipeline_multiple_steps.py``` 

Allows to to train and test a model  
with a loss function that includes multiple steps that can be defined by the user. It saves the model after every epoch
but does not generate the predictions (to save time since it can be done in parallel using the notebook 
```generate_evaluate_predictions.ipynb ```). The parameters are defined inside the main function, although it can be 
adapted to use a config file as in ```full_pipeline_evalution.py```

It is important to remark that the update function that takes care of the weight's update is defined on top
of the file and should be adapted to the number of lead steps taken into account in the loss function.

* ```architecture.py```

Contains pytorch models used for both ``` full_pipeline_multiple_steps.py``` and ``` full_pipeline_evaluation.py``` 
Previous architectures used can be found in the folder ``` modules/old_architectures/```

* ``` plotting.py```

Contains different functions to generate evaluation plots. 

* ``` train_last_model.py```

Contains code to train model with 2step-ahead prediction such as the one used for Iciar2020 results. 

## Notebooks

The main notebooks to explore are:
1. [Train of model using multiple-steps weighted loss][train_notebook]
Contains an example of how to use the functions that train the model that reported the best results mentioned earlier.

1. [Generate predictions using saved models][pred]
Generate values on validation set using the weights of the desired saved model 

1. [Evaluate predictions][plots]
Generate loss plots and comparison plots against different benchmark models

1. [healpix_resampling](https://nbviewer.jupyter.org/github/natbolon/weather_prediction/blob/master/notebooks/healpix_resampling.ipynb)
Generate healpix data from equiangular data

1. [generate_observations](https://nbviewer.jupyter.org/github/natbolon/weather_prediction/blob/master/notebooks/generate_observations.ipynb)
Generate ground-truth data for evaluation of the models 

1. [train_direct_predictions](https://nbviewer.jupyter.org/github/natbolon/weather_prediction/blob/master/notebooks/train_direct_predictions.ipynb)
Train model for direct prediction at a certain time-ahead. Currently is set up to generate predictions at either 72h or 120h ahead. 

1. [full_pipeline](https://nbviewer.jupyter.org/github/natbolon/weather_prediction/blob/master/notebooks/full_pipeline.ipynb)
Notebook-version of the code that can be found in ``` full_pipeline_multiple_steps.py```. **Attention:** code may not be the latest version and therefore it may not match exactly the python file. Do NOT use to try to reproduce the latest results. 

The below notebooks contain all experiments used to create our obtained results reported on the Msc Thesis of [Icíar Lloréns Jover][illorens]. 

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

[train_notebook]: https://nbviewer.jupyter.org/github/natbolon/weather_prediction/blob/master/notebooks/Restarting_weights_per_epoch.ipynb

[pred]: https://nbviewer.jupyter.org/github/natbolon/weather_prediction/blob/master/notebooks/generate_evaluate_predictions.ipynb

[plots]: https://nbviewer.jupyter.org/github/natbolon/weather_prediction/blob/master/notebooks/plot_results.ipynb



## License

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).
