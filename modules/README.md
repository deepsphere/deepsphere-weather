## Modules

* ```full_pipeline_evaluation.py``` 

Allows to train, test, generate predictions and evaluate them for a model trained 
with a loss function that includes 2 steps. All parameters, except GPU configuration, are defined in a config
file such as the ones stored on the folder ```configs/``` .

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
