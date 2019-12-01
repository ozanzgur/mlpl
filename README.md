# mlpl

A machine learning pipeline to speed up data science lifecycle.

Using this library, you can:
- Test new experiments easily and keep track of their results.
- Do different hyperparameter search.
- Create a pipeline that consists of useful steps and save/load it.
- Automatically try different processing steps and use useful ones. (imputations, binning, one-hot encoding, ...)
- Make your predictions more reliable by averaging results obtained from different CV splits and random seeds.

### Start a new pipeline
***
A pipeline consists of  a class and its config and log files.
A pipeline will save your baseline model and the data after the baseline.
For trying new steps, it will load the data from its last state.
It will also automatically compare results of new steps to that of the baseline.
After each useful step, final form of the dataset and hparams will be saved.

**Hyperparameter search**
Pipeline will conduct bayesian/random search for the baseline.
For new steps after that, a simple hyperparameter search will take place.

(The reason for that is you cannot conduct a bayesian search for each experiment.
However, adding a new step will usually change the ideal hyperparameters, so
doing some hyperparameter search is required. Approach in this project is to conduct a bayesian search for the baseline, which will take a lot of time. For testing new steps, a custom simple hyperparameter search method will be used.)

**create new pipeline:
[(example: Titanic competition from Kaggle)](https://www.kaggle.com/c/titanic/overview "(example: Titanic competition from Kaggle)")**
```python
label_name = 'Survived'
trn_path = 'data/train.csv'
test_path = 'data/test.csv'

# Pipeline class will keep track of your processed files, model metrics and experiments. 
lr_pipeline = pipe.Pipeline(label_name = label_name,
                            			   overwrite = True,
                            			   project_path = 'lr_pipeline',
                            			   train_data_path = trn_path,
                            			   test_data_path = test_path,
                            			   minimize_metric = False,
                            			   useful_limit = 0.001,
                            			   line_search_iter = 1,
                            			   n_random_seeds = 1,
                            			   bayesian_search_iter= 50,
                            			   bayesian_search_count = 1,
                            			   final_bayesian_search_iter = 0,
                            			   line_search_patience = 2,
                            			   line_search_params = {'C': (1e-7, 1e3)})
```


### Hyperparameter search using hyperopt
***
- Specify hyperameter search space for each model.

Search is conducted on parameters in search space.
Fixed parameters are parameters that define the model.

This is an example for logistic regression.
**fixed parameters:**
```python
fixed_params_lr = dict(score=accuracy_score,
                       model=sklearn.linear_model.LogisticRegression,                       
                       max_iter=5000,
                       verbose = 0,
                       n_jobs = 3,
                       model_type = 'linear',
                       folds=[KFold(n_splits= 5, shuffle = True, random_state = 42),
                                  KFold(n_splits= 5, shuffle = True, random_state = 13),
                                  KFold(n_splits= 5, shuffle = True, random_state = 100)])
```
**search space:**
```python
lr_search_space = dict(C = hp.loguniform('C', -7, 3),
                       class_weight =  hp.choice('class_weight', ['balanced', None]),
                       solver =  hp.choice('solver ', ['lbfgs', 'sag']))
```
**Averaging results over different splits**
***
By  specifying multiple sklearn folds objects, average predictions over different splits.
(Also available for random_state parameters for models.)

```python
    folds=[KFold(n_splits= 5, shuffle = True, random_state = 42),
                 KFold(n_splits= 5, shuffle = True, random_state = 13),
                 KFold(n_splits= 5, shuffle = True, random_state = 100)]
```

** Creating a baseline model **
***
A baseline step is a step with minimal processing. Preprocessing steps and feature engineering steps in the project will be tested against the metrics of baseline model.

**create the baseline step:**
```python
lr_pipeline.set_baseline_step(model = pmodels.train_sklearn_pipeline,
                                proc = pdefaults.default_sklearn_preprocess,
                                search_model_params= lr_search_space,
                                fixed_model_params = fixed_params_lr
                               )
```

### Run baseline step
***
```python
res = lr_pipeline.run_baseline(return_result = True)
```
- Output contains a javascript that hides details about the step in collapsible boxes.
**output:**
![baseline1](https://user-images.githubusercontent.com/40238324/69920598-f53a9e00-149a-11ea-9fed-45c60fb11b84.PNG)

***

![baseline2](https://user-images.githubusercontent.com/40238324/69920654-7c881180-149b-11ea-8168-3a07f9d28a04.PNG)

***

![baseline3](https://user-images.githubusercontent.com/40238324/69920657-89a50080-149b-11ea-9712-703a5c536042.PNG)

### Create submission and save pipeline
***
**create submission:**
```python
# Convert test_preds to int from probabilities

# Since this competition requires values to be 0 or 1,
# We have to adjust a decision threshold. While selecting this threshold,
# criteria is to make mean value of test_preds to label in training set.
# This step is not necessary in most projects
test_preds = (res['test_preds'] > 0.55).astype('int')

# Prepare submission file
to_sub = sub.copy()
to_sub[label_name] = test_preds
to_sub.to_csv('titanic_sub.csv', index = False)
test_preds.mean()

# Baseline LB score: 0.76555
```
**save pipeline:**
```python
lr_pipeline.save_project()
```

#### Readme for trying new steps will be available in a couple days.