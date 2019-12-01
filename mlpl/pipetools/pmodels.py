from .. import utils, models, prep
from . import putils, pdefaults, dt
import numpy as np
import pandas as pd

def train_lgbm_pipeline(
        train, test,
        label_name, params = None,
        cols_to_drop = None):
    """ Train an lgbm model. This function handles many cumbersome tasks to
    ensure reliability of the metrics. Uses cross validation.
    
    IMPORTANT:
    Number of fold objects, number of random_states and number of bagging_seeds
    must be equal! If you want your results to be reliable, use this functionality.
        
    - Average results for different sets of folds. ex: It can average results for:
        KFold(n_split = 4, random_state = 42),
        KFold(n_split = 4, random_state = 13),
        KFold(n_split = 4, random_state = 666)
    
    - Averaging results for different random seeds and bagging seeds.
    
    - Simple hyperparameter search.
        Tries hparameter sets if your params is a list of hparameter dicts.
        Hyperopt gives better results.
    
    An example for params is:
    
        params = {'num_leaves': 25,
                  'feature_fraction': 1.0,
                  'bagging_fraction': 1.0,
                  'min_data_in_leaf': 25,
                  'objective': 'binary',
                  'max_depth': -1,
                  'learning_rate': 0.03,
                  "boosting_type": "gbdt",
                  "metric": 'auc',
                  "verbosity": -1,
                  "bagging_seed": [11, 22, 33],
                  'random_state': [42, 13, 666],
                  'folds' : [KFold(n_split = 4, random_state = 42),
                             KFold(n_split = 4, random_state = 13),
                             KFold(n_split = 4, random_state = 666)]
                }
                
    Parameters
    ----------
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name for the dataset
    params : dict
        Model parameters. Note that there are additional parameters to the
        standard lgbm parameters. Additionals are taken away before training.
    cols_to_drop : set or list of strings
        Columns not to be used in training. You may or may not include label.
        
    Returns
    -------
    res_pipeline : dictionary
        Training results. Contains:
            - test_preds : pd.series, average test_preds for all random hparameters.
            - best_params : dict, best parameter set (currently identical to params)
            - mean_metric : float, avg. metric for each fold and random parameter set.
            - dev_metric : float, avg. std. of metrics over random parameter set.
            
    """
    # If no parameters were given, get default hparams.
    # Also finds the correct objective for lgbm.
    if params is None:
        params = pdefaults.default_lgbm_hparams(train, label_name)
    
    # Make params a list for builtin hparam search
    if not isinstance(params, list):
        params = [params]
    
    # Create result dictionary
    res_pipeline = dict.fromkeys(['test_pred', 'best_params', 'val_score'])
    
    # Hopefully you will never have a metric worse than that
    # Note that lgbm metrics are always lower-better. Please correct me if this is wrong.
    best_metric = 99999
    
    # cols_to_drop is kept as a set, but lists are also ok.
    if isinstance(cols_to_drop, list):
        cols_to_drop = set(cols_to_drop)
          
    # Columns to keep. Exclude label to prevent error.
    cols_to_keep = test.columns.difference(cols_to_drop.discard(label_name))
    
    # hparam search, loop over all hparameter sets
    for param in params:
        # Check if random_state, bagging_seed and folds have the same length
        putils.lgbm_hparam_seed_check(param)
        
        # Keep mean metrics over all seeds
        mean_seed_metric = 0.0
        mean_seed_metric_std = 0.0
        
        # Keep average test predictions over all seeds
        mean_test_preds = None
        
        # Get number of random_hparams (random_state, bagging_seed, folds)
        n_seeds = len(utils.tolist(params['bagging_seed']))
        
        # Iterate over all random_seed sets.
        for i_seed in range(n_seeds):
            # Copy parameters, as it will be modified.
            run_params = dict(param)
            
            # Pick random_hparams
            run_params['bagging_seed'] = utils.tolist(param['bagging_seed'])[i_seed]
            run_params['random_state'] = utils.tolist(param['random_state'])[i_seed]
            folds = utils.tolist(param['folds'])[i_seed]
            
            # I didn't pass copies of datasets. You should not modify them in train_lgbm.
            res = models.train_lgbm(folds, run_params,
                                    train.loc[:, cols_to_keep],
                                    train[label_name],
                                    test.loc[:, cols_to_keep],
                                    verbose = False)
            
            # Keep avg. metric for random_hparam sets.
            mean_seed_metric += res['mean_metric']
            mean_seed_metric_std += res['dev_metric']
            
            # Keep avg. of test predictions.
            if mean_test_preds is None:
                mean_test_preds = res['test_preds']
            else:
                mean_test_preds = mean_test_preds + res['test_preds']
            
        # Average metric and test predictions over multiple seeds
        mean_seed_metric /= n_seeds
        mean_seed_metric_std /= n_seeds
        mean_test_preds /= n_seeds
        
        # If current set of hparams are better, keep them.
        # Note that this is for hparam search, not for multiple random seeds.
        if mean_seed_metric < best_metric: # assume always minimize
            # Keep results for this set.
            res_pipeline['test_preds'] = mean_test_preds
            res_pipeline['best_params'] = param
            res_pipeline['mean_metric'] = mean_seed_metric
            res_pipeline['dev_metric'] = mean_seed_metric_std
            best_metric = mean_seed_metric
    return res_pipeline

def train_sklearn_pipeline(train, test, label_name, params, cols_to_drop = None):
    """ Train an sklearn model for multiple sklearn fold objects (Optional).
    Averages results and metrics over multiple folds objects.
    
    Parameters
    ----------
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name for the dataset
    params : dict
        Hyperparameters for the sklearn model. Also includes the model and folds.
    cols_to_drop : set or list of strings
        Columns not to use int training
    
    Returns
    -------
    res_pipeline : dictionary
        Result of the training. Contains:
        - test_preds pd.Series, average test predictions
        - mean_metric float, average metric over folds and fold objects
        - best_param identical to params ( will be removed)
    
    """
    run_params = dict(params)
    model = run_params.pop("model")
    score = run_params.pop('score')
    folds = utils.tolist(run_params.pop('folds'))
    predict_proba = run_params.pop('predict_proba', True)
    standardize = run_params.pop('standardize', True)
    model_type = run_params.pop('model_type', 'linear')
    cols_to_keep = test.columns.difference(cols_to_drop)
    nonsparse_cols = test.df.columns.difference(cols_to_drop)
    
    # There must be no null values in data
    # Can't check for sparse
    if not isinstance(train, dt.DataTable):
        raise TypeError('Data must be in DataTables.')
    
    cols_to_keep = utils.tolist(cols_to_keep)
    
    print('NONSPARSE DTYPES:')
    print(train[nonsparse_cols].dtypes)
    
    if model_type == 'linear':
        
        # Drop categoricals with more than 2 distinct values.
        # Linear models can't learn these features
        # They must be either OHEd or dropped.
        high_card_cats = [c for c in nonsparse_cols if str(train[c].dtype) == 'category' and train[c].nunique() > 2]
        
        # Drop high card. cats.
        cols_to_process = nonsparse_cols.difference(high_card_cats)
        
        # Inform user
        if len(high_card_cats) > 0:
            print('Will not use high cardinality categoricals for linear models:')
            print(high_card_cats)
    
        # Standardize numeric features
        # Numerics are columns that are note category or object
        
        numerics = [col for col in cols_to_process if \
                          str(test[col].dtype) not in ['category', 'object']]
        
        # Convert categories to numeric, for standardization.
        # Because, distance based models need all columns to be standardized.
        cats = [col for col in cols_to_process if \
                          str(test[col].dtype) in ['category']]
        
        # Standardize specified columns
        cols_to_std = numerics + cats
        
        # These are the columns that were not standardized before.
        # Ones that were standardized before will be reused.
        new_cols_to_std = []
        
        # Columns that will be used in training
        # Replace originals with their standardized versions, also remove high cards for linear model
        train_col_names = set(cols_to_keep).difference(cols_to_std).difference(high_card_cats)
        train_col_names.update([f'{col}_std' for col in cols_to_std])
        
        # Inform user if will standardize any col
        # Each column will be standardized once.
        # After standardization, columns are added to dataframe and standardized
        # versions will be reused.
        
        orig_cols_to_std = [col for col in cols_to_std if col + '_std' not in train.columns]
        new_std_cols = [col + '_std' for col in orig_cols_to_std]
        if len(new_std_cols) > 0:
            print('STANDARDIZE:')
        
        for col in orig_cols_to_std:
            std_name = col + '_std'
            print(col)
            train[std_name] = train[col].copy()
            test[std_name] = test[col].copy()
            
            # Convert categories to float, because these are note actual categoricals.
            # Real categoricals were either OHEd or dropped.
            if str(test[std_name].dtype) == 'category':
                train[std_name] = train[std_name].astype('float')
                test[std_name] = test[std_name].astype('float')
            
            # Standardized columns will not be directly used in training
            # They will be replaced with their originals in this method
            cols_to_drop.update({std_name})
            
        if len(new_std_cols) > 0:
            prep.standardize_cols([train, test],
                                  cols = new_std_cols,
                                  inplace = True)
        
    # Will average results for each cv
    n_folds = len(folds)
    
    # Keep track of mean metric
    mean_seed_metric = 0.0
    mean_seed_metric_std = 0.0
    mean_test_preds = None
    
    print('TRN COLS TO USE:')
    print(train_col_names)
    print(train[train_col_names].shape)
    
    # Iterate over folds objects
    for folds_obj in folds:
        
        # Train an sklearn model
        res = models.train_sklearn_binary(folds_obj,
                                          train[train_col_names],
                                          run_params,
                                          train[label_name].astype(int),
                                          test[train_col_names],
                                          model,
                                          score,
                                          predict_proba = predict_proba,
                                          true_class = 1)
        # Keep metric for this set of folds.
        mean_seed_metric += res['mean_metric']
        mean_seed_metric_std += res['dev_metric']
        
        # Keep test predictions.
        if mean_test_preds is None:
            mean_test_preds = res['test_preds']
        else:
            mean_test_preds = mean_test_preds + res['test_preds']
    
    # Average metric and test predictions over multiple seeds
    mean_seed_metric /= n_folds
    mean_test_preds /= n_folds
    mean_seed_metric_std /= n_folds
    
    ################################################
    # Create result dictionary
    res_pipeline = {}
    res_pipeline['test_preds'] = mean_test_preds
    res_pipeline['best_param'] = params # TODO: remove best_param
    res_pipeline['mean_metric'] = mean_seed_metric
    res_pipeline['dev_metric'] = mean_seed_metric_std
    return res_pipeline

def rfe_pipeline(
        feature_properties, train,
        test, label_name,
        scoring = None, model = None,
        params = None, cv_scheme = None,
        cols_to_drop = None, step = 2):
    """ Mostly from: https://www.kaggle.com/nroman/recursive-feature-elimination
    Recursive feature elimination for pipeline. Can use lgbm or sklearn models.
    
    default params:
    params = {
          'num_leaves': 100,
          'feature_fraction': 0.5,
          'bagging_fraction': 0.5,
          'min_data_in_leaf': 100,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'random_state': 47f
         }
    
    Parameters
    ----------
    feature_properties : pandas dataframe
        dataframe calculated by vis.
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name for the dataset
    scoring : sklearn scoring
        Scoring method for rfe. default: accuracy
        ex: 'roc_auc', 'mse', ‘accuracy’,...
        see more from: https://scikit-learn.org/stable/modules/model_evaluation.html
    model : sklearn model or lgbm
        Model to be used in rfe.
    params : dictionary
        Model hyperparameters
    cv_scheme : sklearn folds object
        Folds
    cols_to_drop : set or list of strings
        Columns not to use in training
    step : int
        number of features to be dropped in each rfe step.
    
    Returns
    -------
    new_cols_to_drop : list of strings
        Columns that rfe decided to be not useful.
    
    """
    if cols_to_drop is None:
        cols_to_drop = []
    
    for col in test.columns.difference(cols_to_drop):
        col_type = str(train[col].dtype)
        if col_type == 'category':
            train[col] = train[col].astype('int')
            train[col].fillna(-9999, inplace=True)
            train[col] = train[col].astype('category')
        else:
            train[col].fillna(-9999, inplace=True)
    if scoring is None:
        print('Scoring for rfe was not specified. Using accuracy.')
        scoring = 'accuracy'
    
    # Defaults
    if params is None:
        params = {
          'num_leaves': 100,
          'feature_fraction': 0.5,
          'bagging_fraction': 0.5,
          'min_data_in_leaf': 100,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'random_state': 47
         }
        print(f'RFECV model parameters "param" was not specified. Defaults: {params}')
    
    
    if model is None:
        print(f'model was not specified, using lgb.LGBMClassifier')
        model = lgb.LGBMClassifier
    
    run_params = params
    if isinstance(params, list):
        run_params = dict(params[0])
    if isinstance(run_params['bagging_seed'], list):
        run_params['bagging_seed'] = run_params['bagging_seed'][0]
    if isinstance(run_params['random_state'], list):
        run_params['random_state'] = run_params['random_state'][0]
    clf = model(**run_params)
    if cv_scheme is None:
        cv_scheme = StratifiedKFold(n_splits=5)
    
    cols_to_keep = test.columns.difference(cols_to_drop.discard(label_name))
    
    rfe = RFECV(estimator=clf, step=step, cv=cv_scheme, scoring=scoring, verbose=2)
    rfe.fit(train.loc[:,cols_to_keep],  train[label_name])
    
    print('Optimal number of features:', rfe.n_features_)
    print('All features with rank 1: ')
    
    
    
    rank1_cols = train.loc[:,cols_to_keep].columns[rfe.ranking_ == 1]
    for col in rank1_cols: print(col)
    
    plt.figure(figsize=(14, 8))
    plt.xlabel("RFE Steps")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
    plt.show()
    
    y = train[label_name]
    train.drop(label_name, axis = 1, inplace = True)
    print(f'Will keep {len(rank1_cols)} / {len(train.columns.difference(cols_to_drop))}')
    new_cols_to_drop = None
    for df in train, test:
        new_cols_to_drop = list(train.columns.difference(rank1_cols))
        df.drop(df.columns.difference(rank1_cols), axis = 1, inplace = True)
    train[label_name] = y
    return new_cols_to_drop