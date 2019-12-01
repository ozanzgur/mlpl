from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from .. import utils, fe, models, vis, prep
#import pandas as pd

######################## LGBM RELATED ########################

def default_lgbm_preprocess(feature_properties, train, test, label_name):
    """Default preprocessing suited to lgbm. However, other models can continue
    processing after starting with the output of this function.
    
    Steps are:
        - Find id columns and date columns and add them to a list to be removed
            later.
        - Find features that have different characteristics between training and
            test. Also 
            add them to the previous list. They will also be dropped.
        - Label encode object columns
    
    Parameters
    ----------
    feature_properties : Pandas dataframe
        dspipe.vis calculates properties of all features and stores them in this
        dataframe that used for decisions about feature engineering and processing.
    train : Pandas dataframe
        Training data
    test : Pandas dataframe
        Test data
    label_name : string
        Label name in dataset
    
    Returns
    -------
    to_drop : list of strings
        Columns that will not be used in training.
    train : pandas dataframe
        Processed training set
    test : pandas dataframe
        Processed test set
        
    """
    
    # Steps:
    # -  Drop columns that have full_unique_ratio > 0.99. These are usually IDs
    # -  Drop columns that have trn_test_difference > 0.6. These are dates or
    #      inconsistent features.
    # -  Label encode

    # ID columns are cols with all unique values.
    all_uniques = feature_properties.full_unique_ratio > 0.99
    
    # Extract feature names from feauture_properties
    id_cols = feature_properties[all_uniques].feature.values.tolist()
    
    # Print ID columns
    print(f'Don\'t use {id_cols}, as they have full_unique_ratio > 0.99')
    
    # Find inconsistent columns
    # Note: You should not use cols that are different in train and test.
    # (An LGBM model is trained with label=1 for test and label=0 for train.
    #    How well model can determine an example is from train or test describes
    #    The charactericstics difference between training and test.)
    large_diff_cols = feature_properties.trn_test_difference > 0.6
    
    inconsistents = feature_properties[large_diff_cols].feature.values.tolist()
    print(f'\nDon\'t use inconsistent features: {inconsistents}')
    
    # Concatenate two lists
    to_drop = list(set(id_cols + inconsistents))
    
    # Find cols with object type
    objects = feature_properties['dtype'] == 'object'
    
    # Find columns to label encode
    to_le = feature_properties[objects].feature.values.tolist()
    
    # Don't label encode columns that will not be used
    to_le = [col for col in to_le if col not in to_drop]
    
    # Will label encode columns that will be used in training.
    # Why don't we drop columns that will not be used?
    # - Because they will be used in feature engineering in the future.
    #   For this reason, We need to keep originals of all features.
    assert(isinstance(to_le, list))
    to_use = set([col for col in to_le if col not in to_drop])
    
    # Don't label encode label
    to_use.discard(label_name)
    to_label_encode = list(to_use)
    
    # Keep track of originals, because train and test will be mutated
    old_cols = train.columns
    
    print(f'\nLabel encode {to_label_encode}')
    # Label encode some columns
    for col in to_le:
        # utils.utilize is a quite complicated, but very useful decorator.
        # We can write a function that returns a column, then expand its use
        # for many cases using this decorator.
        utils.utilize(mode = 'add',
                      verbose = False)(prep.label_encode)([train, test], col)
        
        # Model will not be trained on these features.
        to_drop.append(col)
        
    # Find newly added columns and print them
    new_cols = train.columns.difference(old_cols)
    print('<b>NEW COLS:</b>')
    for col in new_cols:
        print(col)
    return to_drop, train, test

def default_lgbm_hparams(train, label_name):
    """Get default lgbm hyperparameters. Contains 3 sets of random hyperparameters.
    
    Parameters
    ----------
    train : dataframe
        Training set
    label_name : string
        Label name
        
    Returns
    -------
    defaults : dictionary
        A single dictionary with default hyperparamters for lgbm.
    
    """
    
    
    # Determine which objective must be used.
    obj = determine_lgbm_objective(train, label_name)
    
    # Get suitable metric for the objective.
    # Note that you should always specify your objective and metric.
    metric = None
    if obj == 'binary': metric = 'binary_logloss'
    elif obj == 'multiclass': metric = 'multi_logloss'
    else: metric = 'mse'
    
    defaults =   {'num_leaves': 25,
                  'feature_fraction': 1.0,
                  'bagging_fraction': 0.8,
                  'min_data_in_leaf': 25,
                  'objective': obj,
                  'max_depth': -1,
                  'learning_rate': 0.01,
                  "boosting_type": "gbdt",
                  "metric": metric ,
                  "verbosity": -1,
                  # Select my favourite numbers
                  "bagging_seed": [11, 22, 33],
                  'random_state': [42, 13, 666],
                  'folds' : [KFold(n_split = 4, random_state = 42),
                             KFold(n_split = 4, random_state = 13),
                             KFold(n_split = 4, random_state = 666)]
             }
    # Print defaults
    print(f'\nDetermine objective: {obj}, metric: {metric}')
    return defaults

def determine_lgbm_objective(train, label_name):
    """ Using the target in training data, determines
    which objective lgbm must use.
    
    If distinct value count is 2, it is 'binary'.
    
    Elif target column is an integer, it is 'multiclass'.
    (Also detects float types that are actually integers.)
    
    Else, it is 'regression'.
    
    Parameters
    ----------
    train : A pandas dataframe
        Training data
    label_name: string
        Target column name in training data
        
    Returns
    -------
    objective : string
        lgbm objective to be used in model parameters.
        
    """
    # Get unique value count
    n_unique = train[label_name].nunique()
    
    # label must have at least 2 distinct values.
    if not (n_unique > 1): raise AssertionError()
    
    # Rules are explained in the docstring.
    if n_unique == 2:
        return 'binary'
    if 'int' in str(train[label_name].dtype):
        return 'multiclass'
    if (col_concat - train[label_name]).sum() < 1e-7:
        return 'multiclass'
    else:
        return 'regression'
    
    
######################## SKLEARN RELATED ########################

def default_sklearn_preprocess(feature_properties, train, test, label_name):
    """Default preprocessing for sklearn models. Begins with default_lgbm_preprocess.
    While lgbm can work with missing values, all sklearn models has to have all
    missing values imputed.
    
    Note that this function only adds new columns.
    Steps:
        - Median impute numerics.
        - Mode impute categories. (Need another function for ordinals.)
        - Group outliers in categories. (value_count < 10)
        
    Parameters
    ----------
    feature_properties : Pandas dataframe
        A dataframe calculated by vis.get_all_col_properties
    train : pandas dataframe
        Training data
    test : pandas dataframe
        Test data
    label_name : string
        Label name in dataset
        
    Returns
    -------
    to_drop : list of strings
        Columns that will not be used in training (don't actually drop them.)
    train : pandas dataframe
        Processed train
    test : pandas dataframe
        Processed test
        
    """
    
    # Drop columns with full_unique_ratio > 0.99, as these are usually ID columns
    # Drop columns that have trn_test_difference > 0.6 (date or inconsistent)
    
    # Label encode
    to_drop, train, test = default_lgbm_preprocess(feature_properties, train,
                                                   test, label_name)
    
    # Keep track of original features, because dataframes will be mutated.
    orig_cols = test.columns
    
    # Recalculate feature properties, because data has changed
    feature_properties = vis.get_all_col_properties(train, test, label_name)
    
    # Find numerics to impute. They are floats and ints with null values.
    numerics_to_imp = feature_properties[
                ((feature_properties['dtype'] == 'float')
               | (feature_properties['dtype'] == 'int'))
               & (feature_properties.full_null_ratio > 0.0)].feature.tolist()
    
    # This function median imputes missing numeric values
    prep.process_denses([train, test], numerics_to_imp, mode = 'add', verbose = True)
    
    # New columns have been added. Old ones will be kept, but not used in training.
    to_drop.extend(numerics_to_imp)
    
    # Recalculate feature properties, because data has changed
    feature_properties = vis.get_all_col_properties(train, test, label_name)
    
    # Mode impute categories and group outliers (occurrence < 10)
    # (there should be no object columns in this stage)
    # You can display dataframes using df.to_html()
     
    # Find categories that have null values
    category_to_imp = feature_properties[
                    (feature_properties['dtype'] == 'category')
                  & (feature_properties.full_null_ratio > 0.0)].feature.tolist()
    
    # This function
    # - Mode imputes categories
    # - Groups values with value_count < 10
    # - Label encodes
    prep.process_nominals([train, test],
                          category_to_imp,
                          mode = 'add',
                          verbose = True)
    
    # Recalculate feature properties, because data has changed
    feature_properties = vis.get_all_col_properties(train, test, label_name)
    
    # OHE categoricals if:
    # - n_unique < limit
    # - n_unique > 2
    # & (feature_properties.full_nunique < 200) \
    
    """cats_to_ohe = feature_properties[
                 (feature_properties['dtype'] == 'category') \
               & (feature_properties.full_nunique > 2) \
               & (feature_properties.full_null_ratio == 0.0)].feature.tolist()
    
    # One-hot encoding
    print('OHE CATEGORICALS:')
    for cat in cats_to_ohe:
        print(cat)
        train, test = prep.one_hot_encode([train, test],
                                          cat,
                                          mode = 'add',
                                          sparse = True,
                                          verbose = False)
                                          
    to_drop.extend(cats_to_ohe)"""
    
    to_drop.extend(category_to_imp)
    
    
    # Recalculate feature properties, because data has changed
    #feature_properties = vis.get_all_col_properties(train, test, label_name)
    
    # Print new columns in html
    print(r'<b>New columns:</b>')
    print(list(test.columns.difference(orig_cols)))
    
    # Get specific properties from feature_properties
    fp_slice = feature_properties[['feature', 'dtype', 'full_null_ratio']]
    
    # Mark columns that will be used in training
    fp_slice['to_use'] = ['No' if col in to_drop else 'Yes' \
                          for col in feature_properties.feature.values]
    
    # Sort feature_properties
    fp_slice = fp_slice.sort_values(by = 'to_use', ascending = False)
    
    # Display feature_properties
    print(fp_slice.to_html())
    
    # Remove duplicates, just in case
    to_drop = list(set(to_drop))
    return to_drop, train, test