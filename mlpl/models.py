import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from IPython.display import clear_output
from . import fe
from operator import itemgetter
from sklearn.metrics import accuracy_score, mean_squared_error
import scipy

def metric_dev(scores):
    """Maximum proportional deviation from mean inside scores list.
    
    """
    #return np.max(np.abs(np.array(scores) - np.mean(scores)) / np.mean(scores))
    return np.std(scores)

def train_lgbm(folds, params, X, y, X_test = None, mean_encoding_cols = None, replace_encoded = True, verbose = True):
    """
    Train lgbm model with specified folds. Mean encoding is also an option.
    
    inputs:
    folds - sklearn folds object
            example: folds = StratifiedKFold(n_splits=5)
    params - parameters for lgbm
            example:
            params = {'num_leaves': 50,
                      'min_child_weight': 0.03,
                      'feature_fraction': 0.3,
                      'bagging_fraction': 0.4,
                      'min_data_in_leaf': 100,
                      'objective': 'binary',
                      'max_depth': -1,
                      'learning_rate': 0.006,
                      "boosting_type": "gbdt",
                      "bagging_seed": 11,
                      "metric": 'auc',
                      "verbosity": -1,
                      'reg_alpha': 0.4,
                      'reg_lambda': 0.6,
                      'random_state': 47
                 }
    X,y,X_test - pandas DataFrames
    mean_encoding_cols - list of column names. [col_name]
    replace_encoded - replace mean encoded columns by their originals or add them to the dataset.
    
    """
    test_preds = np.empty(len(X_test)) if X_test is not None else None
    
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns
    metrics = list()

    n_folds = 0
    if not isinstance(folds, list):
        folds = folds.split(X, y)
    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        n_folds += 1
        X_trn_fold = X.iloc[trn_idx]
        X_val_fold = X.iloc[val_idx]
        
        test_fold = None
        if X_test is not None:
            test_fold = X_test.copy()
        
        if mean_encoding_cols is None:
            if verbose:
                print('No mean encoding.')
        else:
            if not (test_fold is not None): raise AssertionError()
            if not (isinstance(mean_encoding_cols, list)): raise AssertionError()
            # Encoding #############
            start_time = time.time()
            #print('Mean encoding...')
            X_trn_fold, X_val_fold, test_fold = mean_encoding_categorical(X_trn_fold, y.iloc[trn_idx], X_val_fold, test_fold,\
                                                                          mean_encoding_cols, replace = replace_encoded)
            if verbose:
                print('Mean encoding took {:.1f}s'.format(time.time() - start_time))
            
            # End of Encoding ######
        if verbose:
            print('Training on fold {}'.format(i_fold + 1))
        
        trn_data = lgb.Dataset(X_trn_fold, label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X_val_fold, label=y.iloc[val_idx])
        
        clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], early_stopping_rounds=500, verbose_eval=False)
    
        feature_importances['fold_{}'.format(i_fold + 1)] = clf.feature_importance()
        if X_test is not None:
            test_preds += clf.predict(test_fold)
        #oof_preds[val_idx] = clf.predict(val_data)
        
        metrics.append(clf.best_score['valid_1'][params['metric']])
    
    if X_test is not None:
        test_preds /= (n_folds + 1)
    if verbose:
        print('Training has finished.')
        print(f'Mean {params["metric"]}: {np.mean(metrics)}')
    
    return {
        'metrics': metrics,
        'feature_importances': feature_importances,
        'test_preds' : test_preds,
        'model' : clf
    }


def train_sklearn_binary(
        folds, X, params, y, X_test,
        model, score = None, classes = None,
        verbose = False, predict_proba = True, true_class = 1):
    n_classes = 2 if classes is None else len(classes)
    sparse = isinstance(X, scipy.sparse.csr.csr_matrix)    
    
    oof_preds = np.empty(X.shape[0])
    test_preds = np.empty(X_test.shape[0])
    mean_label = y.mean()
    
    scores = list()
    
    n_folds = 0
    if not isinstance(folds, list):
        if isinstance(folds, int):
            folds = KFold(n_splits = folds).split(X, y)
        else:
            folds = folds.split(X, y)
    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        n_folds += 1
        if sparse:
            X_trn_fold = X[trn_idx]
            X_val_fold = X[val_idx]
        else:
            X_trn_fold = X.iloc[trn_idx]
            X_val_fold = X.iloc[val_idx]
            
        test_fold = None
        if X_test is not None:
            test_fold = X_test
        
        if verbose:
            print('Training on fold {}'.format(i_fold + 1))
        clf=model(**params).fit(X_trn_fold, y[trn_idx])
        
        if X_test is not None:
            if predict_proba:
                pred_set = clf.predict_proba(test_fold)#[:,1]
                
                # Select predictions from the column that actually corresponds
                # to label = 1. Sklearn sometimes gets confused.
                i_select = None
                for i, class_val in enumerate(clf.classes_):
                    if class_val == true_class:
                        i_select = i
                        break
                #np.argmin(np.abs(pred_set.mean(axis = 0) - mean_label))
                test_preds += pred_set[:, i_select]
            else:
                test_preds += clf.predict(test_fold)
        
        #print(clf.predict_proba(X_val_fold).shape)
        scores.append(clf.score(X_val_fold, y.iloc[val_idx])) #score(y.iloc[val_idx], clf.predict_proba(X_val_fold)[:,1].reshape(-1,1)))
        if verbose:
            print(f'Validation {score}: {scores[-1]}')
    
    if X_test is not None:
        test_preds /= n_folds
    
    if verbose:
        print('Training has finished.')
        print(f'Mean score: {np.mean(scores)}')
        print(f'Std of scores: {np.std(scores)}')
    return {
        'mean_metric': np.mean(scores),
        'dev_metric': metric_dev(scores),
        'test_preds' : test_preds,
        'model' : clf
    }

def mean_encode(train_data, test_data, columns, target_col, reg_method=None,
                alpha=0, add_random=False, rmean=0, rstd=0.1, folds=1):
    length_train = len(train_data)
    '''Returns a DataFrame with encoded columns'''
    encoded_cols = []
    target_mean_global = train_data[target_col].mean()
    for col in columns:
        # Getting means for test data
        nrows_cat = train_data.groupby(col)[target_col].count()
        target_means_cats = train_data.groupby(col)[target_col].mean()
        target_means_cats_adj = (target_means_cats*nrows_cat + 
                                 target_mean_global*alpha)/(nrows_cat+alpha)
        # Mapping means to test data
        encoded_col_test = test_data[col].map(target_means_cats_adj)
        # Getting a train encodings
        if reg_method == 'expanding_mean':
            train_data_shuffled = train_data.sample(frac=1, random_state=1)
            cumsum = train_data_shuffled.groupby(col)[target_col].cumsum() - train_data_shuffled[target_col]
            cumcnt = train_data_shuffled.groupby(col).cumcount()
            encoded_col_train = cumsum/(cumcnt)
            encoded_col_train.fillna(target_mean_global, inplace=True)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))
        elif (reg_method == 'k_fold') and (folds > 1):
            kfold = StratifiedKFold(n_splits = folds, shuffle=True, random_state=1).split(train_data[target_col].values, train_data[target_col])
            parts = []
            for tr_in, val_ind in kfold:
                                # divide data
                    
                
                df_for_estimation, df_estimated = train_data.iloc[tr_in], train_data.iloc[val_ind]
                # getting means on data for estimation (all folds except estimated)
                nrows_cat = df_for_estimation.groupby(col)[target_col].count()
                target_means_cats = df_for_estimation.groupby(col)[target_col].mean()
                target_means_cats_adj = (target_means_cats*nrows_cat + 
                                         target_mean_global*alpha)/(nrows_cat+alpha)
                # Mapping means to estimated fold
                encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
                if add_random:
                    encoded_col_train_part = encoded_col_train_part + normal(loc=rmean, scale=rstd, 
                                                                             size=(encoded_col_train_part.shape[0]))
                # Saving estimated encodings for a fold
                parts.append(encoded_col_train_part)
            encoded_col_train = pd.concat(parts, axis=0)
            encoded_col_train.fillna(target_mean_global, inplace=True)
        else:
            encoded_col_train = train_data[col].map(target_means_cats_adj)
            if add_random:
                encoded_col_train = encoded_col_train + normal(loc=rmean, scale=rstd, 
                                                               size=(encoded_col_train.shape[0]))

        # Saving the column with means
        encoded_col = pd.concat([encoded_col_train, encoded_col_test], axis=0)
        encoded_col[encoded_col.isnull()] = target_mean_global
        encoded_cols.append(pd.DataFrame({'mean_'+target_col+'_'+col:encoded_col}))
    all_encoded = pd.concat(encoded_cols, axis=1)
    #Modified to reindex
    all_encoded = all_encoded.reset_index()
    return (all_encoded.iloc[:length_train].reset_index(drop = True),
            all_encoded.iloc[length_train:].reset_index(drop = True)
           )


def mean_encoding_categorical(train_fold_x, train_fold_y, valid_fold_x, X_test, cols_to_encode, replace = True, n_folds = 4, verbose = False):
    #You can do whatever you want in this function and you won't overfit in CV results. => No bad surprise in submissions.
    #The reason for that is we don't have access to validation data target. Overfitting to CV results from making decisions based on
    #Validation set targets.
    
    #print('Encoding {} columns. (Will replace)'.format(len(cols_to_encode)))
    """for col in cols_to_bin:
        est = KBinsDiscretizer(n_bins=25, encode='ordinal', strategy='quantile') #Can try different things
        est.fit(train_fold_x[col].values.reshape((-1,1)))
        # You may also fit to pd.concat([train_fold_x, valid_fold_x, X_test]) but I'm not sure which one works better

        train_fold_x[col] = est.transform(train_fold_x[col].values.reshape((-1,1)))
        valid_fold_x[col] = est.transform(valid_fold_x[col].values.reshape((-1,1)))
        X_test[col] = est.transform(X_test[col].values.reshape((-1,1)))"""
    
    #Cascaded mean encoding
    if verbose:
        print('Mean encoding {} columns. (Replace columns)...'.format(len(cols_to_encode)))
    num_valid = len(valid_fold_x)
    for col in cols_to_encode:
        train_fold_x['target'] = train_fold_y
        train_encoded, test_encoded = mean_encode(train_fold_x, pd.concat([valid_fold_x, X_test], axis = 0), [col], 'target', reg_method='k_fold',
                alpha=1, add_random=False, rmean=0, rstd=0.1, folds=n_folds)
        train_fold_x.drop('target', axis = 1, inplace = True)
        
        train_encoded.drop('index', axis = 1, inplace = True)
        test_encoded.drop('index', axis = 1, inplace = True)
        
        train_fold_x.reset_index(drop = True, inplace = True)
        valid_fold_x.reset_index(drop = True, inplace = True)
        X_test.reset_index(drop = True, inplace = True)
        
        valid_encoded = test_encoded.iloc[:num_valid].reset_index(drop = True)
        test_encoded = test_encoded.iloc[num_valid:].reset_index(drop = True)
    
        if replace:
            train_fold_x[col] = train_encoded
            valid_fold_x[col] = valid_encoded
            X_test[col] = test_encoded
        else:
            train_fold_x = pd.concat([train_encoded, train_fold_x], axis = 1).reset_index(drop = True)
            valid_fold_x =  pd.concat([valid_encoded, valid_fold_x], axis = 1).reset_index(drop = True)
            X_test =  pd.concat([test_encoded, X_test], axis = 1).reset_index(drop = True)
    
    #Goes back into training
    return [train_fold_x, valid_fold_x, X_test]


def try_mean_encoding_lgbm(X, y, test, folds, params, categorical_cols = None, baseline = None):
    base_perf = baseline
    if categorical_cols is None: 
        categorical_cols = [col for col in X.columns if str(X[col].dtype) in ['object', 'category']]
        
    print('There are {} categorical columns.'.format(len(categorical_cols)))
    folds_list = [fold for fold in folds] # Extract from generator to reuse
    
    # Baseline performance
    if baseline is None:
        print('Get baseline performance...')
        result_lgb = train_lgbm(folds_list, params, X, y, test, mean_encoding_cols = None)
        base_perf = np.mean(result_lgb['aucs'])
    print(f'BASELINE: {base_perf}')
        
    feature_gain = pd.DataFrame(columns = ['feature', 'change from baseline'])
    
    for col in tqdm(categorical_cols):
        print(f'Column name: {col}')
        result_lgb = train_lgbm(folds_list, params, X, y, test, mean_encoding_cols = [col])
        perf = np.mean(result_lgb['aucs'])
        feature_gain.append({'feature': col, 'change from baseline': perf - base_perf}, ignore_index = True)
        
    return feature_gain

def subsampled_idx(folds, X, y, sampling_rates):
    # Subsampling
    if sampling_rates is not None:
        trn_idx = []
        # Downsampling
        for label in sampling_rates:
            class_idx = np.argwhere(y == label).flatten()
            trn_idx = trn_idx + np.random.choice(class_idx, size=int(len(class_idx)*sampling_rates[label]), replace=False).tolist()

        X = X.iloc[trn_idx]
        y = y.iloc[trn_idx]
        if sort_by is not None:
            X['label_col++++'] = y
            X.sort_values(by = sort_by, inplace = True)
            y = X['label_col++++']
            X.drop('label_col++++', inplace = True, axis = 1)
    
    # Keep folds
    folds_list = [fold for fold in folds.split(X, y)]
    print('Set sizes: ')
    print('Train: {}, Validation: {}, Test: {}'.format(len(folds_list[0][0]), len(folds_list[0][1]), len(test)))
    return folds_list

def try_fe_multiple_cols(func, X,
                         y, test,
                         folds, params,
                         cols = None, baseline = None,
                         group_size = 1, sampling_rates = None,
                         sort_by = None, to_drop = None,
                         verbose = False):
    """
    Tries a feature engineering function on specified columns or all columns.
    Trains an lgbm model after each fe. Running this function takes a long time.
    
    inputs:
    func: function, its format must be:  X[col], test[col] = func(X[col], test[col])
    X, y, test  - DataFrames
    folds - sklearn folds generator. (ex: folds = KMeans(n_splits=2))
    params - lgbm parameters
            example:
            params = {'num_leaves': 50,
                      'min_child_weight': 0.03,
                      'feature_fraction': 0.3,
                      'bagging_fraction': 0.4,
                      'min_data_in_leaf': 100,
                      'objective': 'binary',
                      'max_depth': -1,
                      'learning_rate': 0.006,
                      "boosting_type": "gbdt",
                      "bagging_seed": 11,
                      "metric": 'auc',
                      "verbosity": -1,
                      'reg_alpha': 0.4,
                      'reg_lambda': 0.6,
                      'random_state': 47
                 }
    cols - list of column names to try fe on
    baseline - float. precalculated baseline
    group_size - integer. transform columns in groups to speed up process
    sampling_rates - dictionary. {[class : rate]}, downsample classes to speed up execution.
                    (ex:  {1.0: 1.0, 0.0: 0.2})
    sort_by - column name. if you use downsampling, sort data by a column in case you don't use shuffling. Otherwise, lgbm will overfit.
    to_drop - column name(s). columns to drop before training (after fe)
    verbose - Boolean. False: display result dataframe after each step, True: display fold results of each step.
    
    outputs:
    feature_gain - DataFrame, how much each column with fe applied change the baseline performance
    """
    
    base_perf = baseline
    
    if cols is None:
        cols = [col for col in X.columns if (str(X[col].dtype) not in ['object', 'category']) and (col not in to_drop)]
        print('There are {} columns.'.format(len(cols)))
    print('Will try fe on {} columns.'.format(len(cols)))
    
    folds_list = subsampled_idx(folds, X, y, sampling_rates)
    
    if to_drop is None:
        to_drop = []
    # Baseline performance
    if baseline is None:
        print('Get baseline performance...')
        result_lgb = None
        result_lgb = train_lgbm(folds_list, params, X.drop(to_drop, axis = 1), y, test.drop(to_drop, axis = 1))
        base_perf = np.mean(result_lgb['aucs'])
    print(f'BASELINE: {base_perf}')
    
    feature_gain = pd.DataFrame(columns = ['feature', 'change from baseline'])
    
    # Form column groups
    col_groups = [cols[i_group*group_size: (i_group+1)*group_size] for i_group in range(len(cols) // group_size)]
    
    
    if len(cols) / group_size > len(cols) // group_size:
        i_group = len(cols) // group_size
        col_groups.append(cols[i_group*group_size:])
    
    for col_group in tqdm(col_groups):
        print(f'Column name: {col_group}')
        col_orig_X = X[col_group].copy()
        col_orig_test = test[col_group].copy()
        old_cols = X.columns
        if not isinstance(col_group, list):
            col_group = [col_group]
        
        for col in col_group:
            # Column names must be strings.
            if not isinstance(col, str): raise AssertionError()
            X, test = func(X, test, col)
            
        print('X shape:')
        print(X.shape)
        result_lgb = train_lgbm(folds_list, params, X.drop(to_drop, axis = 1), y, test.drop(to_drop, axis = 1))
        perf = np.mean(result_lgb['aucs'])
        
        feature_label = ('+'.join(col_group)) if len(col_group) > 1 else col_group[0]
        feature_gain = feature_gain.append({'feature': feature_label, 'change from baseline': perf - base_perf}, ignore_index = True)
        
        feature_gain.sort_values(by = 'change from baseline', ascending = False, inplace = True)
        if not verbose:
            clear_output()
            display(feature_gain.style.background_gradient(cmap='coolwarm'))
        
        # Revert replaced columns
        X[col_group] = col_orig_X
        test[col_group] = col_orig_test
        
        # Discard created columns
        X = X[old_cols]
        test = test[old_cols]
        
    return feature_gain


def try_new_columns(X, y, test, folds, params, new_cols = None, baseline = None, sampling_rates = None, sort_by = None, to_drop = None, verbose = False):
    if not isinstance(new_cols, list):
        new_cols = [new_cols]
        
    base_perf = baseline
    print('Will try {} new columns.'.format(len(new_cols)))
    
    folds_list = subsampled_idx(folds, X, y, sampling_rates)
    
    # Baseline performance
    if baseline is None:
        print('Get baseline performance...')
        result_lgb = None
        result_lgb = train_lgbm(folds_list, params, X.drop(to_drop, axis = 1), y, test.drop(to_drop, axis = 1))
        base_perf = np.mean(result_lgb['aucs'])
    print(f'BASELINE: {base_perf}')
    
    feature_gain = pd.DataFrame(columns = ['feature', 'change from baseline'])
    
    
    orig_column_names = list(X.columns).remove(new_columns + to_drop)
    # Try all new columns one by one
    for col in tqdm(new_cols):
        print(f'Column name: {col}')
        if not isinstance(col, list):
            col= [col]
        
        for col in col_group:
            # Column names must be strings
            if not (isinstance(col, str)): raise AssertionError()
            X, test = func(X, test, col)
            
        print('X shape:')
        print(X.shape)
        
        result_lgb = train_lgbm(folds_list, params, X[orig_column_names + [col]], y, test[orig_column_names + [col]])
        perf = np.mean(result_lgb['aucs'])
        feature_gain = feature_gain.append({'feature': col, 'change from baseline': perf - base_perf}, ignore_index = True)
        
        feature_gain.sort_values(by = 'change from baseline', ascending = False, inplace = True)
        if not verbose:
            clear_output()
            display(feature_gain.style.background_gradient(cmap='coolwarm'))
        
    return feature_gain

def get_least_important_n(feature_importances, n_folds, n):
    feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(n_folds)]].mean(axis=1)
    return list(feature_importances.sort_values(by = 'average', ascending = True)['feature'].iloc[:n].values)

def drop_least_important_n(dfs, feature_importances, n_folds, n):
    to_drop = get_least_important_n(feature_importances, n_folds, n)
    for df in dfs:
        for col in to_drop:
            try:
                df.drop(col, axis = 1, inplace = True)
            except:
                print(f'{col} does not exist.')
                
def mean_encode_offline(folds, X, y, X_test, col, replace_col = False):
    """
    IMPORTANT: Use the same folds for training a model. Otherwise it will overfit.
    """
    if not isinstance(folds, list):
        folds = list(folds.split(X, y))
        
    outer_splits = len(folds)
    new_train_col = pd.Series(np.empty(len(X)))
    new_test_col = pd.Series(np.zeros(len(X_test)))
    
    start_time = time.time()
    for trn_idx, val_idx in folds:
        X_trn_fold = X.iloc[trn_idx]
        X_val_fold = X.iloc[val_idx]
        test_fold = X_test.copy()
        
        
        X_trn_fold, X_val_fold, test_fold = mean_encoding_categorical(X_trn_fold, y.iloc[trn_idx], X_val_fold, test_fold,\
                                                                      [col], replace = True, n_folds = 8)
        new_train_col.iloc[val_idx] = X_val_fold[col].values
        new_test_col += test_fold[col] / outer_splits
        
        
    print('Mean encoding took {:.1f}s for {}'.format(time.time() - start_time, col))
    if not replace_col:
        return new_train_col, new_test_col
    else:
        X.loc[:, col] = new_train_col
        X_test.loc[:, col] = new_test_col

################################ NN ####################################

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
import tensorflow as tf
import math

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.SparseTensor(indices, coo.data, coo.shape)


def train_nn(folds, X, y, X_test = None, model_params = {}, train_params = {}, sparse_data = False):
    """
    model_params = {
    'layers': None,
    'loss': None,
    'metric': None,
    'optimizer': None
    
    }
    
    train_params = {
    'epochs': 1,
    'verbose': True,
    'batch_size': 32,
    'shuffle': False,
    'test_batch_size': 50
    }
    
    """
    #oof_preds = None
    test_preds = None
    if not isinstance(X, tf.sparse.SparseTensor):
        oof_preds = np.empty(X.shape[0])
        test_preds = np.empty(X_test.shape[0]) if X_test is not None else None
    else:
        oof_preds = np.empty(X.get_shape()[0])
        test_preds = np.empty(X_test.get_shape()[0]) if X_test is not None else None

    scores = list()
    
    n_folds = 0
    if not isinstance(folds, list):
        folds = folds.split(X, y)
    
    i_fold = 0
    for trn_idx, val_idx in folds:
        n_folds += 1
        print('Training on fold {}'.format(i_fold + 1))
        i_fold += 1
        
        X_trn_fold = None
        X_val_fold = None
        y_trn = y.iloc[trn_idx].values
        y_val = y.iloc[val_idx].values
        
        if sparse_data:
            X_trn_fold = X[trn_idx]
            X_val_fold = X[val_idx]
            X_trn_fold = convert_sparse_matrix_to_sparse_tensor(X_trn_fold)
            X_val_fold = convert_sparse_matrix_to_sparse_tensor(X_val_fold)
        else:
            X_trn_fold = X.iloc[trn_idx].values
            X_val_fold = X.iloc[val_idx].values
        
        ## Train keras model ##
        fold_model = NN(**model_params)
        res = fold_model.fit(X_trn_fold, y_trn, X_val_fold, y_val, **train_params)
        
        #val_res = fold_model.evaluate(res['val_metric'])
        if X_test is not None:
            test_preds += np.array(fold_model(X_test.values)).flatten()
        
        scores.append(res['val_metric'])
    
    if X_test is not None:
        test_preds /= n_folds
    print('Training has finished.')
    print('Mean score:', np.mean(scores))
    
    return {
        'scores': scores,
        'test_preds' : test_preds
    }

class NN(Model):
    def __init__(self, custom_layers = None, loss = None, metric = None, optimizer = None, \
                 metric_higher_better = True, lr = 0.001):
        super(NN, self).__init__()
        
        self.lr = lr
        if custom_layers is None:
            self.custom_layers = []
            self.custom_layers.append(Dense(32, activation='relu', dtype = 'float32'))
            self.custom_layers.append(Dense(1, activation='sigmoid', dtype = 'float32'))
        else:
            self.custom_layers = custom_layers
        ##############
        if loss is None:
            self.loss = tf.keras.losses.BinaryCrossentropy()
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr,
                                                      beta_1=0.9,
                                                      beta_2=0.999)
        if metric is None:
            self.train_metric = tf.keras.metrics.AUC(name='train_metric')
            self.test_metric = tf.keras.metrics.AUC(name='test_metric')

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        
        ## Early stopping
        self.best_metric = None
        self.metric_higher_better = metric_higher_better
        
    def early_stopping(self, patience, metric, min_delta = 0.0001):
        if self.best_metric is None:
            self.best_metric = metric
            self.wait_ep = 0
            return False
        else:
            metric_improve = (metric > self.best_metric + min_delta) if self.metric_higher_better else \
                             (metric < self.best_metric - min_delta)
            if metric_improve:
                self.best_metric = metric
                self.wait_ep = 0
                return False
            else:
                self.wait_ep += 1
                if self.wait_ep > patience:
                    self.best_metric = None
                    return True
    
    #@tf.function
    def call(self, x):
        outs = [x]
        for l in self.custom_layers:
            outs.append(l(outs[-1]))
        return outs[-1]
    
    @tf.function
    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            predictions = self.call(X)
            loss = self.loss(y, predictions)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.train_loss(loss)
        self.train_metric(y, predictions)
    
    @tf.function
    def test_step(self, X, y):
        predictions = self.call(X)
        self.test_loss(self.loss(y, predictions))
        self.test_metric(y, predictions)
    
    #@tf.function
    def fit(self, X_tr, y_tr, X_val, y_val, epochs = 1, verbose = 1, batch_size = 32, shuffle = False, test_batch_size = 50, \
            patience = 1, min_delta = 0.0001):
        train_ds = None
        if shuffle:
            train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(X_tr.shape[0]).batch(batch_size)
        else:
            train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(X_tr.shape[0]).batch(batch_size)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(test_batch_size)
        
        res = None
        leave = True if verbose == 2 else False
        for epoch in trange(epochs, desc = 'Epochs'):
            for X, y in tqdm(train_ds, total = math.ceil(X_tr.shape[0] / batch_size), desc = 'Steps', leave = leave):
                self.train_step(X, y)

            for X, y in val_ds:
                self.test_step(X, y)
                
            if verbose > 0:
                template = 'Epoch {}, Loss: {}, Metric: {}, Val Loss: {}, Val Metric: {}'
                print(template.format(epoch+1,
                                      self.train_loss.result(),
                                      self.train_metric.result()*100,
                                      self.test_loss.result(),
                                      self.test_metric.result()*100))
                
            res = {'val_metric': np.array(self.train_metric.result())}
            early_stop = self.early_stopping(patience, res['val_metric'], min_delta = min_delta)
            
            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_metric.reset_states()
            self.test_loss.reset_states()
            self.test_metric.reset_states()
            
            if early_stop:
                print(f'Early stopping at epoch {epoch}.')
                break
            
        return res
    
    def evaluate(self, X_val, y_val, batch_size = 32):
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
        
        for X, y in val_ds:
            self.test_step(X, y)

        res = {'val_metric': self.test_metric.result()}
        self.test_metric.reset_states()
        return res
    
def train_single_col_(
        model, score, params_single,
        df, col, label_name, plot = False):
    """Train an lgbm on a single feature.
    
    """
    # Split data into train, test
    X_train, X_test, y_train, y_test = train_test_split(
        df[col], df[label_name], test_size=0.33, random_state=47)
    
    # Create model
    clf = model(**params_single)
    
    # Fit model
    clf.fit(X_train.to_frame().values, y_train)
    
    if plot:
        # Plot model predictions on the same column
        plt.figure(figsize=(12, 6))
        
        # Get predictions
        x = clf.predict(df[col].sort_values().unique().reshape(-1, 1))
        x = pd.Series(x, index=df[col].sort_values().unique())
        
        # Plot a heatmap
        sns.heatmap(x.to_frame(), cmap='RdBu_r', center=0.0);
        plt.xticks([]);
        plt.title(f'Model predictions for {col}')
        
    # Returns test score
    return score(y_test, clf.predict(X_test.to_frame().values))

def train_single_col_regression(df, col, label_name, plot = False):
    params_single = {'objective': 'regression', "boosting_type": "gbdt",
                     "subsample": 1, "bagging_seed": 11,
                     "metric": 'mse', 'random_state': 47}
    return train_single_col_(lgb.LGBMRegressor, mean_squared_error, params_single,
                             df, col, label_name, plot = plot)

def train_single_col_multiclass(df, col, label_name, plot = False):
    params_single = {'objective': 'multiclass', "boosting_type": "gbdt",
                     "subsample": 1, "bagging_seed": 11,
                     "metric": 'multi_error', 'random_state': 47}
    return train_single_col_(lgb.LGBMClassifier, accuracy_score, params_single,
                             df, col, label_name, plot = plot)