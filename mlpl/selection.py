import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
import lightgbm as lgb

def drop_probably_useless(train, test, label_name):
    """
    A column is probably useless if:
    - Has mostly null values
    - Has a single value
    - Same value is seen 90% of the time
    
    From: https://www.kaggle.com/nroman/recursive-feature-elimination
    """
    one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
    one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

    many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
    many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

    big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

    cols_to_drop = list(set(many_null_cols + many_null_cols_test + \
                            big_top_value_cols + big_top_value_cols_test + \
                            one_value_cols + one_value_cols_test))
    if label_name in cols_to_drop:
        cols_to_drop.remove(label_name)
    
    print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))
    print('Null rate > 0.9 :\n {}'.format(many_null_cols + many_null_cols_test))
    print('Single value:\n {}'.format(one_value_cols + one_value_cols_test))
    print('Same value with gt 0.9 rate:\n {}'.format(big_top_value_cols + big_top_value_cols_test))
    
    train = train.drop(cols_to_drop, axis=1)
    test = test.drop(cols_to_drop, axis=1)
    
def auto_rfe(
        train, test, label_name,
        params = None, cols_not_to_model = None, cols_to_sort = None,
        save_folder = None, scoring = 'roc_auc'):
    """ default parameters:
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
    Mostly from: https://www.kaggle.com/nroman/recursive-feature-elimination
    """
    all_cols = train.columns.difference([label_name])
    train, test = fe.label_encode_obj_cols([train, test], join = True)

    y = train[label_name]
    if label_name not in cols_not_to_model:
        cols_not_to_model = cols_not_to_model + [label_name]
    
    if cols_not_to_model is not None:
        cols_not_to_model = utils.tolist(cols_not_to_model)
        
    cols_to_keep = train.columns.difference(cols_not_to_model)
    if cols_to_sort is not None:
        cols_to_sort = utils.tolist(cols_to_sort)
        for df in train, test:
            for col in cols_to_sort:
                df.sort_values(by = col, inplace = True)
        
    for df in train, test:
        df.drop(df.columns.difference(cols_to_keep), axis = 1, inplace = True)
        for col in cols_to_keep:
            if str(train[col].dtype) == 'category':
                df[col] = df[col].astype('int')
            
    train.fillna(-9999, inplace=True)
    X = train
    
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
    
    clf = lgb.LGBMClassifier(**params)
    rfe = RFECV(estimator=clf, step=10, cv=StratifiedKFold(n_splits=5), scoring=scoring, verbose=2)
    rfe.fit(train, y)
    
    print('Optimal number of features:', rfe.n_features_)
    print('All features with rank 1')
    
    rank1_cols = X.columns[rfe.ranking_ == 1]
    for col in rank1_cols: print(col)
    
    plt.figure(figsize=(14, 8))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
    plt.show()
    
    print(f'Will keep {len(rank1_cols)} features out of {len(all_cols)}')
    for df in train, test:
        df.drop(df.columns.difference(rank1_cols), axis = 1, inplace = True)
        
    train[label_name] = y
    if save_folder is not None:
        print('Will save train and test as "train_rfe.csv" with label and "test_rfe.csv"')
        train.to_csv(os.path.join(save_folder, 'train_rfe.csv'), index = False)
        test.to_csv(os.path.join(save_folder, 'test_rfe.csv'), index = False)
    return {'used_cols': rank1_cols, 'data': (train, test)}

def grouped_feature_ranking(new_fe, X, y, folds, params, group_size, num_runs = 1, to_drop = None):
    def slice_by_index(lst, indexes):
        """Slice list by positional indexes.
        Adapted from https://stackoverflow.com/a/9108109/304209.
        """
        if not lst or not indexes:
            return []
        slice_ = itemgetter(*indexes)(lst)
        if len(indexes) == 1:
            return [slice_]
        return list(slice_)
    
    
    # Keep folds
    folds_list = [fold for fold in folds.split(X, y)]
    
    original_cols = list(X.columns)
    original_cols.remove(to_drop)
    num_all_cols = len(original_cols)
    
    # Create a dictionary that maps column names to fe functions
    #fe_feature_final_names = []
    all_cols = [(col, None) for col in original_cols]
    for fe_func in new_fe:
        if not isinstance(new_fe[fe_func], list):
            new_fe[fe_func] = [new_fe[fe_func]]
            
        num_all_cols += len(new_fe[fe_func])
        if isinstance(fe_func, str):
            if not (fe_func in list(fe_functions.keys())): raise AssertionError()
            for col_name in new_fe[fe_func]:
                all_cols.append((col_name, fe_functions[fe_func]))
                
        else:
            # Must be a function
            if not (hasattr(fe_func, '__call__')): raise AssertionError()
            for col_name in new_fe[fe_func]:
                all_cols.append((col_name, fe_func))
    
    feature_gain = pd.DataFrame(index = np.arange(num_runs))
    for i_run in tqdm(range(num_runs)):
        # Form feature set
        print(f'Number of all features: {num_all_cols}')
        idx = list(np.random.choice(len(all_cols), group_size, replace = False))
        run_cols = slice_by_index(all_cols, idx)
        run_cols_nofe = []
        run_fe_cols_train = []
        run_cols_final_name = []
        for col, func in run_cols:
            if func is not None:
                #fe_col_name = f'{func.__name__}_[{col}]'
                f_col_out = func(X, col)
                run_fe_cols_train.append(f_col_out.to_frame()) #.rename({col: fe_col_name})
                #print(f_col_out.name)
                run_cols_final_name.append(f_col_out.name)
            else:
                run_cols_nofe.append(col)
                run_cols_final_name.append(col)
        
        result_lgb = train_lgbm(folds_list, params, X[run_cols_nofe].append(run_fe_cols_train), y)
        perf = np.mean(result_lgb['aucs'])
        for col in run_cols_final_name:
            if col not in list(feature_gain.columns):
                feature_gain[col] = None
            feature_gain.at[i_run + 1, col] = perf
        #feature_gain.at[i_run + 1, 'run_performance'] = perf
        feature_gain.iloc[0] = feature_gain.mean(axis = 0)
        feature_gain.sort_values(by = 0, axis = 1, ascending = False, inplace = True)
        counts = ((~(feature_gain.iloc[:,:25].isnull())).sum(axis = 0) - 1).transpose()
        clear_output()
        to_display = feature_gain.iloc[0,:25].transpose().to_frame('mean score')
        to_display['mean score'] = to_display['mean score'].astype('float32')
        to_display['run count'] = counts
        display(to_display.style.background_gradient(cmap='coolwarm', subset=["mean score"]))
    return feature_gain