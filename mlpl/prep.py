import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer, Imputer
from tqdm import tqdm_notebook
import math
from . import utils, models
from .pipetools import dt
import scipy
from sklearn.preprocessing import OneHotEncoder

def reduce_mem_usage(dfs, verbose=True):
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    for df in dfs:
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else: df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        
    if len(dfs) > 1:
        return dfs
    else:
        return dfs[0]

def subsample(X, y, sampling_rates = None, sort_by = None):
    """
    Sample from a data in specified rates for each class.
    
    Args:
        X (pd.DataFrame): dataframe
        y (pd.Series): label
        sampling_rates(dict) - dictionary that contains classes(keys) and sampling rates(values) ex: {1.0: 1.0, 0.0:0.2}
    
    Returns:
        X (pd.DataFrame): dataframe output
        y (pd.Series): label output
    """
    
    sample_idx = []
    class_sizes = []
    for label in sampling_rates:
        class_idx = np.argwhere(y == label).flatten()
        size_class = int(len(class_idx)*sampling_rates[label])
        class_sizes.append(size_class)
        sample_idx = sample_idx + np.random.choice(class_idx, size=size_class, replace=False).tolist()
        
    X = X.iloc[sample_idx]
    y = y.iloc[sample_idx]
    
    if sort_by is not None:
        X['label_col++++'] = y
        X.sort_values(by = sort_by, inplace = True)
        y = X['label_col++++']
        X.drop('label_col++++', inplace = True, axis = 1)
        
    print('Class sizes after subsampling: ')
    for label, class_size in zip(sampling_rates, class_sizes):
        print(f'{label}: {class_size}')
        
    return X, y


def impute_col(df, col, method = 'mean'):
    """
    Imputes NaN values by a specified method, in specified columns.
    
    Args:
        dfs (list of pd.DataFrame): dataframes
        cols_to_impute (list of str): columns to impute
        method (str): 'mean', 'median', 'most_frequent'
    
    
    Returns:
        dfs (list of pd.DataFrame): dataframes
    """
    if not (method in ['mode', 'median', 'mean']): raise AssertionError()
        
    fill_val = None
    if method == 'mean':
        fill_val = df[col].mean(skipna = True)
    elif method == 'median':
        fill_val = df[col].median(skipna = True)
    else:
        if not (method == 'mode'): raise AssertionError()
        fill_val = df[col].mode(dropna = True).astype('float').values[0]
        
    if not (fill_val is not None): raise AssertionError()
    null_loc = df[col].isnull()
    if str(df[col].dtype) == 'category':
        new_col = df[col].astype('str')
        #print(new_col.isnull().mean())
        new_col.loc[null_loc] = str(fill_val)
        new_col = new_col.astype('float').astype('int').astype('category')
        
        if not (new_col.isnull().sum() == 0): raise AssertionError()
        return new_col
    else:
        new_col = df[col]
        new_col.loc[null_loc] = fill_val
        if not (new_col.isnull().sum() == 0): raise AssertionError()
        return new_col

def standardize_cols(dfs, cols, return_scaler = False, inplace = False):
    dfs = utils.tolist(dfs)
    
    if not isinstance(dfs[0], dt.DataTable):
        raise TypeError('Inputs must be datatables.')
        
    # Get standardized dataframe columns
    dfs_cols = standardize_dfs([df.df[cols] for df in dfs])
    
    
    # If not inplace, will return a copy
    if not inplace:
        dfs = [df.copy() for df in dfs]
        
    for df, df_cols in zip(dfs, dfs_cols):
        # Replace each standardized column in df
        for col in cols:
            df[col] = df_cols[col]
    
    if not inplace:
        return dfs

def standardize_dfs(dfs, return_scaler = False):
    """
    Modifies dfs.
    Standardizes dataframes after concatenation.
    
    Args:
        dfs (list of pd.DataFrame): dataframes
        return_scaler (bool): return scaler object for further use.
    
    Returns:
        dfs (list of pd.DataFrame): dataframes
    """
    dfs = utils.tolist(dfs)
    
    if not isinstance(dfs[0], pd.DataFrame):
        raise TypeError('Inputs must be dataframes.')
    
    if len(dfs) > 1:
        # Input dataframes must have the same columns
        if not (utils.df_hasthesamecols(dfs)): raise AssertionError()
    
    cols = dfs[0].columns
    scaler = StandardScaler()
    scaler.fit(pd.concat(dfs, axis = 0, ignore_index = True).values)
    dfs = [pd.DataFrame(data = scaler.transform(df.values), columns = cols) for df in dfs]
    
    if return_scaler:
        return dfs, scaler
    else:
        return dfs
    
def one_hot_encode(dfs, col, sparse = False, mode = 'replace', verbose = True):
    """ One hot encode
    Warning: returns a copy of the data.
    Args:
        dfs - list of dataframes
        cols_to_ohe - column name or list of column names
        sparse - true: return sparse
        mode - ['replace', 'return', 'add']
    Returns:
        dfs or dummies - list of dataframes or sparse csrs
    """
    ## DEBUG ##
    sparse = True
    
    
    if verbose:
        print(f'ohe: {col}')
    
    if not (mode in ['replace', 'return', 'add']): raise AssertionError()
    dfs = utils.tolist(dfs)

    if not isinstance(col, str):
        raise TypeError(f'"col" must be of type string.')
            
    col_concat = pd.concat([df[col] for df in dfs], axis = 0, ignore_index = True).to_frame()
    #dummies = pd.get_dummies(col_concat, drop_first=True, sparse=sparse, prefix = col + '_')
    dummies = OneHotEncoder(handle_unknown='ignore',
                            sparse = sparse).fit_transform(col_concat)
    
    sizes = [len(df) for df in dfs]
    sizes = [0] + sizes
    sizes = np.cumsum(np.array(sizes))
    
    # Separate training and test data
    dummies = [dummies[sizes[i]:sizes[i+1]] for i in range(len(dfs))]
    
    if sparse:
        #dummies = [dum.sparse.to_coo() for dum in dummies]
        # Type check
        if not all([isinstance(dum, scipy.sparse.csr.csr_matrix) for dum in dummies]):
            raise TypeError(f'Dummies has type {type(dummies)}')
            
    out = dummies
    
    new_name = col
    if mode == 'add':
        new_name = new_name + '_ohe'
        
    if mode in ['replace', 'add']:
        for df, dum in zip(dfs, dummies):
            df[new_name] = dum
            
        out = dfs
    
    # No need to do anything for mode 'return'
    return out if len(out) > 1 else out[0]

def thermometer_encode(dfs, label_name, cols_to_enc = None, mode = 'replace'):
    """ Thermometer encode
    Warning: returns a copy of the data.
    Args:
        dfs - list of dataframes
        cols_to_ohe - column name or list of column names
        sparse - true: return sparse
        mode - ['replace', 'return']
    Returns:
        dfs or dummies - list of dataframes or sparse csrs
    """
    
    if not (mode in ['replace', 'return']): raise AssertionError()
    dfs = utils.tolist(dfs)
    
    # Columns to encode must be a string, list or set
    if not (isinstance(cols_to_enc, list) \
         or isinstance(cols_to_enc, str) \
         or isinstance(cols_to_enc, set)):
        raise AssertionError()
        
    cols_to_enc = utils.tolist(cols_to_enc)
    if cols_to_enc[0] is None:
        cols_to_enc = dfs[0].columns
    cols_no_process = set(dfs[0].columns.difference(cols_to_enc))
    if len(dfs) > 1:
        cols_no_process.update(set(dfs[0].columns).symmetric_difference(set(dfs[1].columns)))
    
    def get_block(val, val_count, size):
        return np.append(np.ones((size, val)), np.zeros((size, val_count - val)), axis = 1)
    
    df_concat = pd.concat([df[cols_to_enc] for df in dfs], axis = 0, ignore_index = True)
    for col in df_concat.columns:
        if(df_concat[col].isnan().sum() > 0):
            print(f'Column {col} has null values. Imputing null values will improve your metrics.')
            
        values = df_concat[col].unique()
        values.sort()
        therm_cols = []
        for val in values:
            new_col = f'{col}_therm={val}'
            df_concat[new_col] = 0
            therm_cols.append(new_col)
            
        for i_val, val in enumerate(values):
            df_concat.loc[df_concat[col] == val, therm_cols] = get_block(i_val, len(therm_cols), len(df_concat[df_concat[col] == val]))
    
    df_concat = df_concat.drop(cols_to_enc, axis = 1)
    sizes = [len(df) for df in dfs]
    sizes = [0] + sizes
    sizes = np.cumsum(np.array(sizes))
    # separate
    dfs_encoded = [df_concat.iloc[sizes[i]:sizes[i+1]].reset_index(drop = True) for i in range(len(dfs))]
    if mode == 'replace':
        dfs = [pd.concat([df[cols_no_process], df_encoded] for df, df_encoded in zip(dfs, dfs_encoded))]
        return dfs if len(dfs) > 1 else dfs[0]
    else: # mode = 'return'
        return dfs_encoded if len(dfs_encoded) > 1 else dfs_encoded[0]
        
        

def impute_dense(df, col, method = 'median'):
    """
    Imputes NaN values by a specified method, in specified columns.
    
    inputs:
    dfs - dataframes
    col - column to impute
    method: 'mean', 'median', 'most_frequent'
    
    """
    df_col = df[col].copy()
    imputer = Imputer(missing_values = 'NaN', strategy = method, axis = 0)
    res = pd.Series(imputer.fit_transform(df_col.values.reshape(-1, 1)).flatten())
    return res

def impute_mean(df, col):
    return impute_col(df, col, method = 'mean')

def impute_mode(df, col):
    return impute_col(df, col, method = 'mode')

def impute_median(df, col):
    return impute_col(df, col, method = 'median')

def impute_bygroupby(df, cols_to_groupby, col_to_impute, fill_mode = 'median'):
    """
    Impute a column based on groups of other columns.
    
    Args:
        df (pd.DataFrame): input dataframe
        cols_to_groupby (list of strings): columns to groupby
        col_to_impute (str): name of the column to be imputed
        fill_mode (str): {'mode', 'median', 'mean'}, fill mode
        
    Returns:
        imputed_col (pd.Series): copy of imputed column
    """
    null_ratio_orig = df[col_to_impute].isnull().mean()
    
    # col_to_impute must be a string
    if not (isinstance(col_to_impute, str)): raise AssertionError()
    
    df_copy = df[cols_to_groupby if col_to_impute not in cols_to_groupby else df.columns.difference([col_to_impute])]
    for col in df_copy.columns:
        if str(df_copy[col].dtype) == 'category':
            df_copy[col] = df_copy[col].astype('str').fillna('-999').astype('category')
    df_copy = pd.concat([df_copy, df[[col_to_impute]]], axis = 1)
    
    if col_to_impute in cols_to_groupby:
        cols_to_groupby.remove(col_to_impute)
    
    cols_to_groupby = utils.tolist(cols_to_groupby)
    is_category = False
    overall_fill = None
    if str(df[col_to_impute].dtype) == 'category':
        overall_fill = df[col_to_impute].mode().astype('float').astype('str').values[0]
        print(overall_fill)
        is_category = True
        # fill_mode must be 'mode' for category types
        if not (fill_mode == 'mode'): raise AssertionError()
        
    else:
        if fill_mode == 'mode':
            overall_fill = df[col_to_impute].mode().values[0]
        elif fill_mode == 'median':
            overall_fill = df[col_to_impute].median()
        else:
            overall_fill = df[col_to_impute].mean()
    
    mode_to_func = {'mode': lambda x: x.mode(),
                    'median': lambda x: x.median(),
                    'mean': lambda x: x.mean()}
    
    def fill_val(x, fill_mode):
        m = None
        if is_category:
            nulls = (x == 'nan')
            x.loc[nulls] = None
        
        m = mode_to_func[fill_mode](x)
        if isinstance(m, float) or isinstance(m, int):
            return x.fillna(m)
        if m is not None and len(m) > 0:
            return x.fillna(m[0])
        else:
            return x.fillna(overall_fill)
        
    if str(df_copy[col_to_impute].dtype) == 'category':
        df_copy[col_to_impute] = df_copy[col_to_impute].astype('str')
    
    imputed_col = df_copy.groupby(cols_to_groupby)[col_to_impute].transform(lambda x: fill_val(x, fill_mode))
    imputed_col = imputed_col.fillna(overall_fill)
    
    if is_category:
        imputed_col = imputed_col.astype('category')
    
    
    
    # Hopefully you will never see this.
    if not (imputed_col.isnull().mean() == 0): 
        raise AssertionError(f'Col has nulls. Before: {null_ratio_orig}, After: {imputed_col.isnull().mean()}')
    return imputed_col

def full_impute_groupby(
        dfs, cols_to_groupby,
        col_to_impute, fill_mode = 'median'):
    
    """Replaces column in dataframes."""
    #TODO: simplify this, this is ugly
    dfs = utils.tolist(dfs)
    dfs_cols = utils.utilize(mode = 'return', verbose = False, col_to_impute = col_to_impute, fill_mode = fill_mode)\
    (impute_bygroupby)(dfs, cols_to_groupby + [col_to_impute])
    for df, df_col in zip(dfs, dfs_cols):
        df[col_to_impute] = df_col
        
def imputeby_dependents(
        train, test,
        to_impute, feature_properties,
        to_exclude = None, n = 2,
        fill_mode = 'median', lgbm_objective = 'multiclass',
        mode = 'replace'):
    """
    Imputes null values in a column by mean/median/mode of a groupby.
    For numerical columns, set lgbm_objective to 'regression'
    Add label to to_exclude.
    
    Args:
        train (pd.Dataframe): dataframe
        test (pd.Dataframe): dataframe
        to_impute (str): column to be imputed, string
        feature_properties (pd.DataFrame): dataframe that can be obtained by vis.get_all_col_properties()
        to_exclude (str or list of str): features to be excluded, such as label
        unique_limit (int): unique value ratio limit, float
        lgbm_objective (str): lgbm requires a different objective for different data types.
                       'regression' : for integer/float types
                       'multiclass' : for category/object types or supposedly integer types
    """
    # mode must be one of ['add', 'replace', 'return']
    if not (mode in ['add', 'replace', 'return']): raise AssertionError()
        
    impute_metrics = impute_usefulness(train, test, to_impute, feature_properties, to_exclude = to_exclude, lgbm_objective = lgbm_objective)
    to_impute_by = list(impute_metrics['feature'].iloc[:n])
    print('IMPUTE BY:')
    print(to_impute_by)
    print(impute_metrics.iloc[:5].to_html())
    if mode == 'add':
        # Adds imputed column to the dataframe
        new_name = f'imputeby_dependents_{to_impute}'
        train[new_name] = train[to_impute]
        test[new_name] = test[to_impute]
        full_impute_groupby([train, test], to_impute_by, new_name, fill_mode = fill_mode)
    elif mode == 'replace':
        # Mutates column to be imputed
        full_impute_groupby([train, test], to_impute_by, to_impute, fill_mode = fill_mode)
    else: #mode = 'return'
        to_impute_by = utils.tolist(to_impute_by)
        train_col_copy = train[[to_impute] + to_impute_by]
        test_col_copy = train[[to_impute] + to_impute_by]
        full_impute_groupby([train_col_copy, test_col_copy], to_impute_by, to_impute, fill_mode = fill_mode)
        return train_col_copy[to_impute], test_col_copy[to_impute]
    
def full_binning(dfs, col, bins):
    """Quantlie binning.
    Modifies dataframe.
    
    Args:
        dfs (list of pd.DataFrame): dataframes
        col (str): column name
        bins (int): bin count
    """
    @utils.utilize(mode = 'replace')
    def full_bin(df, col):
        return binning(df, col, bins = bins)
    full_bin(dfs, col)

def binning(df, col, strategy = 'quantile', bins = 10):
    """Binning. Returns pd.Series.
    
    Args:
        df (): dataframe
        col (str): column name
        strategy (str): bin strategy for sklearn.KBinsDiscretizer()
        bins (int): bin count
        
    Returns:
        binned_col (pd.Series): result of binning
    """
    
    col = utils.tolist(col)
    binned = pd.Series(KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy).fit_transform(df[col].fillna(-9999).to_numpy()).flatten())
    binned = binned.astype('int')
    return pd.Series(binned)

def binning_values(df, col, strategy = 'quantile', bins = 10):
    """Binning, but returns group values instead of group names. Returns pd.Series.
    
    Args:
        df (pd.DataFrame): dataframe
        col (str): column name 
        strategy (str): bin strategy for sklearn.KBinsDiscretizer()
        bins (int): bin count
        
    Returns:
        binned_col (pd.Series): result of binning, but values
    """
    
    col = utils.tolist(col)
    disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
    binned = disc.fit_transform(df[col].fillna(-99999).to_numpy())
    return pd.Series(disc.inverse_transform(binned).flatten())

def auto_bin_cols(dfs, cols):
    """
    Quantile bins a column if number of unique values is larger than 30.
    bin_count =  - 40 + math.ceil(8 * np.log2(unique_count))
    Modifies dataframe.
    
    Args:
        dfs (list of pd.DataFrame): dataframes
        cols (list of str): column names
    """
    
    cols = utils.tolist(cols)
    dfs = utils.tolist(dfs)
    
    for col in cols:
        unique_count = pd.concat([df[col] for df in dfs], axis = 0, ignore_index = True).nunique()
        if unique_count < 30:
            print(f'Skip {col}, nunique = {unique_count}')
            continue
        
        # This formula seemed more natural
        bin_count =  - 40 + math.ceil(8 * np.log2(unique_count))
        print(f'Binning {col} into {bin_count} bins.')
        full_binning(dfs, col, bin_count)
        
def pipeline_sklearn_preprocess(
        dfs, label_name = None,
        nominals = None, ordinals = None,
        numerics = None, return_sparse = True,
        return_scaler = False, inplace = False):
    """
    * Features must have dtypes integer, float or category.
    * Training set must be the first item in dfs.
    * If return_sparse is True, encoded examples will be returned in type scipy.sparse
    
    Important: Imputation and feature engineering must be completed before this step.
    
    Nominals are one-hot encoded.
    Ordinals are thermometer encoded. (Increasing with integer or float)
    Numerics are standardized.
    (Categoricals must have dtype int or category.)
    Label will not be modified.
    """
    # Type & col check
    
    dfs = utils.tolist(dfs)
    train = dfs[0]
    cols = train.columns
    if label_name is not None:
        cols.drop(label_name)
    
    if return_sparse:
        print('CONVERT CATS TO INT:')
        for i, df in enumerate(dfs):
            for col in df.columns:
                if str(df[col].dtype) == 'category':
                    if i == 0:
                        print(col)
                    df[col] = df[col].astype('int')
        print('DROP OBJECTS:')
        for i, df in enumerate(dfs):
            for col in df.columns:
                if str(df[col].dtype) == 'object':
                    if i == 0:
                        print(col)
                    df.drop(col, axis = 1, inplace = True)
                        
    # Label name must be present
    if label_name is not None:
        if not (isinstance(label_name, str)): raise AssertionError()
    
    numerics = utils.tolist(numerics)
    # Numerics must be float or int
    if numerics[0]  is not None:
        if not(((train[numerics].dtypes == 'int64')  \
              | (train[numerics].dtypes == 'int')    \
              | (train[numerics].dtypes == 'float64')\
              | (train[numerics].dtypes == 'float')
               ).all()):
            raise AssertionError()
    
    ordinals = utils.tolist(ordinals)
    # Ordinals must be int or category
    if ordinals[0]  is not None:
        if not (((train[ordinals].dtypes == 'int') \
               | (train[ordinals].dtypes == 'category')).all()):
            raise AssertionError()
    
    nominals = utils.tolist(nominals)
    # Nominals must be int or category
    if nominals[0] is not None:
        if not (((train[nominals].dtypes == 'int64') \
               | (train[nominals].dtypes == 'int') \
               | (train[nominals].dtypes == 'category')).all()):
            raise AssertionError()
    
    ###########
    
    # Standardize numerics
    scaler = None
    dfs_num = None
    if numerics[0]  is not None:
        print('STANDARDIZE NUMERICS:')
        print(numerics)
        dfs_num = standardize_dfs([df[numerics] for df in dfs], return_scaler)
        for df in dfs_num:
            df.columns = [col + '_standard' for col in df.columns]
        if return_scaler:
            scaler = dfs_num[1]
            dfs_num = dfs_num[0]
        
    # OHE nominals
    dfs_nom = None
    if nominals[0] is not None:
        print('OHE NOMINALS:')
        print(nominals)
        dfs_nom = one_hot_encode([df[nominals] for df in dfs], False)
            
    # Thermometer encode ordinals
    dfs_ord = None
    if ordinals[0] is not None:
        print('THERMOMETER ENCODE ORDINALS:')
        print(ordinals)
        dfs_ord = thermometer_encode([df[ordinals] for df in dfs], label_name)
    to_concat = [df_pieces for df_pieces in [dfs_num, dfs_nom, dfs_ord] if df_pieces is not None]
    
    # Concatenate new columns separetely for each dataframe
    out_dfs = None
    out_dfs = [pd.concat([df_piece[i] for df_piece in to_concat], axis = 1) for i in range(len(dfs))]
    
    if return_sparse:
        dfs = [scipy.sparse.hstack([df, df_out]).tocsr() for df, df_out in zip(dfs, out_dfs)]
    else:
        dfs = [pd.concat([df, df_out], axis = 1) for df, df_out in zip(dfs, out_dfs)]
        
    if return_scaler:
        return dfs, scaler
    else:
        return dfs

    
def process_dense(df, col): #, impute = True, standardize = False
    """
    Fill nans with median
    """
    
    return impute_dense(df, col, method = 'median')


def process_nominal(df, col):
    """
    Fill nans with 'nan'
    Group outliers
    Convert to type 'category' for lgbm
    """

    df_col = impute_col(df, col, method = 'mode')
    df_col = df_col.astype('str')
    #df_col = group_outliers(df_col.to_frame(name = col),  col, limit = limit)
    
    df_col = pd.Series(LabelEncoder().fit_transform(list(df_col.astype(str).values)))
    return df_col.astype('category')

def drop_object_cols(dfs):
    print('DROP OBJECT COLS:')
    for df in dfs:
        for col in df.columns:
            if str(df[col].dtype) == 'object':
                print(col)
                df.drop(col, axis = 1, inplace = True)

def process_nominals(dfs, cols = None, mode = 'replace', verbose = True):
    """
    Fill nans with 'nan'
    Group outliers
    Convert to type 'category' for lgbm
    
    inputs:
    dfs - DataFrames
    cols - list of column names
    limit - value count limit to be considered as outlier. Values with frequencies lower than limit are grouped separetely.
    mode - ['replace', 'add']
    verbose - print steps
    
    """
    if not (mode in ['replace', 'add']): raise AssertionError()
    dfs = utils.tolist(dfs)
    if cols is None:
        cols = list(dfs[0].columns)
    else:
        cols = utils.tolist(cols)
    
    if verbose:
        print('\n<b>PROCESS NOMINAL COLS:</b>')
        print('   - Fillna("mode")')
        #print('   - Group outliers (limit = 10), label = -9999')
        print('   - Label encode')
        print('   - Convert to category')
    
    for col in cols:
        if verbose:
            print(col)
        utils.utilize(mode = mode, verbose = False)(process_nominal)(dfs, col)

        
def process_denses(dfs, cols = None, verbose = True, mode = 'replace'):
    """
    Fill nans with median
    
    inputs:
    dfs - DataFrames
    cols - list of column names
    verbose - print steps
    mode - ['replace', 'add']
    
    """
    if not (mode in ['replace', 'add']): raise AssertionError()
    
    dfs = utils.tolist(dfs)
    if cols is None:
        cols = list(dfs[0].columns)
    
    cols = utils.tolist(cols)
    
    if verbose:
        print('\n<b>PROCESS DENSE COLS: </b>')
        print('   - Impute with median')
    
    
    for col in cols:
        if verbose:
            print(col)
        utils.utilize(mode = mode, verbose = False)(process_dense)(dfs, col)

def label_encode(df, col):
    if isinstance(df, pd.Series):
        nans = df.isnull()
        le = pd.Series(LabelEncoder().fit_transform(list(df.astype(str).values))).astype('category')
        if nans.sum() > 0:
            le.loc[nans] = None
        return le
    else:
        nans = df[col].isnull()
        le = pd.Series(LabelEncoder().fit_transform(list(df[col].astype(str).values))).astype('category')
        if nans.sum() > 0:
            le.loc[nans] = None
        return le
        
def group_outliers_replace(dfs, cols, limit = 10):
    return utils.utilize(mode='replace', limit = limit)(group_outliers)(dfs, cols)

def group_outliers(df, col, limit = 10):
    if isinstance(col, list):
        col = col[0]
    if not (isinstance(col, str)): raise AssertionError()
    col_concat = df[col].copy()
    val_counts = col_concat.value_counts()
    values = val_counts.index.values
    new_values = values.copy()
    is_category = False
    
    if str(new_values.dtype) == 'category':
        new_values = new_values.astype('float')
        
    new_values[np.array(val_counts < limit)] = -9999
    if is_category:
        new_values = new_values.astype('category')
    
    to_map = pd.Series(data = new_values, index = values)
    return col_concat.map(to_map)

def label_encode_obj_cols(dfs, join = False):
    """Label encode all object columns in a DataFrame (REPLACE).
    (Do not use utils.utilize decorator.)
    
    Args:
        dfs (list of pd.DataFrame): list of pandas DataFrames or a single DataFrame
    
    Returns:
        dfs (list of pd.DataFrame): input DataFrames with column(s) replaced.
    """
    if isinstance(dfs, list):
        for df in dfs:
            # Inputs must be dataframes
            if not (isinstance(df, pd.DataFrame)): raise AssertionError()
        
        cols = dfs[0].columns
        for col in cols:
            if dfs[0][col].dtype == 'object':
                val_list = []
                if not join:
                    for df in dfs:
                        val_list = val_list + list(df[col].astype(str).values)
                    le = LabelEncoder().fit(val_list)
                    for df in dfs:
                        df[col] = pd.Series(le.transform(list(df[col].astype(str).values))).astype('category')
                else:
                    col_concat = pd.concat([df[col] for df in dfs], axis = 0, ignore_index = True)
                    le = LabelEncoder().fit(list(col_concat.astype(str).values))
                    for df in dfs:
                        df[col] = pd.Series(le.transform(list(df[col].astype(str).values))).astype('category')
    else:
        # Input must be a dataframe
        if not (isinstance(dfs, pd.DataFrame)): raise AssertionError()
        
        cols = dfs.columns
        for col in cols:
            if dfs[0][col].dtype == 'object':
                dfs[col] = pd.Series(le.transform(list(dfs[col].astype(str).values))).astype('category')
        
    return dfs

def label_encode_sort(train, test, col, label_name):
    mean_label = train[label_name].mean()
    train_sub = train[[col, label_name]]
    
    train_sub[col] = train_sub[col].astype('str').fillna('nan')
    test[col] = test[col].astype('str').fillna('nan')
    val_mean_targets = train_sub.groupby(col)[label_name].mean()
    only_test_values = [val for val in list(test[col].unique()) if val not in val_mean_targets.index]
    val_mean_targets = val_mean_targets.append(pd.Series({val: mean_label for val in only_test_values})).sort_values(ascending = True)
    
    le = preprocessing.LabelEncoder().fit(list(val_mean_targets.index))
    return pd.Series(le.transform(train_sub[col])), pd.Series(le.transform(test[col]))

def impute_usefulness(train, test, to_impute, feature_properties, to_exclude = None, unique_limit = 0.3, lgbm_objective = 'multiclass'):
    """
    Finds how much <to_impute> depends on each column by training an lgbm for each column.
    columns with unique value ratios > unique limit are not considered.
    Inputs:
    - train : dataframe
    - test : dataframe
    - to_impute : column to be imputed, string
    - feature_properties : dataframe that can be obtained by vis.get_all_col_properties()
    - to_exclude : features to be excluded, such as label
    - unique_limit : unique value ratio limit, float
    - lgbm_objective : lgbm requires a different objective for different data types.
                     'regression' : for integer/float types
                     'multiclass' : for category/object types or supposedly integer types
    """
    if to_exclude:
        # to_exclude must be a column name (string) or a set
        if not (isinstance(to_exclude, set) or isinstance(to_exclude, str)): raise AssertionError()
        
    unique_limit = 0.3
    cols_to_try = list(train.columns.difference([to_impute] + utils.tolist(list(to_exclude)) if to_exclude is not None else [to_impute]))
    col_scores = pd.DataFrame({'feature': cols_to_try})
    col_scores['score'] = None
    
    # Don't use sparse columns
    cols_to_try = [col for col in cols_to_try if col in feature_properties.feature.values]
    
    # Eliminate Id columns
    cols_to_try = [col for col in cols_to_try if feature_properties.loc[feature_properties.feature == col, 'full_unique_ratio'].values[0] < unique_limit]
    for col in cols_to_try:
        if '_' + to_impute in col:
            print(f'Skip {col}, as it was possibly generated from {to_impute}')
            cols_to_try.remove(col)
    
    # Label encode if necessary
    trn_orig = train[to_impute].copy()
    test_orig = test[to_impute].copy()
    if str(train[to_impute].dtype) in ['object', 'category']:
        utils.utilize(mode = 'replace')(label_encode)([train, test], to_impute)
        # Don't set lgbm_objective to regression, as dtype for column is category or object
        if not (lgbm_objective != 'regression'): raise AssertionError()
    
    for col in tqdm_notebook(cols_to_try):
        df_concat = pd.concat([train[[col, to_impute]], test[[col, to_impute]]], axis = 0, ignore_index = True).dropna()
        if str(df_concat[col].dtype) in ['object', 'category']:
            df_concat[col] = label_encode(df_concat, col)
        
        if lgbm_objective == 'multiclass':
            col_scores.loc[col_scores.feature == col, 'score'] = models.train_single_col_multiclass(df_concat, col, to_impute)
        elif lgbm_objective == 'regression':
            col_scores.loc[col_scores.feature == col, 'score'] = models.train_single_col_regression(df_concat, col, to_impute)
        else:
            raise Exception(f'lgbm_objective {lgbm_objective} not implemented')
            
    train[to_impute] = trn_orig
    test[to_impute] = test_orig
    
    return col_scores.sort_values(by = 'score', ascending = False).reset_index(drop = True)

def process_ordinal_col(df, col, value_order, limit = 5):
    """
    Label encode in order
    Group outliers
    Impute ordinal column with mode
    Returns result as pd.Series
    
    Args:
        dfs (pd.DataFrame): DataFrame
        col (dict): Dictionary of format {col_name: [val1, val2, ...]}
        limit (int): Value count limit to be considered outlier.
                    (Values with a count lower than this limit are grouped with their neighbor with lower count.)
        value_order (list of values): label encoding takes place in this order. ex: (['A', 'B', 'C'] is converted into [0, 1, 2])
    Returns:
        ordinal_encoded (pd.Series): encoded column as pd.Series
    """
    
    value_map = {val: i for i, val in enumerate(sorted(df[col].unique(), key=value_order))}
    new_col = df[col].map(value_map)
    value_c = new_col.value_counts()
    
    # Add to the neighbor with the lowest count
    for i, (val, count) in enumerate(value_counts.items()):
        if count < limit:
            up_count = value_c[i+1]  if (i+1) < len(value_c) else 999999
            dn_count = value_c[i-1]  if (i-1) > 0 else 999999
            
            if up_count < dn_count:
                new_col = new_col.map({val: value_c.index[i + 1]})
                value_c[value_c.index[i + 1]] += count
            else:
                new_col = new_col.map({val: value_c.index[i - 1]})
                value_c[value_c.index[i - 1]] += count
            #value_c.drop(val)
            value_c = new_col.value_counts()
    new_col = prep.impute_dense(df, col, method= 'most_frequent')
    
    return new_col
    
    
    
def process_ordinal_cols(dfs, cols, limit = 5, verbose = True):
    """Label encode in order
    Group outliers
    Impute ordinal column with mode
    Modifies dataframes.
    Args:
        dfs (list of pd.DataFrames): DataFrames
        cols (dict): Dictionary of format {col_name1: [val1, val2, ...], col_name2: [val1, val2, ...]}
        limit (int): Value count limit to be considered outlier.
                    (Values with a count lower than this limit are grouped with their neighbor with lower count.)
        verbose (bool): print steps
        
    """
    if verbose:
        print('\n<b>PROCESS ORDINAL COLS:</b>')
        print('   - Fillna(mode)')
        print('   - Group outliers (limit = 5), add to less crowded neighboring group')
    for col, val_order in cols.items():
        if verbose:
            print(col)
        @utils.utilize(mode='replace')
        def full_proc_ordinal(df, col):
            return process_ordinal_col(df, col, val_order, limit)
        full_proc_ordinal(df, col)