import numpy as np
import pandas as pd
import os
import functools

def utilize(join_dfs = True, mode = 'replace', verbose = True, **kwargs):
    """
    A decorator for processing dataframes with less code.
    
    Parameters
    ----------
    join_dfs : bool
        if true, calculates feature on concatenated dataframes.
    mode : string
        'replace': replace existing column (len(cols) must be 1.)
        'return_col' : return created column as pd.Series
        'add' : add new column to dataframes
        'verbose' : print what you have done
        
    """
    # Mode must be one of ['replace', 'add', 'return']
    if not (mode in ['replace', 'add', 'return']): raise AssertionError()
        
    def real_decorator(func):
        @functools.wraps(func)
        def func_wrapper(dfs, cols):
            single_col = False
            if isinstance(cols, str):
                single_col = True
            
            cols = tolist(cols) # Convert to list if len == 1
            dfs = tolist(dfs)
            
            # Generate name for new feature
            func_label = func.__name__ + '_' + cols[0] if (len(cols)==1) else '+'.join(cols)
            if len(kwargs) > 0:
                func_label = func_label + '-' + ','.join('%s=%r' % x for x in kwargs.items())
            if join_dfs:
                # Calculate feature on concatenated train and test
                sizes = [len(df) for df in dfs]
                sizes = [0] + sizes
                sizes = np.cumsum(np.array(sizes))
                fe_cols = func(pd.concat([df[cols] for df in dfs], ignore_index = True, axis = 0).reset_index(drop = True), \
                               cols[0] if single_col else cols, **kwargs)
                fe_cols = [fe_cols.iloc[sizes[i]:sizes[i+1]].reset_index(drop = True).rename(func_label) for i in range(len(dfs))]
            else:
                fe_cols = [func(df, cols[0] if single_col else cols, **kwargs).rename(func_label) for df in dfs]
            
            if verbose:
                print('{} : {}'.format(mode ,cols[0]))
            
            if mode == 'return':
                if len(fe_cols) == 1:
                    fe_cols = fe_cols[0]
                return fe_cols
            
            # If mode is add or replace
            for df, fe_col in zip(dfs, fe_cols):
                if mode == 'add':
                    df[func_label] = fe_col
                else: # mode == 'replace':
                    df[cols[0]] = fe_col
        return func_wrapper
    return real_decorator

def df_hasthesamecols(dfs):
    """
    Checks if a list of dataframes have the same columns in the same order.
    """
    
    cols_data = dfs[0].columns
    if not (len(dfs) > 1): raise AssertionError()
    for df in dfs[1:]:
        cols = list(df.columns)
        for col, col_d in zip(cols, cols_data):
            if not (col == col_d):
                return False
    return True

def tolist(dfs):
    if isinstance(dfs, pd.Index):
        dfs = dfs.tolist()
    
    if not isinstance(dfs, list):
        dfs = [dfs]
    return dfs

def load_all(data_folder, extension = '.csv'):
    files = []
    print('# File sizes')
    for f in os.listdir(data_folder):
        if extension in f:
            print(f.ljust(30) + str(round(os.path.getsize(os.path.join(data_folder, f)) / 1024**2, 2)) + 'MB')
            files.append(pd.read_csv(os.path.join(data_folder, f)))
    return files

def dtypes(df, plot = False):
    keys = dict.fromkeys([''.join(i for i in str(t) if not i.isdigit()) for t in set(df.dtypes)])
    types = dict.fromkeys(set(keys))
    for t in types:
        types[t] = pd.Series(df.dtypes[df.dtypes.map(str).apply(lambda x: t in x)].index)
    types_df = pd.DataFrame(types)
    if plot:
        display(types_df.style.highlight_null('red'))
    return  {t:list(types[t].values) for t in types}
