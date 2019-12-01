import pandas as pd
import numpy as np
from . import utils

def modify_first_encounter(df, col):
    col_df = df[col].copy()
    cumcount_f = col_df.groupby(col).cumcount()
    col_df[cumcount_f == 0.0] = -99999
    return col_df[col[0]]

def count_encoding(df, col):
    """Get count(frequency) encoding of a column of any type.
    
    inputs:
    df - pandas dataframe
    col - column name
    
    outputs:
    count encoded column
    """
    if isinstance(col, list):
        col = col[0]
    return df[col].map(df[col].value_counts(dropna=False))

@utils.utilize(mode = 'add')
def full_count(df, col):
    """Replaces column in dfs."""
    return count_encoding(df, col)
    

def mean(df, cols):
    return df[cols].mean(axis = 1)

def std(df, cols):
    #print(cols)
    return df[cols].std(axis = 1)

def kurtosis(df, cols):
    return df[cols].kurtosis(axis = 1)

def absolute(df, col):
    if not (isinstance(col, str)): raise AssertionError()
    return df[cols].abs()

def decimal_part(df, col):
    return ((df[col] - df[col].astype(int)) * 1000).astype(int)

def day_of_week(df, col):
    """Get day of week from a numerical column that represents datetime.
    
    inputs:
    df - pandas DataFrame
    col - column name
    
    outputs:
    new column, pd.Series object
    """
        
    return np.floor((df[col] / (3600 * 24) - 1) % 7)

def hour(df, col):
    """Get hour from a numerical column that represents datetime.
    
    inputs:
    df - pandas DataFrame
    col - column name
    
    outputs:
    new column, pd.Series object
    """
    if not (isinstance(col, str)): raise AssertionError()
    return np.floor(df[col] / 3600) % 24

def combine_cols(df, cols, LE = False):
    new_col = df[cols[0]].astype('str').map(str)
    for col in cols[1:]:
        new_col = new_col.astype('str') + df[col].astype('str').map(str)
    
    if LE:
        return pd.Series(prep.label_encode(new_col)).astype('category')
    else:
        return new_col

def groupby_agg(df, cols, agg_type='mean', groupby_cols = None): #, fillna=True
    """
    Create aggreages with groupby option.
    Groupby a column by other columns and apply aggregate function to each group.
    Returns a new pd.Series.
    
    inputs:
    df - pandas dataframe
    cols - at least 2 column names.
    agg_type - aggregate function for pd.groupby.agg()
    
    """
    # At least 2 column names must be provided as a list
    if not (isinstance(cols, list)): raise AssertionError()
    for col in cols:
        if not (isinstance(col, str)): raise AssertionError()
    
    if groupby_cols is not None:
        groupby_cols = utils.tolist(groupby_cols)
    
    # Join column names in the name if groupby_columns is specified
    new_col_name = '+'.join(cols) + ('_groupby[{}]'.format(groupby_cols) if groupby_cols is not None else '') + '_' + agg_type
    
    temp_pd = None
    if groupby_cols is not None:
        temp_pd = df[groupby_cols + cols]
        temp_pd = temp_pd.groupby(groupby_cols)[cols].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
    else:
        temp_pd = temp_pd[cols].agg(agg_type, axis = 1).rename(new_col_name)
        
    return temp_pd
        
def auto_combine(
        train, test, cols_to_use,
        label_name, stage_add_counts = None, plot = True,
        nunique_limit = 500, cat_nunique_limit = 1000, unique_ratio_limit = 0.5):
    
    if stage_add_counts is None:
        stage_add_counts = [10, 4, 3]
    
    # Label should not be in FE
    if not (label_name not in cols_to_use): raise AssertionError()
    n_examples = len(train)
    added_scores = pd.DataFrame(columns = ['feature', 'gain', 'stage', 'unique_ratio'])
    def combine_cols_createIfNotExists(dfs, n_tops):
        added = []
        for col1, col2 in n_tops:
            for col in [col1, col2]:
                if col not in dfs[0].columns:
                    raise Exception(f'feature does not exist: {col}')
            col_name =  f'[{col1}_{col2}]'
            col_name_alt = f'[{col2}_{col1}]'
            if col_name not in dfs[0].columns or col_name_alt not in dfs[0].columns:
                for df in dfs:
                    df[col_name] = fe.combine_cols(df, [col1, col2])
                print(f'Add {col_name}')
                added.append(col_name)
        return added
    
    all_comb_gains = []
    cols_notused = [col for col in test.columns if col not in cols_to_use]
    train_notused = train[cols_notused]
    test_notused = test[cols_notused]
    
    # Train and test must have the same columns
    if not (df_hasthesamecols([train.drop(label_name, axis = 1), test])): raise AssertionError()
    [train, test], comb_gains_0, top_n, top_n_scores = vis.which_cols_to_combine(
                                                  [train[cols_to_use + [label_name]],
                                                  test[cols_to_use]], label_name,
                                                  plot = False, return_top = stage_add_counts[0],
                                                  nunique_limit = nunique_limit,
                                                  cat_nunique_limit = cat_nunique_limit,
                                                  unique_ratio_limit = unique_ratio_limit)
    
    
    stage_cols_to_use = cols_to_use
    for i_stage, add_count in enumerate(stage_add_counts[1:]):
        #new_col_names = [f'{col1}_{col2}' for col1, col2 in top_n]
        added = combine_cols_createIfNotExists([train, test], top_n)
        print(f'Stage {i_stage} added {len(added)} features')
        cardinality_new = [train[new_f].nunique() / n_examples for new_f in added]
        new_added = pd.DataFrame({'feature': added, 'gain': top_n_scores, 'stage': i_stage + 1, 'unique_ratio': cardinality_new})
        added_scores = pd.concat([added_scores, new_added], axis = 0, ignore_index = True)
        
        stage_cols_to_use = stage_cols_to_use + added # Cols in results
        [train, test], comb_gains, top_n, top_n_scores = vis.which_cols_to_combine_sets([train, test], added,
                                                                                        stage_cols_to_use, label_name,
                                                                                        plot = False, return_top = add_count,
                                                                                        nunique_limit = nunique_limit,
                                                                                        cat_nunique_limit = cat_nunique_limit,
                                                                                        unique_ratio_limit = unique_ratio_limit)
        all_comb_gains.append(comb_gains)
    
    added = combine_cols_createIfNotExists([train, test], top_n)
    print(f'Stage {len(stage_add_counts)} added {len(added)} features')
    cardinality_new = [train[new_f].nunique() / n_examples for new_f in added]
    new_added = pd.DataFrame({'feature': added, 'gain': top_n_scores,
                              'stage': len(stage_add_counts), 'unique_ratio': cardinality_new})
    added_scores = pd.concat([added_scores, new_added], axis = 0, ignore_index = True)
    added_scores = added_scores.sort_values(by = 'gain', ascending = False)
    if plot:
        vis.plot_lower_triangle(comb_gains_0.fillna(0))
        for df in all_comb_gains:
            n_cols = len(df.columns)
            n_rows = len(df.index)
            plt.figure(figsize = (n_cols + 2, n_rows + 2))
            sns.heatmap(df.astype('float'),center=0, linewidths=1, annot=True,fmt='.4f')
        display(added_scores.sort_values(by = 'gain', ascending = False).style.background_gradient(cmap = 'coolwarm'))
    return pd.concat([train, train_notused], axis = 1), pd.concat([test, test_notused], axis = 1), added_scores
                         