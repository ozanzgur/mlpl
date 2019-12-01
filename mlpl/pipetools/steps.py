from .. import prep, utils
from . import putils
from . import dt
import pandas as pd

def inform_mutation(train, col_name):
    """Prints a message about whether a column will be added or an existing one
    will be replaced in future.
    
    Parameters
    ----------
    train: pandas dataframe
        Training set
    col_name : string
        Column name
        
    """
    
    # Find if a column will be replaced
    will_replace = col_name in train.columns
    
    # Inform user
    print(f'replace : {col_name}' if will_replace else f'add : {col_name}')

def get_default_name(replace_default, train, col_name, function_name):
    """Creates a feature name based on conditions. This is used for replacing
    features generated in default preprocessing.
    
    Parameters
    ----------
    replace_default: bool
        User wants to replace feature generated in default preprocessing
    train : pandas dataframe
        Training set
    col_name : string
        Column that will be imputed/processed.
    function_name : string
        Name of the processing step function
    
    Returns
    -------
    col_name : string
        New column name based on conditions
    
    """
    new_name = None
    if replace_default:
        # Infer default column name by dtype
        col_type = str(train[col_name].dtype)
        
        # Default names given in preprocessing
        if col_type == 'category':
            new_name = f'process_nominal_{col_name}'
        else:
            new_name = f'process_dense_{col_name}'
    else:
        # New column name
        new_name = f'{function_name}_{col_name}'
    return new_name
    
def mutate_dataframes(
        train, test, replace_default, trn_col,
        test_col, col_name, function_name):
    """Adds a column to dataframes or replaces.
    
    """
    # Get new column name (Also prints the process to be done.)
    new_name = get_default_name(replace_default, train, col_name, function_name)
    
    # Mutate dataframes
    train[new_name] = trn_col
    test[new_name] = test_col
    
    # Print a message. (REPLACE or ADD w,th column name)
    inform_mutation(train, new_name)
    return train, test

def imputeby_dependents_numeric(
        feature_properties, train, test,
        label_name, col_name, fill_mode = 'median',
        cols_to_drop = [], replace_default = True, binning = False,
        ohe = False, sparse = True, outlier_limit = 10,
        **kwargs):
    """Imputes numeric values using imputeby_dependent function.
    This function finds dependencies by training a decision tree to predict 
    column to be imputed using other columns. It selects the feature with the
    highest score.
    
    Can also binarize and keep the data in sparse form.
    
    Replaces default processed numeric column. These columns have a standard name
    In the format process_dense_{col_name}.
    
    Parameters
    ----------
    feature_properties : pandas dataframe
        Properties dataframe calculated by vis.
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name in dataset
    col_name : string
        Name of the column to be imputed
    fill_mode : string ['median', 'mode', 'mean']
        Method to use for filling. Median is suggested.
    cols_to_drop : set or list of strings
        Columns not to be used in training
    replace_default : string
        If true, will replace feature generated in default preprocessing.
        
    Returns
    -------
    to_drop : list of strings
        New columns not to be used during training
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    
    """
    function_name = 'imputeby_dependents_numeric'
    
    # Get imputed columns
    trn_col, test_col = prep.imputeby_dependents(
        train,
        test,
        col_name,
        feature_properties,
        to_exclude = cols_to_drop.union({label_name}),
        n = 1,
        fill_mode = fill_mode,
        lgbm_objective = 'regression',
        mode = 'return')
    
    # Add imputed columns or replace, depending on arguments
    train, test = mutate_dataframes(train, test, replace_default, trn_col,
                                    test_col, col_name, function_name)
    
    # Optional OHE (sparse or normal), group outliers
    to_drop, train, test = preprocessing_group_(
                        feature_properties, train, test, label_name , replace_default,
                        col_name, function_name, group_outliers, outlier_limit, ohe,
                        sparse, binning)
    return to_drop, train, test

def imputeby_dependents_nominal(
        feature_properties, train, test,
        label_name, col_name, cols_to_drop = [],
        ohe = False, replace_default = True, sparse = True,
        group_outliers = False, outlier_limit = 10, **kwargs):
    """Imputes nominal values using imputeby_dependent function. After that,
    groups outliers (val_count < 10).
    
    Replaces default processed nominal column. These columns have a standard name
    In the format process_nominal_{col_name}-limit=10.
    
    Parameters
    ----------
    feature_properties : pandas dataframe
        Properties dataframe calculated by vis.
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name in dataset
    col_name : string
        Name of the column to be imputed
    cols_to_drop : set or list of strings
        Columns not to be used in training
    replace_default : bool
        If true, will replace feature generated by default preprocessing.
        This feature has name process_nominal_{col_name}-limit=10.
        
    Returns
    -------
    to_drop : list of strings
        New columns not to be used during training
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    
    """
    function_name = 'imputeby_dependents_nominal'
    
    # Get mutated dataframes and cols_to_drop(empty)
    # Get imputed columns
    trn_col, test_col = prep.imputeby_dependents(
        train,
        test,
        col_name,
        feature_properties,
        to_exclude = cols_to_drop.union({label_name}),
        n = 1,
        fill_mode = 'mode',
        lgbm_objective = 'multiclass',
        mode = 'return')
    
    # Optional OHE (sparse or normal), group outliers
    to_drop, train, test = preprocessing_group_(
                        feature_properties, train, test, label_name , replace_default,
                        col_name, function_name, group_outliers, outlier_limit, ohe,
                        sparse, binning = False)
    return to_drop, train, test

def preprocessing_group_(
        feature_properties, train, test, label_name , replace_default,
        col_name, function_name, group_outliers, outlier_limit, ohe,
        sparse, binning):
    """A group of preprocessing steps for nominals. They can be added after every
    imputation step. This is the reason they are in a function.
    This step will replace the column.
    
    
    - Please don't call this function outside the library.
    
    """
    # Will not use these columns in training
    to_drop = []
    
    # Get new column name
    new_name = get_default_name(replace_default, train,
                                col_name, function_name)
    
    # Group values with val_count < limit
    if group_outliers:
        prep.group_outliers_replace([train, test], new_name, limit = outlier_limit)
    
    if binning:
        _, train,test = bin_numeric_auto(feature_properties, train, test,
                                         label_name, new_name, ohe = False,
                                         sparse = True)
    if ohe:
        # Will not use original column in training
        to_drop.append(col_name)
        
        # One-hot encoding
        train, test = prep.one_hot_encode([train, test],
                                          new_name,
                                          mode = 'add',
                                          sparse = sparse)
    return to_drop, train, test

def impute_fixed(
        feature_properties, train, test, label_name,
        col_name, replace_default = True, ohe = False, sparse = True,
        group_outliers = False, outlier_limit = 10, binning = False, **kwargs):
    """Impute missing values with -9999. 
    Replaces column.
    
    """
    function_name = 'impute_fixed'
    
    # Additional steps are required if column is categorical
    is_cat = str(train[col_name].dtype) == 'category'
    
    # Fill both dfs
    for df in [train, test]:
        
        # If col is of category type, we have to fill using a category type.
        # Since this is more difficult, I convert column to numeric, then fill.
        if is_cat:
            # To numeric
            df[col_name] = df[col_name].astype('float')
            
        # Impute using -9999 (hopefully nobody uses that.)
        df[col_name] = df[col_name].fillna(-9999)
        
        # Revert to category type
        if is_cat: 
            df[col_name] = df[col_name].astype('category')
            
    
    # Optional OHE (sparse or normal), group outliers, binning
    to_drop, train, test = preprocessing_group_(
                        feature_properties, train, test, label_name , replace_default,
                        col_name, function_name, group_outliers, outlier_limit, ohe,
                        sparse, binning)
    return to_drop, train, test

def impute_mode(
        feature_properties, train, test, label_name,
        col_name, replace_default = True, ohe = False,  outlier_limit = 10,
        sparse = True, binning = False, **kwargs):
    """Impute missing values with mode.
    
    In default, replaces default processed column. Default columns have standard
    names. Will infer default name by column dtype.
    
    Parameters
    ----------
    feature_properties : pandas dataframe
        Properties dataframe calculated by vis.
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name in dataset
    col_name : string
        Name of the column to be imputed
    ohe : bool
        If true, will one-hot encode
    replace_default : bool
        If true, will replace feature generated by default preprocessing.
        Default name depends on dtype.
        
    Returns
    -------
    to_drop : list of strings
        New columns not to be used during training
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    
    """ 
    function_name = 'impute_mode'
    
    # Get imputed columns
    trn_col, test_col = utils.utilize(mode = 'return') \
    (prep.impute_mode)([train, test], col_name)
    
    # Add imputed columns or replace, depending on arguments
    train, test = mutate_dataframes(train, test, replace_default, trn_col,
                                    test_col, col_name, function_name)

    # Optional OHE (sparse or normal), group outliers, binning
    to_drop, train, test = preprocessing_group_(
                        feature_properties, train, test, label_name , replace_default,
                        col_name, function_name, group_outliers, outlier_limit, ohe,
                        sparse, binning)
    return to_drop, train, test

def impute_mean(
        feature_properties, train, test,
        label_name, col_name, replace_default = True,
        binning = False, ohe = False, sparse = True,
        group_outliers = False, outlier_limit = 10, **kwargs):
    """Impute missing values with mean.
    
    In default, replaces default processed column. You can use this function only
    on numeric columns.
    
    Parameters
    ----------
    feature_properties : pandas dataframe
        Properties dataframe calculated by vis.
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name in dataset
    col_name : string
        Name of the column to be imputed
    replace_default : bool
        If true, will replace feature generated by default preprocessing.
        Default name depends on dtype.
        
    Returns
    -------
    to_drop : list of strings
        New columns not to be used during training
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    
    """
    function_name = 'impute_mean'
    
    # Get imputed columns
    trn_col, test_col = utils.utilize(mode = 'return') \
    (prep.impute_mean)([train, test], col_name)
    
    # Mutate dataframes (with messages)
    train, test = mutate_dataframes(train, test, replace_default, trn_col,
                                    test_col, col_name, 'impute_mean')
    
    # Optional OHE (sparse or normal), group outliers, binning
    to_drop, train, test = preprocessing_group_(
                        feature_properties, train, test, label_name , replace_default,
                        col_name, function_name, group_outliers, outlier_limit, ohe,
                        sparse, binning)
    return to_drop, train, test

def one_hot_encode(
        feature_properties, train, test,
        label_name, col_name, sparse = True,
        **kwargs):
    """One-hot encode column. Will add original to to_drop list, but will not
    drop it.
    
    Parameters
    ----------
    feature_properties : pandas dataframe
        Properties dataframe calculated by vis.
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name in dataset
    col_name : string
        Name of the column to be encoded.
        
    Returns
    -------
    to_drop : list of strings
        New columns not to be used during training
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
        
    """
    train, test = prep.one_hot_encode([train, test],
                                      col_name,
                                      mode = 'add',
                                      sparse = sparse)
    return [col_name], train, test

def bin_numeric_auto(
        feature_properties, train, test,
        label_name, col_name, ohe = False,
        sparse = True, **kwargs):
    """Bin numeric column into a calculated number of bins. Always replaces col.
    
    Parameters
    ----------
    feature_properties : pandas dataframe
        Properties dataframe calculated by vis.
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name in dataset
    col_name : string
        Name of the column to be encoded.
        
    Returns
    -------
    to_drop : list of strings
        New columns not to be used during training
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
        
    """
    # Bins training and test sets together
    prep.auto_bin_cols([train, test], col_name)
    if ohe:
        train, test = prep.one_hot_encode([train, test],
                                          col_name,
                                          mode = 'add',
                                          sparse = sparse)
    return [col_name], train, test

def bin_numeric(
        feature_properties, train, test,
        label_name, col_name, bins,
        ohe = False, sparse = True, **kwargs):
    """Bin numeric column. Always replaces col.
    
    Parameters
    ----------
    feature_properties : pandas dataframe
        Properties dataframe calculated by vis.
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name in dataset
    col_name : string
        Name of the column to be encoded.
    bins : int
        Number of bins
        
    Returns
    -------
    to_drop : list of strings
        New columns not to be used during training
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
        
    """
    # Bins training and test sets together
    prep.full_binning([train, test], col_name, bins = bins)
    
    # One-hot encoding
    if ohe:
        train, test = prep.one_hot_encode([train, test],
                                          col_name,
                                          mode = 'add',
                                          sparse = sparse)
    return [col_name], train, test

def group_outliers(
        feature_properties, train, test,
        label_name, col_name, limit = 10,
        ohe = False, sparse = True, **kwargs):
    """Groups values with val_count < limit under a separate value.
    Always replaces.
    
    Parameters
    ----------
    feature_properties : pandas dataframe
        Properties dataframe calculated by vis.
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
    label_name : string
        Label name in dataset
    col_name : string
        Name of the column to be encoded.
    limit : int
        Values with value_count < limit are grouped.
    ohe : bool
        If true, one-hot encodes.
        
    Returns
    -------
    to_drop : list of strings
        New columns not to be used during training
    train : pandas dataframe
        Training set
    test : pandas dataframe
        Test set
        
    """
    # Group outliers and replace
    prep.group_outliers_replace([train, test], col_name, limit = limit)
    
    # Ohe-hot encoding
    if ohe:
        train, test = prep.one_hot_encode([train, test],
                                          col_name,
                                          mode = 'add',
                                          sparse = sparse)
    return [col_name], train, test




################################### Add steps ##################################

@putils.keep_html
def try_default_nominal_steps(
        pipeline, ohe = False,
        ohe_max_unique = 5000,group_outliers = False):
    """Try default steps for all categorical columns.
    Exclude id columns and inconsistent columns.
    
    Steps for imputation:
        - Most frequent impute
        - Separate impute
        - Impute with group outliers
        - All with OHE
          total: 8
          
    Steps for others:
        - OHE
    """
    cats = (pipeline.original_properties['dtype'] == 'object') \
         & (pipeline.original_properties['is_id_col'] == False) \
         & (pipeline.original_properties['is_inconsistent'] == False)
            
    cats = [col for col in cats if col not in pipeline.cols_to_drop]
    category_cols = pipeline.original_properties[cats].feature.tolist()
    
    # Print category cols
    pipeline.step_html_text('Try Steps for Categoricals: ', islabel = True)
    pipeline.step_html_text(category_cols)
    
    # Leave box that keep_html has created
    pipeline.step_html_end_collapsible()
    
    for col in category_cols:
        # Group steps for this feature under a collapsible cell
        pipeline.step_html_new_collapsible_box()
        pipeline.modify_collapsible_box_title(col)
        
        # Keep the original arg., as we will modify it.
        ohe_col = ohe
        
        # Get properties for this col
        col_props = pipeline.original_properties[pipeline.original_properties.feature == col]
        
        # Get unique count for col
        n_unique_col = col_props['full_nunique'].values[0]
        
        if ohe_max_unique < n_unique_col and ohe:
            # Print that feature will not be ohed 
            pipeline.step_html_text(
                f'Will not OHE {col}, as it has n_unique= {n_unique_col} > {ohe_max_unique}')
            
            # Don't ohe
            ohe_col = False
            
        # Try default steps for this column
        group_name = add_preprocessing('nominal',
                                       col,
                                       pipeline,
                                       pipeline.feature_properties,
                                       binning = False,
                                       ohe = ohe_col,
                                       group_outliers = group_outliers)
        res = None
        if group_name is not None:
            res = pipeline.group_apply_useful(group_name)
        else:
            pipeline.step_html_text(f'No steps available for {col}')
        
        pipeline.step_html_end_collapsible()
    
@putils.keep_html
def try_default_numeric_steps(
        pipeline, ohe = False, binning = False,
        group_outliers = False, ohe_max_unique = 5000,
        bin_min_unique = 25):
    """Try default steps for all numeric columns.
    Exclude id columns and inconsistent columns.
 
    """
    # Numerics:
    # Is type int or float
    # Not id col
    # Not inconsistent
    numerics = ((pipeline.original_properties['dtype'] == 'int') \
              | (pipeline.original_properties['dtype'] == 'float')) \
             & (pipeline.original_properties['is_id_col'] == False) \
             & (pipeline.original_properties['is_inconsistent'] == False)
            
    numerics = [col for col in numerics if col not in pipeline.cols_to_drop]
    numeric_cols = pipeline.original_properties[numerics].feature.tolist()
    
    # Print numeric cols
    pipeline.step_html_text('Try Steps for Numerics: ', islabel = True)
    pipeline.step_html_text(numeric_cols)
    
    # Leave box that keep_html has created
    pipeline.step_html_end_collapsible()
    
    # Loop over all category cols
    for col in numeric_cols:
        # Group steps for this feature under a collapsible cell
        pipeline.step_html_new_collapsible_box()
        pipeline.modify_collapsible_box_title(col)
        
        
        ohe_col = ohe
        bin_col = binning
        
        # Get properties for this col
        col_props = pipeline.original_properties[pipeline.original_properties.feature == col]
        
        # Get unique count for col
        n_unique_col = col_props['full_nunique'].values[0]
        
        if ohe_max_unique < n_unique_col and ohe:
            # Print that feature will not be ohed 
            pipeline.step_html_text(
                f'Will not OHE {col}, as it has n_unique= {n_unique_col} > {ohe_max_unique}')
            
            # Don't ohe
            ohe_col = False
        
        # Don't bin if distinct count is less than bin_min_unique
        if n_unique_col < bin_min_unique and binning:
            bin_col = False
            # Print that feature will not be binned 
            pipeline.step_html_text(
                f'Will not BIN {col}, as it has n_unique= {n_unique_col} < {bin_min_unique}')
        
        # Try default steps for this column
        group_name = add_preprocessing('numeric',
                                       col,
                                       pipeline,
                                       pipeline.feature_properties,
                                       binning = binning,
                                       ohe = ohe_col,
                                       group_outliers = group_outliers)
        
        if group_name is not None:
            res = pipeline.group_apply_useful(group_name)
        else:
            pipeline.step_html_text(f'No steps available for {col}')
            
        pipeline.step_html_end_collapsible()
    
#TODO: fix group_outliers for numerics, as it is a different proecss for numerics.
    
def add_preprocessing(
        col_type, col_to_proc, pipeline, feature_properties,
        binning = False, ohe = False, group_outliers = True): # , sparse = True
    """Try default steps for numeric column.
    
    IMPUTATION:
    
    Steps for others:

    
    """
    # Get name of the column after default preprocessing before imputation
    col_name = None
    if col_type == 'nominal':
        col_name = f'label_encode_{col_to_proc}'
    elif col_type == 'numeric':
        col_name = col_to_proc
        
    group_name =  f'impute_{col_to_proc}'
    
    # Get properties of this feature
    properties_col = feature_properties[feature_properties.feature == col_to_proc]
    
    try:
        # Impute missing:
        has_null = (properties_col['full_null_ratio'] > 0.0).values[0]
        val_count = (properties_col['full_nunique']).values[0]
    except Exception as e:
        # Print which features exist and which one doesn't exist
        e.args += ('col: ', str(col_to_proc), ' not in ', str(feature_properties.feature.values))
        raise e
        
    val_morethan_2 = (val_count > 2)
    
    # Another possible step is to group outliers without imputation,
    # but this is in the default preprocessing.
    if not val_morethan_2 and ohe:
        pipeline.step_html_text(f'Will not OHE, bin or group {col_to_proc} (less than 3 distinct values.)')
        ohe = False
        binning = False
        group_outliers = False
            
    ohe_vals = [False, True] if ohe else [False]
    group_outliers_vals = [False, True] if group_outliers else [False]
    bin_vals = [False, True] if binning else [False]
    
    
    impute_functions = None
    processing_functions = None
    
    if col_type == 'numeric':
        # Use numeric col functions
        # Will try these imputations with combinations of ohe, go, binning, etc.
        impute_functions = [imputeby_dependents_numeric,
                            impute_mean,
                            impute_fixed]

        # May add binning (into a calculated number of bins)
        processing_functions = [bin_numeric_auto] if binning else []
    
    elif col_type == 'nominal':
        # Use nominal col functions
        impute_functions = [impute_mode,
                            imputeby_dependents_nominal,
                            impute_fixed]
        
        processing_functions = [one_hot_encode] if ohe else []
    else:
        # Use 'numeric' or 'nominal' for col_type argument.
        raise AssertionError()
    
    # Determine which set of functions to use
    # Has missing => use impute_functions
    # No missing => use processing_functions
    functions_to_try = impute_functions if has_null else processing_functions
    print(functions_to_try)
    
    # If there is nothing to try, return None
    if len(functions_to_try) == 0:
        return None
    
    #sparse = True
    
    # Try with or without go
    for go_val in group_outliers_vals:
        
        # Try with and without ohe
        for ohe_val in ohe_vals:
            
            # Try with or without binning
            for bin_val in bin_vals:

                # Try all functions
                for f in functions_to_try:
                    
                    # Build parameter set for these parameters
                    proc_params = {'col_name': col_name,
                                   #'sparse': sparse,
                                   'group_outliers': go_val,
                                   'ohe': ohe_val,
                                   'binning': bin_val,
                                   'cols_to_drop': pipeline.cols_to_drop}

                    # Add steps to the pipeline, into the same group
                    pipeline.add_step(proc = f,
                                      group = group_name,
                                      proc_params = proc_params)

    return group_name