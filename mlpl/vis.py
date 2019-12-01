import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import seaborn as sns
import gc
from sklearn.preprocessing import LabelEncoder
from . import fe, utils, encodingpred, prep
from .pipetools import dt
from tqdm import tqdm_notebook
#from IPython.display import clear_output

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def covariate_shift(train_df, test_df, feature):
    params_single = {'objective': 'binary', "boosting_type": "gbdt", "subsample": 1, "bagging_seed": 11, "metric": 'auc', 'random_state': 47}
    
    df_card1_train = pd.DataFrame(data={feature: train_df[feature], 'isTest': 0})
    df_card1_test = pd.DataFrame(data={feature: test_df[feature], 'isTest': 1})

    # Creating a single dataframe
    df = pd.concat([df_card1_train, df_card1_test], ignore_index=True)
    
    # Encoding if feature is categorical
    if str(df[feature].dtype) in ['object', 'category']:
        df[feature] = LabelEncoder().fit_transform(df[feature].astype(str))
    
    # Splitting it to a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(df[feature], df['isTest'], test_size=0.33, random_state=47, stratify=df['isTest'])

    clf = lgb.LGBMClassifier(num_boost_round=500, **params_single)
    clf.fit(X_train.values.reshape(-1, 1), y_train)
    roc_auc =  roc_auc_score(y_test, clf.predict_proba(X_test.values.reshape(-1, 1))[:, 1])

    del df, X_train, y_train, X_test, y_test
    gc.collect();
    
    return roc_auc

def colorful_df(df, cols = None):
    """
    A simple function to display colorful dataframes.
    
    inputs:
    df - DataFrame
    cols - columns to be displayed with color
    """
    display(df.style.background_gradient(cmap='coolwarm', subset = cols))

def train_single_col(df, col, label_name, plot = True):
    params_single = {'objective': 'binary', "boosting_type": "gbdt", "subsample": 1, "bagging_seed": 11, "metric": 'auc', 'random_state': 47}
    X_train, X_test, y_train, y_test = train_test_split(df[col], df[label_name], test_size=0.33, random_state=47, stratify=df[label_name])
    clf = lgb.LGBMClassifier(**params_single)
    clf.fit(X_train.to_frame().values, y_train)
    
    if plot:
        plt.figure(figsize=(12, 6))
        x = clf.predict_proba(df[col].sort_values().unique().reshape(-1, 1))[:, 1]
        x = pd.Series(x, index=df[col].sort_values().unique())
        sns.heatmap(x.to_frame(), cmap='RdBu_r', center=0.0);
        plt.xticks([]);
        plt.title(f'Model predictions for {col}')
    
    return roc_auc_score(y_test, clf.predict_proba(X_test.to_frame().values)[:, 1])

def train_single_col_multiclass(df, col, label_name, plot = False):
    params_single = {'objective': 'multiclass', "boosting_type": "gbdt", "subsample": 1, "bagging_seed": 11, "metric": 'accuracy', 'random_state': 47}
    X_train, X_test, y_train, y_test = train_test_split(df[col], df[label_name], test_size=0.33, random_state=47)
    clf = lgb.LGBMClassifier(**params_single)
    clf.fit(X_train.to_frame().values, y_train)
    
    if plot:
        plt.figure(figsize=(12, 6))
        x = clf.predict(df[col].sort_values().unique().reshape(-1, 1))
        x = pd.Series(x, index=df[col].sort_values().unique())
        sns.heatmap(x.to_frame(), cmap='RdBu_r', center=0.0);
        plt.xticks([]);
        plt.title(f'Model predictions for {col}')
    
    return accuracy_score(y_test, clf.predict(X_test.values.reshape(-1, 1)))

def get_col_properties(train, test, col, label_name, count_list_size = 10):
    # column cannot be the label
    if (col == label_name): raise AssertionError()
        
    full_col = pd.concat([train[col], test[col]], axis = 0, ignore_index = True)
    n_unique = full_col.nunique()
    
    n_examples = len(train)
    n_combined = len(train) + len(test)
    col_properties = {}
    #mean_label = train[label_name].mean()
    col_properties['dtype'] = ''.join(i for i in str(train[col].dtype) if not i.isdigit())
    col_properties['trn_null_count'] = train[col].isnull().sum()
    col_properties['test_null_count'] = test[col].isnull().sum()
    col_properties['trn_null_ratio'] = col_properties['trn_null_count'] / n_examples
    col_properties['full_null_ratio'] = (col_properties['trn_null_count'] + col_properties['test_null_count']) / n_combined
    col_properties['test_null_ratio'] = col_properties['test_null_count'] / len(test)
    col_properties['full_unique_ratio'] = train[col].append(test[col]).nunique() / n_combined
    col_properties['trn_unique_ratio'] = train[col].nunique() / len(train)
    col_properties['test_unique_ratio'] = test[col].nunique() / len(test)
    col_properties['full_nunique'] = n_unique
    col_properties['full_mode'] = full_col.mode().values[0]
    col_properties['full_median'] = full_col.median() if col_properties['dtype'] in ['int', 'float'] else None
    col_properties['full_mean'] = full_col.mean() if col_properties['dtype'] in ['int', 'float'] else None
    
    # ID columns are cols with all unique values.
    col_properties['is_id_col'] = col_properties['full_unique_ratio'] > 0.99
    
    # In order to train lgbm, label encode object and category columns
    trn_col_orig = train[col].copy()
    test_col_orig = test[col].copy()
    col_properties['col_auc_category'] = None
    col_properties['col_auc_numeric'] = None
    
    if col_properties['dtype'] in ['float', 'int']:
        col_properties['trn_test_difference'] = covariate_shift(train, test, col)
        col_properties['col_auc_numeric'] = float(train_single_col(train, col, label_name, plot = False))
        col_properties['is_inconsistent'] = col_properties['trn_test_difference'] > 0.6
    else:
        col_properties['is_inconsistent'] = False
    
    # Label encode
    utils.utilize(mode = 'replace', verbose = False)(prep.label_encode)([train, test], col)
    col_properties['col_auc_category'] = float(train_single_col(train, col, label_name, plot = False))
    
    if col_properties['dtype'] not in ['float', 'int']:
        col_properties['trn_test_difference'] = covariate_shift(train, test, col)
        
    # Revert to original column
    train[col] = trn_col_orig
    test[col] = test_col_orig
    ###############
    
    val_counts = pd.concat([train[col], test[col]], axis = 0, ignore_index = True).value_counts(dropna = False, ascending = False)
        
    ## FE on column properties
    data_len = len(train)
    val_count_ratios = val_counts.values / data_len
    col_properties['fgt10_val_ratio'] = (val_counts > 10).sum() / n_combined
    col_properties['fgt5_val_ratio'] = (val_counts > 5).sum() / n_combined
    col_properties['fgt3_val_ratio'] = (val_counts > 3).sum() / n_combined
    col_properties['feq1_val_ratio'] = (val_counts == 1).sum() / n_combined
    col_properties['low_f_ratio'] = (val_count_ratios < 0.003).sum() / n_combined
    col_properties['mid_f_ratio'] = (val_count_ratios > 0.01).sum() / n_combined
    col_properties['high_f_ratio'] = (val_count_ratios > 0.25).sum() / n_combined
    col_properties['spread'] = col_properties['low_f_ratio'] / (col_properties['high_f_ratio'] + 1e-3)
    col_properties['ratio_top3'] = np.sum(val_counts.values[:3]) / n_combined
    col_properties['ratio_top10'] = np.sum(val_counts.values[:10]) / n_combined
    col_properties['ratio_top1'] = val_counts.values[0] / n_combined
 
    df_temp = pd.concat([train[[col]], test[[col]]], axis = 0, ignore_index = True)#.fillna(-99999)
    df_temp_fill = df_temp.copy()
    df_temp_fill[f'{col}_count'] = fe.count_encoding(df_temp_fill, col)
    df_temp_fill[f'{col}_count'] = df_temp_fill[f'{col}_count'].astype('float').fillna(0)
    if str(train[col].dtype) in ['category', 'object']:
        col_tr = prep.label_encode(df_temp_fill, col)
        df_temp_fill[col] = col_tr
    df_temp_fill = df_temp_fill.iloc[:n_examples].reset_index(drop = True)
    abs_count_corr = np.abs(np.corrcoef(df_temp_fill[f'{col}_count'].values, train[label_name].values)[0,1])
    col_properties['abs_count_correlation'] = abs_count_corr if abs_count_corr != 'nan' else None
    
    df_temp_fill = pd.concat([df_temp_fill, train[label_name].reset_index(drop = True)], axis = 1)
    
    if df_temp_fill[label_name].isnull().any():
        print('Error: null values in target')
    
    col_properties['count_gain'] = useful_score(df_temp_fill.astype('float').fillna(-999), col, f'{col}_count', label_name, from_max = False)
    
    return col_properties

def display_prop_df(prop_df, cols = None, which = 'all'):
    """
    Displays property dataframe in style.
    Inputs:
    cols - columns to display (optional)
    which - which set to display. ('basic': properties used for exploration, 'all': all properties)
    """
    if cols is None and which == 'all':
        display(prop_df.style.highlight_null().background_gradient(cmap = 'coolwarm'))
    elif cols is not None:
        cols = utils.tolist(cols)
        cols = ['feature'] + cols
        df_sub = prop_df[cols].copy().sort_values(by = cols[1], ascending = False)
        display(df_sub.style.highlight_null(null_color='yellow').background_gradient(cmap = 'coolwarm'))
    else:
        # which must be one of ['all', 'basic']
        if not (which == 'basic'): raise AssertionError()
        cols = ['dtype', 'full_nunique', 'full_unique_ratio',
                'trn_null_ratio', 'test_null_ratio', 'full_mode',
                'full_median', 'full_mean', 'ratio_top1',
                'ratio_top3', 'col_auc_numeric', 'col_auc_category',
                'trn_test_difference', 'count_gain', 'pred_count_encoding',
                'feature']
        
        df_sub = prop_df[cols].sort_values(by = cols[1], ascending = False)
        display(df_sub.reset_index(drop = True).style.background_gradient().highlight_null(null_color='yellow'))
        

def get_all_col_properties(train, test, label_name):
    is_datatable = False
    if isinstance(train, dt.DataTable):
        if not isinstance(test, dt.DataTable):
            raise AssertionError('Both dfs must be DataTable.')
        is_datatable = True
        train = train.df
        test = test.df
    
    prop_df = pd.DataFrame()
    
    cols_to_use = list(train.columns)
    if label_name in cols_to_use:
        cols_to_use.remove(label_name)
    
    for col in tqdm_notebook(cols_to_use):
        if col == label_name:
            continue
        prop_df = prop_df.append(get_col_properties(train, test, col, label_name), ignore_index = True)
        
    prop_cols = list(prop_df.columns)
    prop_cols.remove('dtype')
    prop_df['feature'] = cols_to_use
    prop_df = prop_df[['feature', 'dtype'] + prop_cols]
    for col in prop_cols:
        if str(prop_df[col].dtype) not in ['category', 'object']:
            prop_df[col] = prop_df[col].astype('float32')
            
    for col in ['full_median', 'full_mean', 'col_auc_numeric']:
        prop_df[col] = prop_df[col].astype('float32')
    
    prop_df = prop_df.reset_index(drop = True)
    
    # Add count encoding predictions
    encodingpred.add_pred_count_encoding_to_df(prop_df)
    return prop_df
    
    
def describe_single_col(train, test, col, label_name, time_cols = None, count_list_size = 15, brief = False):
    """
    Explore and visualize a feature.
    
    inputs:
    train - training set DataFrame
    test - test set DataFrame
    col - feature name
    label_name - target column name in dataframes
    """
    # col cannot be the label
    if not (col != label_name): raise AssertionError()
    print('#'*100)
    print('_'*100)
    mean_label = train[label_name].mean()
    col_dtype = str(train[col].dtype)
    print(f"{col} : dtype= {col_dtype}")
    print('--------------------------------------')
    print('# values : train= {:,}, test= {:,}'.format(len(train), len(test)))
    train_null_count = train[col].isnull().sum()
    test_null_count = test[col].isnull().sum()
    print(f'# null   : train= {train_null_count}, test= {test_null_count}')
    print('% null   : train= {:.3f}%, test= {:.3f}%'.format(train_null_count / len(train) * 100, test_null_count / len(test) * 100))
    print('# unique values: total= {:,}, train= {:,}, test= {:,}' \
          .format(train[col].append(test[col]).nunique(), train[col].nunique(), test[col].nunique()))
    print('value_counts: ')
    print('######################################')
    val_counts = train[col].value_counts()
    freq_df = pd.DataFrame(index = np.arange(count_list_size if len(val_counts) > count_list_size else len(val_counts)))
    freq_df['value'] = val_counts.index[:count_list_size]
    freq_df['frequency'] = freq_df['value'].map(val_counts.iloc[:count_list_size])
    freq_df['mean label'] = freq_df['value'].apply(lambda x: train[train[col] == x][label_name].mean())
    freq_df = freq_df.append({'value': 'Null', 'frequency': train_null_count, 'mean label': train[train[col].isnull()][label_name].mean()}, ignore_index=True)
    freq_df.sort_values(by = 'frequency', ascending = False, inplace = True)
    freq_df.reset_index(drop=True, inplace = True)
    freq_df['frequency ratio'] = freq_df['frequency'] / len(train)
    
    display(freq_df.style
      .background_gradient(cmap='viridis',subset=['mean label','frequency'])
      .highlight_max(subset=['mean label','frequency'])
      .highlight_null('red')
      .bar(subset=['frequency ratio'], color='#d65f5f'))
    
    print('######################################')
    print('\nHead: ')
    print(train[col][:count_list_size])
    
    
    @utils.utilize(join_dfs = True, mode = 'return')
    def label_encode_functional(df, col):
        return pd.Series(prep.label_encode(df, col))
    
    # Label encode column for lgbm
    train_col_orig = train[col].copy()
    test_col_orig = test[col].copy()
    if col_dtype in ['object', 'category']:
        train[col], test[col] = label_encode_functional([train, test], col)
    
    roc = covariate_shift(train, test, col)
    print(f"Covariate shift (ROC): {roc}")
    print("Large Cov. shift ROC indicates different characteristics between training and test.\n")
    
    roc_model = train_single_col(train, col, label_name)
    print(f"ROC from training a model on this column: {roc_model}")
    
    if col_dtype not in ['object', 'category']: # Numeric data
        hist_multiscale(train, col, bin_sizes = [30, 50, 75, 125])
        distribution_with_label(train, col, label_name)
        
    else: # Categorical data
        train_temp = train[col].copy()
        test_temp = test[col].copy()
        train[col] = train_col_orig
        test[col] = test_col_orig
        bar_plot_with_label(train, test, col, label_name)
        train[col] = train_temp
        test[col] = test_temp
        del train_temp
        del test_temp
        
        
    # Dependency on time
    test[label_name] = -999999
    if time_cols is not None:
        for time_col in time_cols:
            fig, ax = plt.subplots(1,1, figsize = (12, 7))
            sns.scatterplot(data = train, x = time_col, y = col, hue = label_name, ax = ax, alpha = 0.5)
    
    if not brief:
        # Count vs mean label
        df_temp = train[[col, label_name]].copy().fillna(-99999)
        df_temp[col] = fe.count_encoding(df_temp, col)
        count_label_mean = df_temp.groupby(col)[label_name].mean().to_frame(name = 'mean_label').reset_index()
        count_label_mean.columns = ['count', 'mean_label']
        count_label_mean['mean_label'] = count_label_mean['mean_label'] - mean_label
        fig, ax = plt.subplots(1,1, figsize = (10, 7))
        sns.scatterplot(data = count_label_mean, x = 'count', y = 'mean_label', ax = ax)
        ax.set_xlabel('count')
        ax.set_ylabel('deviation from mean target')
        fig.suptitle(' Count vs Mean Target')

        df_temp = train[[col, label_name]].copy()#.fillna(-99999)
        df_temp[f'{col}_count'] = fe.count_encoding(df_temp, col)
        df_temp_fill = df_temp.fillna(-999)
        
        print('### Is count useful? ###')
        depth_dt = 25
        col_sc = dt_accuracy(df_temp_fill, [col], label_name, depth = depth_dt)
        count_sc = dt_accuracy(df_temp_fill, [f'{col}_count'], label_name, depth = depth_dt)
        combined_sc = dt_accuracy(df_temp_fill, [col, f'{col}_count'], label_name, depth = depth_dt)
        print('Useful: {}'.format('Yes' if count_sc < col_sc or combined_sc < col_sc else 'No'))
        print('Col loss: {}'.format(col_sc))
        print('Count loss: {}'.format(count_sc))
        print('Count interaction loss: {}'.format(combined_sc))
        bins_f1 = None if col_dtype in ['object', 'category'] or df_temp[col].nunique() < 20 else 20
        bins_f2 = 20 if df_temp[f'{col}_count'].nunique() > 20 else None
        plot_densitymean([col, f'{col}_count'], df_temp, label_name, bins_f1 = bins_f1, bins_f2 = bins_f2)
        plot_targetmean([col, f'{col}_count'], df_temp, label_name, bins_f1 = bins_f1, bins_f2 = bins_f2)
        plot_density([col, f'{col}_count'], df_temp, label_name, bins_f1 = bins_f1, bins_f2 = bins_f2)

        #plot_density([col, f'{col}_count'], df_temp, label_name, bins_f1 = None if col_dtype in ['object', 'category'] else 20, bins_f2 = 20)

        # Heatmaps
        if time_cols is not None:
            time_cols = utils.tolist(time_cols)
            for time_col in time_cols:
                bins1 = None if col_dtype in ['object', 'category'] or train[col].nunique() < 50 else 50
                bins2 = None if str(train[time_col].dtype) in ['object', 'category'] or train[time_col].nunique()  < 50 else 50
                plot_targetmean([col, time_col], train, label_name, bins_f1 = bins1, bins_f2 = bins2)
                plot_density([col, time_col], train, label_name, bins_f1 = bins1, bins_f2 = bins2)
            
    test.drop(label_name, axis = 1, inplace = True)
    
    train[col] = train_col_orig
    test[col] = test_col_orig
        
def feature_interact_with_others(train, col_to_interact, label_name, targeted_cols = None):
    if not (col_to_interact in set(train.columns)): raise AssertionError()
    if targeted_cols is None:
        targeted_cols = list(set(train.columns))
        targeted_cols.remove(col_to_interact)
    elif not isinstance(targeted_cols, list):
        targeted_cols = [targeted_cols]
    
    if col_to_interact in targeted_cols:
        targeted_cols.remove(col_to_interact)
        
    n_cols = len(targeted_cols)
    y = train[label_name]
    
    print(f'number of plots: {n_cols} ')
    _, ax = plt.subplots(n_cols, 1, figsize = (7, n_cols*7))
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    
    #labels = y.unique()
    for a, col in zip(ax, targeted_cols):
        sns.scatterplot(train[col_to_interact], train[col], hue = y, alpha = 0.5, ax = a)

def pairplot(train, cols, label_name):
    sns.pairplot(train, vars = cols, hue = label_name,
                 diag_kind="kde", markers="+",
                 plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                 height = 6, diag_kws=dict(shade=True))
    
def visualize_feature_importance(feature_importances, n_folds):
    feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(n_folds)]].mean(axis=1)
    #feature_importances.to_csv('feature_importances.csv')

    plt.figure(figsize=(16, 16))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
    plt.title('50 Top feature importance over {} folds average'.format(n_folds));

    plt.figure(figsize=(16, 16))
    sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).tail(50), x='average', y='feature');
    plt.title('50 Bottom feature importance over {} folds average'.format(n_folds));
        
        
def plot_lower_triangle(df):
    """
    https://www.kaggle.com/paulorzp/gmean-of-low-correlation-lb-0-952x
    """
    
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    #n_cols = len(df.columns)

    # Set up the matplotlib figure
    #f, ax = plt.subplots(figsize=(n_cols+2, n_cols+2))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(df,mask=mask,center=0, linewidths=1, #,cmap='prism'
                    annot=True,fmt='.4f') #, cbar_kws={"shrink":.2}
    
    
def display_null_ratios(dfs, cols):
    if not isinstance(dfs, list):
        dfs = [dfs]
        
    plt.figure(figsize = (int(len(cols))*0.75, 6))
    plt.xticks(rotation = 30)
    plt.title('Null ratios')

    nulls = [df[cols].isnull().sum(axis = 0).sort_values(ascending = False).to_frame() / len(df) for df in dfs]
    for i, null_df in enumerate(nulls):
        null_df['df_index'] = i

    null_ratios = pd.concat(nulls, axis = 0, ignore_index = False).reset_index()
    null_ratios.columns = ['col', 'null_rate', 'df_index']

    sns.barplot(data = null_ratios, x = 'col', y = 'null_rate', hue = 'df_index')
    
def bar_plot_with_label(train, test, cols, label_name, ax = None):
    if not isinstance(cols, list):
        cols = [cols]
    
    for col in cols:
        df_train = pd.DataFrame(data={col: train[col], 'isTest': 0})
        df_test = pd.DataFrame(data={col: test[col], 'isTest': 1})
        df = pd.concat([df_train, df_test], ignore_index=True)
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.countplot(data=df.fillna('NaN')[[col, 'isTest']], x=col, hue='isTest', ax=ax[0]);
        sns.countplot(data=train[[col, label_name]].fillna('NaN'), x=col, hue=label_name, ax=ax[1]);
        ax[0].set_title('Train / Test distibution');
        ax[1].set_title(f'Train distibution by {label_name}');
        ax[0].legend(['Train', 'Test']);
        
def bar_plot_train_label(train, col, label_name, ax = None):
    mean_label = train[label_name].mean()
    if not (isinstance(col, str)): raise AssertionError()
    label_set = train[label_name].unique()
    label_cols = list(map(str, label_set))
    col_stats = pd.DataFrame(index = train[col].unique(), columns = label_cols + ['mean_label'])
    mean_stats = train[[label_name, col]].groupby(col)[label_name].mean().to_frame()
    mean_stats['col_value'] = mean_stats.index
    mean_stats.columns = ['mean_label', 'col_value']
    mean_stats['col_value'].fillna('NAN', inplace = True)
    mean_stats['mean_label'].fillna(0, inplace = True)
    mean_stats['count'] = train[col].value_counts()
    
    for label_val in label_set:
        col_stats[str(label_val)] = train[train[label_name] == label_val].groupby(col).count()
        
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    sns.barplot(data=mean_stats, x = 'col_value', y= 'mean_label', hue = 'count', ax=ax);
    #sns.barplot(data=col_stats, x='col_value', y = 'value', hue='variable', ax=ax);
    ax.set_title(f'Train distibution by {label_name}');
    ax.plot([mean_label, -100], [mean_label, 100], linewidth=2)
    #ax_mean = ax.twinx()
    
def hist(df, col, bins = 50, ax = None):
    """
    Plot a matplotlib histogram.
    
    inputs:
    df - dataframe
    col - name of column
    bins - count of bins
    ax - for drawing in a subplot
    """
    if ax is None:
        plt.hist(df[col], bins=bins);
        plt.title(f'{col} : {bins if isinstance(bins, int) else len(bins) - 1} bins');
    else:
        ax.hist(df[col], bins=bins);
        ax.set_title(f'{col} : {bins if isinstance(bins, int) else len(bins) - 1} bins');
        
def distribution_with_label(df, cols, label_name):
    """
    Plots seaborn distplots for multiple columns.
    
    inputs:
    df - dataframe
    col - a column name or a list of column names
    label_name - target column name
    
    """
    
    df_true = df[df[label_name] == 1.0]
    df_false = df[df[label_name] == 0.0]
    
    n_col = 1
    if isinstance(cols, list):
        n_col = len(cols)
    
    fig, ax = plt.subplots(n_col, 4, figsize = (24, 6 * n_col))
    if n_col == 1: # Single column
        fig.suptitle(f'Distribution of {cols} for labels')
        sns.distplot(df_true[cols], hist = False, ax = ax[0])
        sns.distplot(df_false[cols], hist = False, ax = ax[0])
        ax[0].legend([f'{label_name}= 1', f'{label_name}= 0'])
        ax[0].set_title(f'Kde plot for {cols}')
        
        for i, bin_count in enumerate([15, 40, 85]):
            _, bin_edges = np.histogram(df_true[cols], range = (df[cols].min(), df[cols].max()), bins = bin_count)
            hist(df_false, cols, bins = bin_edges, ax = ax[i + 1])
            hist(df_true, cols, bins = bin_edges, ax = ax[i + 1])
            
            ax[i + 1].legend([f'{label_name}= 0', f'{label_name}= 1'])
            
        fig, ax = plt.subplots(1, 1, figsize = (20, 8))
        _, bin_edges = np.histogram(df_true[cols], range = (df[cols].min(), df[cols].max()), bins = 300)
        
        hist(df_false, cols, bins = bin_edges, ax = ax)
        hist(df_true, cols, bins = bin_edges, ax = ax)
        ax.legend([f'{label_name}= 0', f'{label_name}= 1'])
        
        
        
    else: # Multiple columns
        for i_col, col in enumerate(cols): # For each column
            sns.distplot(df_true[cols], hist = False, ax = ax[i_col, 0])
            sns.distplot(df_false[cols], hist = False, ax = ax[i_col, 0])
            ax[i_col, 0].legend([f'{label_name}= 1', f'{label_name}= 0'])
            ax[i_col, 0].set_title(f'Distribution of {col} for labels')
            
            for i, bin_count in enumerate([15, 40, 125]):
                _, bin_edges = np.histogram(df_true[col], range = (df[col].min(), df[col].max()), bins = bin_count)
                hist(df_false, cols, bins = bin_edges, ax = ax[i_col,i + 1])
                hist(df_true, cols, bins = bin_edges, ax = ax[i_col,i + 1])
                ax[i_col, i + 1].legend([f'{label_name}= 0', f'{label_name}= 1'])
                ax[i_col, i + 1].set_title(f'Distribution of {col} for labels')
                
def hist_multiscale(df, cols, bin_sizes = None):
    """
    Plots matplotlib histograms for bins [30, 50, 75, 125] for multiple columns.
    
    inputs:
    df - dataframe
    col - a column name or a list of column names
    bin_sizes
    """
    if bin_sizes is None:
        bin_sizes = [30, 50, 75, 125]
    
    n_col = 1
    if isinstance(cols, list):
        n_col = len(cols)
    
    fig, ax = plt.subplots(n_col, len(bin_sizes), figsize = (24, 6 * n_col))
    
    if n_col == 1: # Single column
        fig.suptitle(f'Distribution of {cols}')
        for a, bins in zip(ax, bin_sizes):
            hist(df, cols, bins, a)
            
    else: # Multiple columns
        for i_col, col in enumerate(cols): # For each column
            for i, bin_size in enumerate(bin_sizes): # For each bin size
                hist(df, col, bin_size, ax[i_col, i])
                
def plot_targetmean(col_pair, df, label_name, bins_f1 = None, bins_f2 = None):
    data_fs = df[col_pair + [label_name]].copy()
    # Input check
    _plot_bin_check(label_name, col_pair, data_fs, bins_f1, bins_f2)
        
    mean_targets = data_fs.groupby(col_pair)[label_name].mean().unstack()
    plt.figure(figsize = (12,12))
    plt.title('Target Mean Heatmap: ' + '+'.join(col_pair))
    try:
        sns.heatmap(mean_targets, cmap="YlGnBu")
    except:
        print('Cannot plot')
        
def plot_density(col_pair, df, label_name, bins_f1 = None, bins_f2 = None):
    data_fs = df[col_pair + [label_name]].copy()
    # Input check
    _plot_bin_check(label_name, col_pair, data_fs, bins_f1, bins_f2)
    
    mean_targets = data_fs.groupby(col_pair)[label_name].count().unstack()
    plt.figure(figsize = (12,12))
    plt.title('Density Heatmap: ' + '+'.join(col_pair))
    try:
        sns.heatmap(mean_targets)
    except:
        print('Cannot plot')
    
        
def plot_densitymean(col_pair, df, label_name, bins_f1 = None, bins_f2 = None):
    n_examples = len(df)
    mean_label = df[label_name].mean()
    data_fs = df[col_pair + [label_name]].copy()
    
    # Input check
    _plot_bin_check(label_name, col_pair, data_fs, bins_f1, bins_f2)
    
    density = data_fs.groupby(col_pair)[label_name].count() / n_examples
    means = data_fs.groupby(col_pair)[label_name].mean() - mean_label
    
    plt.figure(figsize = (12,12))
    plt.title('Density*Mean Heatmap: ' + '+'.join(col_pair))
    try:
        sns.heatmap((means * density).unstack())
    except:
        print('Cannot plot')
        

def _plot_bin_check(label_name, col_pair, data_fs, bins_f1, bins_f2):
    if bins_f1 is not None:
        if not (isinstance(bins_f1, int)): raise AssertionError()
        data_fs[col_pair[0]] = pd.cut(data_fs[col_pair[0]], bins = bins_f1)
        
    if bins_f2 is not None:
        # bin count must be integer
        if not (isinstance(bins_f2, int)): raise AssertionError()
        data_fs[col_pair[1]] = pd.cut(data_fs[col_pair[1]], bins = bins_f2)

def plot_all_interactions_for_col(X, col, label_name):
    cols_to_int = list(set(X.columns))
    cols_to_int.remove(label_name)
    print(f'INTERACTIONS FOR {col}')
    for col_2 in cols_to_int:
        if col == col_2:
            continue
        plot_density([col, col_2], X, 'target')
        plot_targetmean([col, col_2], X, 'target')

def which_cols_to_combine(dfs, label_name, plot = False, return_top = 5, nunique_limit = 1000, cat_nunique_limit = 1000, unique_ratio_limit = 0.5):
    """
    Warning: adds new columns if they are useful.
    """
    @utils.utilize(join_dfs = True, return_col = True)
    def fbin(df, col):
        return prep.binning(df, col, bins = nunique_limit)
    @utils.utilize(join_dfs = True, return_col = True)
    def fcount(df, col):
        return fe.count_encoding(df, col)
    
    train = dfs[0]
    cols_list = list(train.columns)
    cols_list.remove(label_name)
    cols_to_use = [col for col in cols_list]
    
    freq_encoded_cols = []
    binned_cols = []
    
    for col in cols_list:
        if '_count' not in col: # Assumption: count encoded features will never have very large nuniques.
            dfs_cols = fcount(dfs, col)
            col_name = dfs_cols[0].name
            freq_encoded_cols.append(col_name)
            cols_to_use.append(col_name)
            
            for df, encoded_col in zip(dfs, dfs_cols):
                df[col_name] = encoded_col
            
    # Determine if must be skipped or binned, ADD if must be binned
    # Change col1 and col2 (remove not binned and add binned if must_bin = True)
    for col in cols_list:
        #col_dtype = train[col].dtype
        num_unqiue = train[col].nunique()
        if must_skip(train[col], cat_nunique_limit, num_unqiue):
            print(f'Warning: Skip obj column {col}, nunique: {num_unqiue} > {cat_nunique_limit}')
            cols_to_use.remove(col)
        if must_bin(train[col], nunique_limit):
            #print(f'Binning {str(col_dtype)} column {col}, nunique: {num_unqiue} > {nunique_limit}, will add if useful')
            dfs_cols = fbin(dfs, col)
            col_name = dfs_cols[0].name
            binned_cols.append(col_name)
            # If binned, use the binned version
            if col in cols_to_use:
                cols_to_use.remove(col)
                cols_to_use.append(col_name)

            for df, binned_col in zip(dfs, dfs_cols):
                df[col_name] = binned_col
    
    #mean_target = train[label_name].mean()
    #n_examples = len(train)
    result_df = pd.DataFrame(columns = cols_to_use, index = cols_to_use)
    
    #total_loop = len(cols_to_use) * (len(cols_to_use) - 1) / 2
    for col in cols_to_use:
        # Label encode and sort categories by their mean targets
        if str(train[col].dtype) in ['object', 'category']:
            train[col], dfs[1][col] = prep.label_encode_sort(train, dfs[1], col, label_name)
    
    for i_col, col_out in tqdm_notebook(enumerate(cols_to_use), total = len(cols_to_use)):
        for col_in in tqdm_notebook(cols_to_use[i_col+1:], leave = False):

            result_df.at[col_in, col_out] = useful_score(train[[col_in, col_out, label_name]].fillna(-999), col_in, col_out, label_name)
            #usefulness_score(train, col_in, col_out, label_name, mean_target, unique_ratio_limit)
            #(densities * target_means).sum().sum() - max(base_col_in, base_col_out)
    if plot:
        plot_lower_triangle(result_df.fillna(0))
    
    # Determine top n features by their scores
    top_n = [(c1,c2) for c1, c2 in result_df.stack().sort_values(ascending = False).index[:return_top]]
    top_n_scores = [result_df.at[col1, col2] for col1, col2 in top_n]
    # Keep features in top_n
    all_f_in_top_n = []
    for col1, col2 in top_n:
        all_f_in_top_n.append(col1)
        all_f_in_top_n.append(col2)
        
        # Columns must be in dataframe
        if not (col1 in train.columns): raise AssertionError()
        if not (col2 in train.columns): raise AssertionError()
        
    all_f_in_top_n = set(all_f_in_top_n)
    print('Features in top n:')
    print(all_f_in_top_n)
    
    to_drop = [f for f in (freq_encoded_cols + binned_cols) if f not in all_f_in_top_n]
    print('Added:')
    print(freq_encoded_cols + binned_cols)
    
    print('To Drop:')
    print(to_drop)
    print('fbin_card1' in train.columns)
    
    for df in dfs:
        df.drop(to_drop, axis = 1, inplace = True)
    return dfs, result_df, top_n, top_n_scores

def which_cols_to_combine_sets(dfs, cols1, cols2, label_name, plot = False,
                               freq_encode = False, drop_freq_encoded = True,
                               return_top = 20, nunique_limit = 1000, cat_nunique_limit = 1000, unique_ratio_limit = 0.5):
    """
    Warning: adds new columns if they are useful.
    """
    @utils.utilize(join_dfs = True, return_col = True)
    def fbin(df, col):
        return prep.binning(df, col, bins = nunique_limit) 
    
    
    @utils.utilize(join_dfs = True, return_col = True)
    def fcount(df, col):
        return fe.count_encoding(df, col)
    
    train = dfs[0]
    
    # Count features are added to train, test
    col1_encoded = []
    col2_encoded = []
    for col in cols1:
        if '_count' not in col: # Assumption: count encoded features will never have very large nuniques.
            col_name = f'{col}_count'
            dfs_cols = fcount(dfs, col)
            col_name = dfs_cols[0].name
            col1_encoded.append(col_name)
            for df, encoded_col in zip(dfs, dfs_cols):
                df[col_name] = encoded_col
            
    for col in cols2:
        if '_count' not in col:
            col_name = f'{col}_count'
            dfs_cols = fcount(dfs, col)
            col_name = dfs_cols[0].name
            col2_encoded.append(col_name)
            for df, encoded_col in zip(dfs, dfs_cols):
                df[col_name] = encoded_col
    
    freq_encoded_cols = col1_encoded + col2_encoded
    cols1 = cols1 + col1_encoded
    cols2 = cols2 + col2_encoded
    cols1_to_use = [col for col in cols1]
    cols2_to_use = [col for col in cols2]
    cols_to_use_sets = [cols1_to_use, cols2_to_use]
    
    binned_cols = []
    # Determine if must be skipped or binned, ADD if must be binned
    for col in cols1 + cols2:
        #col_dtype = train[col].dtype
        num_unqiue = train[col].nunique()
        
        if must_skip(train[col], cat_nunique_limit, num_unqiue):
            print(f'Warning: Skip obj column {col}, nunique: {num_unqiue} > {cat_nunique_limit}')
            for i_set, col_set in enumerate([cols1, cols2]):
                if col in col_set and col in cols_to_use_sets[i_set]:
                    cols_to_use_sets[i_set].remove(col)
            
        elif must_bin(train[col], nunique_limit):
            #print(f'Binning {str(train[col].dtype)} column {col}, nunique: {num_unqiue} > {nunique_limit}, will add if useful')
            dfs_cols = fbin(dfs, col)
            dfs_cols = utils.tolist(dfs_cols)
            col_name = dfs_cols[0].name
            binned_cols.append(col_name)
            
            # If binned, use the binned version
            for i_set, col_set in enumerate([cols1, cols2]):
                if col in col_set and col in cols_to_use_sets[i_set]:
                    cols_to_use_sets[i_set].remove(col)
                    cols_to_use_sets[i_set].append(col_name)

            for df, binned_col in zip(dfs, dfs_cols):
                df[col_name] = binned_col
    
    cols_to_use_sets[0] = set(cols_to_use_sets[0])
    cols_to_use_sets[1] = set(cols_to_use_sets[1])
    result_df = pd.DataFrame(columns = cols_to_use_sets[1], index = cols_to_use_sets[0])
    
    for col in set(cols_to_use_sets[0] + cols_to_use_sets[1]):
        # Label encode and sort categories by their mean targets
        if str(train[col].dtype) in ['object', 'category']:
            train[col], dfs[1][col] = prep.label_encode_sort(train, dfs[1], col, label_name)
    
    for col_out in tqdm_notebook(cols_to_use_sets[0], total = len(cols_to_use_sets[0])):
        for col_in in tqdm_notebook(cols_to_use_sets[1], leave = False):
            if col_in == col_out:
                result_df.at[col_in, col_out] = 0.0
                continue
            
            result_df.at[col_out, col_in] = useful_score(train[[col_in, col_out, label_name]].fillna(-999), col_in, col_out, label_name)
            #usefulness_score(train, col_in, col_out, label_name, mean_target, unique_ratio_limit)
            #(densities * target_means).sum().sum() - max(base_col_in, base_col_out)
            
    if plot:
        plt.figure()
        _ = sns.heatmap(result_df.astype('float'),center=0, linewidths=1, annot=True,fmt='.4f')

    top_n = [(c1,c2) for c1, c2 in result_df.stack().sort_values(ascending = False).index[:return_top]]
    top_n_scores = [result_df.at[col1, col2] for col1, col2 in top_n]
    # Keep features in top_n
    all_f_in_top_n = []
    for col1, col2 in top_n:
        all_f_in_top_n.append(col1)
        all_f_in_top_n.append(col2)
    all_f_in_top_n = set(all_f_in_top_n)
    
    # Drop added count encodings and binned columns that are not in top n
    to_drop = [f for f in freq_encoded_cols + binned_cols if f not in all_f_in_top_n]
    for df in dfs:
        df.drop(to_drop, axis = 1, inplace = True)
        
    return dfs, result_df, top_n, top_n_scores

# Helper Functions #########################################
def must_bin(col, nunique_limit):
    is_cat = str(col.dtype) in ['object', 'category']
    num_unqiue = col.nunique()
    num_unique_ok = num_unqiue < nunique_limit
    
    if is_cat:
        return False
    elif num_unique_ok:
        return False
    else:
        return True
    
def must_skip(col, cat_nunique_limit, num_unqiue):
    is_cat = str(col.dtype) in ['object', 'category']
    return is_cat and num_unqiue > cat_nunique_limit


def dt_accuracy(df, cols, label_name, depth = None):
    clf = DecisionTreeClassifier(max_depth = depth, class_weight = 'balanced')
    clf = clf.fit(df[cols],df[label_name])
    return metrics.log_loss(df[label_name], clf.predict(df[cols]))

def useful_score(df, col1, col2, label_name, from_max = True): #TODO: remove from_max
    depth_dt = 25
    col1_sc = dt_accuracy(df, [col1], label_name, depth = depth_dt)
    col2_sc = dt_accuracy(df, [col2], label_name, depth = depth_dt)
    combined_sc = dt_accuracy(df, [col1, col2], label_name, depth = depth_dt)
    return max(col1_sc - combined_sc, col1_sc - col2_sc)
    