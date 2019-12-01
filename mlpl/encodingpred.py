import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os
from . import utils, prep

this_dir, this_filename = os.path.split(__file__)
count_encoding_model_path =  os.path.join(this_dir, "count_encoding_model", "countencoding_model_ieee")
count_encoding_model_scaler_path = os.path.join(this_dir, "count_encoding_model", "countencoding_model_ieee_scaler")

def train_count_encoding_model(pred_encoding_model_folder):
    """
    Label name = to_encode
    
    """
    prop_df_concat = utils.get_combined_data(pred_encoding_model_folder)
    to_drop = ['feature', 'to_encode', 'dtype']
    to_use = ['count_gain', 'abs_count_correlation', 'fgt3_val_ratio', 'feq1_val_ratio', 'fgt5_val_ratio', 'fgt10_val_ratio', 'high_f_ratio', 'mid_f_ratio',
                         'low_f_ratio', 'ratio_top10', 'ratio_top3' , 'ratio_top1', 'all_unique_ratio']
    features = prop_df_concat.feature
    trn_data = prop_df_concat.drop(to_drop, axis = 1)[to_use]
    trn_data_std, scaler = prep.standardize_dfs(trn_data, return_scaler = True)
    trn_data_std = trn_data_std[0]

    clf=LogisticRegression(C=0.1, solver="lbfgs", max_iter=100)
    clf.fit(trn_data_std, prop_df_concat['to_encode'])
    pred=clf.predict_proba(trn_data_std)[:,1]

    res_df = pd.DataFrame({'feature': features, 'pred': pred})
    return clf, scaler, res_df.sort_values(by = 'pred', ascending = False)

def pred_count_encoding_model(prop_df):
    #clf = pickle.load(open(os.path.join(pred_encoding_model_folder, 'countencoding_model_ieee'), 'rb'))
    #scaler = pickle.load(open(os.path.join(pred_encoding_model_folder, 'countencoding_model_ieee_scaler'), 'rb'))
    clf = pickle.load(open(count_encoding_model_path, 'rb'))
    #scaler = pickle.load(open(count_encoding_model_scaler_path, 'rb'))
    to_use = ['count_gain', 'abs_count_correlation', 'fgt3_val_ratio',
              'feq1_val_ratio', 'fgt5_val_ratio', 'fgt10_val_ratio',
              'high_f_ratio', 'mid_f_ratio', 'low_f_ratio',
              'ratio_top10', 'ratio_top3' , 'ratio_top1',
              'full_unique_ratio']
    clf_data = prop_df[to_use]
    # TODO: USE scaler, or predictions will be all wrong.
    prep.process_denses(clf_data, verbose = False)
    clf_data_std = prep.standardize_dfs(clf_data, return_scaler = False)
    clf_data_std = clf_data_std[0].values
    pred=clf.predict_proba(clf_data_std)[:,1]
    return pred

def add_pred_count_encoding_to_df(prop_df):
    prop_df['pred_count_encoding'] = pred_count_encoding_model(prop_df)

def get_combined_data(folder):
    files = []
    print('# Files: {}'.format(len(os.listdir(folder))))
    for f in os.listdir(folder):
        if 'csv' in f:
            print(os.path.join(folder, f))
            files.append(files.append(pd.read_csv(os.path.join(folder, f))))
    return pd.concat(files, axis = 0, ignore_index = True)