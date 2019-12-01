import pandas as pd
import numpy as np
from . import selection, prep, models, utils
from hyperopt import tpe, fmin
from hyperopt import Trials
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import copy

def BayesianSearch_lgbm(train, test, label_name, folds, parameters, fixed_parameters, tpe_trials, num_iter = 1, n_classes = None, results_df = None):
    multiclass = False
    if n_classes is None:
        n_classes = 2
    elif n_classes > 2:
        multiclass = True
        
    space  = parameters
    cv_scores = []

    if len(tpe_trials.results) != 0:
        cv_scores = [- ex['loss'] for ex in tpe_trials.results if ex['status'] == 'ok']

    #Don't count previous trials
    num_iter += len(tpe_trials.results)

    highest_cv_score = 0
    best_params = None
    random_params = list(parameters.keys())
    if results_df is None:
        results_df = pd.DataFrame(columns = ['mean_cv_metric', 'all_params'] + random_params)

    def objective(params):
        nonlocal cv_scores
        nonlocal highest_cv_score
        nonlocal best_params
        nonlocal fixed_parameters
        nonlocal results_df

        #Combine fixed and variable parameters
        params_comb = params.copy()
        params_comb.update(fixed_parameters)

        res = models.train_lgbm(folds, params_comb,
                                train.drop(label_name, axis = 1), train[label_name],
                                X_test = test, multiclass = multiclass,
                                n_classes = n_classes)
        
        cv_score = res['mean_val_metrics']
        if cv_score > highest_cv_score:
            highest_cv_score = cv_score
            best_params = params
        
        result_entry = {key:params_comb[key] for key in random_params}
        result_entry['mean_cv_metric'] = cv_score
        result_entry['all_params'] = params_comb
        results_df = results_df.append(result_entry, ignore_index = True).sort_values(by = 'mean_cv_metric')
        cv_scores.append(cv_score)

        clear_output()
        plt.figure(figsize = (15,10))
        plt.show(sns.regplot(list(range(len(cv_scores))),cv_scores))
        print('best params: {}'.format(best_params))
        print('best cv score: {}'.format(highest_cv_score))
        
        display(results_df.style.background_gradient(cmap = 'coolwarm').hide_columns(['all_params']))
        
        print(tpe_trials.results)
        utils.save_pickle(tpe_trials, 'tpe_trials')
        results_df.to_csv('tpe_trial_results.csv', index = False)
        return - cv_score
    
    tpe_algo = tpe.suggest
    tpe_best = fmin(fn=objective, space=space,\
            algo=tpe_algo, trials=tpe_trials,\
            max_evals=num_iter)
    
    utils.save_pickle(tpe_trials, 'tpe_trials')
    return {'results':results_df, 'tpe_trials': tpe_trials, 'tpe_best':tpe_best}

def bayesian_search(model, search_params, fixed_params, num_iter = 25, mode = 'bayesian', minimize = True, tpe_trials = None, **kwargs):
    space = search_params
    results = dict.fromkeys(['test_preds', 'best_params', 'mean_metric', 'dev_metric'])
    results['mean_metric'] = None
    
    # mode must be one of ['bayesian', 'random']
    if not (mode in ['bayesian', 'random']): raise AssertionError()
    
    def objective(params):
        #Combine fixed and variable parameters
        params_comb = params.copy()
        params_comb.update(fixed_params)

        param_res = model(params = params_comb, **kwargs)
        
        improve = False
        if results['mean_metric'] is None:
            improve = True
        elif minimize and (param_res['mean_metric'] < results['mean_metric'] - 1e-6):
            improve = True
        elif not minimize and (param_res['mean_metric'] > results['mean_metric'] + 1e-6):
            improve = True
            
        if improve:
            results['dev_metric'] = param_res['dev_metric']
            results['mean_metric'] = param_res['mean_metric']
            results['best_params'] = params_comb
            results['test_preds'] = param_res['test_preds']
        
        if minimize:
            return param_res['mean_metric']
        else:
            return - param_res['mean_metric']
    
    if tpe_trials is None:
        tpe_trials = Trials()
    else:
        print(f'Continue previous bayesian search from iteration: {len(tpe_trials.results)}')
    
    # Start hparam search
    tpe_algo = tpe.suggest if mode == 'bayesian' else tpe.random.suggest
    fmin(fn=objective, space=space, 
         algo=tpe_algo, trials=tpe_trials,
         max_evals=num_iter)
    
    print(f'Best mean metric: {results["mean_metric"]}')
    print(f'Dev. of metric: {results["dev_metric"]}')
    print('<b>SELECTED HPARAMETERS: </b>')
    print(results['best_params'])
    
    results['tpe_trials'] = tpe_trials
    return results

def metric_is_better(metric, best_metric, minimize):
    return (minimize and metric < best_metric - 1e-7) \
        or (not minimize and metric > best_metric + 1e-7)

def line_search(
        model, params_to_search, model_params,
        try_count = 3, change_rate = 0.2, patience = 3,
        minimize = True, **kwargs):
    
    def rectify_change(x, x_orig):
        x_isint = isinstance(x_orig, int) or isinstance(x_orig, np.integer)
        # x must be greater than 0
        if not (x > 0): raise AssertionError()
        if x < 1.0 and x_isint:
            x = 1.0
        if x_isint:
            x = int(x)
        return x

    orig_params_fixed = {key: val for key, val in model_params.items() if key not in list(params_to_search.keys())}
    orig_params_search = {key: val for key, val in model_params.items() if key in list(params_to_search.keys())}
    
    # Loop over search hyperparameters
    for key, val in orig_params_search.items():
        # Hyperparameter must be a float or int
        if not (isinstance(val, int) \
             or isinstance(val, float) \
             or isinstance(val, np.integer) \
             or isinstance(val, np.floating)):
            raise AssertionError()
            
        min_val, max_val = params_to_search[key]
        # Original parameter must be in the search space
        if not (val > min_val and val < max_val):
            print(f'Warning: parameter {key} is outside the search space. It will be changed to the nearest limit. Expanding the search space is recommended.')
        
    for key, (dn_limit, up_limit) in params_to_search.items():
        if not (dn_limit < up_limit): raise AssertionError()

    res_param = model(params = model_params, **kwargs)
    best_metric = res_param['mean_metric']
    best_param_search_middle = copy.deepcopy(orig_params_search)
    result = {'mean_metric': res_param['mean_metric'], 'best_params': model_params, 'test_preds': res_param['test_preds']}
    
    def search_onedir_(search_dir, p):
        new_search_params = copy.deepcopy(best_param_search_middle)
        best_search_params = copy.deepcopy(best_param_search_middle)
        if  (new_search_params[p] == params_to_search[p][1]) and (search_dir == 'up'):
            return None
        if (new_search_params[p] == params_to_search[p][0]) and (search_dir == 'down'):
            return None
        param_change = rectify_change(new_search_params[p] * change_rate, new_search_params[p])
        
        new_search_params[p] = new_search_params[p] + (param_change if search_dir=='up' else -param_change)
        # Check if hyperparameter is in limits
        if (new_search_params[p] < params_to_search[p][0]):
            new_search_params[p] = params_to_search[p][0]
        elif (new_search_params[p] > params_to_search[p][1]):
            new_search_params[p] = params_to_search[p][1]
        
        res_param = model(params = {**orig_params_fixed, **new_search_params}, **kwargs)
        metric = res_param['mean_metric']
        res_direction = {'mean_metric': result['mean_metric'], 'best_params': model_params, 'test_preds': result['test_preds']}
        best_metric_dir = best_metric
        n_trials = 0
        
        metric_better = metric_is_better(metric, best_metric_dir, minimize)
        
        while metric_better or n_trials < patience:
            if metric_better:
                best_metric_dir = metric
                best_search_params = copy.deepcopy(new_search_params)
                res_direction['test_preds'] = res_param['test_preds']
                n_trials = 0
            else:
                n_trials += 1
            ###
            
            # Check if hyperparameter is in limits
            if (new_search_params[p] == params_to_search[p][0]) or (new_search_params[p] == params_to_search[p][1]):
                break
            
            # Modify hyperparameter in specified direction
            param_change = rectify_change(new_search_params[p] * change_rate, new_search_params[p])
            new_search_params[p] = new_search_params[p] + (param_change if search_dir=='up' else -param_change)
            
            # Check if hyperparameter is in limits
            if (new_search_params[p] < params_to_search[p][0]):
                new_search_params[p] = params_to_search[p][0]
            elif (new_search_params[p] > params_to_search[p][1]):
                new_search_params[p] = params_to_search[p][1]
            
            
            # Calculate metric
            res_param = model(params = {**orig_params_fixed, **new_search_params}, **kwargs)
            metric = res_param['mean_metric']
            
            # Find if new hparams improved the result
            metric_better = metric_is_better(metric, best_metric_dir, minimize)
        
        res_direction['dev_metric'] = best_metric_dir
        res_direction['mean_metric'] = best_metric_dir
        res_direction['best_params'] = {**orig_params_fixed, **best_search_params}
        return res_direction
    
    prev_round_improve = True
    for _ in range(try_count):
        if not prev_round_improve:
            break
        prev_round_improve = False
        for parameter in params_to_search:
            up_result = search_onedir_('up', parameter)
            down_result = search_onedir_('down', parameter)
            
            if minimize:
                if up_result is not None and (up_result['mean_metric'] < best_metric - 1e-6):
                    best_metric = up_result['mean_metric']
                    best_param_search_middle = up_result['best_params']
                    prev_round_improve = True
                if down_result is not None and (down_result['mean_metric'] < best_metric - 1e-6):
                    best_metric = down_result['mean_metric']
                    best_param_search_middle = down_result['best_params']
                    prev_round_improve = True
            else:
                if up_result is not None and (up_result['mean_metric'] > best_metric + 1e-6):
                    best_metric = up_result['mean_metric']
                    best_param_search_middle = up_result['best_params']
                    prev_round_improve = True
                if down_result is not None and (down_result['mean_metric'] > best_metric + 1e-6):
                    best_metric = down_result['mean_metric']
                    best_param_search_middle = down_result['best_params']
                    prev_round_improve = True
    
    result['mean_metric'] = best_metric
    result['best_params'] = {**orig_params_fixed, **best_param_search_middle}
    print('SELECTED PARAMETERS:')
    print(result['best_params'])
    return result