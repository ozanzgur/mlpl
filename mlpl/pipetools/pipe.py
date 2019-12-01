from . import putils, pdefaults, dt
from .. import utils, fe, models, vis, search
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import RFECV
from IPython.display import display, HTML, clear_output
from IPython.utils import io
import pickle
import time
import math
import os
import scipy.sparse

import functools
import dill

def make_step_name(proc_params):
    """Creates a step tag from processing parameters.
    Don't add cols_to_drop as its not necessary and its too long
    Don't add params with False, as they are not necessary
    If value is True, only add the key.
    
    Ex proc_params:
        {'col_name': 'asdf', 'bin': True, 'ohe': False}
    Ex result:
        'function_name_(col_name=asdf)_bin'
        
    """
    tokens = []
    for key, val in proc_params.items():
        if key != 'cols_to_drop' and val != False:
            if (val == True) or (val == 'True'):
                tokens.append(key)
            else:
                tokens.append(f'({key}={val})')
                
    return '_'.join(tokens)


######################################## PIPELINE_CLASS ######################################################

class Pipeline:
    def __init__(
        self, *, project_path = 'pipeline_project',
        label_name = None, n_random_seeds = 1, feature_properties = None,
        steps = None, baseline_step = None, baseline_metric = None,
        train_data_path = None, test_data_path = None, minimize_metric = True,
        bayesian_search_iter = 25, line_search_iter = 2, bayesian_search_count = 2,
        line_search_params = None, final_bayesian_search_iter = None, overwrite = False,
        useful_limit = 3e-3, line_search_patience = 3):
        """
        Optionally add baseline step, other steps or baseline metric.
        Important:
            - Steps must have the format, f_name(feature_properties, train, test, label_name, cols_to_drop=None, ...)
            - Each step must return a dictionary that contains 'mean_metric' and 'test_preds' and must modify train, test.
            - Baseline must have the format, f_name(feature_properties, train, test, label_name, ...)
            - (Optional) Baseline should also return 'cols_to_drop'. It must modify train, test for further use in following steps.
                This modified file will be saved. train_data_path and test_data_path will be changed to the path of the modified file.
        Args:
            steps (dict, {func_name1: params1, func_name2: params2, ...}): A dictionary defining each step in the project.
            baseline_step (dict, {func_name1: params1}): A dictionary defining baseline step. Metric of each step will be compared to that of this step.
            baseline_metric (float): Baseline metric to be compared, in case you had already ran the baseline before.
        
        """
        if steps is None:
            self.steps = {}
            self.step_results = pd.DataFrame(columns = ['diff_from_baseline', 'mean_metric'])
        else:
            self.steps = steps
            self.step_results = pd.DataFrame(columns = ['diff_from_baseline', 'mean_metric'], index = steps.keys())
        
        self.model_steps = {}
        self.model_results = pd.DataFrame(columns = ['diff_from_baseline', 'mean_metric'])
        
        self.baseline_step = None
        if baseline_step is not None:
            self.set_baseline_step(baseline_step)
        self.step_groups = {}
                   
        # HTML
        self.step_html = ''
        self.keephtml = False
        self.div_level = 1
        self.js = """<script>
                    var coll = document.getElementsByClassName("collapsible");
                    var i;
        
                    for (i = 0; i < coll.length; i++) {
                      coll[i].addEventListener("click", function() {
                        this.classList.toggle("active");
                        var content = this.nextElementSibling;
                        if (content.style.display === "none") {
                          content.style.display = "block";
                          this.style.borderBottomLeftRadius = "1px";
                          this.style.borderBottomRightRadius = "1px";
                          content.style.borderTopLeftRadius = "1px";
                          content.style.borderTopRightRadius = "1px";
                          
                        } else {
                            if (content.style.display === "block") {
                              content.style.display = "none";
                              this.style.borderBottomLeftRadius = "5px";
                              this.style.borderBottomRightRadius = "5px";
                              } else {
                                  content.style.display = "block";
                                  this.style.borderBottomLeftRadius = "1px";
                                  this.style.borderBottomRightRadius = "1px";
                                  content.style.borderTopLeftRadius = "1px";
                                  content.style.borderTopRightRadius = "1px";
                              }
                        }
                      });
                    }
                    </script>"""
        self.step_html_reset()
        
        ### LOAD FROM FILE ###
        # Project path must be a non-empty string
        if not (isinstance(project_path, str)): raise AssertionError()
        if not (len(project_path) > 0): raise AssertionError()
        if len(project_path) > 0:
            if os.path.exists(project_path) and not overwrite:
                print('A project exists on this path, loading...')
                self.load_project(project_path)
                return
            else:
                # Create project directories and set paths
                self.project_path = project_path
                self.test_preds_path = os.path.join(self.project_path, 'test_preds')
                self.log_dir = os.path.join(self.project_path, 'logs')
                self.prop_path = os.path.join(self.log_dir, 'datascitools_proj_prop')
                for path in [self.project_path, self.test_preds_path, self.log_dir]:
                    if not os.path.exists(path): os.makedirs(path)
            
        print('No project found, creating...')
                
        # Search
        self.line_search_params = line_search_params
        self.baseline_metric = baseline_metric
        self.minimize_metric = minimize_metric
        self.bayesian_search_iter = bayesian_search_iter
        self.line_search_iter = line_search_iter
        self.bayesian_search_count = bayesian_search_count
        self.final_bayesian_search_iter = final_bayesian_search_iter
        self.model_run_time = None
        self.n_random_seeds = n_random_seeds
        self.min_run_count = self.get_step_run_count()
        self.useful_limit = useful_limit
        self.line_search_patience = line_search_patience
        
        # Data
        self.col_dtypes = None
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.label_name = label_name
        self.feature_properties = feature_properties
        self.cols_to_drop = set()
        self.final_params = None
        self.original_properties = None
        self.cols_to_drop_path = None
        
        # Parallelism
        #self.multiprocessing_count = 2
        #self.create_project_properties()
        
        print('Loading data...')
        train, test = self.load_files()
        self.mean_label = train[self.label_name].mean()
        self.mode_label = train[self.label_name].mode()
        self.median_label = train[self.label_name].median()
        del train
        del test
        if overwrite:
            self.save_project()
        
    def save_project(self):
        proj_props = {}
        proj_props['mean_label'] = self.mean_label
        proj_props['median_label'] = self.median_label
        proj_props['mode_label'] = self.mode_label
        proj_props['baseline_metric'] = self.baseline_metric
        proj_props['final_params'] = self.final_params
        #proj_props['cols_to_drop'] = self.cols_to_drop
        proj_props['minimize_metric'] = self.minimize_metric
        proj_props['line_search_iter'] = self.line_search_iter
        proj_props['n_random_seeds'] = self.n_random_seeds
        proj_props['bayesian_search_iter'] = self.bayesian_search_iter
        proj_props['bayesian_search_count'] = self.bayesian_search_count
        proj_props['final_bayesian_search_iter'] = self.final_bayesian_search_iter
        proj_props['line_search_params'] = self.line_search_params
        proj_props['col_dtypes'] = self.col_dtypes
        proj_props['label_name'] = self.label_name
        proj_props['useful_limit'] = self.useful_limit
        
        # Paths
        proj_props['trn_path'] = self.train_data_path
        proj_props['test_path'] = self.test_data_path
        proj_props['project_path'] = self.project_path
        proj_props['test_preds_path'] = self.test_preds_path
        proj_props['log_dir'] = self.log_dir
        proj_props['prop_path'] = self.prop_path
        proj_props['cols_to_drop_path'] = self.cols_to_drop_path
        proj_props['line_search_patience'] = self.line_search_patience
        
        # save baseline function
        print('Saving baseline function')
        try:
            proj_props['baseline_serialized'] = dill.dumps(self.baseline_step['model'])
        except:
            print('No baseline model was found.')
        # Some hyperparameters need serialization, such as folds.
        try:
            proj_props['model_params'] = {key: dill.dumps(val) for key, val in self.final_params.items()}
        except:
            print('No model_params was found.')
        
        with open(self.prop_path, 'wb') as fp:
            pickle.dump(proj_props, fp)
        print(f'Saved project to {self.project_path}')
        if self.feature_properties is not None:
            self.feature_properties.to_csv(os.path.join(self.log_dir, 'feature_properties.csv'), index = False)
        if self.original_properties is not None:
            self.original_properties.to_csv(os.path.join(self.log_dir, 'original_properties.csv'), index = False)
    
    def get_feature_properties():
        # Calculate properties for all columns
        fp = vis.get_all_col_properties(train, test, label_name)
        
        # If not calculated before, self.dtypes will be None
        if self.dtypes is None:
            self.dtypes = fp['dtypes']
        else:
            new_dtypes = fp['dtypes']
                
    def load_project(self, proj_path):
        self.project_path = proj_path
        self.test_preds_path = os.path.join(self.project_path, 'test_preds')
        self.log_dir = os.path.join(self.project_path, 'logs')
        self.prop_path = os.path.join(self.log_dir, 'datascitools_proj_prop')
        
        try:
            with open(self.prop_path, 'rb') as fp:
                proj_props = pickle.load(fp)
        except:
            print('Could not load project.')
            return
        
        self.mean_label = proj_props.get('mean_label')
        self.median_label = proj_props.get('median_label')
        self.mode_label = proj_props.get('mode_label')
        self.baseline_metric = proj_props.get('baseline_metric')
        self.final_params = proj_props.get('final_params')
        #self.cols_to_drop = proj_props.get('cols_to_drop')
        self.col_dtypes = proj_props.get('col_dtypes')
        self.minimize_metric = proj_props.get('minimize_metric')
        self.line_search_iter = proj_props.get('line_search_iter')
        self.n_random_seeds = proj_props.get('n_random_seeds')
        self.bayesian_search_iter = proj_props.get('bayesian_search_iter')
        self.bayesian_search_count = proj_props.get('bayesian_search_count')
        self.final_bayesian_search_iter = proj_props.get('final_bayesian_search_iter')
        self.line_search_params = proj_props.get('line_search_params')
        self.label_name = proj_props.get('label_name')
        self.useful_limit = proj_props.get('useful_limit')
        self.line_search_patience = proj_props.get('line_search_patience')
        
        #Paths
        self.train_data_path = proj_props.get('trn_path')
        self.test_data_path = proj_props.get('test_path')
        self.project_path = proj_props.get('project_path')
        self.test_preds_path  = proj_props.get('test_preds_path')
        self.log_dir  = proj_props.get('log_dir')
        self.prop_path  = proj_props.get('prop_path')
        self.cols_to_drop_path = proj_props.get('cols_to_drop_path')
        
        # Load cols_to_drop
        self.load_cols_to_drop()
        
        if 'baseline_serialized' in proj_props:
            print('baseline model found.')
            if self.baseline_step is None:
                self.baseline_step = {}
            self.baseline_step['model'] = dill.loads(proj_props['baseline_serialized'])
        else:
            print('baseline model not found.')
            
        if 'model_params' in proj_props:
            print('loading model params')
            self.final_params = {key: dill.loads(val) for key, val in proj_props['model_params'].items()}
            self.baseline_step['model_params'] = dict(self.final_params)
        else:
            print('model params not found.')
        try:
            self.feature_properties = pd.read_csv(os.path.join(self.log_dir, 'feature_properties.csv'))
        except:
            train, test = self.load_files()
            self.step_html_text('feature_properties not provided, calculating feature_properties...')
            self.feature_properties = vis.get_all_col_properties(train, test, self.label_name)
        try:
            self.original_properties = pd.read_csv(os.path.join(self.log_dir, 'original_properties.csv'))
        except:
            print('Could not load original properties.')
        
        print(f'Project loaded from {self.project_path}')
    
    
    def get_step_run_count(self):
        # random seed count must be specified.
        if self.n_random_seeds is None: raise AssertionError()
        random_seed_count = self.n_random_seeds
        
        min_run_count = 2
        line_search_param_count = len(list(self.line_search_params.keys())) if self.line_search_params is not None else 1
        
        min_run_count *= self.line_search_iter
        min_run_count *= line_search_param_count
        min_run_count *= random_seed_count
        print(f'random_seed_count: {random_seed_count}')
        print(f'line_search_param_count: {line_search_param_count}')
        print(f'line_search_iteration_count: {self.line_search_iter}')
        print(f'\nModel will run at least {min_run_count} times for each step.')
        print('Formula is: 2 * line_search_iteration_count * random_seed_count * line_search_param_count')
        return min_run_count
    
    def load_cols_to_drop(self):
        if self.cols_to_drop_path is None:
            print('cols_to_drop does not exist.')
        else:
            # Load cols_to_drop
            try:
                print(f'Load cols_to_drop from: {self.cols_to_drop_path}')
                with open(self.cols_to_drop_path, 'rb') as fp:
                    self.cols_to_drop = pickle.load(fp)
            except FileNotFoundError:
                    self.cols_to_drop = set()
                    print('cols_to_drop does not exist.')
                    return
    
    def load_files(self, train = None, test = None, train_path = None, test_path = None):
        if train is None and test is None:
            self.load_cols_to_drop()
        
        if train is None:
            curr_name = self.train_data_path
            if train_path is None:
                # if training data is None, data path must be provided
                if not (self.train_data_path is not None): raise AssertionError()
            else:
                curr_path = train_path
                
            # Laod csv if file is csv
            if curr_name[-4:] == '.csv':
                train = dt.DataTable(pd.read_csv(curr_name))
                
            else:
                train = dt.DataTable(curr_name)
            
        if test is None:
            curr_name = self.test_data_path
            if test_path is None:
                # if test data is None, data path must be provided
                if not (self.test_data_path is not None): raise AssertionError()
            else:
                curr_name = test_path
                
            if curr_name[-4:] == '.csv':
                test = dt.DataTable(pd.read_csv(curr_name))
            else:
                test = dt.DataTable(curr_name)
        
        return train, test
    
    @putils.format_html
    def read_baseline_data(self):
        default_data_path = os.path.join(self.log_dir, 'BASELINE_{}_pipeline.csv')
        default_hparam_path = os.path.join(self.log_dir, 'BASELINE_hparams')
        trn_path = default_data_path.format('train')
        test_path = default_data_path.format('test')
        
        self.step_html_labeled_text('New train path: ', trn_path)
        self.step_html_labeled_text('New test path: ', test_path)
        self.train_data_path = trn_path
        self.test_data_path = test_path
        
        with open(default_hparam_path, 'wb') as fp:
            new_hparam = pickle.load(fp)
            self.baseline_step['model_params'] = new_hparam
        self.step_html_labeled_text('New hparams: ', self.baseline_step['model_params'])
    
    def set_baseline_step(self, model = None, proc = None, search_model_params = None, fixed_model_params = {}, proc_params = {}):
        self.baseline_step = {'proc': proc, 'tag': model.__name__, 'model': model, 'proc_params': proc_params,
                              'fixed_model_params': fixed_model_params, 'search_model_params': search_model_params, 'model_params': None}
    
    #@putils.keep_html
    def add_step_apply_if_useful(self, tag = None, model = None, proc = None, model_params = None, proc_params = {}, search_params = None):
        self.add_step(tag, model, proc, model_params, proc_params, search_params)
        if tag is None:
            if model is None:
                tag = proc.__name__
            else:
                tag = model.__name__
            # process parameters must be present
            if not (proc_params is not None): raise AssertionError()
                
            if len(proc_params.keys()) > 0:
                tag = tag + '_' + make_step_name(proc_params)
        return self.process_apply_if_useful(tag)
    
    def add_eval_step(self, tag = None, model = None, proc = None, model_params = None, proc_params = {}, search_params = None, group = None):
        self.add_step(tag, model, proc, model_params, proc_params, search_params, group)
        if tag is None:
            if model is None:
                tag = proc.__name__
            else:
                tag = model.__name__
            if not (proc_params is not None): raise AssertionError()
            tag = tag + '_' + make_step_name(proc_params)
        
        self.run_step(tag)
    
    def add_step(self, tag = None, model = None, proc = None, model_params = None, proc_params = {}, search_params = None, group = None):
        """
        Add a step to the pipeline.
        Steps may modify training and test data. They will be deleted at the end of each step.
        Important: 
            - Each step must return a dictionary that contains 'mean_metric' and 'test_preds'.
            - Each step must have the format, f_name(feature_properties, train, test, label_name, ...)
        
        Args:
            tag (str): step name to be displayed
            function (function): step function
            params (dict): parameters for the step
        """
        # Input check
        if model_params is not None:
            if not (model is not None): raise AssertionError()
        if proc_params is not None:
            if not (proc is not None): raise AssertionError()
        
        if search_params is None:
            search_params = self.line_search_params
        if tag is None:
            if model is None:
                tag = proc.__name__
            else:
                tag = model.__name__
            
            # Process params must be present.
            if (proc_params is None): raise AssertionError()
            if len(proc_params.keys()) > 0:
                tag = tag + '_' + make_step_name(proc_params)
        if tag in self.steps:
            self.step_html_text(f'{tag} already exists, replacing.')
            self.step_results.loc[tag, 'diff_from_baseline'] = None
            self.step_results.loc[tag, 'mean_metric'] = None
        else:
            self.step_results = pd.concat([self.step_results, pd.DataFrame(columns = ['diff_from_baseline', 'mean_metric'], index = [tag])], axis = 0)
        
        if group is not None:
            if group not in self.step_groups:
                self.step_groups[group] = {'best_step': None, 'best_metric': None, 'steps': [tag]}
            else:
                if tag not in self.step_groups[group]['steps']:
                    self.step_groups[group]['steps'].append(tag)
        
        
        self.steps[tag] = {'tag': tag, 'model': model,
                           'proc': proc, 'model_params': model_params,
                           'proc_params': proc_params, 'search_params': search_params,
                           'group': group}
        
    def group_steps(group_tag, tag_list):
        for tag in tag_list:
            # Tag must be in steps
            if not (tag in self.steps.keys()): raise AssertionError()
            self.steps[tag]['group'] = group_tag
        self.step_groups[group_tag] = {'best_step': None, 'best_metric': None, 'steps': tag_list}
    
    def show_step_results(self):
        self.step_results['diff_from_baseline'] = self.step_results['diff_from_baseline'].astype('float')
        self.step_results['mean_metric'] = self.step_results['mean_metric'].astype('float')
        display(self.step_results.style.background_gradient().highlight_null())
        
    def show_model_results(self):
        display(self.model_results.style.background_gradient().highlight_null())
    
    @putils.format_html
    def show_group(self, group_name):
        self.step_html_text(f'{group_name}:', islabel = True)
        for tag in self.step_groups[group_name]['steps']:
            self.step_html_text(tag)
    
    @putils.keep_html
    def show_steps(self):
        self.step_html = ''
        self.step_html_newbox(color = 'PaleGreen', display = 'block')
        self.step_html_labeled_text('Step Groups: ', '(Only the best step in a group will be applied.)')

        for group_tag in self.step_groups:
            self.show_group(group_tag)
        
        self.step_html_newbox(color = 'PaleGreen', display = 'block')
        steps_no_group = [tag for tag in self.steps if self.steps[tag]['group'] is None]
        self.step_html_labeled_text('Steps without a group: ', '(All will be applied if useful.)')
        for tag in steps_no_group:
            self.step_html_text(tag)
        
        
    @putils.keep_html
    def run_baseline(
            self, train = None, test = None,
            train_path = None, test_path = None, change_data = True,
            return_result = False, save_hparams = True, hparam_search_mode = 'bayesian'):
        
        # Baseline step must be present
        if not (self.baseline_step is not None): raise AssertionError()
        train, test = self.load_files(train, test, train_path, test_path)
        
        self.step_html_text('Baseline Step', islabel = True)
        if self.feature_properties is None:
            self.step_html_text('feature_properties not provided, calculating feature_properties...')
            self.feature_properties = vis.get_all_col_properties(train, test, self.label_name)
            self.original_properties = self.feature_properties.copy()
        self.step_html = self.step_html + '</div>'
        
        res_bl, train, test = self.run_step(None, train, test, baseline = True, hparam_search_mode = hparam_search_mode, return_data = True)
        self.baseline_metric = res_bl['mean_metric']
        
        self.set_cols_to_drop(res_bl['cols_to_drop'])
        self.step_html_line()
        self.step_html_labeled_text('Baseline metric: ', str(self.baseline_metric))
        self.modify_collapsible_box_title(f'Baseline: {self.baseline_metric}')
        self.save_data(train, test, 'BASELINE')
        self.save_hparams(res_bl['best_params'], 'BASELINE', set_as_final = True)
            
        del train
        del test
        if return_result:
            return res_bl
        
    def get_baseline_params(self):
        return self.baseline_step['model_params']
    
    def set_cols_to_drop(self, to_drop_new): # Handles Nones and duplicates
        if to_drop_new is not None:
            if self.cols_to_drop is None:
                self.cols_to_drop = set(to_drop_new)
            else:
                self.cols_to_drop.update(to_drop_new)
            self.cols_to_drop.discard(self.label_name)
    
    @putils.format_html
    def run_step(
            self, step_tag, train = None, test = None,
            train_path = None, test_path = None, baseline = False, hyperparam_search = True,
            hparam_search_mode = 'bayesian', save = False, return_data = True):
        
        start_time = time.time()
        if not (self.steps is not None) or \
           not (self.step_results is not None) or \
           not (self.feature_properties is not None):
            raise AssertionError()
        
        #self.step_html_labeled_text('Step Tag: ', str(step_tag))
        self.modify_collapsible_box_title('Baseline' if baseline else str(step_tag))
        step_dict = None
        if not baseline:
            step_dict = self.steps[step_tag]
        else:
            step_dict = self.baseline_step
        
        train, test = self.load_files(train, test, train_path, test_path)
        
        proc_param = step_dict['proc_params']
        self.step_html_start_collapsible('Processing')
        to_drop_new, train, test = self.capture_prints()(step_dict['proc'])(self.feature_properties, train, test, self.label_name, **proc_param)
        
        self.step_html_end_collapsible()
        
        self.set_cols_to_drop(to_drop_new)
        if save:
            self.save_data(train, test, step_dict["tag"])

        self.step_html_line()
        self.step_html_start_collapsible('Training')
        
        clear_output()
        display(HTML(self.step_html))
        
        cols_to_use_display = list(test.columns.difference(self.cols_to_drop))
        if len(cols_to_use_display) > 40:
            cols_to_use_display = cols_to_use_display[:40] + ['...']
        self.step_html_labeled_text('Columns to use: ', str(cols_to_use_display))
        res_step = None
        
        do_search = False
        if baseline: # This will run for the baseline step
            
            # Will use baseline model if model was not specified
            model_to_use = self.baseline_step['model'] \
                            if step_dict['model'] is None else step_dict['model']
            
            # Don't do hparam search
            if step_dict['search_model_params'] is None or not hyperparam_search:
                self.step_html_text('Use given hparameters.')
                
                # Run step model
                res_step = self.capture_prints()(model_to_use)(
                    train, test, self.label_name,
                    step_dict['fixed_model_params'], self.cols_to_drop)
                
            else:
                do_search = True
                
                # Do bayesian search
                self.step_html_text('Do bayesian search')
                res_step = self.do_bayesian_search(
                    model_to_use, step_dict, train,
                    test, self.cols_to_drop, hparam_search_mode = hparam_search_mode)
                
                # Print selected hparams after bayesian search
                self.step_html_labeled_text('Selected Params: ', str(res_step['best_params']))
                
            # Get results
            self.baseline_step['model_params'] = res_step['best_params']
        else: # If not a baseline step.
            # Use baseline model if model was not specified
            model_to_use = step_dict['model'] if step_dict['model'] is not None else self.baseline_step['model']
            
            model_exists = step_dict['model'] is not None and step_dict['model_params'] is not None
            params_to_use = step_dict['model_params'] if model_exists else self.baseline_step['model_params']
            
            if step_dict['search_params'] is None or not hyperparam_search:
                # No hparam search
                self.step_html_text('Use given hyperparameters')
                res_step = self.capture_prints()(model_to_use)(train, test, self.label_name, params_to_use, self.cols_to_drop)
            else:
                # Do line search using line_search(model, params_to_search, model_params, try_count = 3, change_rate = 0.2, minimize = True, **kwargs)
                do_search = True
                self.step_html_text('Do line search')
                res_step = self.capture_prints()(search.line_search)(model_to_use, step_dict['search_params'], params_to_use, \
                                                                     try_count = self.line_search_iter, minimize = self.minimize_metric,\
                                                                     train = train, test = test, label_name = self.label_name,
                                                                     cols_to_drop = self.cols_to_drop, patience = self.line_search_patience)
        self.step_html_end_collapsible()
        mean_metric = res_step['mean_metric']
        diff_bl = None
        if not baseline:
            self.step_html_line()
            diff_bl = mean_metric - self.baseline_metric
            
            # Baseline metric must be present
            if not (self.baseline_metric is not None): raise AssertionError()
                
            self.step_results.loc[step_tag, 'diff_from_baseline'] = diff_bl
            self.step_results.loc[step_tag, 'mean_metric'] = mean_metric
            self.step_html_text(f'Mean metric: ', islabel = True)
            self.step_html_text(str(mean_metric), newline = False)
            #self.step_html_text(f'Dev. of metric: ', islabel = True)
            #self.step_html_text(str(res_step['dev_metric']), newline = False)
            # Adjust color of the label
            better = (diff_bl > self.useful_limit and not self.minimize_metric) or (diff_bl < self.useful_limit and self.minimize_metric)
            label_color = 'green' if better else 'red'
            self.step_html_labeled_text(f'Difference from baseline: ', str(diff_bl), color = label_color)
            """label_color = '#095600' if better else '#680000'
            
            char_sign = ''
            if diff_bl > 0:
                char_sign = '+'
                
            self.modify_collapsible_box_title(f'{step_tag} : {char_sign}{diff_bl}', color = label_color)"""
            self.step_html_collapsible_add_metric(step_tag,
                                                  mean_metric,
                                                  self.baseline_metric)
            
            if step_dict['group'] is not None:
                group_best_metric = self.step_groups[step_dict['group']]['best_metric']
                better = (group_best_metric is None) or \
                         (mean_metric > group_best_metric and not self.minimize_metric) or \
                         (mean_metric < group_best_metric and self.minimize_metric)
                
                if better:
                    self.step_groups[step_dict['group']]['best_metric'] = mean_metric
                    self.step_groups[step_dict['group']]['best_step'] = step_dict['tag']
        
        if save:
            self.step_html_text(f'Final metric: {res_step["mean_metric"]}')
            self.step_html_text(f'Changing baseline metric to: {res_step["mean_metric"]}')
            self.baseline_metric = res_step['mean_metric']
            if do_search:
                self.save_hparams(res_step['best_params'], step_tag, set_as_final = True)
        
        res = {'mean_metric': mean_metric,
               'diff_from_baseline': diff_bl,
               'test_preds': res_step['test_preds'],
               'best_params': res_step.get('best_params', None),
               'cols_to_drop': to_drop_new}
        
        if res['best_params'] is not None:
            self.save_hparams(res['best_params'], step_dict["tag"])
        
        time_taken = time.time() - start_time
        if time_taken > 60:
            self.step_html_text(f'Step took {int(time_taken//60)}m {math.floor(time_taken%60)}s')
        else:
            self.step_html_text('Step took {:.1f}s'.format(time_taken))
        if return_data:
            return res, train, test
        else:
            return res
    
    def save_data(self, train, test, tag):
        self.step_html_text('Saving, will change paths to:')
        
        # Create paths
        self.train_data_path = os.path.join(self.log_dir, f'{tag}_train_pipeline')
        self.test_data_path = os.path.join(self.log_dir, f'{tag}_test_pipeline')
        self.cols_to_drop_path = os.path.join(self.log_dir, f'{tag}_colstodrop')
        
        # Display paths
        self.step_html_text(self.train_data_path)
        self.step_html_text(self.test_data_path)
        
        # Save train, test (They are DataTables, not dataframes)
        train.save(self.train_data_path)
        test.save(self.test_data_path)
        
        # Also save cols_to_drop, because cols_to_drop must be also reloaded
        # Before each step. The reason for that is if previous step was not useful,
        # it would not be applied and data would be rolled back. In this case,
        # cols_to_drop must be also rolled back.
        self.cols_to_drop_path = f'{tag}_colstodrop'
        with open(self.cols_to_drop_path, 'wb') as fp:
            pickle.dump(self.cols_to_drop, fp)
        
        self.col_dtypes = {col: train[col].dtype for col in test.columns}
        
        # Data is not saved unless it has changed and changes were useful.
        # For future feature engineering, recalculate feature properties
        self.step_html_text('Calculate feature properties, as data was changed.')
        self.feature_properties = vis.get_all_col_properties(train, test, self.label_name)
        del train; del test
        
    def save_hparams(self, params, tag, set_as_final = False):
        self.step_html_text('Saving hparams.')
        path = os.path.join(self.log_dir, f'{tag}_hparams')
        with open(path, 'wb') as fp:
            pickle.dump(params, fp)
        if set_as_final:
            self.step_html_text('Set parameters as final parameters.')
            self.final_params = params
    
    def step_html_collapsible_add_metric(self, step_tag, metric, baseline_metric):
        # Difference from baseline
        diff_bl = metric - baseline_metric
        
        # If we aim to decrease the metric, lower is better and vice versa.
        better = (diff_bl > self.useful_limit and not self.minimize_metric) \
              or (diff_bl < self.useful_limit and self.minimize_metric)
        
        # Set a color for worse and better, respectively
        label_color = '#095600' if better else '#680000'
        
        # Add a + sign if difference is positive
        char_sign = ''
        if diff_bl > 0:
            char_sign = '+'
            
        self.modify_collapsible_box_title(f'{step_tag} : {char_sign}{diff_bl}',
                                          color = label_color)
    
    def do_bayesian_search(
            self, model_to_use,
            step_dict, train,
            test, cols_to_drop,
            hparam_search_mode = 'bayesian'):
        
        """Please use search.bayesian_search for bayesian search. This function is not supposed to be called outside the class."""
        best_bayesian_result = None
        
        # First, run bayesian search in short lengths
        # The reason for that is bayesian search gets stuck on suboptimal regions if it has an unlucky start
        for i_search in range(self.bayesian_search_count):
            self.step_html_text('Bayesian search:', islabel = True)
            self.step_html_text(f'{i_search + 1}/{self.bayesian_search_count}', newline = False)
            bay_result = search.bayesian_search(model_to_use,
                                              step_dict['search_model_params'],
                                              step_dict['fixed_model_params'],
                                              num_iter = self.bayesian_search_iter,
                                              minimize = self.minimize_metric,
                                              train = train.copy(), test = test.copy(), label_name = self.label_name,
                                              cols_to_drop = cols_to_drop,
                                              tpe_trials = None) #these are kwargs
            # Decide if this bayesian run got better results
            if self.minimize_metric:
                if best_bayesian_result is None or best_bayesian_result['mean_metric'] > bay_result['mean_metric']:
                    best_bayesian_result = bay_result
            else:
                if best_bayesian_result is None or best_bayesian_result['mean_metric'] < bay_result['mean_metric']:
                    best_bayesian_result = bay_result
            
            # Display bayesian result
            self.step_html_labeled_text('Mean metric: ', str(bay_result['mean_metric']))
            self.step_html_labeled_text('Dev. of metric: ', str(bay_result['dev_metric']))
        
        # Run a final long search
        if self.final_bayesian_search_iter > 0:
            final_result = search.bayesian_search(model_to_use,
                                                  step_dict['search_model_params'],
                                                  step_dict['fixed_model_params'],
                                                  num_iter = self.final_bayesian_search_iter + self.bayesian_search_iter,
                                                  minimize = self.minimize_metric,
                                                  train = train, test = test, label_name = self.label_name,
                                                  cols_to_drop = cols_to_drop,
                                                  tpe_trials = best_bayesian_result['tpe_trials']) # Start from the end of the best search

            self.step_html_labeled_text('Final mean metric: ', str(final_result['mean_metric']))
            return final_result
        else:
            return best_bayesian_result
    
    @putils.format_html
    def apply_processing(
            self, step_tag,
            train = None, test = None,
            train_path = None, test_path = None,
            baseline = False, return_data = False,
            save = True):
        
        if not (self.steps is not None) or \
           not (self.step_results is not None) or \
           not (self.feature_properties is not None):
            raise AssertionError()
        
        step_dict = None
        if not baseline:
            step_dict = self.steps[step_tag]
        else:
            step_dict = self.baseline_step
        
        train, test = self.load_files(train, test, train_path, test_path)
        proc_param = step_dict['proc_params']
        
        self.step_html_labeled_text(f'Processing: ', f'{step_tag}')
        to_drop_new, train, test = self.capture_prints()(step_dict['proc'])(self.feature_properties, train, test, self.label_name, **proc_param)
        
        if save:  
            self.save_data(train, test, step_tag)
            self.set_cols_to_drop(to_drop_new)
        if return_data:
            return to_drop_new, train, test
    
    @putils.keep_html
    def apply_useful_processing(
            self, train = None,
            test = None, train_path = None,
            test_path = None):
        
        diff_bl = self.step_results.diff_from_baseline
        usefuls = self.step_results[diff_bl < -self.useful_limit] if self.minimize_metric else self.step_results[diff_bl > self.useful_limit]
        if len(usefuls) == 0:
            self.step_html_text('No useful steps were found.')
            return
        else:
            usefuls = list(zip(usefuls.index, usefuls.diff_from_baseline))
            
            # Eliminate steps that are not the best in their groups
            def is_best_in_group(step_tag):
                is_best = (self.steps[step_tag]['group'] is None) or (self.step_groups[self.steps[step_tag]['group']]['best_step'] == step_tag)
                if is_best:
                    self.step_html_text(f'Using {step_tag}')
                else:
                    self.step_html_text(f'Dropping {step_tag}')
                return is_best
            usefuls = [tag for tag, _ in usefuls if is_best_in_group(tag)]
            
            train, test = self.load_files(train, test, train_path, test_path)
            for tag in usefuls:
                # Don't evaluate, don't save to disk
                train, test = self.apply_processing(train, test, save = False, return_data = True)
            
            # Evaluate, save to disk and change baseline metric
            self.run_step(usefuls[-1], train, test, hyperparam_search = True, return_data = False, save = True)
        
    #def save_step_data(self, step_dict, train, test):
    def apply_baseline_processing(
            self, save = True,
            train = None, test = None,
            train_path = None, test_path = None):
        
         # Baseline step must be present
        if not (self.baseline_step is not None): raise AssertionError()
            
        train, test = self.load_files(train, test, train_path, test_path)
        if self.feature_properties is None:
            self.step_html_text('feature_properties not provided, calculating feature_properties...')
            self.feature_properties = vis.get_all_col_properties(train, test, self.label_name)
        
        cols_to_drop_new, train, test = self.apply_processing(None, train, test, baseline = True, save = False, return_data = True)
        self.set_cols_to_drop(cols_to_drop_new)
        self.save_data(train, test, self.baseline_step['tag'])
    
    @putils.keep_html
    def eval_all_steps(self):
        if not (self.steps is not None) or \
           not (self.step_results is not None) or \
           not (self.feature_properties is not None) or \
           not (len(list(self.steps.keys())) > 0):
            raise AssertionError()
            
        self.step_html_text('Try all steps', islabel = True)
        for tag in self.steps:
            self.run_step(tag, save = False, return_data = False)
        
        self.show_step_results()
    
    @putils.format_html
    def add_model(
            self, tag,
            model = None, fixed_hparams = None,
            search_hparams = None, model_type = 'linear'):
        
        if model is None:
            self.step_html_text(f'Using baseline model.')
            model = self.baseline_step['model']
            
        if tag is None:
            tag = model.__name__
        if tag in self.model_steps:
            self.step_html_text(f'{tag} already exists, replacing.')
            self.model_results.loc[tag, 'diff_from_baseline'] = None
            self.model_results.loc[tag, 'mean_metric'] = None
        else:
            self.model_results = pd.concat([self.model_results, pd.DataFrame(columns = ['diff_from_baseline', 'mean_metric'], index = [tag])], axis = 0)
            
        self.model_steps[tag] = {
            'tag': tag,
            'model': model,
            'fixed_model_params': fixed_hparams,
            'search_model_params': search_hparams,
            'model_params': None
        }
        
    @putils.format_html
    def run_model(
            self, step_tag, train = None, test = None,
            train_path = None, test_path = None, save_test_pred = True,
            hyperparam_search = True, return_pred = True, use_final_params = False):
        
        self.modify_collapsible_box_title(step_tag)
        
        # Model steps and step results must be present
        if not (len(list(self.model_steps.keys())) > 0) or \
           not (self.step_results is not None):
            raise AssertionError()
        
        step_dict = None
        step_dict = self.model_steps[step_tag]
        
        train, test = self.load_files(train, test, train_path, test_path)
        self.step_html_text(f'Training {step_tag}', islabel = True)
        self.step_html_line()
        res_step = None
        self.step_html_labeled_text('Columns to use: ',f'{list(train.columns.difference(self.cols_to_drop))}')
        self.step_html_labeled_text('Columns not to use: ', f'{self.cols_to_drop}')
        
        if not hyperparam_search:
            if step_dict['search_model_params'] is not None and step_dict['model_params'] is None:
                self.step_html_text('***  Warning: Search parameters will not be used, as they were not evaluated in hyperparameter search. ***')
                
            # Select parameter set
            params_to_use = step_dict['model_params'] if step_dict['model_params'] is not None else step_dict['fixed_model_params']
            if use_final_params:
                if self.final_params is None:
                    # No hyperparameters found
                    self.step_html_text('use_final_params is True, but baseline parameters have not changed. Using baseline parameters.', islabel = True)
                    
                    # Baseline step and its parameters must be present
                    if not (self.baseline_step is not None) or \
                       not (self.baseline_step['model_params'] is not None):
                        raise AssertionError()
                        
                    params_to_use = self.baseline_step['model_params']
                else:
                    params_to_use = self.final_params
                self.step_html_labeled_text('Hyperparameters: ', str(params_to_use))
                
            # Run model
            res_step = self.capture_prints()(step_dict['model'])(train, test, self.label_name, params_to_use, self.cols_to_drop)
            
        else: # Do bayesian search
            if (step_dict['search_model_params'] is None): raise AssertionError()
            best_bayesian_result = None
            for i_search in range(self.bayesian_search_count):
                self.step_html_text(f'Bayesian search: {i_search + 1}/{self.bayesian_search_count}')
                bay_result = search.bayesian_search(step_dict['model'],
                                                  step_dict['search_model_params'],
                                                  step_dict['fixed_model_params'],
                                                  num_iter = self.bayesian_search_iter,
                                                  minimize = self.minimize_metric,
                                                 train = train, test = test, label_name = self.label_name, cols_to_drop = self.cols_to_drop) #these are kwargs
                if (best_bayesian_result is None) or (self.minimize_metric and best_bayesian_result['mean_metric'] > bay_result['mean_metric']) or \
                                                     (not self.minimize_metric and best_bayesian_result['mean_metric'] < bay_result['mean_metric'])  :
                    best_bayesian_result = bay_result
                self.step_html_text(f'Mean metric: {bay_result["mean_metric"]}')
            res_step = best_bayesian_result
            step_dict['model_params'] = res_step['best_params']
            self.step_html_labeled_text('Selected Params: ', str(res_step['best_params']))
        
        mean_metric = res_step['mean_metric']
        self.step_results.loc[step_tag, 'mean_metric'] = mean_metric
        diff_bl = None
        if self.baseline_metric is not None:
            diff_bl = mean_metric - self.baseline_metric
            better = (diff_bl < 0) if self.minimize_metric else (diff_bl > 0)
            if better:
                self.save_hparams(res_step['best_params'], step_tag, set_as_final = True)
                self.step_html_labeled_text('Mean Metric: ', f'{mean_metric}', color = 'Green')
            else:
                self.step_html_labeled_text('Mean Metric: ', f'{mean_metric}', color = 'Red')
                
            self.step_results.loc[step_tag, 'diff_from_baseline'] = diff_bl
            self.step_html_labeled_text(f'Difference from baseline: ', f'{diff_bl}')
        else:
            self.step_html_labeled_text('Mean Metric: ', f'{mean_metric} (No baseline metric was found.)')
        
        res = {'mean_metric': mean_metric,
               'diff_from_baseline': diff_bl,
               'test_preds': res_step['test_preds']
              }
                
        self.step_html_text('Saving test predictions:')
        pred_path = f'{self.test_preds_path}\\{step_dict["tag"]}_{mean_metric}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
        self.step_html_text(pred_path)
        display(pd.Series(res_step['test_preds']))
        pd.Series(res_step['test_preds']).to_csv(pred_path, index = False)
        
        # Show the result in title of collapsible
        self.step_html_collapsible_add_metric(step_dict['tag'],
                                              mean_metric,
                                              self.baseline_metric)
        if return_pred:
            return res
    
    def train_blend_all_models(
            self, train_path = None,
            test_path = None, save_folder = None):
        
        train, test = self.load_files(None, None, train_path, test_path)
        test_results = []
        
        for step_tag in self.model_steps.keys():
            res = self.run_model(step_tag, train, test, hyperparam_search = True)
            test_results.append(res['test_preds'])
        
        test_results = np.hstack([np.reshape(test_res, (-1,1)) for test_res in test_results])
        test_results = np.mean(test_results, axis = 1)
        return pd.Series(test_results)
    
    @putils.keep_html
    def process_apply_if_useful(self, step_tag):
        self.step_html_text('Apply if useful', islabel = True)
        res, train, test = self.run_step(step_tag, save = False, return_data = True)
        
        best_metric = self.baseline_metric
        step_useful = (res['mean_metric'] < best_metric) if self.minimize_metric else (res['mean_metric'] > best_metric)
        if step_useful:
            self.save_data(train, test, step_tag)
            self.step_html_text(f'Changing baseline metric to: {res["mean_metric"]}')
            self.baseline_metric = res['mean_metric']
        else:
            self.step_html_text(f'Step {step_tag} was not useful. Best metric was: {best_metric}')
        return res
    
    @putils.keep_html
    def group_apply_useful(self, group_name):
        # Baseline metric must be present
        if not (self.baseline_metric is not None): raise AssertionError()
        # Find steps in the group
        steps_to_try = [self.steps[step]['tag'] for step in self.steps if self.steps[step]['group'] == group_name]
        
        # At least one step must be present in the specified group.
        if not (len(steps_to_try) > 0): raise AssertionError()
            
        self.step_html_labeled_text('Steps to evaluate: ', str(steps_to_try))
        
        best_step = None
        best_diff = None
        # Try steps in the group
        for tag in steps_to_try:
            res = self.run_step(tag, return_data = False, save = False)
            diff = res['diff_from_baseline']
            if (best_diff is None) or (diff < best_diff and self.minimize_metric) or (diff > best_diff and not self.minimize_metric):
                    best_step = tag
                    best_diff = diff

        self.step_html_labeled_text('Best Diff: ', str(best_diff))
        # Determine if any of the steps is useful
        if not ((self.minimize_metric and best_diff < -self.useful_limit) or (not self.minimize_metric and best_diff > self.useful_limit)):
            self.step_html_newbox(display = 'block')
            self.step_html_text('None of the steps were useful.')
            self.step_html_end_collapsible()
            
            return None
        else:
            # Apply best step
            self.step_html_newbox(display = 'block')
            self.step_html_text(f'Apply {best_step}', islabel = True)
            # Run last step and evaluate
            res, train, test = self.run_step(best_step, train = None, test = None, hyperparam_search = True, return_data = True, save = True)
            self.step_html_end_collapsible()
            
            return {'test_preds': res['test_preds']}
    
    def step_html_reset(self):
        if not self.keephtml:
            self.step_html = """
            <style>
                .collapsible {
                  background-color: #777;
                  color: white;
                  cursor: pointer;
                  padding: 18px;
                  text-align: left;
                  outline: none;
                  font-size: 15px;
                  font-weight: 700;
                  border-radius: 5px 5px 5px 5px;
                }
                .active, .collapsible:hover {
                  background-color: #555;
                }
                
                .container {
                  display: none;
                  background-color: inherit;
                }
                </style>
                """ + self.js
    
    def step_html_text(
            self, text,
            islabel = False, newline = True,
            color = None):
        
        bold =  islabel
        large = islabel
        html_template = '<{} style= "font-size:{};color:{};font-weight:{}">{}</{}>'
        
        weight = 700 if bold else 400
        size = 'large' if large else 'initial'
        start = 'div' if newline else 'span'
        
        html_new = html_template.format(start, size, color, weight, text, start)
        
        level = self.div_level * 6 + len(self.js)
        self.step_html = self.step_html[:-level] + html_new + self.step_html[-level:]

    
    def modify_collapsible_box_title(self, text, color = None):
        html_new = '<span style= "font-size:{};'
        if color is not None:
            html_new = html_new + f'color:{color}'
        html_new = html_new + f';font-weight:700">{text}</span>'
        
        to_find = 'collapsible_box">'
        place_i = self.step_html.rindex(to_find) + len(to_find)
        after_text_i = self.step_html[place_i:].index('</button>') + place_i
        
        # Insert new text
        self.step_html = self.step_html[:place_i] \
                       + html_new \
                       + self.step_html[after_text_i:]
        
        clear_output()
        display(HTML(self.step_html))
    
    def step_html_new_collapsible_box(self):
        level = self.div_level * 6 + + len(self.js)
        self.step_html = self.step_html[:-level] \
            + f'<div></div><button type="button" style="width: 97%; border: 1px solid #4E4E4E" class="collapsible collapsible_box"></button>'\
            + self.step_html[-level:]
            
        self.step_html_newbox()
    
    def step_html_newbox(self, color = None, display = 'none'):
        level = self.div_level * 6 + len(self.js)
        self.step_html = self.step_html[:-level] \
            + f'<div style="border:2px solid; border-radius: 15px; display: {display}; padding:2em; background: {color}"></div>'\
            + self.step_html[-level:]
        self.div_level += 1
    
    def step_html_start_collapsible(self, title):
        level = self.div_level * 6 + len(self.js)
        self.step_html = self.step_html[:-level] + \
        f'<div></div><button type="button" style="width: 97%" class="collapsible">{str(title)}</button>'+\
        '<div class="div.input_area container"></div>' + self.step_html[-level:]
        
        self.div_level += 1
        
    def step_html_end_collapsible(self):
        self.div_level -= 1
        # Show collapsible
        clear_output()
        display(HTML(self.step_html))
    
    def step_html_line(self):
        linehtml = """
        <style>
        hr { 
          display: block;
          margin-top: 0.5em;
          margin-bottom: 0.5em;
          margin-left: auto;
          margin-right: auto;
          border-style: inset;
          border-width: 1px;
        } 
        </style> <hr>"""
        level = 6 * self.div_level + len(self.js)
        self.step_html = self.step_html[:-level] + linehtml + self.step_html[-level:]
    
    def step_html_labeled_text(self, label, text, color = None): #'black'
        self.step_html_text(label, islabel = True, color = color)
        self.step_html_text(text, newline = False, color = color)
    
    def capture_prints(self):
        def real_decorator(func):
            @functools.wraps(func)
            def func_wrapper(*args, **kwargs):
                with io.capture_output(display = True) as captured:
                    out = func(*args, **kwargs)
                    
                seperated = str(captured).split('\n')
                for line in seperated:
                    if len(line) > 1500:
                        line = line[:1500] + ' ...'
                    if isinstance(line, str):
                        if line.isupper():
                            line = f'<b>{line}</b>'
                        self.step_html_text(line) # , newline = True
                    else:
                        # Line must be a string
                        raise AssertionError('Somehow you printed something that\'s not a string')
                return out
            return func_wrapper
        return real_decorator
    