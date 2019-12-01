from .. import utils
import numpy as np
import pandas as pd
import os
from IPython.display import display, HTML, clear_output
from functools import wraps

def keep_html(func):
    """ A decorator that prevents format_html from cleaning the output html by
    setting self.keephtml to True before the func. step_html_reset calls
    will not clear output if keep_html is used before a function.
    
    Use this before functions that contain loops of functions that
    use format_html.
    
    Parameters
    ----------
    func : A function
        Function to be called in the decorator.
        
    """
    # Handle decorators in func
    @wraps(func)
    
    # Inner loop for decorator
    def inner1(*args, **kwargs):
        # Clear html output
        args[0].step_html_reset()
        
        # If keep_html was already active in an outer loop, will not change it.
        active_in_outer = args[0].keephtml
        
        # Creating a new box increases div_level by one. This means that
        # Printing a new text will go into this box. It will return to the original
        # Div level after calling func.
        orig_div_level = args[0].div_level
        
        # Prevents format_html from clearing the output
        args[0].keephtml = True
        
        # Create an html div (it's a box.)
        if not active_in_outer:
            args[0].step_html_newbox(display = 'block')
            
        # Run function
        out = func(*args, **kwargs)
        
        # Enable step_html_reset to clear again
        if not active_in_outer:
            args[0].keephtml = False
        
        # Return to original div
        args[0].div_level = orig_div_level
        
        # Clear after running the function
        args[0].step_html_reset()
        return out
    return inner1

def format_html(func):
    """ A decorator to help functions with html output.
    Before the call of each function, creates an html box.
    After the call, resets html variable that is kept in the class
    as step_html.
    
    Parameters
    ----------
    func : A function
        Function to be called in the decorator.
        
    """
    # Handle decorators in func
    @wraps(func)
    
    # Inner loop for decorator
    def inner1(*args, **kwargs):
        # If user has interrupted the notebook, cleans remaining html
        args[0].step_html_reset()
        
        # Create an html div/box
        args[0].step_html_new_collapsible_box()
        out = func(*args, **kwargs)
        
        # Leave that collapsible box
        args[0].step_html_end_collapsible()
        
        clear_output()
        display(HTML(args[0].step_html))
        # Reset html string in class, because it was already displayed.
        args[0].step_html_reset()
        return out
    return inner1

def blend_from_csv(files = None, directory = ''):
    """Loads test predictions from csvs and averages them. Can do weighted
    averaging if files is a dictionary in format {[filename : weight]}.
    If you specify directory, give only filenames in files.
    
    Parameters
    ----------
    files : list of strings, dict or None
        if files is a list of paths or filenames, these files are blended.
        if files is a dict of {[filename : weight]}, weighted average is calculated.
        if files is None, files in directory are averaged.
    directory : string
        A directory that contains test predictions as csv. If files is none,
        this must be provided. This will be concatenated with filenames in files.
        
    Returns
    -------
    blend_result : pandas series
        Blending result
    
    """
    
    # If no filenames are given, get all files in directory.
    if files is None:
        files = os.listdir(directory)
        files = [f for f in files if f[-4:] == '.csv']
    
    # Keep track of the result
    blend_result = None
    
    # Load from dict
    filenames = None
    files_isdict = isinstance(files, dict)
    
    # Build filenames list
    if files_isdict:
        filenames = list(files.keys())
    else:
        # If not a dict, files must be either a string or a list of strings
        if not (isinstance(files, list) or isinstance(files, str)):
            raise AssertionError()
        
        # Convert to list if a string
        filenames = utils.tolist(files)
    
    # Iterate over all weights and filenames
    for f in filenames:
        # Get weight for this file
        weight = files[f] if files_isdict else 1 / len(list(files))
        
        # Filenames must be strings
        if not (isinstance(f, str)): raise AssertionError()
        
        # Load test predictions in this file
        path = os.path.join(directory,f)
        test_pred = np.reshape(pd.read_csv(path, header = None).values, (-1,1))
        
        if blend_result is None:
            # Set blending results
            blend_result = test_pred * weight
        else:
            # Add to blending results
            blend_result += test_pred * weight
    return pd.Series(blend_result.flatten())


def lgbm_hparam_seed_check(param):
    """Checks if number of random_states, bagging_seeds and folds are equal.
    Raises AssertionError if hparameters are not in correct count.
    
    Parameters
    ----------
    param : dictionary
        Hyperparameters for lgbm
    
    """
    seeds_islist = isinstance(param['random_state'], list)
    rs_islist = isinstance(param['bagging_seed'], list)
    bs_islist = isinstance(param['folds'], list)
    
    # All 3 random variable paramters must be either list or a single value.
    if not (all([not seeds_islist, not rs_islist, not bs_islist]) \
         or all([seeds_islist, rs_islist, bs_islist])):
        raise AssertionError()
    
    # Length check
    if seeds_islist:
        # Folds, random_state and bagging_seed must have the same length
        random_state_count = len(param['random_state'])
        len_equal = (len(param['bagging_seed']) == random_state_count) \
                and (len(param['folds']) == random_state_count)
        # Finally, check condition
        if not len_equal: raise AssertionError()