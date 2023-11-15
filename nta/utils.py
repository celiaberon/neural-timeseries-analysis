import functools
import os

import numpy as np


def repeat_and_store(num_reps):

    '''
    Decorator to repeat function call and stores each call's output in a list.

    Args:
        num_reps:
            Number of times to call inner function.

    Returns:
        multi_output:
            List of len()=num_reps with output from inner function.
    '''

    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            multi_output = []
            for rep in range(num_reps):
                single_output = func(seed=rep, *args, **kwargs)
                multi_output.append(single_output)
            return multi_output
        return wrapper_repeat
    return decorator_repeat


def save_plot_metadata(func):

    def inner(*args, **kwargs):

        # Skip if not saving figure.
        if not kwargs.get('save', False):
            return func(*args, **kwargs)

        if func.__name__ == 'plot_trial_type_comparison':
            write_metadata_lineplots(*args, **kwargs)

        elif func.__name__ == 'plot_heatmap_wrapper':
            print('No metadata saving function')

        elif func.__name__ == 'plot_peaks_wrapper':
            write_metadata_peak_plots(*args, **kwargs)

        return func(*args, **kwargs)

    return inner


def write_metadata_lineplots(*args, **kwargs):

    fname = kwargs.get('fname')
    new_plot = kwargs.get('n_iters', (1, 0))[1] == 0
    exploded_trials = args[0].copy()

    # Number of sessions per mouse.
    sess_per_mouse = np.array(exploded_trials
                              .groupby("Mouse", as_index=False)["session"]
                              .nunique())
    grp_on = (kwargs.get('column')
              if not kwargs.get('ls_col', False)
              else [kwargs.get('column'), kwargs.get('ls_col')])

    # Number of trials per condition (trace) in plot.
    trials_per_cond = np.array(exploded_trials
                               .groupby(grp_on, as_index=False)["nTrial"]
                               .nunique(), dtype='str')

    metadata = [f'filename = {fname}',
                f'supblot = {kwargs.get("y_col").split("_")[0]}',
                f'channel = {"_".join(kwargs.get("y_col").split("_")[1:])}',
                f'n_trials = {exploded_trials.nTrial.nunique()}',
                f'n_sessions = {exploded_trials.Session.nunique()}',
                f'mice = {exploded_trials.Mouse.unique()}',
                f'sessions/mouse = {sess_per_mouse}',
                f'conditions = {grp_on}',
                f'trials/condition = \n{trials_per_cond}',
                '\n',
                ]

    if new_plot & ('continuity_broken' in exploded_trials.columns):
        metadata.insert(0, '\n')
        metadata.insert(0, 'prep_data_params ='
                        f'{kwargs.get("prep_data_params", "DEFAULT")}')

    # Create new metadata text file or append to existing file for each
    # subplot.
    metadata_fname = f'{fname[:-4]}_metadata.txt'
    write_style = ('a' if (os.path.exists(metadata_fname) and not new_plot)
                   else 'w')
    with open(metadata_fname, write_style) as f:
        for line in metadata:
            f.write(line)
            f.write('\n')


def write_metadata_peak_plots(*args, **kwargs):

    fname = kwargs.get('fname')
    peaks = args[0].copy()

    # Number of sessions per mouse.
    sess_per_mouse = np.array(peaks
                              .groupby("Mouse", as_index=False)["session"]
                              .nunique())

    # Number of trials per condition (trace) in plot.
    grp_on = ['Reward', kwargs.get('x_col')]
    trials_per_cond = np.array(peaks
                               .groupby(grp_on, as_index=False)["nTrial"]
                               .nunique(), dtype='str')

    metadata = [f'filename = {fname}',
                f'peak_metrics = {kwargs.get("metrics")}',
                f'show_outliers = {kwargs.get("show_outliers", False)}',
                f'channel = {kwargs.get("channel")}',
                f'n_trials = {peaks.nTrial.nunique()}',
                f'n_sessions = {peaks.Session.nunique()}',
                f'mice = {peaks.Mouse.unique()}',
                f'sessions/mouse = {sess_per_mouse}',
                f'conditions = {grp_on}',
                f'trials/condition = \n{trials_per_cond}',
                '\n',
                ]

    if 'continuity_broken' in peaks.columns:
        metadata.insert(0, '\n')
        metadata.insert(0, 'prep_data_params ='
                        f'{kwargs.get("prep_data_params", "DEFAULT")}')

    # Create new metadata text file and insert info.
    metadata_fname = f'{fname[:-4]}_metadata.txt'
    with open(metadata_fname, 'w') as f:
        for line in metadata:
            f.write(line)
            f.write('\n')
