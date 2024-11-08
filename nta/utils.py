import configparser
import functools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfkit
import seaborn as sns
from IPython.core.magic import register_cell_magic


def load_config_variables(path_to_file: str,
                          section: str = 'color_palette') -> dict:

    '''
    Create dictionary containing parameter values that will be repeated
    across notebooks

    Args:
        section:
           Section name within configuration file.

    Returns:
        config_variables:
            Dictionary containing variables and assigned values from config
            file.
    '''

    # For color palette configuration only
    import matplotlib as mpl
    cpal = mpl.cm.RdBu_r(np.linspace(0, 1, 8))

    if not os.path.isfile(os.path.join(path_to_file, 'plot_config.ini')):
        path_to_file = os.getcwd()

    config_file = configparser.ConfigParser()
    config_file.read(os.path.join(path_to_file, 'plot_config.ini'))

    # Create dictionary with key:value for each config item
    config_variables = {}
    for key in config_file[section]:
        config_variables[key] = eval(config_file[section][key])

    return config_variables


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


def single_session(func):
    def inner(*args, **kwargs):

        trials = args[0]
        assert trials.Session.dropna().nunique() == 1, (
            'Not written for multi-session processing')
        output = func(*args, **kwargs)
        return output
    return inner


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

        elif func.__name__ == 'multiclass_roc_curves':
            write_metadata_roc(*args, **kwargs)

        return func(*args, **kwargs)

    return inner


def write_metadata(file_path, metadata, new_plot=True):

    # Create new metadata text file or append to existing file for each
    # subplot.
    metadata_path = file_path.split('/')
    metadata_path.insert(-1, 'metadata')
    metadata_path = '/'.join(metadata_path)

    metadata_fname = f'{metadata_path[:-4]}_metadata.txt'
    write_style = ('a' if (os.path.exists(metadata_fname) and not new_plot)
                   else 'w')
    with open(metadata_fname, write_style) as f:
        for line in metadata:
            f.write(line)
            f.write('\n')


def write_metadata_lineplots(*args, **kwargs):

    fname = kwargs.get('fname')
    new_plot = kwargs.get('n_iters', (1, 0))[1] == 0
    exploded_trials = args[0].copy()

    # Number of sessions per mouse.
    sess_per_mouse = np.array(exploded_trials
                              .groupby("Mouse", as_index=False, observed=True)["Session"]
                              .nunique())
    grp_on = (kwargs.get('column')
              if not kwargs.get('ls_col', False)
              else [kwargs.get('column'), kwargs.get('ls_col')])

    # Number of trials per condition (trace) in plot.
    trials_per_cond = np.array(exploded_trials
                               .groupby(grp_on, as_index=False, observed=True)["nTrial"]
                               .nunique(), dtype='str')

    metadata = [f'filename = {fname}',
                f'subplot = {kwargs.get("y_col").split("_")[0]}',
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
    write_metadata(fname, metadata, new_plot)


def write_metadata_peak_plots(*args, **kwargs):

    fname = kwargs.get('fname')
    peaks = args[0].copy()

    # Number of sessions per mouse.
    sess_per_mouse = np.array(peaks
                              .groupby("Mouse", as_index=False, observed=True)["Session"]
                              .nunique())

    # Number of trials per condition (trace) in plot.
    grp_on = ['Reward', kwargs.get('x_col')]
    trials_per_cond = np.array(peaks
                               .groupby(grp_on, as_index=False, observed=True)["nTrial"]
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

    write_metadata(fname, metadata)


def write_metadata_roc(*args, **kwargs):

    fname = kwargs.get('fname')
    trials = kwargs.get('trials').copy()

    # Number of sessions per mouse.
    sess_per_mouse = np.array(trials
                              .groupby("Mouse", as_index=False, observed=True)["Session"]
                              .nunique())

    # Number of trials per condition (trace) in plot.
    grp_on = [kwargs.get('trial_type', 'Reward'), kwargs.get('pred_behavior')]
    trials_per_cond = np.array(trials
                               .groupby(grp_on, as_index=False, observed=True)["nTrial"]
                               .nunique(), dtype='str')

    metadata = [f'filename = {fname}',
                f'neural_event = {kwargs.get("neural_event")}',
                f'predicted_behavior = {kwargs.get("pred_behavior")}',
                f'conditioned_event = {kwargs.get("trial_type", "Reward")}',
                f'n_samples_per_rep = {kwargs.get("n_samples")}',
                f'n_trials = {trials.nTrial.nunique()}',
                f'n_sessions = {trials.Session.nunique()}',
                f'mice = {trials.Mouse.unique()}',
                f'sessions/mouse = {sess_per_mouse}',
                f'conditions = {grp_on}',
                f'trials/condition = \n{trials_per_cond}',
                '\n',
                ]

    if 'continuity_broken' in trials.columns:
        metadata.insert(0, '\n')
        metadata.insert(0, 'prep_data_params ='
                        f'{kwargs.get("prep_data_params", "DEFAULT")}')

    write_metadata(fname, metadata)


def downcast_all_numeric(df):

    fcols = df.select_dtypes('float').columns
    icols = df.select_dtypes('integer').columns

    df[fcols] = df[fcols].apply(pd.to_numeric, downcast='float')
    df[icols] = df[icols].apply(pd.to_numeric, downcast='integer')

    return df


def cast_object_to_category(df):

    cols = df.select_dtypes('object').columns
    for col in cols:
        df[col] = df[col].astype('category')

    return df


def loop_cell_adjust_legend(flag, sp_idx, ax, n_groups, **kwargs):
    if (sp_idx == (n_groups - 1)) or (not flag):
        ax.get_legend().set(**kwargs)
    else:
        ax.get_legend().remove()


def loop_cell_create_axes(flag, figsize, ncols=3, nrows=1, **kwargs):
    w, h = figsize
    ncols = ncols if flag else 1
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows,
                            figsize=(w * ncols, h * nrows),
                            layout='constrained', **kwargs)
    if not flag:
        axs = [axs]
    return fig, axs


def label_outer_axes(fig, axs, xlabel, ylabel):

    if not isinstance(fig, plt.Figure):
        fig = fig.fig
    else:
        [ax_.set(xlabel='', ylabel='') for ax_ in axs.flatten()]
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                    left=False, right=False)
    plt.ylabel(xlabel, labelpad=10)
    plt.xlabel(ylabel)
    plt.tight_layout()
    return fig, axs


def html_to_pageless_pdf(input_html, output_pdf):
    options = {
        'page-size': 'A1',                  # Larger page size for continuous flow
        'disable-smart-shrinking': '',      # Avoids shrinking content, useful for continuous layouts
        'no-outline': None,                 # Prevents automatic outlines that may add spaces
        'zoom': '1.8',                     # Adjust zoom to ensure images scale without large spaces
        'margin-top': '0',
        'margin-bottom': '0',
        'viewport-size': '1280x1024',       # Sets viewport for better handling of embedded images

    }
    pdfkit.from_file(input_html, output_pdf, options=options)


def set_notebook_params(grp_key, notebook_id, root='.'):

    config_file = configparser.ConfigParser()
    config_file.read(os.path.join(root, 'prep_data_params.ini'))

    # Create dictionary with key:value for each config item
    data_loading_params = {}
    for key in config_file['data_loading']:
        data_loading_params[key] = eval(config_file['data_loading'][key])

    # Create dictionary with key:value for each config item
    data_cleaning_params = {}
    for key in config_file['data_cleaning']:
        data_cleaning_params[key] = eval(config_file['data_cleaning'][key])

    config_file.read('../mouse_cohorts.ini')
    data_loading_params['mice'] = eval(config_file['cohorts'].get(grp_key.lower()))
    data_loading_params['label'] = f'{grp_key}/{notebook_id}'

    return data_loading_params, data_cleaning_params
