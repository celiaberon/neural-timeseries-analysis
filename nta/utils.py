import configparser
import functools
import os
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdfkit
import seaborn as sns
from IPython.core.magic import register_cell_magic


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


def calc_sess_per_mouse(trials):
    sess_per_mouse = (trials
                       .groupby("Mouse", observed=True)["Session"]
                       .nunique())
    return f'\n{" " * 5}{sess_per_mouse.to_string().replace(chr(10), chr(10) + " " * 5)}'


def calc_trials_per_cond(trials, grp_on):
    trials_per_cond = (trials
                       .groupby(grp_on, observed=True)["nTrial"]
                       .nunique())

    return f'\n{" " * 5}{trials_per_cond.to_string().replace(chr(10), chr(10) + " " * 5)}'


def record_continuity_broken(df, metadata, **kwargs):
    if 'continuity_broken' in df.columns:
        metadata.insert(0, '\n')
        metadata.insert(0, 'prep_data_params ='
                        f'{kwargs.get("prep_data_params", "DEFAULT")}')
    return metadata


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
    metadata_path = file_path.split('\\') if platform.system() == 'Windows' else file_path.split('/')
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

    # For trials per condition.
    grp_on = (kwargs.get('column')
              if not kwargs.get('ls_col', False)
              else [kwargs.get('column'), kwargs.get('ls_col')])

    metadata = [f'filename = {fname}',
                f'subplot = {kwargs.get("y_col").split("_")[0]}',
                f'channel = {"_".join(kwargs.get("y_col").split("_")[1:])}',
                f'n_trials = {exploded_trials.nTrial.nunique()}',
                f'n_sessions = {exploded_trials.Session.nunique()}',
                f'mice = {exploded_trials.Mouse.unique()}',
                f'sessions/mouse = {calc_sess_per_mouse(exploded_trials)}',
                f'conditions = {grp_on}',
                f'trials/condition = \n{calc_trials_per_cond(exploded_trials, grp_on)}',
                '\n',
                ]

    if new_plot:
        metadata = record_continuity_broken(exploded_trials, metadata, **kwargs)

    write_metadata(fname, metadata, new_plot)


def write_metadata_peak_plots(*args, **kwargs):

    fname = kwargs.get('fname')
    peaks = args[0].copy()

    # Number of trials per condition (trace) in plot.
    grp_on = ['Reward', kwargs.get('x_col')]

    metadata = [f'filename = {fname}',
                f'peak_metrics = {kwargs.get("metrics")}',
                f'show_outliers = {kwargs.get("show_outliers", False)}',
                f'channel = {kwargs.get("channel")}',
                f'n_trials = {peaks.nTrial.nunique()}',
                f'n_sessions = {peaks.Session.nunique()}',
                f'mice = {peaks.Mouse.unique()}',
                f'sessions/mouse = {calc_sess_per_mouse(peaks)}',
                f'conditions = {grp_on}',
                f'trials/condition = \n{calc_trials_per_cond(peaks, grp_on)}',
                '\n',
                ]

    metadata = record_continuity_broken(peaks, metadata, **kwargs)

    write_metadata(fname, metadata)


def write_metadata_roc(*args, **kwargs):

    fname = kwargs.get('fname')
    trials = kwargs.get('trials').copy()

    # Number of trials per condition (trace) in plot.
    grp_on = [kwargs.get('trial_type', 'Reward'), kwargs.get('pred_behavior')]

    metadata = [f'filename = {fname}',
                f'neural_event = {kwargs.get("neural_event")}',
                f'predicted_behavior = {kwargs.get("pred_behavior")}',
                f'conditioned_event = {kwargs.get("trial_type", "Reward")}',
                f'n_samples_per_rep = {kwargs.get("n_samples")}',
                f'n_trials = {trials.nTrial.nunique()}',
                f'n_sessions = {trials.Session.nunique()}',
                f'mice = {trials.Mouse.unique()}',
                f'sessions/mouse = {calc_sess_per_mouse(trials)}',
                f'conditions = {grp_on}',
                f'trials/condition = \n{calc_trials_per_cond(trials, grp_on)}',
                '\n',
                ]

    metadata = record_continuity_broken(trials, metadata, **kwargs)

    write_metadata(fname, metadata)


def write_metadata_glm(*args, **kwargs):

    fname = kwargs.get('fname')
    trials = kwargs.get('trials').copy()
    ts = kwargs.get('ts').copy()

    events_per_predictor = ts.sum(axis=0).drop([col for col in ts.columns if col.startswith('z_')]).astype('int')

    metadata = [f'filename = {fname}',
                f'n_trials = {trials.nTrial.nunique()}',
                f'n_sessions = {trials.Session.nunique()}',
                f'mice = {trials.Mouse.unique()}',
                f'sessions/mouse = {calc_sess_per_mouse(trials)}',
                f'trials/mouse = {calc_trials_per_cond(trials, "Mouse")}',
                f'events/predictor = \n{" " * 5}{events_per_predictor.to_string().replace(chr(10), chr(10) + " " * 5)}',
                '\n',
                ]

    metadata = record_continuity_broken(trials, metadata, **kwargs)

    with open(fname, 'w') as f:
        for line in metadata:
            f.write(line)
            f.write('\n')


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

    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                    left=False, right=False)
    plt.ylabel(ylabel, labelpad=10)
    plt.xlabel(xlabel, labelpad=10)

    return fig, axs


def html_to_pageless_pdf(input_html, output_pdf):
    
    # Platform-specific wkhtmltopdf path
    if platform.system() == 'Windows':
        wkhtmltopdf_path = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    else:
        # Original Mac path (likely /usr/local/bin/wkhtmltopdf or similar)
        wkhtmltopdf_path = '/usr/local/bin/wkhtmltopdf'  # Adjust if different
            
    # Create configuration with explicit path
    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
    
    options = {
        'page-size': 'A1',                  # Larger page size for continuous flow
        'disable-smart-shrinking': '',      # Avoids shrinking content, useful for continuous layouts
        'no-outline': None,                 # Prevents automatic outlines that may add spaces
        'zoom': '1.8',                     # Adjust zoom to ensure images scale without large spaces
        'margin-top': '0',
        'margin-bottom': '0',
        'viewport-size': '1280x1024',       # Sets viewport for better handling of embedded images

    }
    pdfkit.from_file(input_html, output_pdf, options=options, configuration=config)


def set_notebook_params(grp_key, notebook_id, root='.'):

    config_file = configparser.ConfigParser()
    config_file.read(os.path.join(root, f'{grp_key.lower()}_params.ini'))
    # Create dictionary with key:value for each config item
    data_loading_params = {}
    for key in config_file['data_loading']:
        data_loading_params[key] = eval(config_file['data_loading'][key])

    # Create dictionary with key:value for each config item
    data_cleaning_params = {}
    for key in config_file['data_cleaning']:
        data_cleaning_params[key] = eval(config_file['data_cleaning'][key])

    config_file.read(os.path.join(root, 'mouse_cohorts.ini'))
    data_loading_params['mice'] = eval(config_file['cohorts'].get(grp_key.lower()))
    data_loading_params['label'] = f'{grp_key.lower()}/{notebook_id.lower()}'

    return data_loading_params, data_cleaning_params


def label_hemi(ch, all_channels):
    hemisphere_labels = {'L': 'Left Hemi', 'R': 'Right Hemi'}

    single_color = len(set([ch[-4:-1] for ch in all_channels])) == 1
    if single_color:
        label = hemisphere_labels.get(ch[-1])
    else:
        label = f'{hemisphere_labels.get(ch[-1])}: {ch.split("_")[1][:3]}'

    return label
