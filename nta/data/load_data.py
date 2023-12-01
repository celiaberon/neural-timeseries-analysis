"""
Created on Tue Apr 26 20:17:37 2022

@author: celiaberon
"""

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import nta.preprocessing.quality_control as qc
from nta.features.behavior_features import add_behavior_cols
from nta.features.design_mat import make_design_mat
from nta.utils import load_config_variables


def sessions_to_load(mouse: str,
                     dataset: str,
                     *,
                     root: str = '',
                     probs: int = 9010,
                     QC_pass: bool = True,
                     ) -> list:

    '''
    Make list of sessions to include for designated mouse

    Args:
        mouse (str):
            Mouse ID.
        root (str, path):
            Path to reference sheet containing preprocessing info for each
            session.
        prob_set (int):
            Filter bandit data by probability conditions.
        QC_pass (bool):
            Whether to take sessions passing quality control (True) or failing
            (False).

    Returns:
        dates_list (list):
            List of dates to load in for mouse.
    '''

    root = set_data_overview_path(root)

    if dataset in ['standard', 'colab']:
        session_log = 'session_log_all_cohorts.csv'
    elif dataset == 'dan':
        session_log = 'session_log_dan.csv'

    file_path = root / session_log
    ref_df = pd.read_csv(file_path)
    probs = str(probs) if not isinstance(probs, str) else probs
    if dataset in ['standard', 'colab']:
        ref_df = ref_df.query('Mouse == @mouse \
                              & Condition == @probs & Pass == @QC_pass')
    elif dataset == 'dan':
        ref_df = ref_df.query('Mouse == @mouse \
                              & Condition == @probs & trt_grp=="WT"')
    return list(set(ref_df.Date.values))


def get_max_trial(full_sessions: dict) -> int:

    '''
    Get maximum trial ID to use for unique trial ID assignment. Importantly,
    also confirm that max trial matches between dataframes.

    Args:
        full_sessions:
            Dictionary containing and trial- and timeseries-based data.

    Returns:
        max_trial:
            Number corresponding to maximum trial value.
    '''

    try:
        max_trial_trials = full_sessions['trials'].nTrial.max()
        max_trial_ts = full_sessions['ts'].nTrial.max()
        assert max_trial_trials == max_trial_ts
        max_trial = max_trial_trials
    except AttributeError:
        max_trial = 0

    return max_trial


def concat_sessions(*,
                    sub_sessions: dict = None,
                    full_sessions: dict = None,
                    **kwargs):

    '''
    Aggregate multiple sessions by renumbering trials to provide unique trial
    ID for each trial. Store original id in separate column.

    Args:
        sub_sessions:
            Smaller unit to be concatenated onto growing aggregate dataframe.
        full_sessions:
            Core larger unit updated with aggregating data.

    Returns:
        full_sessions:
            Original full_sessions data now also containing sub_sessions data.
    '''

    max_trial = get_max_trial(full_sessions)

    # Iterate over both trial and timeseries data.
    for key, ss_vals in sub_sessions.items():

        # Store original trial ID before updating with unique value.
        if 'nTrial_orig' not in ss_vals.columns:
            ss_vals['nTrial_orig'] = ss_vals['nTrial'].copy()

        # Create session column to match across dataframes.
        if 'session' not in ss_vals.columns:
            mouse = kwargs.get('mouse', None)
            session_date = kwargs.get('session_date', None)
            ss_vals['session'] = '_'.join([mouse, session_date])

        # Add max current trial value to all new trials before concatenation.
        tmp_copy = ss_vals.copy()
        tmp_copy['nTrial'] += max_trial
        full_sessions[key] = pd.concat((full_sessions[key], tmp_copy))
        full_sessions[key] = full_sessions[key].reset_index(drop=True)

    # Use function to assert that new dataframes have matching max trial ID.
    _ = get_max_trial(full_sessions)

    return full_sessions


def read_multi_mice(mice: list[str],
                    root: str = '',
                    **kwargs) -> dict:

    '''
    Load in sessions by mouse and concatenate into one large dataframe keeping
    every trial id unique.

    Args:
        mice:
            List of mice from which to load data.
        root:
            Path to root directory containing Mouse data folder.

    Returns:
        multi_mice (dict):
            {'trials': trials data, 'timeseries': timeseries data}
    '''
    if not root:
        root = input('Please provide a path to the data:')

    if not isinstance(mice, list):
        mice = [mice]

    multi_mice = {key: pd.DataFrame() for key in ['trials', 'ts']}

    for mouse in mice:

        multi_sessions = read_multi_sessions(mouse=mouse, root=root, **kwargs)

        if len(multi_sessions['trials']) < 1:
            continue  # skip mouse if no sessions returned

        multi_mice = concat_sessions(sub_sessions=multi_sessions,
                                     full_sessions=multi_mice)

    print(f'{multi_mice["trials"].Session.nunique()} total sessions loaded in')
    return multi_mice


def read_multi_sessions(mouse: str,
                        root: str = '',
                        *,
                        prob_set: int = 9010,
                        QC_pass: bool = True,
                        dataset: str = None,
                        qc_photo: bool = True,
                        **dm_kwargs) -> dict:

    if not root:
        root = input('Please provide a path to the data:')
    if dataset is None:
        dataset = 'standard'
    root = Path(root)
    cohort = load_cohort_dict(root, dataset)
    # define list of files to work through (by session and preprocessing date)
    dates_list = sessions_to_load(mouse, probs=prob_set, QC_pass=QC_pass,
                                  root=root, dataset=dataset)
    multi_sessions = {key: pd.DataFrame() for key in ['trials', 'ts']}

    # Loop through files to be processed
    for session_date in tqdm(dates_list, mouse, disable=False):

        file_path = set_session_path(root, mouse=mouse, session=session_date,
                                     dataset=dataset)

        ts_path = file_path / f'{mouse}_{session_date}_timeseries.parquet.gzip'
        trials_path = file_path / f'{mouse}_trials.csv'

        if not (ts_path.exists() & trials_path.exists()):
            print(f'skipped {mouse} {session_date}')
            continue

        ts = pd.read_parquet(ts_path)
        trials = pd.read_csv(trials_path, index_col=0)

        trials, ts = add_behavior_cols(trials, ts)

        # If no photometry channels passed QC, move on to next session.
        channels = {'z_grnL', 'z_grnR'}
        if qc_photo:
            sig_cols = {ch for ch in channels
                        if not qc.is_normal(ts.get(ch, None),
                                            sensor=cohort.get(mouse))}
        else:
            sig_cols = channels
        if not sig_cols:
            continue
        # Replace channels without signal with NaNs.
        ts[list(channels - sig_cols)] = np.nan

        if dm_kwargs:
            ts = make_design_mat(ts, trials, **dm_kwargs)

        # Trial level quality control needs to come at the end.
        trials_matched = qc.QC_included_trials(ts,
                                               trials,
                                               allow_discontinuity=False,
                                               drop_enlP=False)

        multi_sessions = concat_sessions(sub_sessions=trials_matched,
                                         full_sessions=multi_sessions,
                                         mouse=mouse,
                                         session_date=session_date)

    # QC all mice sessions by ENL penalty rate set per mouse

    return multi_sessions


###################################
# Functions to set some data paths
###################################

def set_session_path(root, dataset, *, mouse: str = '', session: str = ''):

    if dataset == 'standard':
        mid_path = root / 'headfixed_DAB_data/preprocessed_data'
    elif dataset == 'dan':
        mid_path = root / 'headfixed_DAB_data/Dan_data/preprocessed_data'
    elif dataset == 'colab':
        mid_path = root
    else:
        raise NotImplementedError('No path to dataset provided')

    full_path = mid_path / mouse / session

    return full_path


def set_data_overview_path(root):

    full_path = root / 'data_overviews'

    return full_path


def load_cohort_dict(root, dataset):

    import os
    if dataset in ['standard', 'colab']:
        cohort = load_config_variables(os.getcwd(), 'cohort')['cohort']
    else:
        raise NotImplementedError

    return cohort
