#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 20:17:37 2022

@author: celiaberon
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import nta.preprocessing.quality_control as qc
from nta.features.behavior_features import add_behavior_cols
# from nta.features.pp_design_mat import make_design_mat


def sessions_to_load(mouse: str,
                     root: str,
                     prob_set: int=9010,
                     QC_pass: bool=True) -> list:

    '''
    Make list of sessions to include for designated mouse

    Args:
        mouse (str):
            Mouse ID.
        root (str, path):
            Path to reference sheet containing preprocessing info for each session.
        prob_set (int):
            Filter bandit data by probability conditions.
        QC_pass (bool):
            Whether to take sessions passing quality control (True) or failing (False).

    Returns:
        dates_list (list):
            List of dates to load in for mouse.
    '''

    if mouse in [f'C{i}' for i in range(30,37)]:
        # old method without session log
        from datetime import date, datetime

        ref_df = pd.read_csv(f'{root}photometry_round1_pp.csv', index_col=None)
        updated_version = date(2022, 6, 22)

        status_col = f'states_{updated_version}'
        ref_df = ref_df.loc[(ref_df.Mouse==mouse) 
                            & (~pd.isnull(ref_df[status_col])) 
                            & ~(ref_df[status_col].isin(['False', 'FALSE', False]))
                            & (ref_df.Condition.isin([prob_set, str(prob_set)]))]

        date_format = '%m/%d/%y'
        dates_mask = [(datetime.strptime(row[status_col], date_format).date()-updated_version).days >= 0 for _, row in ref_df.iterrows()]
        dates_list = ref_df[dates_mask].Date.values

        return dates_list

    if mouse in [f'C{i}' for i in range(44,48)]:
        ref_df = pd.read_csv(f'{root}session_log_QC_photometry_dms.csv', index_col=0)
    else:
        ref_df = pd.read_csv(f'{root}session_log_QC_photometry.csv', index_col=0)

    ref_df = ref_df.loc[(ref_df.Mouse==mouse)
                        & (ref_df.Condition==prob_set)
                        & (ref_df.Pass==QC_pass)
                        & (ref_df.enlp_pass==QC_pass)
                        ]
    dates_list = [session[4:] for session in ref_df.Session.values]

    return dates_list


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
        max_trial_bdf = full_sessions['bdf'].nTrial.max()
        max_trial_analog = full_sessions['analog'].nTrial.max()
        assert max_trial_bdf == max_trial_analog
        max_trial = max_trial_bdf
    except KeyError:
        max_trial = 0

    return max_trial


def concat_sessions(*,
                    sub_sessions: dict=None,
                    full_sessions: dict=None,
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
        if 'session' not in sub_sessions.columns:
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


def read_in_multi_mice(mice: list, **kwargs) -> dict:

    '''
    Load in sessions by mouse and concatenate into one large dataframe keeping
    every trial id unique.

    Args:
        mice (list of str)
        pp_style (str):
            'zscore', 'z_dFF', or 'dFF'

    Returns:
        multi_mice (dict):
            {'bdf': trials data, 'analog': timeseries data}
    '''

    multi_mice = {key:pd.DataFrame() for key in ['bdf','analog']}

    for mouse in mice:

        multi_sessions = read_in_multi_sessions(mouse=mouse, **kwargs)

        if len(multi_sessions['bdf'])<1:
            continue # skip mouse if no sessions returned

        multi_mice = concat_sessions(subsessions=multi_sessions,
                                     full_sessions=multi_mice)

    print(f'{multi_mice["bdf"].Session.nunique()} total sessions loaded in')
    return multi_mice


def read_in_multi_sessions(mouse: str,
                           root: str,
                           *,
                           # pp_style: bool=False,
                           prob_set: int=9010,
                           fname_suffix: str='states_50Hz',
                           QC_pass: bool=True,
                           **dm_kwargs) -> dict:

    # define list of files to work through (by session and preprocessing date)
    dates_list = sessions_to_load(mouse, prob_set, QC_pass=QC_pass, root=root)

    multi_sessions = {key:pd.DataFrame() for key in ['bdf','analog']}

    # Loop through files to be processed
    for session_date in tqdm(dates_list, mouse, disable=False):

        file_path = os.path.join(root, mouse, session_date)
        try:
            filename = f'{mouse}_{session_date}_{fname_suffix}.parquet.gzip'
            df_single = pd.read_parquet(os.path.join(file_path, filename))

            bdf = pd.read_csv(os.path.join(file_path, f'{mouse}_behavior_df_full.csv' ))
            bdf = bdf.drop(columns=[col for col in bdf.columns if 'Unnamed' in col])

            fs = int(fname_suffix.split('_')[1][:-2]) # assumes structure of 'states_XXHz_...'
            bdf, df_single = add_behavior_cols(bdf, df_single, fs)

            qc_eval = qc.QC_session_performance(bdf.query('flagBlocks==False'),
                                                df_single,
                                                filename_suffix='photometry')
            if not qc_eval==QC_pass:
                continue

        except FileNotFoundError:
            print(f'skipped {filename}')
            continue

        if dm_kwargs:
            # df_single = make_design_mat(df_single, bdf, **dm_kwargs)
            raise NotImplementedError

        # If no photometry channels passed QC, move on to next session.
        df_single = qc.QC_photometry_signal(df_single, mouse, session_date)
        if df_single is None:
            continue

        trials_matched = qc.QC_included_trials(df_single,
                                            bdf,
                                            allow_discontinuity=False,
                                            drop_enlP=False)

        multi_sessions = concat_sessions(subsessions=trials_matched,
                                     full_sessions=multi_sessions,
                                     mouse=mouse,
                                     session_date=session_date)

    # QC all mice sessions by ENL penalty rate set per mouse

    return multi_sessions
