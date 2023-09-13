"""
Created on Tue Apr 26 20:17:37 2022

@author: celiaberon
"""

import os

import numpy as np
import pandas as pd

from nta.preprocessing.signal_processing import snr_photo_signal


def QC_included_trials(analog_single: pd.DataFrame,
                       bdf_single: pd.DataFrame,
                       allow_discontinuity: bool = False,
                       drop_enlP: bool = False,
                       drop_timeouts: bool = False) -> dict:

    '''
    Quality control within sessions to match timeseries and trial dataframes
    -- remove high timeout blocks

    Args:
        analog (pandas DF):
            Timeseries containing states and photometry data.
        mouse (str):
            Mouse ID.
        session_date (str):
            Session ID.

    Returns:
        trials_matched (dict):
            {'bdf': trial data, 'analog': timeseries data}
            Contain only identical trials.
    '''

    flagged_trials = bdf_single.loc[bdf_single.flagBlocks].nTrial.values

    if allow_discontinuity:
        # flagged_trials stays as is
        if drop_enlP:  # take only trials without enl penalty
            ntrials = len(bdf_single)
            bdf_single = bdf_single.loc[bdf_single.n_ENL == 1]
            print(ntrials-len(bdf_single), 'penalty trials dropped')

        if drop_timeouts:  # take only trials with choice lick
            ntrials = len(bdf_single)
            bdf_single = bdf_single.loc[~bdf_single.timeout]
            print(ntrials-len(bdf_single), 'timeout trials dropped')

    else:
        # flag only blocks that don't disrupt photometry timeseries
        min_trial_good_block = bdf_single.query('~flagBlocks').nTrial.min()
        max_trial_good_block = bdf_single.query('~flagBlocks').nTrial.max()
        flagged_trials[(flagged_trials < min_trial_good_block)
                       | (flagged_trials > max_trial_good_block)]

    bdf_single = bdf_single.loc[~bdf_single.nTrial.isin(flagged_trials)]

    # these blocks occur too infrequently -- less than 10 sessions
    bdf_single = bdf_single.loc[bdf_single.iBlock <= 16]

    included_trials = (set(analog_single.nTrial.astype('int').unique())
                       .intersection(bdf_single.nTrial.unique())
                       )
    # take only specified trials to match both dfs
    analog_QC = analog_single.loc[analog_single.nTrial.isin(included_trials)]
    bdf_QC = bdf_single.loc[bdf_single.nTrial.isin(included_trials)]

    return {'bdf': bdf_QC, 'analog': analog_QC}


def QC_enl_penalty_rate(trials: pd.DataFrame) -> list:

    '''
    Generate list of sessions satisfying ENL penalty criteria, defined
    as no greater than 2 std above mean penalty rate in final sessions.

    Args:
        trials (pandas.DataFrame):
            Trial data

    Returns:
        qc_sessions (list):
            Sessions that pass ENLP quality control criteria.
    '''

    trials_ = trials.copy()
    trials_['penalty'] = trials_['n_ENL'] > 1
    trials_['Date'] = pd.to_datetime(trials_['Date'], format='%Y_%m_%d')
    penalties = (trials_
                 .sort_values(by='Date')
                 .groupby(['Mouse', 'Date', 'Session'], as_index=False)
                 .penalty.mean())

    for mouse, mouse_penalties in penalties.groupby('Mouse'):
        late_sessions_dates = np.sort(mouse_penalties.Date.unique())[-6:]
        late_sessions = mouse_penalties.query('Date.isin(@late_sessions_dates)')['penalty']
        late_sessions_mean = np.nanmean(late_sessions)
        late_sessions_std = np.nanstd(late_sessions)
        qc_criteria = late_sessions_mean + (2 * late_sessions_std)
        penalties.loc[penalties.Mouse == mouse, 'QC_criteria'] = qc_criteria

    penalties['Pass'] = penalties['penalty'] <= penalties['QC_criteria']
    qc_sessions = penalties.query('Pass == True').Session.values

    return qc_sessions


def get_sess_val(trials, trial_variable):

    val = (trials
           .groupby('Session')
           .apply(lambda x: x[trial_variable].unique())
           .squeeze().item())

    return val


def QC_session_performance(trials: pd.DataFrame,
                           analog: pd.DataFrame,
                           update_log: bool = False,
                           filename_suffix: str = 'photometry',
                           **kwargs) -> bool:

    '''
    Filter out sessions that don't meet certain criteria:
        target_avg_threshold:
            Proportion of trials to high value port, must exceed
        side_bias_threshold:
            Proportion of trials that can favor one side (above/below 0.5)
        spout_bias_threshold:
            Proportion of licks to one spout (slightly more inclusive than
            choice)

    Args:
        trials (pandas.DataFrame):
            Trial data.
        analog (pandas.DataFrame):
            Timeseries data.
        update_log (bool):
            TRUE if saving .csv overview of session qc stats.
        filename_suffix (str):
            Suffix for session log filename.

    Returns:
        criteria_met (bool):
            TRUE if passes quality control.
    '''

    criteria_met = True

    # Set thresholds for session-level behavioral performance.
    condition_perf_thresh = {9010: 0.7,
                             '9010': 0.7,
                             8020: 0.6,
                             '8020': 0.6}
    TARGET_AVG = condition_perf_thresh.get(trials.Condition.unique()[0])
    SIDE_BIAS = 0.1
    SPOUT_BIAS = 0.15
    MIN_TRIALS = 100

    # Evaluate spout bias on same trials as trial-level QC (i.e., not
    # including flagged blocks).
    trial_ids = trials.nTrial.unique()
    analog = analog.copy().query('nTrial.isin(@trial_ids)')

    n_valid_trials = (trials
                      .query('flag_block==False & timeout==False')['nTrial']
                      .nunique())

    target_avg = trials.selHigh.mean()
    if target_avg < TARGET_AVG:
        criteria_met = False

    right_avg = trials.direction.mean()
    if np.abs(right_avg-0.5) > SIDE_BIAS:
        criteria_met = False

    spout_avg = (analog.query('iSpout.ne(0)')
                 .iSpout.value_counts(normalize=True)[2.])
    if np.abs(spout_avg - 0.5) > SPOUT_BIAS:
        criteria_met = False
    if n_valid_trials < MIN_TRIALS:
        criteria_met = False

    if update_log:
        enlp_rate = np.mean(trials['n_ENL'] > 1)
        sess_summary = pd.DataFrame(
                                    {'Mouse': get_sess_val(trials, 'Mouse'),
                                     'Date': get_sess_val(trials, 'Date'),
                                     'Session': get_sess_val(trials, 'Session'),
                                    #  'Condition': trials.Condition.unique()[0],
                                     'P(right)': round(right_avg, 2),
                                     'P(high)': round(target_avg, 2),
                                     'P(spout)': round(spout_avg, 2),
                                     'N_valid_trials': n_valid_trials,
                                     'enl_penalty_rate': round(enlp_rate, 2),
                                     'Pass': criteria_met},
                                    index=[0])

        # save_session_log(sess_summary, filename_suffix)

    return criteria_met, sess_summary


def load_session_log(path_to_file: str):

    '''
    Load existing session log if it exists, otherwise initialize a new one.

    Args:
        path_to_file:
            Path including filename to try loading in file.

    Returns:
        session_log:
            DataFrame containing session summary quality control stats.
        previous sessions:
            List of sessions already included in session_log.
    '''

    try:
        session_log = pd.read_csv(path_to_file, index_col=0)
        previous_sessions = session_log.Session.values
    except FileNotFoundError:
        session_log = pd.DataFrame()
        previous_sessions = []

    return session_log, previous_sessions


def save_session_log(session_qc_summary: pd.DataFrame,
                     filename_suffix: str = 'photometry',
                     root: str = ''):

    '''
    Save summary of session quality control metrics.

    Args:
        session_qc_summary:
            DataFrame containing quality control metrics for single session.
        filename_suffix:
            Suffix for session_log filename.

    Returns:
        None
    '''

    if not root:
        root = input('Please provide a path for logging:')
    filename = f'session_log_qc_{filename_suffix}.csv'
    path_to_file = os.path.join(root, filename)

    session_log, previous_sessions = load_session_log(path_to_file)
    updated_log = pd.merge(session_log, session_qc_summary,
                           on=['Mouse', 'Date'], how='left')
    updated_log.to_csv(path_to_file)
    # if session_qc_summary.Session.squeeze() not in previous_sessions:
    #     session_log = pd.concat((session_log, session_qc_summary),
    #                             ignore_index=True)
    #     session_log.to_csv(path_to_file)
    # else:
    #     print('duplicate found, skipping logging')


def QC_photometry_signal(timeseries: pd.DataFrame,
                         mouse: str,
                         session_date: str,
                         ) -> pd.DataFrame:

    '''
    Quality control on signal strength in photometry channels using FFT
    method. If bilateral signals pass, take delta between right and left.

    Args:
        timeseries (pandas.DataFrame):
            Timeseries data containing photometry signal.
        mouse (str):
            Mouse ID.
        session_date (str):
            Session ID.
        pp_style (bool):
            Deprecated, if changing standardization method for photometry.

    Returns:
        timeseries (pandas.DataFrame):
            Copy of input but replace data with NaNs where QC fails.
            ALTERNATE: early return with FALSE if no channels pass QC.
    '''

    ts_ = timeseries.copy()

    # always use z-scored data just for QC consistency
    y_cols = ts_.columns.intersection(['z_grnR', 'z_grnL'])

    # need different thresholds for dLight vs GRAB-DA
    qc_thresh = 2 if mouse in ['C32', 'C33', 'C34', 'C35'] else 5
    y_cols = [col for col in y_cols if snr_photo_signal(ts_, col) < qc_thresh]

    if len(y_cols) == 2:
        print(f'insufficient photometry data...discarding {session_date}')
        return None

    for y_col in y_cols:
        ts_[y_col] = np.nan  # NaNs for cols that don't pass QC
    if len(y_cols) == 2:
        ts_['z_grnDelta'] = ts_['z_grnR'] - ts_['z_grnL']

    return ts_
