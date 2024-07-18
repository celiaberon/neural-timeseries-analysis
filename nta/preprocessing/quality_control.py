"""
Created on Tue Apr 26 20:17:37 2022

@author: celiaberon
"""

import os

import numpy as np
import pandas as pd
import scipy

from ..features.select_trials import match_trial_ids
from ..preprocessing.signal_processing import snr_photo_signal


def QC_included_trials(ts: pd.DataFrame,
                       trials: pd.DataFrame,
                       allow_discontinuity: bool = False,
                       **kwargs) -> dict:

    '''
    Quality control within sessions to match timeseries and trial dataframes
    starting and ending trial ID values. Also remove -- remove high timeout
    blocks.

    Args:
        ts:
            Timeseries containing states and neural data for a single
            session.
        trials:
            Trial data for single session.
        allow_discontinuity:
            Whether or not continuous trial structure can be broken. If False,
            only trials from beginning/end can be excluded. If False, also
            permits mismatch in trial ids between min and max trial.

    Returns:
        trials_matched (dict):
            {'trials': trial data, 'ts': timeseries data}
            Contain only identical trial sets.
    '''

    assert ts.Session.dropna().nunique() == 1, (
        'multi-session not implemented')
    assert trials.Session.dropna().nunique() == 1, (
        'multi-session not implemented')

    trials_ = trials.copy()
    ts_ = ts.copy()

    # flag only blocks that don't disrupt photometry timeseries
    min_trial = trials_.query('flag_block == 0').nTrial.min()
    max_trial = trials_.query('flag_block == 0').nTrial.max()
    flagged_blocks = (trials_
                      .query('~nTrial.between(@min_trial, @max_trial)')
                      .nTrial.dropna().values)

    trials_ = trials_.loc[~trials_.nTrial.isin(flagged_blocks)]
    # Match min and max trial IDs only (can differ internally but offset will)
    # be consistent.
    trials_, ts_ = match_trial_ids(trials_, ts_,
                                   allow_discontinuity=allow_discontinuity)

    return {'trials': trials_, 'ts': ts_}


def QC_enl_penalty_rate(trials: pd.DataFrame) -> list:

    '''
    Generate list of sessions satisfying ENL penalty criteria, defined
    as no greater than 2 std above mean penalty rate in final sessions.

    Args:
        trials (pandas.DataFrame):
            Trial data.

    Returns:
        qc_sessions (list):
            Sessions that pass ENLP quality control criteria.
    '''

    trials_ = trials.copy()
    trials_['penalty'] = trials_['n_ENL'] > 1
    trials_['Date'] = pd.to_datetime(trials_['Date'], format='%Y_%m_%d')
    penalties = (trials_
                 .sort_values(by='Date')
                 .groupby(['Mouse', 'Date', 'Session'], as_index=False, observed=True)
                 .penalty.mean())

    for mouse, mouse_penalties in penalties.groupby('Mouse', observed=True):
        late_dates = np.sort(mouse_penalties.Date.unique())[-6:]
        late_sessions = mouse_penalties.query('Date.isin(@late_dates)')
        late_sessions_mean = np.nanmean(late_sessions['penalty'])
        late_sessions_std = np.nanstd(late_sessions['penalty'])
        qc_criteria = late_sessions_mean + (2 * late_sessions_std)
        penalties.loc[penalties.Mouse == mouse, 'QC_criteria'] = qc_criteria

    penalties['Pass'] = penalties['penalty'] <= penalties['QC_criteria']
    qc_sessions = penalties.query('Pass == True').Session.values

    return qc_sessions


def get_sess_val(trials, trial_variable):

    val = (trials
           .groupby('Session', observed=True)
           .apply(lambda x: x[trial_variable].unique())
           .squeeze().item())

    return val


def QC_session_performance(trials: pd.DataFrame,
                           ts: pd.DataFrame,
                           update_log: bool = False,
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
        ts (pandas.DataFrame):
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
                             '8020': 0.6,
                             '100-1': 0.7}
    TARGET_AVG = condition_perf_thresh.get(trials.Condition.unique()[0])
    SIDE_BIAS = 0.1
    SPOUT_BIAS = 0.15
    MIN_TRIALS = 100

    # Evaluate spout bias on same trials as trial-level QC (i.e., not
    # including flagged blocks).
    trial_ids = trials.nTrial.unique()
    ts_ = ts.copy().query('nTrial.isin(@trial_ids)')

    n_valid_trials = (trials
                      .query('flag_block==False & timeout==False')['nTrial']
                      .nunique())

    target_avg = trials.selHigh.mean()
    if target_avg < TARGET_AVG:
        criteria_met = False

    right_avg = trials.direction.mean()
    if np.abs(right_avg - 0.5) > SIDE_BIAS:
        criteria_met = False

    spout_avg = (ts_.query('iSpout.ne(0)')
                 .iSpout.value_counts(normalize=True)[2.])
    if np.abs(spout_avg - 0.5) > SPOUT_BIAS:
        criteria_met = False
    if n_valid_trials < MIN_TRIALS:
        criteria_met = False

    if update_log:
        enlp_rate = np.mean(trials['n_ENL'] > 1)
        qc_summary = pd.DataFrame({'Mouse': get_sess_val(trials, 'Mouse'),
                                   'Date': get_sess_val(trials, 'Date'),
                                   'Session': get_sess_val(trials, 'Session'),
                                   'P(right)': round(right_avg, 2),
                                   'P(high)': round(target_avg, 2),
                                   'P(spout)': round(spout_avg, 2),
                                   'N_valid_trials': n_valid_trials,
                                   'enl_penalty_rate': round(enlp_rate, 2),
                                   'Pass': criteria_met},
                                  index=[0])
    if kwargs.get('fname_suffix') == 'Kevin':
        for col in ['Condition', 'age_at_session', 'Age_Group', 'retrained']:
            qc_summary[col] = get_sess_val(trials, col)

        save_session_log(qc_summary, **kwargs)

    return criteria_met


def load_session_log(path_to_file: str):

    '''
    Load existing session log if it exists, otherwise initialize a new one.

    Args:
        path_to_file:
            Path including filename to try loading in file.

    Returns:
        session_log:
            DataFrame containing session summary quality control stats.
        logged_sessions:
            List of sessions already included in session_log.
    '''

    try:
        session_log = pd.read_csv(path_to_file, index_col=None)
        logged_sessions = ['_'.join(sess)
                           for sess in session_log[['Mouse', 'Date']].values]
        return session_log, logged_sessions
    except FileNotFoundError:
        return pd.DataFrame(), []


def save_session_log(sess_qc: pd.DataFrame,
                     fname_suffix: str = 'photometry',
                     root: str = '',
                     **kwargs):

    '''
    Save summary of session quality control metrics.

    Args:
        sess_qc:
            DataFrame containing quality control metrics for single session.
        filename_suffix:
            Suffix for session_log filename.

    Returns:
        None
    '''

    if not root:
        root = input('Please provide a path for logging:')
    filename = f'session_log_{fname_suffix}.csv'
    path_to_file = os.path.join(root, filename)
    session_log, logged_sessions = load_session_log(path_to_file)

    curr_session = f'{sess_qc.Mouse.item()}_{sess_qc.Date.item()}'
    if curr_session not in logged_sessions:
        tmp_log = pd.DataFrame({'Mouse': sess_qc['Mouse'].item(),
                               'Date': sess_qc['Date'].item()},
                               index=[0])
        session_log = pd.concat((session_log, tmp_log)).reset_index(drop=True)

    if 'N_valid_trials' not in session_log.columns:
        updated_log = pd.merge(session_log, sess_qc,
                               on=['Mouse', 'Date'], how='left')
    else:
        updated_log = session_log.copy()
        for col in sess_qc.columns.drop(['Mouse', 'Date']):
            mouse, date, val = sess_qc[['Mouse', 'Date', col]].iloc[0].values
            idx = updated_log.query('Mouse == @mouse & Date == @date').index
            updated_log.loc[idx, col] = val

    updated_log.to_csv(path_to_file, index=False)


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

    y_cols_pass = {'z_grnR', 'z_grnL'} - set(y_cols)
    return ts_, y_cols_pass


def is_normal(ts, include_score=False, verbose=False, thresh_score=0,
              sensor='grabda_vls'):

    '''
    Test for normality as a measure of signal to noise. Result of normally
    distributed data fails to pass QC protocol. Normality is determined via
    collective evaluation of skew, kurtosis, and K-S test p-value.

    Args:
        ts:
            Timeseries to be evaluated.
        include_score:
            Whether or not to include the number of metrics that passed as
            normally distributed.
        thresh_score:
            Threshold for permissable tests that return normality=True.
            Default of 1 means any test returning normal dist is sufficient to
            accept dist as normal (most conservative inclusion level).

    Returns:
        result:
            True if any metric is consistent with normal distribution. False if
            all metrics deviate from normal distribution.
        score:
            Number of metrics that are consistent with normal distribution (0
            to 3).
    '''

    if ts is None:  # needs to be a distribution to have signal
        return True

    thresholds = {'grabda_vls': (0.5, 0.8),
                  'grabda_dms': (0.1, 0.2),
                  'grabda_kev': (0.1, 0.2),
                  'rDA': (0.5, 0.8)}

    skew_thresh, kurt_thresh = thresholds.get(sensor)

    # Should be a bit skewed (although for DA not necessarily a lot).
    skew = np.abs(ts.skew()) < skew_thresh  # True when fails
    kurtosis = np.abs(ts.kurtosis()) < kurt_thresh  # True when fails

    rand_normal = np.random.normal(0, np.nanstd(ts), len(ts))
    _, p_value = scipy.stats.ks_2samp(ts, rand_normal, alternative="two-sided")

    ks_test = p_value > 0.05  # True when fails

    # Check for any odd peak in counts (usually at 0, corresponding to noise).
    counts, _ = np.histogram(ts.dropna(), bins=500)
    smooth_counts = any(np.diff(counts)[5:] > 1000)

    score = sum((skew, kurtosis, ks_test, smooth_counts))
    # print(skew, kurtosis, ks_test)

    result = score > thresh_score

    if include_score:
        if verbose:
            print(f'skew = {np.abs(ts.skew())}\n',
                  f'kurtosis = {np.abs(ts.kurtosis())}\n',
                  f'p_value = {p_value}')
        return result, score
    else:
        if verbose:
            print(f'skew = {np.abs(ts.skew())}\n',
                  f'kurtosis = {np.abs(ts.kurtosis())}\n',
                  f'p_value = {p_value}')
        return result
