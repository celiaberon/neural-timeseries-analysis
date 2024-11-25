#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 20:53:07 2020

@author: celiaberon
"""

from functools import partial

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy import stats as st

from ..utils import single_session


def set_analog_headers(beh_timeseries: pd.DataFrame) -> pd.DataFrame:

    '''
    Handle differing orders that behavior timeseries columns can take.

    Args:
        beh_timeseries:
            Timeseries of behavior data at 200 Hz.

    Returns:
        ts_:
            Timeseries with column headers filled in correctly.
    '''

    ts_ = beh_timeseries.copy()
    # State columns will have mode 53 (consumption period) or 30 (ENL).

    mode_values, mode_counts = st.mode(ts_, keepdims=False)
    cols_in_play = [(i, val, count) for i, (val, count)
                    in enumerate(zip(mode_values, mode_counts))
                    if val in [30, 48, 53]]
    cols_in_play.sort(key=lambda x: x[2], reverse=True)
    state_columns = [i for i, _, _ in cols_in_play[:2]]
    state_columns.sort()
    # Match state columns position to the appropriate column header ordering.
    match state_columns:
        case [4, 5]:
            ts_.columns = ['nTrial', 'iBlock', 'iTrial', 'iOccurrence',
                           'iState_start', 'iState_end', 'analog1', 'analog2']
        case [3, 4]:
            ts_.columns = ['iBlock', 'iTrial', 'iOccurrence', 'iState_start',
                           'iState_end', 'analog1', 'analog2', 'nTrial']
        case _:
            raise ValueError

    return ts_


def trim_to_session_start(beh_timeseries: pd.DataFrame) -> pd.DataFrame:

    '''
    Trim timeseries to where the session really starts (first trial onset).

    Args:
        beh_timeseries:
            Timeseries of behavior data collected at 200 Hz.

    Returns:
        trimmed_data:
            Behavior timeseries trimmed at beginning to first trial onset.
    '''

    if beh_timeseries['iOccurrence'][0] > 1:  # in case doesn't start at 0
        first_trial_idx = beh_timeseries.query('iOccurrence == 0').index[0]
        trimmed_data = beh_timeseries.loc[first_trial_idx:].copy()

        return trimmed_data.reset_index(drop=True)

    else:
        return beh_timeseries.reset_index(drop=True)


def handshake_sync_pulses(photo_timeseries: pd.DataFrame) -> pd.DataFrame:

    '''
    Find first incoming and outgoing pulses to and from photometry system and
    trim timeseries data to onset of each.

    Args:
        photo_timeseries:
            Timeseries of photometry data.

    Returns:
        ts_:
            Copy of photo_timeseries trimmed to first detected pulse from each
            (photometry and behavior) system.
    '''

    ts_ = photo_timeseries.copy()

    # Confirm recording system is sending signal to behavior.
    first_pulse_to_behavior = ts_.query('toBehSys == 1').index[0]
    ts_ = ts_.loc[first_pulse_to_behavior:]  # trim to pulse onset

    # Confirm behavior system sends signal back at trial starts (ENL onset)
    first_pulse_to_photometry = ts_.loc[ts_['fromBehSys'] == 0].index[0]
    ts_ = ts_.loc[first_pulse_to_photometry:].reset_index(drop=True)

    return ts_


def trim_at_sync_pulses(timeseries: pd.DataFrame,
                        nth_pulse: int = 1,
                        sync_col: str = 'sync_pulse') -> pd.DataFrame:

    '''
    Crop timeseries between first (or nth) and last (or -nth) sync pulses.

    Args:
        timeseries:
            Timeseries containing sync pulses.
        nth_pulses:
            Nth sync pulse at which to crop dataframe (beginning and end).
        sync_col:
            Column name containing binary sync pulses.

    Returns:
        ts_:
            Cropped timeseries dataframe.
    '''

    print('reminder, trimming to first 0->1, new from 1->0  with fromBehSys')
    ts_ = timeseries.copy()

    pulse_onsets = ts_.query(f'{sync_col} == 1').index.values
    ts_ = ts_.loc[pulse_onsets[nth_pulse - 1]: pulse_onsets[-nth_pulse]]
    ts_ = ts_.reset_index(drop=True)

    return ts_


def calculate_trial_length(ts: pd.DataFrame, trial_starts: list) -> list:

    '''
    Calculate length of trials in number of timesteps from list of indices
    corresponding to trial start times.

    Args:
        ts:
            Timeseries data used for last trial length.
        trial_starts:
            List of trial start indices.

    Returns:
        trial_lengths:
            List of trial lengths as number of samples between trial starts.
    '''

    trial_lengths = np.diff(trial_starts, append=len(ts)).tolist()

    return trial_lengths


def bins_per_trial_behavior(beh_timeseries: pd.DataFrame) -> tuple:

    '''
    Count number of timesteps between ENL onsets in behavior timeseries as
    measure of trial length.

    Args:
        beh_timeseries:
            Timeseries of behavior data collected at 200 Hz.

    Returns:
        trial_lengths:
            List of trial lengths as the number of samples between ENL state
            onsets. Note: For trials with ENL penalties, use first ENL.
        trial_starts:
            Indices of first ENL onset for each trial in behavior timeseries.
    '''

    ts_ = beh_timeseries.copy()

    try:
        trial_starts = ts_.query('sync_pulse==1').index.tolist()

    except Exception:
        print('no sync pulse for 23-29, using old ENL method')

        # Get list of indices for ENL onsets defined by state transition.
        trial_starts = (ts_
                        .query('ENL == 1')
                        .groupby('nTrial')
                        .nth(0)
                        .index
                        .tolist())

    # Calculate length of trials (in 5ms samples) from intervals between ENLs.
    trial_lengths = calculate_trial_length(ts_, trial_starts)

    return trial_lengths, trial_starts


def bins_per_trial_photo(photo_timeseries) -> tuple:

    '''
    Count number of timesteps between ENL onsets pulses in photometry
    timeseries as measure of trial length.

    Args:
        photo_timeseries:
            Timeseries of photometry data (likely downsampled from original
            sampling freq during demodulation).

    Returns:
        trial_lengths:
            List of trial lengths as the number of samples between ENL state
            onsets. Note: For trials with ENL penalties, use first ENL.
        trial_starts:
            Indices of first ENL onset for each trial in behavior timeseries.
    '''

    ts_ = photo_timeseries.copy()

    # Trial starts defined as drop from 1->0 in pulse from behavior system.
    trial_starts = ts_.loc[ts_.fromBehSys.diff() == -1].index.tolist()
    # Insert pulses at beginning and end of df because trimmed to pulses.
    trial_starts = np.insert(trial_starts, 0, 0)
    trial_starts = np.insert(trial_starts, -1, ts_.index[-1])

    # Calculate length of trials from intervals between pulses.
    trial_lengths = calculate_trial_length(ts_, trial_starts)

    return trial_lengths, trial_starts


def sliding_corr(list1: list, list2: list, lag_range: int = 30,
                 max_corr_thresh: float = 0.9999) -> tuple:

    '''
    Slide lists across each other over offset range and get correlation for
    each offset position and select offset that maximizes correlation.

    Args:
        list1, list2:
            Lists to maximize correlation between.
        lag_range:
            Halfwidth of offset positions over which to calculate correlation
            between lists.

    Returns:
        corr_lst:
            Full list of correlation coefficients between offsets within
            range(-offset_range, offset_range).
        lag:
            Offset position that maximizes correlation between list1, list2.

    Note:
        Max correlation must be very close to 1. If not, return AssertionError
        because no offset provides satisfactory match between lists.
    '''

    shorter_list = min(len(list1), len(list2)) - 1

    # To avoid indexing errors if lag range would shift past list boundary.
    if shorter_list <= lag_range:
        lag_range = shorter_list - 5

    corr_lst = []
    for i in range(-lag_range, lag_range):
        if i < 0:
            corr = np.corrcoef(list1[np.abs(i):shorter_list],
                               list2[:shorter_list - np.abs(i)])
        if i >= 0:
            corr = np.corrcoef(list1[:shorter_list - i],
                               list2[i:shorter_list])
        corr_lst.append(corr[1, 0])
    # Lag that maximizes correlation within lag range.
    lag = range(-lag_range, lag_range)[np.argmax(corr_lst)]

    # Trim each list based on max corr position and recalculate max corr.
    trimmed_list1 = list1[np.max((0, -lag)): shorter_list - np.max((0, lag))]
    trimmed_list2 = list2[np.max((0, lag)): shorter_list - np.max((0, -lag))]
    max_corr_trimmed_lists = np.corrcoef(trimmed_list1, trimmed_list2)[1, 0]

    print(max_corr_trimmed_lists)
    # Fails if maximum correlation isn't essentially perfect.
    assert max_corr_trimmed_lists > max_corr_thresh

    return corr_lst, lag


def resample_and_align(beh_timeseries: pd.DataFrame,
                       photo_timeseries: pd.DataFrame,
                       channels: list[str] = ['grnR', 'redR', 'grnL', 'redL'],
                       by_trial: bool = False,
                       **kwargs) -> pd.DataFrame:

    '''
    Downsamples photometry data to match behavior data sampling rate and
    aligns photometry channel timeseries with behavior data timeseries.

    Args:
        beh_timeseries:
            Timeseries of behavior data collected at 200 Hz.
        photo_timeseries:
            Timeseries of photometry data (likely downsampled from original
            sampling freq during demodulation).
        channels:
            List of photometry channels to align with behavior data.
        by_trial:
            Whether or not to perform downsampling on photometry by trial or
            across the full session (default).

    Returns:
        aligned_df:
            Timeseries of behavior and photometry data aligned at 200Hz.
    '''

    beh_ts_ = beh_timeseries.copy()
    photo_ts_ = photo_timeseries.copy()

    # Get trial start indices and trial lengths for each timeseries.
    behavior_trial_lengths, beh_trial_idx = bins_per_trial_behavior(beh_ts_)
    photo_trial_lengths, photo_trial_idx = bins_per_trial_photo(photo_ts_)

    # Find offset between photometry and behavior (number of trials between
    # data collection onsets). NOTE: order matters for later trimming.
    _, offset = sliding_corr(list1=photo_trial_lengths,
                             list2=behavior_trial_lengths,
                             **kwargs)

    # Maximum number of trials that can be aligned.
    shorter_list = min(len(photo_trial_idx), len(beh_trial_idx)) - 1

    if by_trial:
        raise NotImplementedError

    else:
        # Trim each dataframe to align trials based on offset to perfectly
        # correlate trial lengths and list order for corr func.
        first_photo_idx = photo_trial_idx[np.max((0, -offset))]
        last_photo_idx = photo_trial_idx[shorter_list - np.max((0, offset))]
        photo_ts_trimmed = photo_ts_.loc[first_photo_idx:last_photo_idx]

        first_beh_idx = beh_trial_idx[np.max((0, offset))]
        last_beh_idx = beh_trial_idx[shorter_list - np.max((0, -offset))]
        beh_ts_trimmed = (beh_ts_
                          .loc[first_beh_idx:last_beh_idx]
                          .reset_index(drop=True))

        # Ensure that length of photometry timeseries is essentially integer
        # multiple of behavior timeseries.
        print('downsampling by ', len(photo_ts_trimmed) / len(beh_ts_trimmed))
        assert np.abs(3 - len(photo_ts_trimmed) / len(beh_ts_trimmed)) < 1e-5

        # Downsample photometry data to match sampling rate of behavior data.
        ds_photometry = signal.resample(photo_ts_trimmed[channels],
                                        len(beh_ts_trimmed))
        ds_photometry_df = pd.DataFrame(ds_photometry, columns=channels)

        # Can now easily join aligned and sampling rate-matched dataframes.
        aligned_df = pd.concat([beh_ts_trimmed, ds_photometry_df], axis=1)
        aligned_df = aligned_df.reset_index(drop=True)

        # Time in seconds needed to shift for alignment (sanity check).
        aligned_start_time = beh_ts_trimmed.session_clock.iloc[0]
        true_start_time = beh_ts_.session_clock.iloc[0]
        offset_time = aligned_start_time - true_start_time
        print(f'shift into behavior by {offset_time} seconds')

    return aligned_df


def set_to_same_clock(ts1: pd.DataFrame, ts2: pd.DataFrame) -> pd.DataFrame:

    '''
    Reset session clock times after alignment of dataframes to run on
    synchronized clocks. Necessary for mapping between dataframes by time.

    Args:
        ts1:
            Timeseries of data collection started first.
        ts2:
            Timeseries of data collection started second (used as official
            sync clock).

    Returns:
        ts1_:
            Copy of ts1 with `session_clock` column synchronized to that of
            ts2.
    '''

    ts1_, ts2_ = ts1.copy(), ts2.copy()
    aligned_start_time = ts1_.session_clock.iloc[0]
    real_clock_start = ts2_.session_clock.iloc[0]
    adjustment_time = aligned_start_time - real_clock_start
    ts1_['session_clock'] -= round(adjustment_time, 3)

    return ts1_


def align_behav_photo(beh_timeseries: pd.DataFrame,
                      photo_timeseries: pd.DataFrame,
                      **kwargs) -> pd.DataFrame:

    '''
    Crops photometry and/or behvior dataframes to start and end at the same
    instantaneous events. Uses correlation in sync pulses between dataframes
    to match alignments. Dataframes can be provided with different sampling
    rates.

    Args:
        beh_timeseries:
            Timeseries of behavior data collected at 200 Hz.
        photo_timeseries:
            Timeseries of photometry data (likely downsampled from original
            sampling freq during demodulation).

    Returns:
        beh_ts_trimmed, photo_ts_trimmed:
            Timeseries of behavior and photometry data aligned to single clock
            and starting timepoint.
    '''

    beh_ts_ = beh_timeseries.copy()
    photo_ts_ = photo_timeseries.copy()

    # Get trial start indices and trial lengths for each timeseries.
    behavior_trial_lengths, beh_trial_idx = bins_per_trial_behavior(beh_ts_)
    photo_trial_lengths, photo_trial_idx = bins_per_trial_photo(photo_ts_)

    # Find offset between photometry and behavior (number of trials between
    # data collection onsets). NOTE: order matters for later trimming.
    # try:
    _, offset = sliding_corr(list1=photo_trial_lengths,
                             list2=behavior_trial_lengths,
                             **kwargs)

    # Maximum number of trials that can be aligned.
    shorter_list = min(len(photo_trial_idx), len(beh_trial_idx)) - 1

    # Trim each dataframe to align trials based on offset to perfectly
    # correlate trial lengths and list order for corr func.
    first_photo_idx = photo_trial_idx[np.max((0, -offset))]
    last_photo_idx = photo_trial_idx[shorter_list - np.max((0, offset))]
    photo_ts_trimmed = photo_ts_.loc[first_photo_idx:last_photo_idx]

    first_beh_idx = beh_trial_idx[np.max((0, offset))]
    last_beh_idx = beh_trial_idx[shorter_list - np.max((0, -offset))]
    beh_ts_trimmed = (beh_ts_
                      .loc[first_beh_idx:last_beh_idx]
                      .reset_index(drop=True))

    # Time in seconds needed to shift for alignment (sanity check).
    aligned_start_time = beh_ts_trimmed.session_clock.iloc[0]
    true_start_time = beh_ts_.session_clock.iloc[0]
    offset_time = aligned_start_time - true_start_time
    print(f'shift into behavior by {offset_time} seconds')

    # Match clocks start time once dataframes are aligned.
    beh_ts_trimmed = set_to_same_clock(beh_ts_trimmed, photo_ts_trimmed)

    return beh_ts_trimmed, photo_ts_trimmed


def find_nearest_time(all_times: np.array, x: float, err=0.1) -> int:

    '''
    Returns time that minimizes differences between instantaneous timepoint
    and a series of timepoints.

    Args:
        all_times:
            Array of timestamps (for mapping into).
        x:
            Instantaneous timepoint (to be mapped):
    Returns:
            Index of nearest timepoint in `all_times` to `x`.
    '''

    nearest_timepoint = np.argmin(np.abs(all_times - x))
    acceptable_err = np.min(np.abs(all_times - x)) < err
    if not acceptable_err:
        if nearest_timepoint == (len(all_times)-1):
            return -1  #  -1 for last timepoint not matching
        else:
            raise AssertionError('no matching timepoint found within error limit')

    return nearest_timepoint


@single_session
def map_events_by_time(target_ts: pd.DataFrame,
                       orig_ts: pd.DataFrame,
                       event_col: str) -> pd.DataFrame:

    '''
    Map events by timestamp into index of new dataframe with nearest
    timestamp.

    Args:
        target_ts:
            Timeseries events are being mapped into.
        orig_ts:
            Timeseries containing events to be mapped.
        event_col:
            Name of column containing events to be mapped.

    Returns:
        target_ts_:
            Copy of target_ts with new column for mapped events.
    '''

    # Define set of instantaneous events.
    onset_only_LUT = {'iSpout': True,
                      'Select': True,
                      'ENLP': True,
                      'CueP': True,
                      'outcome_licks': True,
                      'fromBehSys': True,
                      'sync_pulse': True}
    onset_only = onset_only_LUT.get(event_col, False)

    target_ts_ = target_ts.copy()

    if onset_only:
        events = orig_ts.query(f'{event_col}.diff() > 0')
    else:
        events = orig_ts.query(f'{event_col}.diff().ne(0)')

    # Get arrays indices at nearest timepoint to event time in new array.
    event_times = events.session_clock.values
    event_ids = events[event_col].values

    find_nearest_time_ = partial(find_nearest_time, target_ts_.session_clock)
    event_idcs = np.array(list(map(find_nearest_time_, event_times)))
    # Filter out bad match on last trial if necessary.
    event_ids = event_ids[event_idcs >= 0]
    event_idcs = event_idcs[event_idcs >= 0]
    event_idcs = target_ts_.iloc[event_idcs].index.values

    # For impulse events where on/off indices are the same.
    if onset_only:
        target_ts_[event_col] = 0
        target_ts_.loc[event_idcs, event_col] = event_ids
        assert np.allclose(sum(event_ids), target_ts_[event_col].sum(), atol=1), (
            'failed to map accurately')

    # For step functions need to fill between onset and offset.
    else:
        target_ts_[event_col] = np.nan
        target_ts_.loc[event_idcs, event_col] = events[event_col].values
        target_ts_[event_col] = target_ts_[event_col].ffill()

        assert np.allclose(sum(event_ids), target_ts_.loc[event_idcs, event_col].sum(), atol=1), (
            'failed to map accurately')

    return target_ts_
