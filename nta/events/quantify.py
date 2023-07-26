"""
Created on Fri Aug 12 13:30:18 2022

@author: celiaberon
"""
import pandas as pd
from collections import defaultdict
import numpy as np
import itertools
from typing import Callable

def create_combo_col(data: pd.DataFrame, grouping_levels: list):

    '''
    Create column with unique identifier for every combination of conditions
    across multiple columns.

    Args:
        data:
            Data (trials or timeseries form) containing features.
        grouping_levels:
            List of features to be combined in single column representation.

    Returns:
        data_:
            Data with new column 'combo_col' jointly representing each
            condition across grouping_levels.
    '''

    data_ = data.copy()
    data_['combo_col'] = ''
    for i, col in enumerate(grouping_levels):
        prefix = '' if i==0 else '_'
        data_['combo_col'] += prefix + data_[col].astype(str)

    return data_


def set_peak_function(rew: (int | bool),
                      state: str,
                      sensor: str,
                      override: Callable=None,
                      **kwargs) -> Callable:
    '''
    Select function to use for peak calculation based on task state and
    fluorescent sensor or use provided function.

    Args:
        rew:
            Whether or not the state was rewarded.
        state:
            Task state peak is calculated within.
        sensor:
            Fluorescent sensor.
        override:
            Function to override rules with for peak calculation.

    Returns:
        peak_func:
            Function used for quantifying magnitude of event-based peaks.
    '''

    if override is not None:
        peak_func = override
        return peak_func

    # detecting peaks with opposite sign for reward/no reward
    peak_funcs = {1:np.argmax, 0:np.argmin, -1:np.argmin}
    if sensor != 'DA':
        peak_func = np.argmax
    else:
        peak_func = peak_funcs[rew] if state=='Consumption' else np.argmax

    return peak_func


def find_group_peak_time(ts: pd.DataFrame,
                         peak_func: Callable,
                         times_col: str,
                         channel_col: str,
                         max_peak_delay: float=0.5):
    
    '''
    Use the aggregate (mean) trace for a trial type to find the location (in 
    time) of the average "peak" fluorescence.

    Args:
        ts:
            Timeseries form of the data.
        peak_func:
            Function used to define the "peak".
        times_col:
            Name of the column containing time relative to an event.
        channel_col:
            Name of the column containing the neural data.
        max_peak_delay:
            Upper limit of time relative to event before which peak needs (is
            expected) to have occurred.

    Returns:
        peak_time:
            Time (in seconds) of the average peak for the given trial type.
    '''

    # Find group mean peak time to use for trial peak calculations.
    ts_peak_time_window = ts.loc[ts[times_col].between(0, max_peak_delay)]
    mean_trace = ts_peak_time_window.groupby(times_col)[channel_col].mean()
    center_idx = peak_func(mean_trace) # peak time index
    peak_time = mean_trace.index[center_idx] # peak time in seconds

    return peak_time


def group_peak_metrics(trials: pd.DataFrame,
                     grouping_levels: list[str],
                     channel: str,
                     sensor: str='DA',
                     states: list[str]=None,
                     **kwargs) -> pd.DataFrame:

    '''
    Find mean peak timing for group of trials and use this to calculate metric
    for individual trial peak magnitudes.

    Args:
        trials:
            Trial-based data containing pre-aligned neural traces.
        grouping_levels:
            Variables on which to group trials to determine average peak
            timing.
        channel:
            Label of channel containing neural data (e.g. 'grnL', 'redR').
        sensor:
            Label denoting fluorophore measured. 'DA' indicates dopamine
            sensor, anticipating oppositely-signed neural response to reward
            and necessitating appropriate min/max functions. Other sensors use
            max peak finding only.
        states:
            List of events neural data is pre-aligned to that should be used
            as basis for peak timing/magnitude.

    Returns:
        trials_:
            Trial data containing peak metrics and group peak timing.

    Notes:
        Future features:
            - Individual peak timing returned.
            - Offset from extrapolated consumption baseline.
    '''

    peak_times = defaultdict(lambda: defaultdict(list))
    trials_ = trials.copy()

    if states is None:
        states = ['Cue', 'Consumption']

    for state in states:

        channel_col = f'{state}_{channel}'
        times_col = f'{state}_times'

        # Convert to longform timeseries and drop NaNs from grouping columns.
        exp_trials = (trials_
                     .explode(column=[channel_col, times_col])
                     .dropna(subset=grouping_levels +['Reward']))
        exp_trials = create_combo_col(exp_trials, grouping_levels)
        
        # Iterate over each unique condition.
        for peak_group_id, peak_group in exp_trials.groupby('combo_col'): 

            # Iterate over each reward outcome in condition and use
            # appropriate function to detect peak for group, storing mean
            # value and peak time.
            for rew, peak_group_outcome in peak_group.groupby('Reward'):
                peak_func = set_peak_function(rew, state, sensor)

                # Find group mean peak time for trial peak calculations.
                peak_time = find_group_peak_time(peak_group_outcome, 
                                                    peak_func,
                                                    times_col,
                                                    channel_col)

                # Store group peak time by trial for single trial peak calc.
                trials_ids = peak_group_outcome.nTrial.unique()
                trial_times = np.repeat(peak_time, len(trials_ids))
                peak_times[peak_group_id][times_col].extend(trial_times)
                peak_times[peak_group_id][f'{state}_trials'].extend(trials_ids)
            
        # Peak times indexed by trial for mapping into original df. Flattened
        # across combination conditions.
        peak_times_state = (pd.DataFrame(peak_times)
                    .T
                    .explode(column=[times_col, f'{state}_trials'])
                    .set_index(f'{state}_trials'))
        trials_[f'{state}_peak_time'] = trials_['nTrial'].map(peak_times_state[times_col])

        # Calculate peak magnitude around mean peak time for individual trials.
        trials_ = group_peak_quantification(trials_,state, channel, **kwargs)

    return trials_


def group_peak_quantification(trials: pd.DataFrame,
                              state: str,
                              channel: str,
                              *,
                              offset: bool=True,
                              hw: int=2,
                              agg_funcs: list[str]=['mean']) -> pd.DataFrame:

    '''
    Calculate peak magnitude for each trial based on average timing of sensor
    peak (max or min) across trials in trial type. Individual peak magnitudes
    are calculated based on `agg_func` within a small window with provided
    halfwidth around group-averaged peak timing.

    Note: If offset from baseline, simply subtract initial value from peak and
    store offset value for future convenience.

    Args:
        trials:
            Trial-based data containing pre-aligned neural traces.
        state:
            Event neural data is aligned to.
        channel:
            Label of channel containing neural data (e.g. 'grnL', 'redR').
        offset:
            Whether or not to offset peak magnitude by baseline neural value
            immediately preceding aligned event.
        hw:
            Number of sample steps (bins @ 1/sampling freq) to form half-width
            of window around group-mean peak time. Peak magnitude calculated
            from data spanning (group peak time - hw, group peak time + hw).
        agg_func:
            List of functions used to reduce snippet of neural data around
            group mean peak time to single summary metric for each trial.

    Returns:
        trials_:
            Trial-based data containing columns for peak summary value aligned
            to state for every summary metric in `agg_func`.
    '''

    # Define timepoint (in seconds) preceding align event to offset baseline.
    T_BASELINE = -0.04 

    channel_col = f'{state}_{channel}'
    times_col = f'{state}_times'

    trials_ = trials.copy()
    exp_trials = (trials_
                  .explode(column=[channel_col, times_col])
                  .reset_index(drop=True)) 

    # Get index of rows at group peak time and expand to permissible range for
    # individual trial peak time. 
    group_peak_times = exp_trials[times_col]==exp_trials[f'{state}_peak_time']
    idcs = exp_trials.loc[group_peak_times].index.values
    snippet_idcs = list(itertools.chain(*[np.arange(x-hw, x+hw) for x in idcs]))

    for af in agg_funcs:
        peak_col = f'{state}_{af}'

        # If column previously calculated, make sure it's fully overwritten.
        if peak_col in trials_.columns:
            trials_ = trials_.drop(columns=[peak_col])
        
        # Grab subset of rows for peak averaging.
        agg_df = (exp_trials.loc[snippet_idcs]
              .groupby('nTrial', as_index=True)
              .agg({channel_col:af})
              .rename(columns={channel_col:peak_col}))

        # Add column to trial data mapping peak metric for each trial.
        trials_ = trials_.merge(agg_df, left_on='nTrial', 
                                right_index=True, how='left')
        trials_[peak_col] = trials_[peak_col].astype('float')

    if offset:
        # At the moment offset everything by timepoint preceding cue.
        if (state!='Cue') and ('Cue_offset' in exp_trials.columns):
            trials_[peak_col] = trials_[peak_col] - trials_[f'Cue_offset']
        else:
            offset_df = (exp_trials.loc[exp_trials[times_col]==T_BASELINE]
                         .set_index('nTrial')[[channel_col]]
                         .rename(columns={channel_col: f'{state}_offset'}))
            trials_ = trials_.merge(offset_df, left_on='nTrial', 
                                    right_index=True, how='left')
            trials_[peak_col] -= trials_[f'{state}_offset']

    return trials_
