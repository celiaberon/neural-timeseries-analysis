'''
@author: celiaberon
'''

from functools import partial

import numpy as np
import pandas as pd

from nta.events.quantify import group_peak_metrics


def get_event_indices(timeseries: pd.DataFrame,
                      *,
                      aligned_event: str = None,
                      align_to: str = 'onset',
                      ) -> dict[int | float, np.array]:

    '''
    Returns indices of specified events for each trial in a lookup table.

    Args:
        timeseries (pandas.DataFrame):
            Timeseries data.
        aligned_event (str):
            Task event to align to (e.g. Select, Cue,Consumption, ENL...).
        align_to (str):
            Align to either 'onset' or 'offset' of event.

    Returns:
        event_idcs_by_trial (list):
            Dict of {trial:indices} for all event (onset/offset) instances in
            timeseries.
    '''

    # Use first or last instance of each event for alignment.
    nth_val = 0 if align_to == 'onset' else -1

    # event alignment position per trial as index
    event_idcs = (timeseries
                  .loc[timeseries[aligned_event] == 1]
                  .groupby('nTrial', as_index=False)
                  .nth(nth_val)
                  .index
                  .values
                  )

    # get trial ids to validate mapping into trial dataframe later
    trial_ids = timeseries.loc[event_idcs].nTrial.values

    # lookup table for event index by trial ID
    event_idcs_by_trial = dict(zip(trial_ids, event_idcs))

    return event_idcs_by_trial


def align_single_trace(idx: int,
                       *,
                       ts: pd.DataFrame = None,
                       window_interp: list[int | float] = None,
                       y_col: str = None,
                       ):

    '''
    Pull snippet of photometry timeseries with specified window around
    event-based index. Exclude any windows missing data.

    Args:
        ts:
            Timeseries of neural trace.
        idx:
            Alignment index in timeseries.
        window_interp:
            From align_photomery_epoch(), idx range to pull for plotting
            aligned window.
        y_col:
            Column with neural data in ts.

    Returns:
        (np.array):
            Timeseries of signal aligned by index
            OR NaN if ANY missing values in aligned signal snippet.
    '''

    if idx is None:
        return np.nan
    try:
        # Get list of indices from timeseries to pull aligned photometry trace
        idcs = window_interp+idx
        if ts.loc[idcs, 'session'].nunique() != 1:
            return np.nan  # don't allow crossing session boundaries
        trace = ts.loc[idcs, y_col].values

        return np.nan if np.any(np.isnan(trace)) else trace

    except KeyError:
        return np.nan  # if missing any values full trace fails


def align_photometry_to_event(trials: pd.DataFrame,
                              ts_full: pd.DataFrame,
                              *,
                              channel: str | list[str] = None,
                              aligned_event: str = None,
                              window: tuple[int | float, int | float] = (1, 3),
                              sampling_freq: int = 50,
                              quantify_peaks: bool = True,
                              **kwargs
                              ):

    '''
    Align photometry timeseries to behavior/task event and store snippets by
    trial.

    Args:
        trials:
            Trial dataframe with behavior/task variables.
        ts_full:
            Timeseries data with unbroken photometry stream used for
            alignment.
        channel:
            Specifying color and hemisphere of brain e.g. ('grnR', 'grnL',
            'redL', 'redR').
        aligned_event:
            Task event neural data will be aligned to.
        window:
            Photometry window as (seconds before, seconds after)
            ALIGNED_EVENT.
        sampling_freq:
            Photometry sampling frequency in Hz.
        quantify_peaks:
            Option to quantify peaks with some default arguments for grouping.

    Returns:
        trials_:
            Copy of trials with columns containing photometry snippet and
            corresponding timestamps relative to aligned_event.
    '''

    # Ensure that our photometry timeseries has unbroken trial continuity.
    if ts_full.session.nunique() > 1:
        assert ts_full.groupby('session').nTrial.diff().max() == 1
    else:
        assert ts_full.nTrial.diff().max() == 1

    if isinstance(channel, str):
        channel = [channel]

    trials_ = trials.copy()

    # Get list of indices for timeseries by trial for each instance of event.
    idx = get_event_indices(ts_full, aligned_event=aligned_event, **kwargs)
    window_interp, timesteps = interpolate_window(window, sampling_freq)

    for ch in channel:
        # Need full unbroken timeseries of photometry data to properly address
        # trial continuity.
        align_trace_partial = partial(align_single_trace,
                                      ts=ts_full,
                                      window_interp=window_interp,
                                      y_col=ch)

        photo_column = f'{aligned_event}_{ch}'
        times_column = f'{aligned_event}_{ch}_times'

        # Initialize as NaNs to handle trials without epoch.
        trials_[photo_column] = np.nan
        trials_[times_column] = np.nan

        # Store snippet of photometry data alongside trial data.
        trials_[photo_column] = (trials_['nTrial']
                            .apply(lambda trial: align_trace_partial(idx=idx.get(trial, None))))

        # Map times into full trial table only for trials with complete photometry snippets.
        trials_with_data = trials_.dropna(subset=[photo_column]).copy()
        trials_with_data[times_column] = [timesteps]*len(trials_with_data)
        trials_[times_column] = (trials_['nTrial']
                                 .map(trials_with_data.set_index('nTrial')[times_column]))

        if quantify_peaks:
            trials_ = group_peak_metrics(trials_,
                                         grouping_levels=['Session'],
                                         channel=ch,
                                         states=[aligned_event],
                                         agg_funcs=['mean', 'min', 'max'],
                                         offset=False)

    return trials_


def interpolate_window(window: tuple[int | float, int | float] = (1, 3),
                       sampling_freq: int = 50):

    '''
    Generates index interpolation for pre- and post-event window boundaries
    (in seconds) based on sampling frequency

    Args:
        window (tuple):
            Time in seconds before and after alignment event.
        sampling_freq (int):
            Sampling rate in Hz.

    Returns:
        window_interp (list):
            List of sample number relative to alignment event in a single
            window.
        timesteps (list):
            Timesteps in seconds corresponding to sample number.
    '''

    pre_window, post_window = window
    pre_window *= sampling_freq  # convert to number of sampling steps
    pre_window = int(pre_window)
    post_window *= sampling_freq  # convert to number of sampling steps
    post_window = int(post_window)

    window_interp = np.arange(-pre_window, post_window)  # num steps across window
    timesteps = window_interp / sampling_freq  # convert to units of seconds

    return window_interp, timesteps


def trials_licks_in_epoch(ts: pd.DataFrame, epoch: str):

    '''
    Get list of trials containing any licks within task epoch.

    Args:
        ts:
            Timeseries data containing spout contacts and task epochs.
        epoch:
            Task state of interest (e.g. Cue, Consumption, Select).

    Returns:
        trials:
            Array of trial IDs with licks in designated state.
    '''

    trials = ts.loc[(ts[epoch] == 1) & (~np.isnan(ts.iSpout))].nTrial.unique()

    return trials


def trim_trials_without_licks(timeseries: pd.DataFrame, trials: pd.DataFrame):

    '''
    Filter data to include only trials that contain cue, selection lick, and
    at least one lick in the consumption period.

    Args:
        timeseries:
            Timeseries data containing spout contacts and task epochs.
        trials:
            Trial data to match with timeseries data.

    Returns:
        timeseries_trimmed:
            Trimmed to include completed trials only.
        trials_trimmed:
            Trimmed to include completed trials only.
    '''

    trials_trimmed = trials.copy()
    timeseries_trimmed = timeseries.copy()

    # Get list of trials with completed task state for each epoch.
    select_trials = trials_licks_in_epoch(timeseries_trimmed, 'Select')
    consumption_trials = trials_licks_in_epoch(timeseries_trimmed, 'Consumption')
    cue_trials = timeseries_trimmed.loc[timeseries_trimmed.Cue == 1].nTrial.unique()

    # Take the set of trials meeting criteria in all epochs.
    included_trials = list(set(select_trials)
                           .intersection(set((consumption_trials)))
                           .intersection(set(cue_trials)))

    # Subset each dataframe by set of trials.
    timeseries_trimmed = timeseries_trimmed.query('nTrial.isin(@included_trials)').copy()
    trials_trimmed = trials_trimmed.query('nTrial.isin(@included_trials)')

    return timeseries_trimmed, trials_trimmed


def get_lick_times(timeseries: pd.DataFrame,
                   align_to: str = 'onset',
                   **kwargs
                   ) -> pd.DataFrame:

    '''
    Create dataframe containing event/lick times on trial based clock (in
    seconds) with row for each trial. Includes Cue, penalties, Selection lick,
    ENL state, first Consumption lick.

    Args:
        timeseries:
            Timeseries version of data containing analog states and spout
            contacts.
        align_to:
            'onset' or 'offset' to designate storing time of first or last
            instance of event per trial.

    Returns:
        lick_times:
            Dataframe containing event/lick times within each trial relative
            to trial start (beginning of first ENL period).
    '''

    # Create a dataframe containing index as nTrial for mapping into.
    lick_times = pd.DataFrame(index=timeseries.nTrial.unique())
    lick_times.index.name = 'nTrial'

    nth_val = 0 if align_to == 'onset' else -1

    # Get trial times for nth occurrence of event in each trial
    for event in ['Cue', 'ENLP', 'Select', 'ENL']:
        if event not in timeseries.columns:
            continue
        data_during_event = timeseries.loc[timeseries[event] == 1].copy()
        trials_with_event = data_during_event.nTrial.unique()
        lick_times.loc[trials_with_event, event] = (data_during_event
                                                    .groupby('nTrial')
                                                    .nth(nth_val)['trial_clock'].values)

    # As above for Consumption, for first lick only
    # (requires extra conditioning on spout contact).
    first_outcome_lick_idx = (timeseries
                              .query('Consumption==1 & ~iSpout.isna()')
                              .groupby('nTrial', as_index=False)
                              .nth(0).index)
    first_outcome_lick_trials = timeseries.loc[first_outcome_lick_idx].nTrial.unique()
    first_outcome_lick_times = timeseries.loc[first_outcome_lick_idx].trial_clock.values
    lick_times.loc[first_outcome_lick_trials, 'Consumption'] = first_outcome_lick_times

    return lick_times.reset_index()


def trials_by_time_array(trials: pd.DataFrame,
                         channel: str,
                         align_event: str,
                         win: tuple[int | float, int | float] = None,
                         FS: int = 50):

    '''
    Create simple array containing event-aligned neural traces stacked by
    trial. Includes list of trials identifying each row in array.

    Args:
        trials:
            Trial-based data already including event-aligned traces by trial.
        channel:
            Specifying color and hemisphere of brain e.g. ('grnR', 'grnL',
            'redL', 'redR').
        align_event:
            Behavior or task event neural traces are aligned to.
        win:
            Duration of time to plot before and after align_event, in seconds.
        FS:
            Sampling frequency in Hz of neural data.

    Returns:
        trials_by_time (np.array):
            Array of neural traces aligned to event and stacked by trial.
        timestamps (np.array):
            1D array of timestamps labeling horizontal axis of trials_by_time.
        trial_clean (pd.DataFrame):
            Trial data corresponding to traces in heatmap array. nTrial values
            give trial IDs for rows in trial_by_time.

    Notes:
        Expects column in trials containing pre-aligned neural data, such that
        trial_by_time will have num_cols = window length of each pre-aligned
        trace.
    '''

    # Drop trials that don't have photometry data.
    photo_col = f'{align_event}_{channel}'
    time_col = f'{align_event}_{channel}_times'

    all_photo_cols = [col for col in trials.columns if f'_{channel}' in col]
    trials_clean = (trials.copy()
                          .dropna(subset=all_photo_cols)
                          .reset_index(drop=True))

    # Stack event-aligned timeseries into array: timepoints x trials.
    exploded_trials = trials_clean.explode(column=[photo_col, time_col])

    if win:
        within_window = exploded_trials[time_col].between(-win[0], win[1],
                                                          inclusive='both')
        exploded_trials = exploded_trials.loc[within_window]
        n_timepoints = len(np.arange(-win[0], win[1]+(1/FS), step=1/FS))
    else:
        n_timepoints = len(trials_clean[photo_col].iloc[0])

    timestamps = exploded_trials[time_col].unique()
    n_trials = exploded_trials.nTrial.nunique()
    exploded_traces = exploded_trials[photo_col].astype('float')
    trials_by_time = (np.array(exploded_traces.dropna())
                        .reshape(n_trials, n_timepoints)
                      )

    return trials_by_time, timestamps, trials_clean


def sort_by_trial_type(trials: pd.DataFrame,
                       stacked_ts_traces: np.array,
                       task_variable: str):

    '''
    Sort trial table and array of aligned timeseries together on a given trial
    variable. Always also sort first on selection time and then by reward
    outcome.

    Args:
        trials:
            Dataframe containing trial level information.
        stacked_ts_traces:
            Array containing timeseries traces aligned to given behavior event
            and trimmed to window boundaries.
        task_variable:
            Column containing task variable that defines unique trial types.

    Returns:
        trials_sorted:
            trials variable sorted by selection time, Reward, task_variable.
        stacked_ts_traces:
            Timeseries traces sorted (as rows) to match trials_sorted indices
            for cross referencing.
    '''

    trials_ = trials.copy()
    time_col = 'tSelection' if 'tSelection' in trials_.columns else 't_cue_to_sel'

    # Sort trials in trial table by selection time, reward outcome, and task
    # variable.
    trials_sorted = trials_.sort_values(by=['Reward', task_variable, time_col])
    idx_sorted = trials_sorted.index.values

    # Sort neural traces as timeseries to match trial table.
    stacked_ts_traces = stacked_ts_traces[idx_sorted]
    trials_sorted['ngroup'] = (trials_sorted
                               .groupby(task_variable, sort=False)
                               .ngroup()
                               )
    return trials_sorted, stacked_ts_traces
