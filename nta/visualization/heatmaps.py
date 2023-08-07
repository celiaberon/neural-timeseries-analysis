from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nta.events.align import sort_by_trial_type, trials_by_time_array
from nta.features.select_trials import subsample_trial_types

sns.set(style='whitegrid',
        rc={'axes.labelsize': 11,
            'axes.titlesize': 11,
            'legend.fontsize': 11,
            'savefig.transparent': True})


def add_relative_timing_columns(timeseries: pd.DataFrame,
                                trials: pd.DataFrame):

    '''
    Create columns for relative timing of major behavior/task events, with all
    times in seconds. NOTE: for cue, all times are relative to cue onset.

    Args:
        timeseries:
            Timeseries data, used for calculating latency to first consumption
            lick.
        trials:
            Trial data.

    Returns:
        trials_:
            Trial data containing columns for each relative timing event.
    '''

    CUE_DURATION = 0.08

    trials_ = trials.copy()
    trials_['t_cue_to_sel'] = (trials_['tSelection'] / 1000) + CUE_DURATION

    # Calculate delay between selection and consumption licks as difference
    # in lengths between true and effective consumption states.
    lick_delay = (timeseries
                  .groupby('nTrial', as_index=False)
                  .agg({'Consumption': np.sum, 'stateConsumption': np.sum}))
    lick_delay['t_sel_to_cons'] = (lick_delay.stateConsumption
                                   - lick_delay.Consumption) * (1/50)

    lick_delay_trial_lut = lick_delay.set_index('nTrial')['t_sel_to_cons']
    trials_['t_sel_to_cons'] = (trials_['nTrial'].map(lick_delay_trial_lut))
    trials_['t_cue_to_cons'] = (trials_['t_cue_to_sel']
                                + trials_['t_sel_to_cons'])

    # Relative events preceding later events in a trial.
    trials_['t_sel_pre_cons'] = -trials_['t_sel_to_cons']
    trials_['t_cue_pre_cons'] = -trials_['t_cue_to_cons']
    trials_['t_cue_pre_sel'] = -trials_['t_cue_to_sel']

    return trials_


def define_relative_events(align_event: str):

    '''
    Define task/behavioral events that will be plotted relative to the aligned
    event based on column headers and timing necessary for proper alignment.

    Args:
        align_event:
            The name of the event aligned at x=0 for plotting/timing.

    Returns:
        other_event_cols:
            Column headers corresponding to other events to be plotted.
        labels:
            Generic labels for other events consistent across plots.
    '''

    match align_event:
        case 'Cue':
            other_event_cols = ['t_cue_to_sel', 't_cue_to_cons']
            labels = ['selection lick', 'first consumption lick']
        case 'Consumption':
            other_event_cols = ['t_sel_pre_cons', 't_cue_pre_cons']
            labels = ['selection lick', 'cue onset']
        case 'Select':
            other_event_cols = ['t_cue_pre_sel', 't_sel_to_cons']
            labels = ['cue onset', 'first consumption lick']

    return other_event_cols, labels


def scatter_behavior_events(trials: pd.DataFrame,
                            ax,
                            align_event: str,
                            window: tuple,
                            fs: int = 50
                            ):

    '''
    Plot timing of major task/behavioral events (Cue, Selection, first
    Consumption lick) for each trial.

    Args:
        trials:
            Trial data containing timing information.
        ax:
            Axis object on which to plot event times.
        align_event:
            Event at x=0 to which other events times will be relative.
        window:
            Duration plotted before and after event at x=0, in seconds.
        fs:
            Sampling rate across window (in Hz).

    Returns:
        ax:
            Updated axis object containing scatterplot of event times.
    '''

    tstep = 1/fs  # timestep in seconds
    color_dict = {'selection lick': 'w',
                  'first consumption lick': 'k',
                  'cue onset': sns.color_palette('colorblind')[3]}

    other_events, labels = define_relative_events(align_event)

    basic_scatterplot = partial(sns.scatterplot,
                                y=np.arange(0.5, len(trials)+0.5),
                                ax=ax,
                                marker='.',
                                edgecolor=None,
                                size=1)

    # Plot other task events relative to aligned task event at x=0.
    for label, task_event in zip(labels, other_events):

        # Offset task event position by pre-window duration and cue_offset in
        # addition to relative event timing.
        event_times = (trials[task_event].values + window[0]) / tstep
        basic_scatterplot(x=event_times,
                          color=color_dict[label],
                          label=label
                          )

    # Scatterplot for align_event as x=0 for each trial.
    align_event_label = [k for k in color_dict if align_event.lower() in k][0]
    basic_scatterplot(x=window[0]/tstep,
                      color=color_dict[align_event_label],
                      label=align_event_label
                      )

    ax.legend().remove()

    return ax


def label_trial_types(ax,
                      *,
                      trials: pd.DataFrame = None,
                      task_variable: str = None,
                      trial_type_palette=None,
                      tstep: float = 1/50,
                      include_label: bool = True):

    '''
    Add vertical lines with text labels to denote collections of trials within
    each unique trial type.

    Args:
        trials:
            Trial data containing trial type information.
        task_variable:
            Variable on which unique trial types are conditioned.
        ax:
            Current axes for matplotlib figure object.
        trial_type_palette:
            Optionally provide palette for color mapping trial types.
        tstep:
            Time in seconds for each x-axis step.
        include_label:
            Whether or not to include annotation of trial type labels.

    Returns:
        ax:
            Updated axes with color-coded and labeled trial types.
    '''

    if (task_variable is None) or (task_variable == 'Trial'):
        return ax  # nothing to label in these cases

    ymax = 0  # initialize at origin

    x_scale_factor = np.abs(ax.get_xlim()).sum()
    x_offset = x_scale_factor / (500*tstep)

    for i, (key, grp) in enumerate(trials.groupby(task_variable, sort=False)):

        # Select color for trial type labeling and annotation.
        if trial_type_palette is None:
            trial_color = sns.color_palette('deep')[i]
        else:
            trial_color = trial_type_palette[key]

        # Set y-bounds for trial type bar label and vertically center text.
        ymin = ymax
        ymax += len(grp)
        text_y = (ymax+ymin)/2
        ax.vlines(x=-.5,
                  ymin=ymin,
                  ymax=ymax,
                  color=trial_color,
                  label=None,
                  lw=12
                  )

        if include_label:
            ax.annotate(key,
                        (-x_offset, text_y),
                        va='center',
                        color=trial_color,
                        fontsize=13,
                        annotation_clip=False,
                        rotation=90
                        )
            ax.yaxis.set_label_coords(-0.1, 0.5)

    return ax


def get_cmap_range(trials: pd.DataFrame,
                   states: list = None,
                   channel: str = None):

    '''
    Get range of values for neural traces to normalize colormap
    onto. Likely overkill to do it by state-aligned window unless
    the window limits are not overlapping.

    Args:
        trials:
            Trial data containing pre-aligned neural timeseries.
        states:
            States for which state-aligned data will be plotted.
        channel:
            'L' or 'R' hemisphere from which neural data was obtained.

    Returns:
        vmin, vmax:
            Min and max values across all state-aligned neural timeseries.
    '''

    vmin = np.inf
    vmax = -np.inf

    if states is None:
        states = ['Cue', 'Select', 'Consumption']

    for state in states:
        state_range = trials[f'{state}_{channel}'].explode()

        vmin = np.min((vmin, state_range.min()))
        vmax = np.max((vmin, state_range.max()))

    return vmin, vmax


def center_xticks_around_zero(tstamps: list,
                              freq: float | int,
                              tick_interval: int = None):

    '''
    Get list of xticks and xticklabels that are centered at x=0 seconds and
    spaced out by a given interval in seconds.

    Args:
        tstamps:
            List of timestamps spanning the x-axis.
        freq:
            Sampling frequency of timestamps in Hz.
        tick_interval:
            Spacing desired between ticks, in seconds.

    Returns:
        xticks:
            Position (as indices) for ticks on x-axis.
        xticklabels:
            Labels corresponding to xticks.
    '''

    # Set tick interval in units of seconds, defaults to 1s.
    if tick_interval is None:
        tick_interval = freq
    else:
        tick_interval *= freq
        tick_interval = int(tick_interval)

    # Find index at which x=0 seconds.
    center_idx = np.where(tstamps == 0)[0][0]

    # Piecewise create list of indices for tick positions at tick_interval.
    pre_window_idcs = np.arange(center_idx, -1, step=-tick_interval)
    pre_window_idcs = pre_window_idcs[pre_window_idcs >= 0]  # positive idcs
    post_window_idcs = np.arange(center_idx, len(tstamps), step=tick_interval)
    xticks = np.unique(np.concatenate((pre_window_idcs, post_window_idcs)))

    # Get tick labels from timestamp list.
    xtick_labels = tstamps[xticks]

    return xticks, xtick_labels


def plot_heatmap(heatmap_array: np.array,
                 align_event: str,
                 *,
                 tstamps: list = None,
                 ax=None,
                 hm_kwargs=None,
                 **kwargs,
                 ):

    '''
    Plot a heatmap where each row contains event-aligned data from a single
    trial, the x-axis spans a designated window of time, and the color
    intensity corresponds to the magnitude of a neural timeseries at each
    timepoint.

    Args:
        heatmap_array:
            Array containing neural data for N trials x M timespoints.
        align_event:
            Task or behavioral event aligned to x=0 seconds.
        tstamps:
            List of timestamps corresponding to M timepoints in heatmap_array.
        ax:
            Axis object on which to plot heatmap.
        hm_kwargs:
            Keyword arguments to pass through to sns.heatmap function.
        **kwargs:
            Keyword arguments for labeling trial types.

    Returns:
        ax:
            Axis object containing heatmap.
    '''

    np.random.seed(0)  # seed for consistent subsampling.

    # Convert window boundaries from time (secs) to index points for x-axis.
    freq = round(1/np.diff(tstamps[:2])[0])  # number of samples per second

    # Alignment errors if window length dims not specified properly.
    assert len(tstamps) == heatmap_array.shape[1]

    # Create figure and axes.
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    if hm_kwargs is None:
        hm_kwargs = {}

    # Plot heatmap.
    sns.heatmap(heatmap_array,
                cmap='viridis',
                ax=ax,
                rasterized=True,
                cbar=False,
                **hm_kwargs,
                label=None)

    # Label groups of trial types.
    ax = label_trial_types(ax=ax,
                           tstep=1/freq,
                           **kwargs)

    xticks, xticklabels = center_xticks_around_zero(tstamps,
                                                    freq,
                                                    tick_interval=1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=0)
    ax.set(ylabel='nTrial', xlabel='time (s)', yticks=[],
           title=f'aligned to {align_event}')

    return ax


def create_scaled_colorbar(fig, axs, vmin: float, vmax: float):

    '''
    Create colorbar mapping range of values and position properly on figure.

    Args:
        fig:
            Figure object on which to create colorbar.
        axs:
            Axes for which color mapping applies.
        vmin, vmax:
            Min and max values for mapped color gradient.

    Returns:
        fig, axs:
            Updated figure and axes objects containing colorbar.
    '''

    cmap = 'viridis'
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # spectrogram min/max
    (fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  ax=axs,
                  anchor=(1.0, 0.5),
                  panchor=(0, 0),
                  shrink=0.15,
                  label='z-score',
                  orientation='horizontal',
                  )
        .outline.set_visible(False))

    return fig, axs


def plot_heatmap_wrapper(trials: pd.DataFrame,
                         *,
                         alignment_states: list = None,
                         channel: str = None,
                         task_variable: str = None,
                         subsample: int = None,
                         win: tuple = (1, 2),
                         figsize: tuple = None,
                         **kwargs):

    '''
    Wrapper to combine typical sets of components into heatmaps plotting
    neural activitiy by trial over time, overlaid with relative timing of task
    and behavioral events. If given, trials are sorted by trial type.

    Args:
        trials:
            Trial data containing pre-aligned neural timeseries.
        alignment_states:
            List of states to include as subplots, within which x=0 marks
            timing of state.
        channel:
            'L' or 'R' hemisphere from which neural data was obtained.
        task_variable:
            Variable on which unique trial types are conditioned.
        subsample:
            Target sample size per group within task_variable to downsample
            (w/o replacement) number of trials to.
        win:
            Duration plotted before and after event at x=0, in seconds.
        figsize:
            Tuple of figure size is (width, height).

    Returns:
        fig, axs:
            Figure and axes objects containing plotted data.
    '''

    # Default to plotting 3 main trial events.
    if alignment_states is None:
        alignment_states = ['Cue', 'Select', 'Consumption']

    if figsize is None:
        figsize = (3.*len(alignment_states), 2.0)

    fig, axs = plt.subplots(ncols=len(alignment_states),
                            figsize=figsize,
                            sharey=True,
                            sharex=True)

    # Set min and max values for color-mapped range.
    vmin, vmax = get_cmap_range(trials, channel=channel)

    for i, (ax, state) in enumerate(zip(axs, alignment_states)):

        # Create image-like array of event-aligned timeseries.
        hm_array, timestamps, trials_ = trials_by_time_array(trials,
                                                             channel=channel,
                                                             align_event=state,
                                                             win=win)

        # Sort trials by task variable and subsample to target num trials
        # within condition. Needs to be after setting trials for heatmap array.
        if task_variable is not None:
            if subsample:
                trials_ = subsample_trial_types(trials_,
                                                task_variable,
                                                subsample)
            else:
                trials_ = trials_.copy().reset_index(drop=True)
            trials_, hm_array = sort_by_trial_type(trials_,
                                                   hm_array,
                                                   task_variable)

        # Plot actual heatmap.
        ax = plot_heatmap(hm_array,
                          trials=trials_,
                          align_event=state,
                          task_variable=task_variable,
                          tstamps=timestamps,
                          ax=ax,
                          include_label=i < 1,
                          hm_kwargs={'vmin': vmin, 'vmax': vmax},
                          **kwargs)

        # Convert window boundaries from time (secs) to idx points for x-axis.
        freq = round(1/np.diff(timestamps[:2])[0])  # num samples per second

        # Overlay scatterplot of behavior events for each trial's timeseries.
        ax = scatter_behavior_events(trials_, ax, state, win, fs=freq)

    # Rescale colorbar to fit plot.
    fig, axs = create_scaled_colorbar(fig, axs, vmin, vmax)

    # Edit legend to give white points a black border (but not in plot itself).
    h, labels = axs[-1].get_legend_handles_labels()
    h[labels.index('selection lick')].set_edgecolor('k')
    real_legend_items = [(h_, l_) for h_, l_ in zip(h, labels) if len(l_) > 2]
    axs[-1].legend(*list(zip(*real_legend_items)),
                   bbox_to_anchor=(1, 1),
                   markerscale=2.,
                   edgecolor='white')
    h[labels.index('selection lick')].set_edgecolor('white')

    plt.tight_layout()

    return fig, axs
