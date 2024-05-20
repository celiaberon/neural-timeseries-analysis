#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 15:22:38 2022

@author: celiaberon
"""

from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype

from ..events.align import get_lick_times
from ..utils import save_plot_metadata


def set_new_axes(n_iters: list,
                 *,
                 behavior_hist: bool = True,
                 figsize: tuple = None,
                 **kwargs):

    '''
    Configure axes for event alignment and possibly behavior distributions.
    Size figure and individual axes based on set axes dims and number of
    subplots total. Arrange for including behavior distributions if specified.

    Args:
        n_iters (list):
            [total number of subplot groups, current subplot group]
        behavior_hist (bool):
            Whether or not to plot behavior distributions.
        figsize:
            (width, height) dimensions.

    Returns:
        fig (matplotlib.figure object)
        axs (matplotlib.axes object)
    '''

    if figsize is None:
        subplot_width = 4.0
        subplot_height = 3.0 + behavior_hist
    else:
        subplot_width, subplot_height = figsize

    nrows = int(np.ceil(n_iters[0] / 3))
    ncols = min(n_iters[0], 3)

    dims = (subplot_width * ncols, subplot_height * nrows)
    if behavior_hist:  # 2 plots per alignment with behavior distributions
        fig, axs = plt.subplots(nrows=2 * nrows, ncols=ncols,
                                figsize=dims,
                                sharex=True,
                                gridspec_kw={'height_ratios': (3, 1)})
    else:  # Otherwise single layer of lineplots for each alignment
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                figsize=dims,
                                sharey=False, sharex=True)
        if n_iters[0] == 1:
            axs = [axs, None]  # consistent dims for handling in plotting func

    return fig, axs


def set_current_axes(axs,
                     *,
                     n_iters: list = None,
                     behavior_hist: bool = None,
                     ):

    '''
    Select current subplot axes for lineplot and behavior distributions
    '''

    ncols = min(n_iters[0], 3)

    if n_iters[0] > 1:
        ax1 = axs.flatten()[n_iters[1]]
        ax2 = axs.flatten()[n_iters[1] + ncols] if behavior_hist else None
        # n_iters[1] += 1 # mutable update globally
    else:
        ax1, ax2 = axs

    return ax1, ax2


def subsample_trial_types(trials: pd.DataFrame,
                          column: str,
                          target_size: int = None) -> pd.DataFrame:

    '''
    Subsample to target size for each trial type to compare effects of rare
    trial types.

    Args:
        trials:
            Trial-based dataframe.
        column:
            Behavior variable to subsample condiitons within.
        target_size:
            Number of repetitions to sample from each group within column.

    Returns:
        subsampled_trials:
            Dataframe as subset of trials containing
            N trials = target_size * number of unique conditions in column.
    '''

    trials_ = trials.copy()

    # Default to minimum number of reps across conditions in column.
    min_trial_type = trials_[column].value_counts().min()
    if (target_size is None) or (min_trial_type < target_size):
        target_size = min_trial_type

    subsampled_trials = trials_.groupby(column).sample(n=target_size)

    return subsampled_trials


def config_plot_cpal(ts, column, *, cmap_colors=None, **kwargs):

    '''Set color palette with flexible input'''

    match cmap_colors:
        case int():  # Sequential palette for numerical conditions
            cpal = mpl.cm.RdBu(np.linspace(0, 1, cmap_colors))
            cpal = mpl.colors.LinearSegmentedColormap.from_list('cpal', cpal,
                                                                N=cmap_colors)
            cpal = [cpal(i) for i in range(cmap_colors)]
        case (dict() | list()):  # Use palette if given explicitly
            cpal = cmap_colors
        case str():
            cpal = sns.color_palette(cmap_colors, n_colors=ts[column].nunique())

        case _:
            cpal = sns.color_palette('deep', n_colors=ts[column].nunique())

    if (not isinstance(cpal, dict)) & is_numeric_dtype(ts[column]):
        labels = np.sort(ts[column].dropna().unique())
        if any(labels < 0) & any(labels > 0) & ~any(labels == 0):
            labels = np.sort(np.insert(labels, 0, 0))
        cpal = {label: color for label, color in zip(labels, cpal)}
    kwargs.update({'cpal': cpal})

    return cpal, kwargs


@save_plot_metadata
def plot_trial_type_comparison(ts: pd.DataFrame,
                               *,
                               column: str = None,
                               y_col: str = None,
                               trial_units: bool = False,
                               behavior_hist: bool = False,
                               n_iters: list = None,
                               axs=None,
                               fig=None,
                               error=('ci', 95),
                               **kwargs):

    '''
    Plot lineplots of neural data aligned to behavioral/task events. Option
    to also plot alongside distribution of behavior events.

    Args:
        ts:
            Timeseries of neural data also containing trial type information.
        column:
            Column within timeseries to condition plot color palette on.
        align_event:
            Behavior/task event neural data is aligned to (0 on x-axis).
        y_col:
            Hemisphere (R/L) and/or color (grn/red) to use for y-values.
        trial_units:
            Whether to plot individual traces for each trial.
        behavior_hist:
            Whether to include plots for behavior timing distributions.
        n_iters:
            [number of total plot iterations, current plot iteration]
        axs:
            matplotlib.axes object to add plots into.
        fig:
            matplolib.figure object.
        error:
            (error type, significance level)
        **kwargs:
            ls_col:
                Second column to condition plotting on (line style).
            show_leg:
                Whether or not to include legend in plot.
            ...

    Returns:
        fig:
            New/updated figure object.
        axs:
            New/updated axes object.
    '''

    sns.set(style='ticks', font_scale=1.0, rc={'axes.labelsize': 11,
                                               'axes.titlesize': 11,
                                               'savefig.transparent': True,
                                               'legend.title_fontsize': 11,
                                               'legend.fontsize': 10,
                                               'legend.borderpad': 0.2,
                                               'figure.titlesize': 11,
                                               'figure.subplot.wspace': 0.1,
                                               })

    if n_iters is None:
        n_iters = [1, 0]

    # Plot aesthetics and designate current ax1: lineplot, ax2: behavior.
    cpal, kwargs = config_plot_cpal(ts, column, **kwargs)
    if n_iters[1] == 0:
        fig, axs = set_new_axes(n_iters=n_iters,
                                behavior_hist=behavior_hist,
                                **kwargs)
    ax1, ax2 = set_current_axes(axs,
                                n_iters=n_iters,
                                behavior_hist=behavior_hist)

    # Core plotting function regardless of individual trial traces or group
    # mean.
    lineplot_core = partial(sns.lineplot,
                            data=ts,
                            x=f'{y_col.split("_")[0]}_times',
                            y=y_col,
                            hue=column,
                            ax=ax1)

    # Actual lineplot for event-aligned neural data.
    if trial_units:
        ax1 = lineplot_core(units='nTrial',
                            estimator=None,
                            alpha=0.9)
    else:
        ax1 = lineplot_core(style=kwargs.get('ls_col', None),
                            n_boot=kwargs.get('n_boot', 1000),
                            errorbar=error,
                            err_kws={'alpha': 0.3},
                            estimator=kwargs.get('estimator', 'mean'),
                            palette=cpal)

    fig, ax1 = config_plot(ax1, fig, ts, y_col, column, **kwargs)

    # Plot distribution of behavioral/task events relative to alignment event.
    if behavior_hist:
        ax2 = behavior_event_distributions(ts, y_col, ax2,
                                           column=column, **kwargs)

    plt.tight_layout()
    return fig, axs


def plotting_wrapper(trials: pd.DataFrame,
                     alignment_states: list = None,
                     channel: str = None,
                     **kwargs):

    '''
    Plot panel of photometry figures.

    Args:
        trials:
            Trial-based data containing aligned photometry snippets and
            timestamps.
        alignment_states:
            List of events to plot aligned neural data to.
        channel:
            L or R hemisphere photometry channel.
        **kwargs:
            See plot_trial_type_comparison().

    Returns:
        fig:
            Figure object containing created and filled plot.
        axs:
            Axes object containing created and filled plot.
    '''

    axs = None
    fig = None

    # Default to plotting 3 main trial events.
    if alignment_states is None:
        alignment_states = ['Cue', 'Select', 'Consumption']

    # Iteratively fill subplots with each event-alignd photometry trace.
    for plot_iter, event in enumerate(alignment_states, start=1):
        n_iters = [len(alignment_states), plot_iter - 1]

        leg = kwargs.get('leg_override', plot_iter >= n_iters[0])
        photometry_column = f'{event}_{channel}'
        exploded_trials = (trials.copy()
                           .dropna(subset=[photometry_column])
                           .explode(column=[photometry_column,
                                            f'{event}_times'])
                           )
        fig, axs = plot_trial_type_comparison(exploded_trials,
                                              align_event=event,
                                              y_col=photometry_column,
                                              n_iters=n_iters,
                                              show_leg=leg,
                                              fig=fig,
                                              axs=axs,
                                              **kwargs)

    if kwargs.get('save', False):
        fig.savefig(kwargs.get('fname'), dpi=200, bbox_inches='tight')

    return fig, axs


def config_plot(ax,
                fig,
                ts: pd.DataFrame,
                y_col: str,
                column: str,
                show_leg: bool = False,
                ylim: tuple = None,
                ls_col=False,
                window: tuple = (1, 3),
                title: str = None,
                **kwargs):

    '''
    Configure plot aesthetics such as labeling 0 positions on axes
    and setting limits/legends/tick labels consistently.
    '''

    ts_channel = ts[y_col]
    align_event = y_col.split('_')[0]
    if ylim is None:
        ymin = np.mean(ts_channel) - 1 * np.abs(np.min(ts_channel))
        ymax = np.mean(ts_channel) + 0.8 * np.max(ts_channel)
        ylim = (ymin, ymax)
        print(ylim)
    ax.axvline(x=0, color='k', ls='-', lw=0.8, alpha=1.0, zorder=0,
               label=None)
    ax.axhline(y=0, color='k', ls='-', lw=0.8, alpha=1.0, zorder=0)
    ax.set(xlabel='Time (s)', ylabel='z-score', ylim=ylim)
    ax.text(x=-0.05, y=ylim[1] + 0.1 * sum(ylim), s=align_event)
    ax.set(xlim=(-window[0], window[1]))
    ticks = ax.get_xticks()
    ax.set_xticks([int(tick) for tick in ticks if tick.is_integer()])
    ax.set(xlim=(-window[0], window[1]))

    if title:
        fig.suptitle(title, y=0.9)

    if not show_leg:
        ax.legend().set_visible(False)
    else:
        # Replace legend with colorbar if conditioning on numeric variable
        # that contains more than two conditions.
        if is_numeric_dtype(ts[column]) & (ts[column].dropna().nunique() > 3):
            fig, ax = convert_leg_to_cbar(fig, ax, **kwargs)
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().set_visible(False)
            fig.legend(handles, labels, bbox_to_anchor=(1, 0.9),
                       loc='upper left', frameon=False,
                       title='' if ls_col else column)

    sns.despine()

    return fig, ax


def label_legend_unique_handles(ax, **kwargs):

    handles, labels = ax.get_legend_handles_labels()
    legend_reduced = dict(zip(labels, handles))
    ax.legend(legend_reduced.values(), legend_reduced.keys(),
              bbox_to_anchor=(1, 1), frameon=False, **kwargs)
    return ax


def convert_leg_to_cbar(fig, ax, labels=None, cpal=None, **kwargs):

    '''
    Create colorbar to replace legend. Should be called for sequential,
    numeric labels only.

    Args:
        fig, ax:
            Matplotlib figure and axis (current axis) objects.
        labels:
            Colormap labels that will be used to label colorbar.
        cpal:
            Color palette containing at least color values, but ideally
            pre-connected to color labels.
    Returns:
        fig, ax:
            Figure and axis objects with added colorbar and removed legend.
    '''

    hw = len(cpal) // 2

    if labels is None:
        labels = np.array(list(cpal.keys()))

    if any(labels > 0) & any(labels < 0):
        cm_min, cm_max = (-hw - 1, hw + 1)
    else:
        cm_min, cm_max = (0, len(cpal) + 1)

    i_color = range(cm_min, cm_max)
    color_vals = np.array(list(cpal.values()))
    cmap, norm = mpl.colors.from_levels_and_colors(i_color, color_vals)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    h, lab = ax.get_legend_handles_labels()
    ax.legend().set_visible(False)

    # Add cbar to figure to avoid resizing subplots to accomodate.
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False,
                    left=False, right=False)
    cbar = plt.colorbar(sm, anchor=(1.3, 0), shrink=0.6)
    cbar.ax.tick_params(size=0)
    if len(labels) <= 7:
        cbar.set_ticks(np.arange(cm_min + 0.5, cm_max - 0.5),
                       labels=[int(label) for label in labels])
    else:
        cbar.set_ticks([cm_min + 0.5, cm_max - 1.5],
                       labels=[labels[0], labels[-1]])

    if 'Rewarded' in lab:
        plt.legend(*list(zip(*[(h_, l_) for h_, l_ in zip(h, lab)
                               if l_ in ['Rewarded', 'Unrewarded']])),
                   bbox_to_anchor=(1.5, 1), markerscale=1., fontsize=13,
                   edgecolor='white')

    return fig, ax


def behavior_event_distributions(ts,
                                 y_col,
                                 ax,
                                 *,
                                 graded_cue: bool = False,
                                 show_leg: bool = False,
                                 **kwargs):

    all_events_ts = kwargs.pop('lick_ts')
    all_events_ts = all_events_ts.query('nTrial.isin(@ts.nTrial.unique())')
    lick_times = get_lick_times(lick_ts=all_events_ts, **kwargs)

    # Default approach, calculate relative timing of 2 distributions to
    # aligned event.
    state_colors = {'Cue': sns.color_palette('colorblind')[3],
                    'Select': 'darkgray',
                    'Consumption': 'k'}
    plotting_kwargs = {'color': True}

    align_event = y_col.split('_')[0]

    if 'ENL' in align_event:
        ts_ = ts.copy().groupby('nTrial')['T_ENL'].nth(0)
        lick_times['T_ENL'] = lick_times['nTrial'].map(ts_)
        plotting_kwargs = {'palette': sns.color_palette('Greys', 6)[1:],
                           'hue': 'T_ENL'}

    elif 'fake_event' in align_event:  # For null comparisons
        fake_events = ts.query('fake_event == 1')
        event_times = (fake_events
                       .groupby('nTrial')
                       .nth(0)['trial_clock'].values)
        lick_times['fake_event'] = np.nan
        trial_ids = fake_events.nTrial.unique()
        lick_times.loc[lick_times.nTrial.isin(trial_ids),
                       'fake_event'] = event_times

    ylim_ub = []
    for event in ['Cue', 'Select', 'Consumption']:

        if graded_cue:
            raise NotImplementedError
            # plotting_kwargs = {'hue': column, 'palette': kwargs.get('cpal')}
            # ts_ = ts.copy().groupby('nTrial')[column].nth(0)
            # lick_times[column] = lick_times['nTrial'].map(ts_)
        elif plotting_kwargs.get('color', None) is not None:
            plotting_kwargs['color'] = state_colors[event]

        if event == align_event:
            ax.axvline(x=0, color=state_colors[event])
        else:
            lick_times[f'{event} time'] = (lick_times[event]
                                           - lick_times[align_event])
            sns.histplot(data=lick_times, x=f'{event} time', ax=ax,
                         binwidth=0.05, alpha=0.7,
                         label=event, stat='probability',
                         element='step', linewidth=0, edgecolor=None,
                         **plotting_kwargs)
            ylim_ub.append(ax.get_ylim()[1])

    ax.set(ylabel='fraction\ntrials', xlabel='Time (s)',
           ylim=(0, max(ylim_ub)))

    if not show_leg:
        ax.legend().set_visible(False)

    else:
        # if graded_cue:
        #     legend_labels,legend_handles=ax.get_legend_handles_labels()
        #     ax.legend(legend_labels[:: len(legend_labels) // 2],
        #             legend_handles[:: len(legend_labels) // 2],
        #             frameon=False, fontsize=12, markerscale=0.8)
        # else:
        def custom_patch(key, color):
            from matplotlib.patches import Patch
            return Patch(facecolor=color, edgecolor=None, label=key, alpha=0.7)
        leg_elements = [custom_patch(key, col)
                        for key, col in state_colors.items()]
        ax.legend(handles=leg_elements, bbox_to_anchor=(1, 1.5),
                  loc='upper left', frameon=False, markerscale=0.5,
                  )

    return ax
