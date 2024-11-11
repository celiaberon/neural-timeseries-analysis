#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:30:18 2022

@author: celiaberon
"""

from math import modf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import save_plot_metadata


@save_plot_metadata
def plot_peaks_wrapper(peaks: pd.DataFrame,
                       x_col: str = None,
                       channel: str = None,
                       metrics: str | dict[str, str] = 'mean',
                       plot_func=sns.boxplot,
                       show_outliers: bool = True,
                       plot_func_kws: dict = None,
                       mosaic_kws: dict = None,
                       ignore_reward: bool = False,
                       states: list = None,
                       **kwargs):

    '''
    Plot panel of photometry summary statistic figures, with subplot for each
    aligned event, potentially split by reward outcome.

    Args:
        peaks:
            Trial-based data containing summary statistic for event-aligned
            photometry.
        x_col:
            Column name of variable to plot on x-axis of each subplot.
        channel:
            L or R hemisphere photometry channel.
        metrics:
            Summary statistic to plot, treated as suffix of column along with
            channel.
        plot_func:
            Seaborn plotting function, commonly:
                sns.barplot(),
                sns.pointplot(),
                sns.violinplot()
        show_outliers:
            Whether or not to show outliers in plot. Defaults to True.
        plot_func_kws:
            Kwargs recognized by plot_func(). Includes arguments like `hue`,
            `palette`, etc.
        mosaic_kws:
            See initialize_peak_fig(); kwargs recognized by subplot_mosaic().
        ignore_reward:
            Whether or not to plot rewarded and unrewarded trials separately
            for Consumption metrics.
        states:
            Task states to include summary statistics for in plot, defaults to
            Cue and Consumption.
        **kwargs:
            See plot_trial_type_comparison().

    Returns:
        fig, axs:
            Figure and axes objects containing created and filled plot.
    '''

    if plot_func_kws is None:
        plot_func_kws = {}
    if mosaic_kws is None:
        mosaic_kws = {}

    fig, ax = initialize_peak_fig(states, **mosaic_kws)

    # Set color palette to be passed as a keyword argument to plotting func.
    plot_func_kws['palette'] = set_color_palette(peaks,
                                                 x_col,
                                                 **plot_func_kws)
    if states is None:
        states = ['Consumption', 'Cue']
    if isinstance(metrics, str):
        metrics = {state: metrics for state in states}
        if not ignore_reward and ('Consumption' in states):
            metrics['no reward'] = metrics['Consumption']
            metrics['reward'] = metrics['Consumption']

    ax_modifiers = {0: 'no ', 1: ''}
    for state in states:

        # For Cue, of if ignore_reward then also for Consumption, plot
        # photometry metric vs. x_col for all trials.
        if (state != 'Consumption') or ignore_reward:
            metric = metrics[state]
            y_col = f'{state}_{channel}_{metric}'

            # Drop outliers but take a look at how many there are first.
            if not show_outliers:
                peaks = exclude_outliers(peaks, x_col, y_col)
            if (state != 'Consumption') or ignore_reward:
                plot_func(data=peaks, x=x_col, y=y_col, ax=ax[state],
                          **plot_func_kws)

        # Split consumption trials into rewarded and unrewarded.
        else:
            for reward_id, reward_group in peaks.groupby('Reward'):
                label = f'{ax_modifiers[reward_id]}reward'
                y_col = f'{state}_{channel}_{metrics[label]}'

                if not show_outliers:
                    peaks = exclude_outliers(peaks, x_col, y_col)
                plot_func(data=reward_group,
                          x=x_col,
                          y=y_col,
                          ax=ax[label],
                          **plot_func_kws)

    ax = config_plot(ax, channel, metrics, **kwargs)

    # If x-axis variable and hue are different, recolor plot for x-variable
    # and use hatching for hue variable.
    if (x_col != plot_func_kws.get('hue', x_col)) & (plot_func == sns.barplot):
        assert plot_func_kws.get('hue_order') == [0, 1], 'order will not match labels'
        ax = hatch_and_recolor(ax, peaks, x_col, cpal=plot_func_kws['palette'], **kwargs)

    if kwargs.get('save'):
        if not isinstance(fig, plt.Figure):
            print('skipping save, not main figure')
        else:
            fig.savefig(kwargs.get('fname'), dpi=200, bbox_inches='tight')

    return fig, ax


def initialize_peak_fig(states,
                        *,
                        figsize: tuple[int | float, int | float] = (3.5, 4.),
                        axes_style: str = None,
                        flatten_layout: bool = False,
                        **mosaic_kwargs,
                        ):

    sns.set(style='ticks',
            rc={'axes.labelsize': 12,
                'axes.titlesize': 12,
                'savefig.transparent': True,
                'legend.frameon': False
                })

    if states:
        layout = [states]
    elif flatten_layout:
        layout = [['Cue', 'reward', 'no reward']]
    else:
        layout = [['.', 'reward'],
                  ['Cue', 'reward'],
                  ['Cue', 'no reward'],
                  ['.', 'no reward']]
    if figsize == (3.5, 4.) and (states or flatten_layout):
        figsize = (2 * len(layout[0]), 2)

    if 'subfig' in mosaic_kwargs:
        fig = mosaic_kwargs.pop('subfig')
        ax = fig.subplot_mosaic(layout, **mosaic_kwargs,
                                gridspec_kw={
                                    "bottom": -0.05,
                                    "top": 0.85,
                                    "left": 0.2,
                                    "right": 0.8,
                                    "wspace": 0.8,
                                    "hspace": 3})
    else:
        fig, ax = plt.subplot_mosaic(layout, figsize=figsize, **mosaic_kwargs)

    if states is None:
        ax['no reward'].set(title='Unrewarded',)
        ax['reward'].set(title='Rewarded', xlabel='')
        ax['Cue'].set(title='Cue')
    else:
        for state in states:
            ax[state].set(title=state)

    match axes_style:
        case 'constant':
            ax['Cue'].set(ylim=(-2.5, 4.5))
            ax['reward'].set(ylim=(-2.5, 4.5))
            ax['no reward'].set(ylim=(-2.5, 4.5))
        case 'pos-neg':
            ax['Cue'].set(ylim=(0, 4.5))
            ax['reward'].set(ylim=(0, 4.5))
            ax['no reward'].set(ylim=(-2.5, 0))
        case dict():
            [ax[key].set(ylim=val) for key, val in axes_style.items()]
        case _:
            pass
    sns.despine()

    return fig, ax


def hatch_and_recolor(axs, peaks, x_col, hatch_labels={0: 'Stay', 1: 'Switch'},
                      hatch_colors={0: 'white', 1: 'white'}, cpal=None, **kwargs):

    '''Recolor plot for x-variable and use hatching for hue variable.'''

    import matplotlib.patches as mpatches

    hatch_styles = {0: '', 1: '//'}

    def cpal_key(i, nbars):
        val = (i % nbars) - (nbars // 2) + (i % nbars >= np.ceil(nbars / 2))
        return val

    if cpal is None:
        cpal = set_color_palette(peaks, x_col, palette=peaks[x_col].nunique())
    nbars = len(cpal)
    for label, ax_ in axs.items():
        for bar in ax_.patches[:-2]:

            rem, j = np.round(modf(bar._x0 - 0.1), decimals=1)
            if rem < 0:
                j -= 1
                rem = 1 + rem
            bar.set_hatch(hatch_styles[rem > 0.5])  # because of hue_order
            bar.set_facecolor(cpal[cpal_key(int(j+1), nbars)])
            bar.set_alpha(0.8)
            bar.set_edgecolor('k')


    legend_elements = [mpatches.Patch(facecolor=hatch_colors.get(k),
                                      edgecolor='k',
                                      label=hatch_labels.get(k),
                                      hatch=hatch_styles.get(k))
                       for k in hatch_labels]
    ax_.legend(handles=legend_elements, bbox_to_anchor=(1, 1))

    return axs


def set_color_palette(peaks, x_col, palette=None, hue=None, **kwargs):

    '''
    Set color palette, defaulting to using x-axis variable to color if no hue
    is provided.
    '''

    if hue is None:
        hue_col = x_col
    else:
        hue_col = hue

    labels = np.sort(peaks[hue_col].dropna().unique())
    match palette:
        case dict():
            return palette
            # labels = palette.keys()
        case str():
            return sns.color_palette(palette, n_colors=len(labels))
        case None:
            return None
        case list():
            return dict(zip(labels, palette))
        case int():
            palette = sns.color_palette('RdBu', n_colors=len(labels))
            return dict(zip(labels, palette))


def exclude_outliers(peaks: pd.DataFrame,
                     x_col: str,
                     y_col: str) -> pd.DataFrame:

    '''
    Remove outliers from distribution of y variable for each x variable.
    '''
    def get_num_outliers(column):
        '''Count number of outliers in distribution.'''
        outliers = 1 - not_outlier(column)
        print(f'{round(np.mean(outliers),3)} points removed as outliers')

    def not_outlier(column):
        '''Return boolean of whether each value in distribution is otulier.'''
        q1 = np.percentile(column, 25)
        q3 = np.percentile(column, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return (column > lower_bound) & (column < upper_bound)

    peaks_ = peaks.dropna(subset=[y_col]).copy()
    peaks_.groupby(x_col)[y_col].agg([get_num_outliers])
    peaks_ = (peaks_.loc[peaks_.groupby(x_col, group_keys=False)[y_col]
                               .apply(not_outlier)]
                    .reset_index(drop=True))

    return peaks_


def config_plot(ax, channel, metrics, col_id: str = '', **kwargs):

    ylab = channel.split('_')[0]

    try:
        ax['reward'].set(xlabel='')
        ticks = [int(ax['Cue'].get_xticks()[0]),
                 int(ax['Cue'].get_xticks()[-1])]

        ax['Cue'].set(xlabel=col_id, ylabel=ylab, xticks=ticks)
        ax['reward'].set(xlabel='', ylabel=ylab, xticks=ticks)
        ax['no reward'].set(xlabel=col_id, yticks=[0, -1, -2], xticks=ticks,
                            ylim=(-3, 0), ylabel=ylab)
        ax['no reward'].axhline(y=0, color='k', lw=2)
    except KeyError:
        pass

    first_ax = list(ax.keys())[0]
    if np.any(ax[first_ax].get_legend_handles_labels()):
        [ax_.legend().remove() for _, ax_ in ax.items()]

    return ax


def plot_correlation(r, col0, col1, ax=None, hue=None, palette=None, legend=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    sns.barplot(data=r, x='channel', y='r', hue=hue, palette=palette, ax=ax, legend=legend)
    ax.axhline(y=0, color='k')
    ax.set(ylim=(-1, 1), title=f'correlation {col0}\nvs. {col1}', xlabel='hemisphere')
    plt.legend(bbox_to_anchor=(1.8, 1))
    sns.despine()


def calc_grouped_corr(trials, col0, col1, grouping_variable):
    rs = {}
    for i, val in enumerate(trials[grouping_variable].dropna().unique()):
        rs[i] = {
            'r': (trials
                  .groupby(grouping_variable)[[col0, col1]]
                  .corr()
                  .reset_index()
                  .query(f'level_1 == @col1 & {grouping_variable} == @val')[col0].item()),
            grouping_variable: val
        }
    return pd.DataFrame(rs).T


def plot_swarm_and_point(peaks_agg, Data, hue, palette,
                         size=3, add_pointplot=False, ylim=(-1,7.5)):

    fig = plt.figure(figsize=(6, 3), layout='constrained')
    subfigs = fig.subfigures(ncols=2, wspace=0.2)

    for subfig, ch in zip(subfigs, Data.sig_channels):

        axs = subfig.subplots(ncols=2, sharey=True)

        for ax, event in zip(axs.flatten(), ['Cue', 'Consumption']):
            sns.swarmplot(
                data=peaks_agg, x='Reward', y=f'{event}_{ch}_mean', hue=hue,
                palette=palette, ax=ax, size=size, legend=False,
                alpha=0.6 if add_pointplot else 1)
            if add_pointplot:
                sns.pointplot(
                    data=peaks_agg, x='Reward', y=f'{event}_{ch}_mean',
                    hue=hue, palette=palette, ax=ax, markersize=size*2,
                    legend=False, linestyle='none', dodge=True, zorder=3,
                    errorbar=None)
            ax.axhline(y=0, color='k', ls='--')

            if peaks_agg.Reward.nunique() > 1:
                tick_labels = [Data.palettes['reward_pal_labels'][v] for v in ax.get_xticks()]
            else:
                tick_labels = ['' for v in ax.get_xticks()]
            ax.set_xticks(
                ax.get_xticks(), tick_labels,
                rotation=45, ha='right'
                )
            ax.set(xlabel='', title=event, ylabel='session means (z)',
                   ylim=ylim)

            # if pd.api.types.is_numeric_dtype(peaks_agg[hue]) & (peaks_agg[hue].nunique() > 3):
            #     print('true')
            #     avg_plots.convert_leg_to_cbar(fig, ax, cpal=palette)
            subfig.suptitle(Data.hemi_labels.get(ch[-1]), fontsize=12,
                            ha='right')
            sns.despine()
    return fig
