#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:30:18 2022

@author: celiaberon
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nta.utils import save_plot_metadata


def initialize_peak_fig(states,
                        *,
                        figsize: tuple[int | float, int | float] = (3.5, 4.),
                        axes_style: str = None,
                        flatten_layout: bool = False,
                        **mosaic_kwargs):

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
            print('defaulting to autofit axis limits')
    sns.despine()

    return fig, ax


def plot_and_recolor(plot_func,
                     data,
                     ax,
                     color_palette,
                     y_col: str = '',
                     cols: tuple[str, str] = None,
                     **kwargs):

    hatch_styles = {0: '', 1: '//', 2: '+', 3: '*'}

    col1, col2 = cols
    # order = list(color_palette)
    order = [val for val in color_palette if val in data[col1].unique()]

    hue_factor = data[col2].dropna().nunique()

    # reduce order to only those needed
    plot_func(data=data, x=col1, hue=col2, y=y_col, ax=ax, showfliers=False,
              order=order)

    for i, box in enumerate(ax.artists):
        key = order[i // hue_factor]
        box.set_facecolor(color_palette[key])
        box.set_hatch(hatch_styles[i % hue_factor])

    return ax


def set_color_palette(peaks, x_col, palette=None, hue=None, **kwargs):

    if hue is None:
        hue_col = x_col
    else:
        hue_col = hue

    labels = np.sort(peaks[hue_col].dropna().unique())
    match palette:
        case dict():
            labels = palette.keys()
        case str():
            palette = sns.color_palette(palette, n_colors=len(labels))
        case None:
            palette = None
        case list():
            palette = dict(zip(labels, palette))
        case _:
            palette = sns.color_palette('RdBu', n_colors=len(labels))
            palette = dict(zip(labels, palette))

    return palette


def exclude_outliers(peaks, x_col, y_col):

    def get_num_outliers(column):

        outliers = 1 - not_outlier(column)
        print(f'{round(np.mean(outliers),3)} points removed as outliers')

    def not_outlier(column):

        q1 = np.percentile(column, 25)
        q3 = np.percentile(column, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return (column > lower_bound) & (column < upper_bound)

    peaks_ = peaks.dropna(subset=[y_col]).copy()
    peaks_.groupby(x_col, observed=False)[y_col].agg([get_num_outliers])
    peaks_ = (peaks_.loc[peaks_.groupby(x_col, group_keys=False, observed=False)[y_col]
                               .apply(not_outlier)]
                    .reset_index(drop=True))

    return peaks_


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
        if not ignore_reward and 'Consumption' in states:
            metrics['no reward'] = metrics['Consumption']
            metrics['reward'] = metrics['Consumption']

    ax_modifiers = {0: 'no ', 1: ''}
    for state in states:
        if (state != 'Consumption') or ignore_reward:
            metric = metrics[state]
            y_col = f'{state}_{channel}_{metric}'

            # Drop outliers but take a look at how many there are first.
            if not show_outliers:
                peaks = exclude_outliers(peaks, x_col, y_col)
            if (state != 'Consumption') or ignore_reward:
                plot_func(data=peaks, x=x_col, y=y_col, ax=ax[state],
                          **plot_func_kws)
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

    if kwargs.get('save'):
        fig.savefig(kwargs.get('fname'), dpi=200, bbox_inches='tight')

    return fig, ax


def config_plot(ax, channel, metrics, col_id: str = '', **kwargs):

    ylab = channel.split('_')[0]
    # [ax_.set(ylabel=f'{metrics[k]} {ylab}') for k, ax_ in ax.items()]

    try:
        ax['reward'].set(xlabel='')
        ticks = [ax['Cue'].get_xticks()[0], ax['Cue'].get_xticks()[-1]]
        # tick_labels = [int(i) for i in ticks]

        ax['Cue'].set(xlabel=col_id, ylabel=ylab, xticks=ticks)#, xticklabels=tick_labels)
        ax['reward'].set(xlabel='', ylabel=ylab, xticks=ticks)#, xticklabels=tick_labels)
        ax['no reward'].set(xlabel=col_id, yticks=[0, -1, -2], xticks=ticks,
                            ylim=(-3, 0), ylabel=ylab)
        ax['no reward'].axhline(y=0, color='k', lw=2)
    except KeyError:
        pass

    first_ax = list(ax.keys())[0]
    if np.any(ax[first_ax].get_legend_handles_labels()):
        [ax_.legend().remove() for _, ax_ in ax.items()]
    plt.tight_layout()

    return ax
