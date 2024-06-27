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

    if x_col != plot_func_kws.get('hue', x_col):
        ax = hatch_and_recolor(ax, peaks, x_col, **kwargs)

    if kwargs.get('save'):
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
            print('defaulting to autofit axis limits')
    sns.despine()

    return fig, ax


def hatch_and_recolor(axs, peaks, x_col, hatch_labels={0: 'Stay', 1: 'Switch'},
                      **kwargs):

    import matplotlib.patches as mpatches

    hatch_styles = {0: '', 1: '//', 2: '+', 3: '*'}

    def cpal_key(i, nbars):
        return (i % nbars) - (nbars // 2) + (i % nbars >= (nbars // 2))

    cpal = set_color_palette(peaks, x_col, palette=peaks[x_col].nunique())

    nbars = len(cpal)
    for ax_ in [axs['Cue'], axs['reward'], axs['no reward']]:
        for i, bar in enumerate(ax_.patches):
            bar.set_hatch(hatch_styles[i // nbars])
            bar.set_facecolor(cpal[cpal_key(i, nbars)])
            bar.set_alpha(0.8)
            bar.set_edgecolor('k')

    legend_elements = [mpatches.Patch(facecolor='white', edgecolor='k',
                                      label=hatch_labels.get(k), hatch=hatch_styles.get(k))
                       for k in hatch_labels]
    axs['reward'].legend(handles=legend_elements, bbox_to_anchor=(1, 1))

    return axs


# def plot_and_recolor(plot_func,
#                      data,
#                      ax,
#                      color_palette,
#                      y_col: str = '',
#                      cols: tuple[str, str] = None,
#                      **kwargs):

#     hatch_styles = {0: '', 1: '//', 2: '+', 3: '*'}

#     col1, col2 = cols
#     # order = list(color_palette)
#     order = [val for val in color_palette if val in data[col1].unique()]

#     hue_factor = data[col2].dropna().nunique()

#     # reduce order to only those needed
#     plot_func(data=data, x=col1, hue=col2, y=y_col, ax=ax, showfliers=False,
#               order=order)

#     for i, box in enumerate(ax.artists):
#         key = order[i // hue_factor]
#         box.set_facecolor(color_palette[key])
#         box.set_hatch(hatch_styles[i % hue_factor])

#     return ax


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
    peaks_.groupby(x_col)[y_col].agg([get_num_outliers])
    peaks_ = (peaks_.loc[peaks_.groupby(x_col, group_keys=False)[y_col]
                               .apply(not_outlier)]
                    .reset_index(drop=True))

    return peaks_


def config_plot(ax, channel, metrics, col_id: str = '', **kwargs):

    ylab = channel.split('_')[0]

    try:
        ax['reward'].set(xlabel='')
        ticks = [int(ax['Cue'].get_xticks()[0]), int(ax['Cue'].get_xticks()[-1])]

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
