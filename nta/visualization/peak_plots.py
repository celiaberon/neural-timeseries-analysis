#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:30:18 2022

@author: celiaberon
"""

import os
from math import modf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import label_hemi, save_plot_metadata
from ..visualization import avg_plots


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
                       existing_ax: dict=None,
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
    
    if existing_ax is not None:
        ax = existing_ax
        fig = next(iter(ax.values())).figure
    else:
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
            pass
        else:
            fig.savefig(kwargs.get('fname'), dpi=200, bbox_inches='tight')

    return fig, ax


def initialize_peak_fig(states,
                        *,
                        figsize: tuple[int | float, int | float] = (2.5, 2.5),
                        axes_style: str = None,
                        flatten_layout: bool = False,
                        **mosaic_kwargs,
                        ):

    sns.set_theme(style='ticks',
        font_scale=1.0,
        rc={'axes.labelsize': 10,
            'axes.titlesize': 10,
            'savefig.transparent': True,
            'legend.title_fontsize': 10,
            'legend.fontsize': 10,
            'legend.borderpad': 0.2,
            'figure.titlesize': 10,
            'figure.subplot.wspace': 0.1,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.frameon': False,
            'hatch.linewidth': 1.0,
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
    if figsize == (2.5, 2.5) and (states or flatten_layout):
        figsize = (1.2 * len(layout[0]), 2.5)

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
        ax['no reward'].set(title='Unrewarded', ylabel='')
        ax['reward'].set(title='Rewarded', xlabel='', ylabel='')
        ax['Cue'].set(title='Cue', ylabel='z-score')
    else:
        for state in states:
            ax[state].set(title=state)

    match axes_style:
        case 'constant':
            ax['Cue'].set(ylim=(-2.5, 4.5), ylabel='z-score')
            ax['reward'].set(ylim=(-2.5, 4.5))
            ax['no reward'].set(ylim=(-2.5, 4.5))
        case 'pos-neg':
            ax['Cue'].set(ylim=(0, 4.5), ylabel='z-score')
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
    if ylab == 'z':
        ylab = 'z-score'

    try:
        ax['reward'].set(xlabel='')
        ticks = [int(ax['Cue'].get_xticks()[0]),
                 int(ax['Cue'].get_xticks()[-1])]

        ax['Cue'].set(xlabel=col_id, ylabel=ylab, xticks=ticks)
        ax['reward'].set(xlabel='', ylabel='', xticks=ticks)
        ax['no reward'].set(xlabel=col_id, yticks=[0, -1, -2], xticks=ticks,
                            ylim=(-3, 0), ylabel='')
        ax['no reward'].axhline(y=0, color='k', lw=2)
    except KeyError:
        pass

    first_ax = list(ax.keys())[0]
    if np.any(ax[first_ax].get_legend_handles_labels()):
        [ax_.legend().remove() for _, ax_ in ax.items()]

    return ax


def plot_correlation(r, ax=None, hue=None,
                     palette=None, legend=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if palette is None:
        palette = sns.color_palette('deep', n_colors=len(r[hue].unique()))
    if 'hatch' not in r.columns:
        r['hatch'] = ' '

    # Calculate number of hue levels and total bars per x position
    n_hue_levels = len(r[hue].unique()) if hue else 1
    n_bars_per_hue = r.get('hatch').nunique()  # number of hatch styles
    total_width = 0.8   # total width available for all bars at each x position
    bar_width = total_width / (n_hue_levels * n_bars_per_hue)
    
    for i, channel in enumerate(r['channel'].unique()):
        channel_data = r.query('channel == @channel')
        
        # For each hue value
        for j, hue_val in enumerate(channel_data[hue].unique()):
            hue_data = channel_data.query(f'{hue} == {hue_val}')
            
            # Base position for this hue group
            hue_center = i + (j - (n_hue_levels-1)/2) * (bar_width * n_bars_per_hue)
            
            # Plot non-hatched bar
            non_hatched = hue_data.query('hatch == " "')
            if not non_hatched.empty:
                ax.bar(hue_center - bar_width/2, 
                      non_hatched['r'].values[0],
                      width=bar_width,
                      color=palette[hue_val] if isinstance(palette, dict) else palette[j])
            
            # Plot hatched bars
            for hatch in hue_data['hatch'].unique():
                if hatch == ' ':
                    continue
                hatched = hue_data.query('hatch == @hatch')
                if not hatched.empty:
                    ax.bar(hue_center + bar_width/2,
                        hatched['r'].values[0],
                        width=bar_width,
                        color=palette[hue_val] if isinstance(palette, dict) else palette[j],
                        hatch=hatch)

    ax.axhline(y=0, color='k')
    ax.set(ylim=(-1, 1),
           xlabel='hemisphere')
    ax.set_ylabel('correlation', labelpad=0)

    import matplotlib.patches as mpatches
    
    if r.hatch.nunique() == 1:
        legend_elements = ax.legend().legendHandles
    else:
        # Create legend elements for each combination of hue and hatch
        legend_elements = []
        hue_values = r[hue].unique()
        hatch_values = dict(zip(r['trial_type'].unique(), r['hatch'].unique()))
        for j, hue_val in enumerate(hue_values):
            color = palette[hue_val] if isinstance(palette, dict) else palette[j]
            for hatch_label, hatch in hatch_values.items():
                patch = mpatches.Patch(
                    facecolor=color,
                    hatch=hatch,
                    label=f'{hue_val} - {hatch_label}'
                )
                legend_elements.append(patch)

    if legend:
        ax.legend(handles=legend_elements, bbox_to_anchor=(1., 1))
    sns.despine()
    
    # Set x-ticks at channel positions
    ax.set_xticks(range(len(r['channel'].unique())))
    ax.set_xticklabels(r['channel'].unique())
    
    return ax

def calc_grouped_corr(trials: pd.DataFrame,
                      col0: str,
                      col1: str,
                      grouping_variable: str) -> pd.DataFrame:

    '''
    Calculate the pairwise correlation coefficients between two columns
    within groups defined by a grouping variable in a DataFrame.

    Args:
        trials
            The input DataFrame containing the data.
        col0, col1
            The names of the columns for which to calculate the correlation.
        grouping_variable
            The column name used to group the data before calculating correlations.

    Returns:
        A DataFrame containing the correlation coefficients (`r`) for each
        group and the corresponding value of the grouping variable. Each row
        corresponds to a unique group, with the correlation coefficient and
        the group value.

    Notes:
    - The function drops missing values (`NaN`) in the grouping variable before
      processing.
    - Correlation is computed using the `.corr()` method of pandas, which
      defaults to Pearson correlation.
    - Assumes that `trials` contains numeric data in `col0` and `col1`.
    '''

    rs = {}
    for i, val in enumerate(trials[grouping_variable].dropna().unique()):
        rs[i] = {
            'r': (trials
                  .groupby(grouping_variable, observed=True)[[col0, col1]]
                  .corr()
                  .reset_index()
                  .query(f'level_1 == @col1 & {grouping_variable} == @val', engine='python')[col0].item()),
            grouping_variable: val
        }
    return pd.DataFrame(rs).T


def plot_swarm_and_point(peaks_agg, Data, hue, palette,
                         size=3, add_pointplot=False, ylim=(-1, 7.5),
                         events=['Cue', 'Consumption'], **kwargs):

    n_events = len(events)
    x_col = kwargs.get('x', 'Reward')
    scale_factor = 1.2 if peaks_agg[x_col].nunique() == 1 else 0.5
    width = (peaks_agg[x_col].nunique() + scale_factor) * n_events * np.max((len(Data.sig_channels)-1.5, 1)) * 1.
    fig = plt.figure(figsize=(width, 2), layout='constrained')
    subfigs = fig.subfigures(ncols=len(Data.sig_channels), wspace=0.2)
    if len(Data.sig_channels) == 1:
        subfigs = [subfigs]

    for subfig, ch in zip(subfigs, Data.sig_channels):
        axs = subfig.subplots(ncols=n_events, sharey=True)

        axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]
        for ax, event in zip(axs, events):
            y_col = f'{event}_{ch}_{kwargs.get("y_suffix", "mean")}'
            sns.swarmplot(
                data=peaks_agg, x=x_col, y=y_col,
                hue=hue, palette=palette, ax=ax, size=size,  # legend=False,
                alpha=0.6 if add_pointplot else 1)
            if add_pointplot:
                sns.pointplot(
                    data=peaks_agg, x=x_col, y=y_col,
                    hue=hue, palette=palette, ax=ax, markersize=size*2,
                    legend=False, linestyle='none', dodge=True, zorder=3,
                    errorbar=None)
            ax.axhline(y=0, color='k', lw=0.5)

            if x_col != 'Reward':
                tick_labels = ax.get_xticklabels()
            elif peaks_agg['Reward'].nunique() > 1:
                tick_labels = [Data.palettes['reward_pal_labels_short'][v] for v in ax.get_xticks()]
                ax.set(xlabel='Outcome')
                ax.set_xticks(
                ax.get_xticks(), tick_labels,
                # rotation=45, ha='right'
                )
            else:
                # tick_labels = ['' for v in ax.get_xticks()]
                ax.set_xticks([])
                ax.set_xlabel('')
            
            ax.set(title=event, ylabel='session means (z)',
                   ylim=ylim)

            if pd.api.types.is_numeric_dtype(peaks_agg[hue]) & (peaks_agg[hue].nunique() > 3):
                avg_plots.convert_leg_to_cbar(fig, ax, cpal=palette)
            else:
                ax.legend().remove()
            subfig.suptitle(label_hemi(ch, Data.sig_channels), fontsize=11,
                            ha='right')
            sns.despine()
    return fig


def plot_baseline_vs_TENL(Data, trials, base_args, args, plot_id):

    '''Convenient function for common peak plot'''
    base_args['plot_func_kws'].update(args['plot_func_kws'])
    fig = plt.figure(figsize=(len(Data.sig_channels)*2.15, 1.3), layout='constrained')
    subfigs = fig.subfigures(ncols=len(Data.sig_channels)+1)

    corrs = pd.DataFrame()
    for subfig, ch in zip(subfigs, Data.sig_channels):
        base_args['mosaic_kws'] = {'subfig': subfig}
        if isinstance(base_args['style'], str):
            style = base_args['style']

            for i, (ls, val) in enumerate(zip(['--', '-'], [1, 0])):
                base_args['plot_func_kws']['ls'] = ls
                _, ax = plot_peaks_wrapper(
                    trials.query(f'{style} == {val}'),
                    channel=ch,
                    fname=os.path.join(Data.save_path, f'{plot_id}_{ch}.svg'),
                    **base_args,
                    existing_ax=ax if i>0 else None,  # Pass existing axes
                )
        else:
            _, ax = plot_peaks_wrapper(
                trials,
                channel=ch,
                fname=os.path.join(Data.save_path, f'{plot_id}_{ch}.svg'),
                **base_args
            )
        ax['Cue'].set_xticks([0, 2, 4], [1, 1.5, 2])
        ax['Cue'].set(ylim=(-2, 1), ylabel='avg z-score', title='', xlabel='ENL (s)')

        # Annotate subfigure with hemisphere label
        hemi_label = label_hemi(ch, Data.sig_channels)
        subfig.suptitle(hemi_label, fontsize=11, y=1.1)

        if isinstance(style, str):
            for i, (hatch, val) in enumerate(zip(['////', ' '], [1, 0])):
                # Calculate correlation for the channel and trial type
                channel_corr = calc_grouped_corr(
                    trials.query(f'{style} == {val}'),
                    grouping_variable=args['plot_func_kws']['hue'],
                    col0=base_args['x_col'],
                    col1=f'Cue_{ch}_offset'
                )
                channel_corr['channel'] = hemi_label.split()[0]
                channel_corr['trial_type'] = val
                channel_corr['hatch'] = hatch
                corrs = pd.concat((corrs, channel_corr)).reset_index(drop=True)
        
        else:
            # Calculate correlation for the channel
            channel_corr = calc_grouped_corr(
                trials,
                grouping_variable=args['plot_func_kws']['hue'],
                col0=base_args['x_col'],
                col1=f'Cue_{ch}_offset'
            )
            channel_corr['channel'] = hemi_label.split()[0]
            corrs = pd.concat((corrs, channel_corr)).reset_index(drop=True)

    plot_correlation(corrs,
                     ax=subfigs[-1].subplots(), legend=True,
                     **args['plot_func_kws'])
    return fig, corrs
