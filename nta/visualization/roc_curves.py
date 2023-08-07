import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from nta.features.select_trials import resample_and_balance
from nta.utils import repeat_and_store

sns.set(style='ticks',
        rc={'axes.labelsize':12, 'axes.titlesize':12,
            'savefig.transparent':True, 'legend.frameon':False}
        )

def calc_dprime(dists: np.array) -> float:

    '''
    Calculate index of sensitivity, i.e. the d-prime metric, as:
    d' = |u1 - u2| / sqrt(0.5*(var1 + var2))

    Args:
        dists:
            2xN array containing N measurements for each of 2 mutually
            exclusive classes.

    Returns:
        dprime:
            Sensitivity index of discrimination between two distributions.
    '''

    mu_dists = np.mean(dists, axis=1)
    var_dists = np.var(dists, axis=1)

    dprime_numerator = np.abs(np.diff(mu_dists)[0]) 

    # No assumptions of equal variance: square root of average variance.
    dprime_denominator = np.sqrt(np.mean(var_dists))

    dprime = round(dprime_numerator / dprime_denominator, 2)

    return dprime


# Repeatedly sample with replacement to get bootstrapped distribution of 
# d-prime values.
@repeat_and_store(100) 
def calc_dprime_sample(*,
                       seed: int=None,
                       trials: pd.DataFrame=None,
                       neural_event: str='Consumption_mean',
                       pred_behavior: str='',
                       **kwargs) -> float:

    '''
    Calculate d-prime index for resampled trials balanced with respect to a
    designated trial variable.

    Args:
        seed:
            Seed position for sampling with PRNG.
            Note: often provided by decorator.
        trials:
            Trial data.
        neural_event:
            Column label for trial-based metric of neural activity on which d'
            metric will reflect predictive power.
        pred_behavior:
            Trial variable (defined by binary and mutually exclusive values)
            that is the object of classification from neural_event.

    Returns:
        sample_dprime:
            d' metric from balanced and resampled data.
    '''

    # Sample with replacement to obtain a balanced dataset containing equal
    # number of trials for each value within pred_behavior.
    balanced_data = resample_and_balance(trials,
                                         trial_type=pred_behavior,
                                         necessary_cols=[neural_event],
                                         seed=seed,
                                         **kwargs)
    
    # Store distribution of neural events within each class of pred_behavior.
    neural_dists_by_class = np.zeros((2, len(balanced_data)//2))
    for i, grp in balanced_data.groupby(pred_behavior):
        neural_dists_by_class[i, :] = grp[neural_event].values

    # Calculate dprime between two class-defined distributions of neural
    # measurements.
    sample_dprime = calc_dprime(neural_dists_by_class)

    return sample_dprime


# Repeatedly sample with replacement to get bootstrapped distribution of ROC 
# values.
@repeat_and_store(100)
def calc_roc_sample(*,
                    seed: int=None,
                    trials: pd.DataFrame=None,
                    neural_event: str='Consumption_grnL_mean',
                    pred_behavior: str='',
                    **kwargs) -> tuple[np.array, np.array]:
    
    '''
    Calculate Receiver Operating Characteristic (ROC) curve for resampled
    trials balanced with respect to a designated trial variable. ROC curve
    shows relationship between False Positive Rate (independent variable) and
    True Positive Rate (dependent variable).

    Args:
        seed:
            Seed position for sampling with PRNG.
            Note: often provided by decorator.
        trials:
            Trial data.
        neural_event:
            Column label for trial-based metric of neural activity on which d'
            metric will reflect predictive power.
        pred_behavior:
            Trial variable (defined by binary and mutually exclusive values)
            that is the object of classification from neural_event.

    Returns:
        (mean_fpr, interp_tpr):
            Array of (false positive rates, true positive rates) making up x
            and y values of ROC curve.
    '''

    # Sample with replacement to obtain a balanced dataset containing equal
    # number of trials for each value within pred_behavior.
    balanced_data = resample_and_balance(trials,
                                         trial_type=pred_behavior,
                                         necessary_cols=[neural_event],
                                         seed=seed,
                                         **kwargs)
    
    # Set FPR against which TPR will be evaluated.
    mean_fpr = np.linspace(-1, 1, 100)

    # Calculate FPR and TPR predicting positive cases (=1) of behavior event
    # using values of neural measurements. 
    fpr, tpr, _ = metrics.roc_curve(balanced_data[pred_behavior]==1,
                                    balanced_data[neural_event])
    
    # Interpolate TPR into fixed FPR steps.
    interp_tpr = np.interp(mean_fpr, fpr, tpr)

    return (mean_fpr, interp_tpr)


def plot_roc_with_auc(fpr: np.array,
                      tprs: list[np.array],
                      ax=None,
                      palette: dict[str, np.array]=None,
                      label: str=None,
                      text_offset: int|float=0,
                      plot_auc: bool=True,
                      **kwargs):
    
    '''
    Plot ROC curve as True Positive Rate vs. False Positive Rate.

    Args:
        fpr:
            Array of False Positive Rates as x-variable.
            Note: FPR given at fixed positions is identical across samples, 
            so single array is sufficient for multiple samples.
        tprs:
            Array or list of arrays from multiple samples, containing True
            Positive Rate.
        ax:
            Matplotlib axis object on which to plot ROC curve.
        palette:
            Dictionary as {class label: color RGB sequence} for assigning
            colors.
        label:
            Class label as key for `palette`.
        text_offset:
            y-positional offest for text annotation. Necessary when plotting
            multiple traces on single `ax` with `plot_auc`=TRUE.
        plot_auc:
            Whether to include area under the curve (AUC) metric on plot.

    Returns:
        ax:
            Populated matplotlib axis object.
    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5,2.5))
    if palette is None:
        palette = {}

    color = palette.get(label, 'k')

    # Calculate mean TPR across bootstrapped samples.
    N = len(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    sem_tpr = np.std(tprs, axis=0) / np.sqrt(N)

    # Lineplot representing bootstrapped mean TPR across fixed FPR values.
    ax.plot(fpr, mean_tpr, label=label, lw=0.6, color=color)

    # Shading representing SEM from boostrapped mean for TPR.
    ax.fill_between(fpr, y1=mean_tpr+sem_tpr, y2=mean_tpr-sem_tpr, alpha=0.5,
                    color=color)
    
    if plot_auc:
        AUC = metrics.auc(x=fpr, y=mean_tpr)
        custom_label = f'{label}={round(AUC, 2)}'
        ax.text(x=0.5, y=0.1+text_offset, s=custom_label, color=color,
                size=12)

    ax.plot([0,1], [0,1], color='k', ls='--', lw=0.5)
    ax.set(xlabel='False Positive Rate',
           xticks=[0, 0.5, 1],
           xlim=(-0.02, 1),
           ylabel='True Positive Rate',
           yticks=[0,0.5, 1],
           ylim=(-0.02, 1.02),
           )
    
    return ax


def multiclass_roc_curves(trials: pd.DataFrame,
                          neural_event: str='Consumption_grnL_mean',
                          pred_behavior: str='',
                          trial_variable: str='Reward',
                          **kwargs):

    '''
    Plot multiple ROC curves on same axes, each conditioned on single class
    within a given trial variable.

    Args:
        trials:
            Trial data.
        neural_event:
            Column label for trial-based metric of neural activity on which d'
            metric will reflect predictive power.
        pred_behavior:
            Trial variable (defined by binary and mutually exclusive values)
            that is the object of classification from neural_event.
        trial_variable:
            Trial variable on which to condition each ROC curve.
        
    Returns:
        ax:
            Matplotlib axes object containing and ROC curve for each unique
            class within `trial_variable` of `trials`.
    '''

    ax=None
    for i, (label, trial_type) in enumerate(trials.groupby(trial_variable, dropna=True)):

        # Bootstrap (FPR, TPR) for each trial type defined by trial_variable.
        bootstrapped_rocs = calc_roc_sample(trials=trial_type,
                                            neural_event=neural_event,
                                            pred_behavior=pred_behavior,
                                            **kwargs)        
        # Unpack bootstrapped arrays into their respective lists.
        tprs = []
        [tprs.append(tpr) for _, tpr,  in bootstrapped_rocs]
        
        # All FPRs across samples are identical, so onkly need single array.
        fpr = bootstrapped_rocs[0][0]

        text_offset = i * (1/6)
        ax = plot_roc_with_auc(fpr, tprs, label=label, ax=ax,
                               text_offset=text_offset, **kwargs)

    sns.despine()
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1,1), loc='upper left', title=trial_variable)
    
    return ax
