import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import pandas as pd
from nta.features.select_trials import resample_and_balance

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


# def calc_dprime_bootstrap(trials: pd.DataFrame,
#                           neural_event: str,
#                           pred_behavior: str,
#                           n_reps: int):

#     '''
    
#     print d-prime for two distributions predicting binary event (i.e., next switch) '''

#     dprimes=[]
#     for rep in range(n_reps):
#         rebalanced_data = resample_and_balance(trials, neural_event, pred_behavior, seed=rep)
        
#         neural_dists_by_class = np.zeros((2, len(rebalanced_data)//2))
#         for i, grp in rebalanced_data.groupby(pred_behavior):
#             neural_dists_by_class[i, :] = grp[neural_event].values

#         dprimes.append(calc_dprime(neural_dists_by_class))
                       
#     mean_dprime = round(np.mean(dprimes), 2)
#     # sem_dprime = round(np.std(dprimes) / np.sqrt(n_reps), 2)
#     print(f'{mean_dprime=}')

#     return dprimes

import functools
def repeat_and_store(num_reps):
    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            multi_output=[]
            for rep in range(num_reps):
                single_output = func(seed=rep, *args, **kwargs)
                multi_output.append(single_output)
            return multi_output
        return wrapper_repeat
    return decorator_repeat

@repeat_and_store(100)
def calc_dprime_sample(*,
                       seed: int=None,
                       trials: pd.DataFrame=None,
                       neural_event: str='Consumption_mean',
                       pred_behavior: str='',
                       **kwargs) -> float:

    '''
    print d-prime for two distributions predicting binary event (i.e., next switch) '''

    balanced_data = resample_and_balance(trials,
                                         trial_type=pred_behavior,
                                         necessary_cols=[neural_event],
                                         seed=seed,
                                         **kwargs)
    
    neural_dists_by_class = np.zeros((2, len(balanced_data)//2))
    for i, grp in balanced_data.groupby(pred_behavior):
        neural_dists_by_class[i, :] = grp[neural_event].values

    sample_dprime = calc_dprime(neural_dists_by_class)

    return sample_dprime

@repeat_and_store(100)
def calc_roc_sample(*,
                    seed: int=None,
                    trials: pd.DataFrame=None,
                    neural_event: str='Consumption_mean',
                    pred_behavior: str='',
                    **kwargs) -> tuple[np.array, np.array]:

    balanced_data = resample_and_balance(trials,
                                         trial_type=pred_behavior,
                                         necessary_cols=[neural_event],
                                         seed=seed,
                                         **kwargs)
    mean_fpr = np.linspace(-1, 1, 100)
    fpr, tpr, _ = metrics.roc_curve(balanced_data[pred_behavior]==1,
                                    balanced_data[neural_event])
    interp_tpr = np.interp(mean_fpr, fpr, tpr)

    return (mean_fpr, interp_tpr)


# def calc_roc_bootstrap(trials: pd.DataFrame,
#                        neural_event: str,
#                        pred_behavior: str,
#                        n_reps: int):

#     mean_fpr = np.linspace(-1, 1, 100)
#     tprs = []

#     for rep in range(n_reps):

#         balanced_data = resample_and_balance(trials,
#                                              trial_type=pred_behavior,
#                                              necessary_cols=[neural_event],
#                                              n_samples=20,
#                                              seed=rep)
#         fpr, tpr, _ = metrics.roc_curve(balanced_data[pred_behavior]==1,
#                                         balanced_data[neural_event])
#         interp_tpr = np.interp(mean_fpr, fpr, tpr)
#         tprs.append(interp_tpr)

#     return mean_fpr, tprs


def plot_roc_with_auc(fpr: np.array,
                      tprs: list[np.array],
                      ax=None,
                      palette=None,
                      label: str=None,
                      text_offset: int|float=0,
                      plot_auc: bool=True,
                      **kwargs):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(2.5,2.5))
    if palette is None:
        palette = {}

    color = palette.get(label, 'k')

    N = len(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    sem_tpr = np.std(tprs, axis=0) / np.sqrt(N)

    ax.plot(fpr, mean_tpr, label=label, lw=0.6, color=color)
    ax.fill_between(fpr, y1=mean_tpr+sem_tpr, y2=mean_tpr-sem_tpr, alpha=0.5, color=color)
    
    if plot_auc:
        AUC = metrics.auc(x=fpr, y=mean_tpr)
        custom_label = f'{label}={round(AUC, 2)}'
        ax.text(x=0.5, y=0.1+text_offset, s=custom_label, color=color, size=12)

    ax.plot([0,1], [0,1], color='k', ls='--')
    ax.set(xlabel='False Positive Rate',
           xticks=[0, 0.5, 1],
           xlim=(-0.02,1),
           ylabel='True Positive Rate',
           yticks=[0,0.5, 1]
           )
    
    return ax


def multiclass_roc_curves(trials: pd.DataFrame,
                      neural_event: str='Consumption_mean',
                      pred_behavior: str='+1sw',
                      trial_variable: str='Reward',
                    #   n_reps: int=1,
                      **kwargs):

    ax=None
    for i, (label, trial_type) in enumerate(trials.groupby(trial_variable, dropna=True)):

        bootstrapped_rocs = calc_roc_sample(trials=trial_type,
                                            neural_event=neural_event,
                                            pred_behavior=pred_behavior,
                                            **kwargs)        
        tprs = []
        [tprs.append(tpr) for _, tpr,  in bootstrapped_rocs]
        fpr = bootstrapped_rocs[0][0]

        text_offset = i * (1/6)
        ax = plot_roc_with_auc(fpr, tprs, label=label, ax=ax, text_offset=text_offset, **kwargs)

    sns.despine()
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1,1), loc='upper left', title=trial_variable)
    
    return ax
