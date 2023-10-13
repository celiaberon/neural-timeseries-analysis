import numpy as np
import pandas as pd


def check_group_size(grouped_data,
                     n_samples: int,
                     min_frac_samples: float=0.5):

    '''
    Confirm each group contains at least X% trials of target sample size.

    Args:
        grouped_data:
            Groupby object containing groups within which sampling will occur.
        n_samples:
            Number of samples desired from each group (w/ or w/o replacement).
        min_frac_samples:
            Minimum accepted proportion of sample size from which sampling is
            permitted.
            Note: if sampling w/o replacement, necessarily equals 1.0.

    Returns:
        AssertionError if number of trials from all groups does not exceed
        given threshold.
    '''

    min_trial_count = round(n_samples * min_frac_samples)
    assert np.all(grouped_data.size() > min_trial_count), f'Sample size > {min_frac_samples} group size'


def resample_and_balance(trial_data: pd.DataFrame,
                         trial_type: str,
                         *,
                         n_samples: int=100,
                         seed: int=0,
                         necessary_cols: list[str]=None,
                         **kwargs) -> pd.DataFrame:

    '''
    Balance trial types based on trial variable, using sampling with
    replacement within each trial type to reach target sample number for each
    group.

    Args:
        trial_data:
            Original trial dataframe with trial type events occurring at
            natural rate.
        trial_type:
            Trial variable within which to balance groups.
        n_samples:
            Number of samples to take (with replacement) for each value
            existing for trial_type variable.
        seed:
            Seed to set for sampling PRNG.
        necessary_cols:
            Any additional columns that must contain data in balanced df.

    Returns:
        balanced_data:
            Trial data containing of length = n_samples x n_groups, where
            n_groups is number of unique values of trial_type.
    '''

    if necessary_cols is None:
        necessary_cols = []

    imbalanced_data = (trial_data
                       .copy()
                       .dropna(subset=[trial_type] + necessary_cols)
                       .groupby(trial_type))

    check_group_size(imbalanced_data, n_samples=n_samples)

    balanced_data = (imbalanced_data
                     .sample(n=n_samples, replace=True, random_state=seed))

    return balanced_data


def subsample_trial_types(trials: pd.DataFrame,
                          task_variable: str,
                          n_samples: int,
                          seed: int=0) -> pd.DataFrame:

    '''
    Sample from each trial type without replacement up to target number of
    trials.

    Args:
        trials:
            Dataframe containing trial level information.
        task_variable:
            Column on which to group and sample trials.
        n_samples:
            Number of trials to sample up to within each unique condition of
            task_variable.

    Returns:
        sampled_trials:
            Subsampled trial table of length 
            N = n_samples x task_variable.nunique()
    '''

    grp_trials = (trials
                  .copy()
                  .reset_index(drop=True)
                  .groupby(task_variable))

    try:
        check_group_size(grp_trials, n_samples=n_samples, min_frac_samples=1.0)
        sampled_trials = grp_trials.sample(n=n_samples, random_state=seed, replace=False)

    except AssertionError:
        print('Under sampling target, using all trials for some groups')
        sampled_trials = pd.DataFrame()
        for _, grp in grp_trials:
            N = min((len(grp), n_samples))
            sampled_trials = pd.concat((sampled_trials, grp.sample(n=N)))              

    return sampled_trials
