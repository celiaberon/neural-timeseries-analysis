import numpy as np
import pandas as pd


def check_group_size(grouped_data,
                     n_samples: int,
                     min_frac_samples: float = 0.5):

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
    assert np.all(grouped_data.size() > min_trial_count), (
        f'Sample size > {min_frac_samples} group size')


def resample_and_balance(trial_data: pd.DataFrame,
                         trial_type: str,
                         *,
                         n_samples: int = 100,
                         seed: int = 0,
                         necessary_cols: list[str] = None,
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
                          seed: int = 0) -> pd.DataFrame:

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
        sampled_trials = grp_trials.sample(n=n_samples, random_state=seed,
                                           replace=False)

    except AssertionError:
        print('Under sampling target, using all trials for some groups')
        sampled_trials = pd.DataFrame()
        for _, grp in grp_trials:
            N = min((len(grp), n_samples))
            sampled_trials = pd.concat((sampled_trials, grp.sample(n=N)))

    return sampled_trials


def match_trial_ids(*args, allow_discontinuity: bool = False):

    if len(args) == 2:
        included_trials = (set(args[0].nTrial.astype('int').values)
                           .intersection(args[1].nTrial))
    if len(args) > 2:
        included_trials = (set(args[0].nTrial.astype('int').values)
                           .intersection(*[set(arg.nTrial.values)
                                           for arg in args[1:]]))

    if not allow_discontinuity:
        orig_set = included_trials
        included_trials = np.arange(min(included_trials),
                                    max(included_trials) + 1).astype('int')

        if len(included_trials) - len(orig_set) > 0:
            print(f'filled in {len(included_trials) - len(orig_set)} trials '
                  'for continuity')

    matched_args = []
    for arg in args:

        arg_ = (arg.loc[arg.nTrial.isin(included_trials)].copy()
                .reset_index(drop=True))
        matched_args.append(arg_)

    return matched_args


def clean_data(trials: pd.DataFrame,
               ts: pd.DataFrame = None,
               allow_discontinuity: bool = False,
               drop_penalties: bool = False,
               drop_timeout: bool = False,
               clip_blocks: tuple[int, int] = (2, 15),
               store_results: bool = True,
               **kwargs) -> dict:

    '''
    Quality control within sessions to match timeseries and trial dataframes
    -- remove high timeout blocks

    Args:
        ts:
            Timeseries containing states and neural data for a single
            session.
        trials:
            Trial data for single session.
        allow_discontinuity:
            Whether or not continuous trial structure can be broken. If false,
            only trials from beginning/end can be excluded.
        drop_penalties:
            Whether or not to exclude trials with ENL and Cue penalties.
        drop_timeouts:
            Whether or not to exclude trials with selection timeouts.
        clip_blocks:
            Minumum and maximum block ID to include.
        store_results:
            Store a log (dictionary) of trials dropped during cleaning.

    Returns:
        trials_, ts_:
            Original dataframes with dropped trials. Should have identical
            trial IDs.
    '''

    trials_ = trials.copy()
    include_ts = isinstance(ts, pd.DataFrame)
    if include_ts:
        ts_ = ts.copy()
    results = {}

    if allow_discontinuity:
        # flagged_trials stays as is
        if drop_penalties:  # take only trials without enl penalty
            ntrials = len(trials_)
            trials_ = trials_.query('enlp_trial == False')
            enlp_dropped = ntrials - len(trials_)
            print(f'{enlp_dropped = }')

            ntrials = len(trials_)
            trials_ = trials_.query('n_Cue == 1')
            cuep_dropped = ntrials - len(trials_)
            print(f'{cuep_dropped=}')
            if store_results:
                results['ENLP_dropped'] = enlp_dropped
                results['CueP_dropped'] = cuep_dropped

        if drop_timeout:  # take only trials with choice lick
            ntrials = len(trials_)
            trials_ = trials_.query('timeout == False')
            timeouts_dropped = ntrials - len(trials_)
            print(f'{timeouts_dropped = }')
            if store_results:
                results['timeouts_dropped'] = timeouts_dropped

        trials_ = trials_.query('flag_block == 0').copy()
        trials_['continuity_broken'] = True
        if include_ts:
            ts_['continuity_broken'] = True

    # these blocks occur too infrequently -- less than 10 sessions
    min_block, max_block = clip_blocks
    trials_ = trials_.query('iBlock.between(@min_block, @max_block)')
    print(f'trimmed data between block {min_block} and {max_block}')

    if include_ts:
        trials_, ts_ = match_trial_ids(trials_, ts_, allow_discontinuity=True)

    if store_results:
        if include_ts:
            return trials_, ts_, results
        return trials_, results
    if include_ts:
        return trials_, ts_
    return trials_
