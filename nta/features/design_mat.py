import pandas as pd

from nta.events.quantify import create_combo_col

'''reference new_make_design_mat.ipynb for function/build testing grounds'''


def classify_lick_state(timeseries: pd.DataFrame,
                        states: str | set[str]) -> pd.DataFrame:

    '''
    Define discrete categories of licks such that each lick is mutually
    exclusively labeled by its state. New columns populated by interaction
    between (1) state, and (2) presence of a lick.

    Args:
        timeseries:
            Timeseries form of behavior/neural data.
        states:
            List of states for lick classification.
            Note: states not listed result in licks discarded from matrix.
    Returns:
        ts_:
            Copied of original timeseries containing N=len(state) new columns.
        lick_cols:
            List of columns containing classified licks.
    '''

    if isinstance(states, str):
        states = list(states)

    lick_states = [s for s in states if s not in ['ENL', 'Cue']]

    ts_ = timeseries.copy()
    lick_cols = []
    for s in lick_states:

        # new column containing lick events within each state
        lick_cols.append(f'{s[:3].lower()}_lick')
        ts_[lick_cols[-1]] = ts_[s] * ts_['Lick']

    return ts_, lick_cols


def pull_lick_from_bout(timeseries: pd.DataFrame,
                        lick_pos: list[int],
                        state: str = 'Consumption',
                        only_nth_lick: bool = False) -> pd.DataFrame:

    '''
    Seperate single lick from lick bout (typically first lick).

    Args:
        timeseries:
            Timeseries form of behavior/neural data. Expects licks labeled in
            states.
        lick_pos:
            Starting at 1 for column labeling (will adjust for 0 indexing)
        state:
            Will only apply function to licks within designated state (might
            only ever be "Consumption" because assumes bout).
        only_nth_lick:
            If True, drop original column containing rest of lick bout. False,
            original column contains remaining licks only (no duplicates
            across columns).

    Returns:
        ts_:
            Copy of timeseries, with new columns for each lick in `lick_pos`
            (N=len(lick_pos)), minus 1 if keep_only_nth_lick is True (original
            lick column, most likely "con_lick").
    '''

    # Reference columns generated by `classify_lick_state`.
    state_licks = f'{state.lower()[:3]}_lick'

    if len(lick_pos) > 1:
        # Reverse sort licks to pull from later to earlier licks. Avoids
        # errors in counting after dropping licks.
        lick_pos = sorted(lick_pos)[::-1]

    ts_ = timeseries.copy()

    # Iterate over licks to isolate, starting with later licks first.
    for nth_lick in lick_pos:
        new_col = f'{state_licks}_{nth_lick}'

        # Get index for nth bout lick for designated state.
        lick_subset = ts_.loc[ts_[state_licks] == 1]
        nth_lick_idcs = (lick_subset
                         .groupby('nTrial', as_index=False)
                         .nth(nth_lick - 1)
                         .index)

        # Create column containing nth lick events only.
        ts_[new_col] = 0  # zero outside of licks
        ts_.loc[nth_lick_idcs, new_col] = 1

        # Reset these licks to zero in original column.
        ts_.loc[nth_lick_idcs, state_licks] = 0

        ts_[new_col] = ts_[new_col].astype('int8')

    if only_nth_lick:
        ts_ = ts_.drop(columns=[state_licks])  # drop original column

    return ts_


def event_interactions_dummies(timeseries: pd.DataFrame,
                               trials: pd.DataFrame,
                               states: str | list[str],
                               trial_type: str,
                               as_dummy: bool = True,
                               drop_non_interaction: bool = False
                               ) -> pd.DataFrame:

    '''
    Define interaction terms to further classify lick identities based on
    multiple behavior/task variables.

    Args:
        timeseries:
            Timeseries form of behavior/neural data. Expects licks labeled in
            states.
        trials:
            Trial-based behavior/task data that can be mapped to timeseries.
        states:
            List of lick-states to interact with trial variables.
        trial_type:
            Column to interact with lick-states (e.g. 'Reward' or 'seq2').
        as_dummy:
            If True, create dummy variables for each value in `trial_type`
            column; if False (only in case of binary variable), symmetric
            representation for 1s and 0s.
        drop_non_interaction:
            If True, drop original lick column and leave only interaction term.

    Returns:
        t_:
            Copy of timeseries, with new columns for interaction terms.
    '''

    ts_ = timeseries.copy()

    # Filter ts by columns containing lick-states to interact using format
    # from classify_lick_state().
    licks_interact = [state.lower()[:3] for state in states]
    licks_interact = '|'.join(licks_interact)  # joined by | conditional
    ts_ = ts_.filter(regex=(licks_interact))

    # Make two lists - one for interacting, the other to store and add back in
    # after interacting (e.g. nTrial)
    cols_interact = ts_.columns
    cols_hold = [col for col in timeseries.columns if col not in ts_.columns]

    if as_dummy:
        dummies = pd.get_dummies(trials[trial_type], prefix=trial_type)
    else:
        dummies = trials[[trial_type]]  # same form as dummies; idx is nTrial

    for dummy_col in dummies.columns:

        # Timeseries representation of trial variable to interact with
        # actual timeseries events.
        trial_ts = timeseries['nTrial'].map(dummies[dummy_col])

        # {trial_type_class}_{trial_type_dummy}_{orig_lick_column_name}
        root1 = trial_type.lower()[:3]  # trial type class
        root2 = dummy_col.split("_")[-1]  # trial type dummy
        new_cols = [f'{root1}_{root2}_{col}' for col in cols_interact]

        # interact (multiply) dummy variable with all specified lick columns
        ts_[new_cols] = ts_[cols_interact].multiply(trial_ts, axis='index')

    if drop_non_interaction:
        ts_ = ts_.drop(columns=cols_interact)  # drop original lick columns

    # Add columns to final df that weren't interacted but should be kept.
    ts_[cols_hold] = timeseries[cols_hold].copy()

    return ts_


def get_state_onset(timeseries, state: str):

    state_onset_times = (timeseries['nTrial']
                         .map(timeseries.loc[timeseries[state] == 1]
                              .groupby('nTrial')['trial_clock'].first()))

    return state_onset_times


def add_heatmap_columns(timeseries, trials):

    ts_ = timeseries.copy()

    ts_['hm_t_cue_off_to_sel'] = ts_['nTrial'].map(trials['tSelection'])

    # Timer running relative to trial cue onset.
    cue_on_times = get_state_onset(ts_, state='Cue')
    ts_['hm_t_from_cue_on'] = ts_['trial_clock'] - cue_on_times

    # Timer running relative to trial first consumption lick.
    cons_on_times = get_state_onset(ts_, state='Consumption')
    ts_['hm_t_from_cons_on'] = ts_['trial_clock'] - cons_on_times

    # Calculate latency between selection and first consumption licks.
    latency = (ts_
               .groupby('nTrial')
               .agg({'Consumption': sum, 'stateConsumption': sum})
               .diff(axis='columns')
               .rename(columns={'stateConsumption': 't_sel_to_cons'}))
    ts_['hm_t_sel_to_cons'] = (ts_['nTrial']
                               .map(latency['t_sel_to_cons'] * (1000 / 50)))
    ts_['hm_t_cue_off_to_cons'] = (ts_['hm_t_sel_to_cons']
                                   + ts_['hm_t_cue_off_to_sel'])

    # Store additional columns for post-model conditioning.
    hm_cols = ['Reward', 'seq2', 'outcome_seq',  # 'outcome_seq_history', 'h2',
               'seq2_+1switch']  # 'h2_+1sw
    for col in hm_cols:
        ts_[f'hm_{col}'] = ts_['nTrial'].map(trials[col])

    return ts_[[col for col in ts_.columns if str(col).startswith('hm')]]


def label_active_rows(design_matrix: pd.DataFrame,
                      features: list[str] = None):

    '''
    Flag rows with active coefficients for trimming trials down. Note, for
    model selection to keep trial boundaries constant across different feature
    set comparisons, this should be done with full task and lick state
    variables included. For final models, should be done with reduced feature
    set.

    Args:
        design_matrix:
            Timeseries data with columns for independent variables and
            photometry signal.
        features:
            Column headers of features to include in trimming

    Returns:
        - binary list of len(design_matrix) with 1s where any timepoint in
          features columns is 1
    '''

    all_cols = design_matrix.columns
    active_cols = [col for col in all_cols if col.startswith(tuple(features))]

    active_rows = design_matrix[active_cols].values.any(axis=1)
    active_rows_mask = (active_rows > 0).astype('int8')

    return active_rows_mask


def track_enl_period(timeseries: pd.DataFrame,
                     enl_col: str = 'ENL',
                     method: str = 'linear'):

    '''
    Set counter for time within ENL onset (can be linear or exponential)
    option to add: can also step forward past ENL period
    '''

    ts_ = timeseries.copy()
    ts_[f't_to_{enl_col.lower()}_on'] = 0  # 0 outside of ENL period
    enl = ts_.query(f'{enl_col} == 1')

    # Scaling factor to keep within similar range of other features.

    if method == 'linear':
        enl_tracker = enl.groupby('nTrial').cumcount()
    elif method == 'exponential':
        factor = 5000
        enl_tracker = (enl.groupby('nTrial').cumcount()**2) / factor
    else:
        raise NotImplementedError

    ts_.loc[enl.index, f't_to_{enl_col.lower()}_on'] = enl_tracker

    return ts_


def reduce_cue_to_onset(timeseries: pd.DataFrame) -> pd.Series:

    cue_ts_ = timeseries.copy()
    cue_onsets = (cue_ts_.query('Cue == 1')
                         .groupby('nTrial', as_index=False)
                         .nth(0).index.values)
    cue_ts_['cue'] = 0
    cue_ts_.loc[cue_onsets, 'cue'] = 1

    return cue_ts_['cue']


def make_design_mat(timeseries: pd.DataFrame,
                    trials: pd.DataFrame,
                    states: set[str] = None,
                    nth_licks: list[int] = None,
                    interactions: dict[str, str] = None,
                    feature_comparison: bool = False) -> pd.DataFrame:

    '''
    Wrapper to generate design matrix with desired features classifying each
    lick.

    Args:
        timeseries:
            Timeseries form of behavior/neural data.
        trials:
            Trial-based behavior/task data that can be mapped to timeseries.
        states:
            States to define licks by (only keep licks in specified states).
        nth_licks:
            Lick index in bout to isolate from bout.
            e.g. to take only first lick -> [1].
        interactions:
            {trial_type: states} are types of features to interact
        feature_comparison:
            Whether or not model fit on design matrix is part of broader
            feature comparison (requiring consistency in active rows across
            feature sets).

    Returns:
        design_mat:
            Dataframe with N rows for timepoints across session and M columns
            for features (each lick uniquely classified).

    Notes:
        TO-DO: currently only functional for cases where lick gets pulled from
        bout; add option to include whole original bout
    '''

    ts_ = timeseries.copy()

    if states is None:
        states = {'Select', 'Consumption', 'ENLP', 'Cue', 'ENL'}
    if nth_licks is None:
        nth_licks = [1]  # default behavior is take only first lick

    tmp_lick_states = set()
    if feature_comparison:
        # need Consumption period for keeping trial durations constant
        tmp_lick_states = {'Consumption'} - states

    trials = create_combo_col(trials, grouping_levels=['seq2', '+1switch'],
                              generic_col_name=False)

    trials = (trials.set_index('nTrial')  # for mapping to timeseries
                    .convert_dtypes())  # maintains consistency across sessions

    # Store some columns to add to design matrix at end.
    photo_cols = [col for col in ts_ if 'z_grn' in col]
    hm_columns = add_heatmap_columns(ts_, trials)

    # Make column with 1s where licks occurred.
    ts_['Lick'] = ts_.iSpout.ne(0).astype('int8')

    # Create binary column for each state-defined lick class.
    ts_, lick_cols = classify_lick_state(ts_, states | tmp_lick_states)

    ts_ = track_enl_period(ts_, method='exponential')

    # Make design matrix only containing licks and essential trial IDs.
    dm_cols = (lick_cols
               + ['nTrial', 'iBlock', 't_to_enl_on', 'session',
                  'session_clock']
               + photo_cols)
    dm = ts_[dm_cols].copy()

    # Represent cue as impulse occurring at cue onset time.
    dm['cue'] = reduce_cue_to_onset(ts_)

    if 'ENLP' in states:
        # Track ENL period preceding a penalty
        ts_ = track_enl_period(ts_, enl_col='state_ENLP', method='exponential')
        dm['t_to_enlp_on'] = ts_['t_to_state_enlp_on'].copy()

    # Split licks by nth position in bout (as dummies).
    dm = pull_lick_from_bout(dm, nth_licks, only_nth_lick=False)

    if interactions is not None:
        # 'flag' column for any NaNs in trial type (including timeouts).
        dm['flag'] = 0
        for trial_type, s_ in interactions.items():
            dm['flag'] += dm['nTrial'].map(trials[trial_type].isna())
            dm = event_interactions_dummies(dm, trials, states=s_,
                                            trial_type=trial_type)
            dm['flag'] = dm['flag'].clip(0, 1).astype('int8')
    else:
        dm['flag'] = dm['nTrial'].map(trials['timeout'].astype('int8'))

    # Flag trials where cue got cut off (first/last trials of a session).
    cue_cols = [col for col in dm.columns if col.endswith('cue')]
    missing_cue = (dm
                   .groupby('nTrial')[cue_cols]
                   .sum().sum(axis=1)
                   .eq(0))
    missing_cue_trials = missing_cue[missing_cue].index.values
    dm.loc[dm.nTrial.isin(missing_cue_trials), 'flag'] = 1

    if feature_comparison:
        features = dm.columns.drop(['nTrial', 'iBlock', 'flag'] + photo_cols)
        dm['active_rows'] = label_active_rows(dm, features.tolist())

    cols_to_drop = []
    if 'Cue' not in states:
        cols_to_drop.append('cue')
    if 'ENL' not in states:
        cols_to_drop.append('t_to_enl_on')
    if tmp_lick_states:
        cols_to_drop.extend([col for col in dm.columns if 'con' in col])
    dm = dm.drop(columns=cols_to_drop)

    dm[hm_columns.columns] = hm_columns

    return dm
