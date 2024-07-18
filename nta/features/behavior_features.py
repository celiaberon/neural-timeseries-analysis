"""
@author: celiaberon
"""

import numpy as np
import pandas as pd

from ..utils import single_session


def shift_trial_feature(trials: pd.DataFrame,
                        col: str,
                        n_shift: int,
                        shift_forward: bool = True,
                        new_col: str = ''):

    '''
    Shift trial feature forward or backward to relate past or future events to
    current trial, respectively.

    Args:
        trials:
            Trial-based dataframe.
        col:
            Column to be shifted.
        n_shift:
            Number of trials to shift the column.
        shift_forward:
            When True, shift column forward. New column will represent history
            of current trial.
        new_col:
            Can provide custom name for new column.

    Returns:
        trials_:
            Copy of original data with new column for shifted feature.
    '''

    trials_ = trials.copy()
    if not new_col:
        new_col = col.lower()

    if shift_forward:
        # Use (-) for trial histories (N trials back from current trial).
        trials_[f'-{n_shift}{new_col}'] = trials_.groupby('Session', observed=True)[col].shift(n_shift)
    else:
        # Use (+) for subsequent trials (N trials forward from current).
        trials_[f'+{n_shift}{new_col}'] = trials_.groupby('Session', observed=True)[col].shift(-n_shift)

    return trials_


def shift_trial_with_cleanup(trials, col, shift=1, new_col=None):

    '''
    Shift feature value relative to trial label in multi-session dataframe. Can
    be applied with broken trial continuity.

    Args:
        trials:
            Trial-based dataframe, rows for trials and columns for features.
        col:
            Column to shift.
        shift:
            Number of trials by which to shift `col`. Positive shifts forward
            (such that associated value corresponds to previous trial).
        new_col:
            New column label.

    Returns:
        trials_:
            Trial table containing `new_col` as a shifted version of `col`.
    '''

    trials_ = trials.copy()

    if new_col is None:
        prefix = f'prev{abs(shift)}_' if shift > 0 else f'next{abs(shift)}_'
        new_col = prefix + col

    trials_[new_col] = trials_.groupby('Session', observed=True)[col].shift(shift)

    # Find discontinuous trials within session and prevent shift mislabeling.
    trials_.loc[trials_.groupby('Session', observed=True).nTrial.diff() > 1, new_col] = np.nan

    return trials_


def encode_choice_reward_pairing(trials: pd.DataFrame):

    '''
    Summarize each trial with single symbol that captures choice direction
    (Right or Left) and reward outcome (uppercase for reward).

    Args:
        trials:
            Trial-based dataframe.

    Returns:
        trials_:
            Copy of trials containing new column with encoded trial types.

    Notes:
        k{N} symbol system uses N=1 for current trial and N=1+i for the ith
        trial back.
    '''

    trials_ = trials.copy()

    # Mappings from (Choice, Reward) tuples to letter-symbol code.
    trial_mappings = {(1.0, 1.0): 'R',
                      (1.0, 0.0): 'r',
                      (0.0, 1.0): 'L',
                      (0.0, 0.0): 'l'}

    # Look up mapping for each trial with data (not null) for both reward and
    # direction.
    twd_ = trials_.copy().dropna(subset=['Reward', 'direction'])
    trials_['k1'] = None
    trials_.loc[twd_.index, 'k1'] = [trial_mappings[h] for h in
                                     zip(twd_.direction, twd_.Reward)]

    return trials_


@single_session
def build_history_sequence(trials: pd.DataFrame,
                           sequence_length: int):

    '''
    Combine encoded trial identities into sequential code that includes the
    current trial's action-outcome combination. Preserve left and right
    identity of choices.

    Args:
        trials:
            Trial-based dataframe.
        sequence_length:
            Number of trials to combine into a history sequence.

    Returns:
        trials_:
            Copy of trials containing new column with sequence history.
    '''

    trials_ = trials.copy()

    if 'k1' not in trials_.columns:
        # Create column with choice-outcome code to construct trial histories.
        trials_ = encode_choice_reward_pairing(trials_)

    for h_length in range(1, sequence_length):
        if f'k{h_length + 1}' in trials_.columns:
            continue
        trials_[f'k{h_length + 1}'] = trials_['k1'].shift(h_length)

    # Initialize column with current trial code.
    trials_[f'RL_seq{sequence_length}'] = trials_[f'k{sequence_length}']
    N = sequence_length - 1

    # Backtrack through previous trials and build up sequential code.
    while N > 0:
        trials_[f'RL_seq{sequence_length}'] += trials_[f'k{N}']
        N -= 1

    return trials_


def convert_to_AB_sequence(trials: pd.DataFrame,
                           sequence_length: int):

    '''
    Project left-right sequences onto direction-agnostic axis using laterally
    abstracted code. Here, A denotes first direction in the sequence and B the
    alternative direction.

    Args:
        trials:
            Trial-based dataframe.
        sequence_length:
            Number of trials to combine into a history sequence.

    Returns:
        trials_:
            Copy of trials containing new column with AB sequence history.
    '''

    trials_ = trials.copy()

    # Mappings based on first direction chosen in the history sequence.
    mappings_ref_L = {'L': 'A', 'l': 'a', 'R': 'B', 'r': 'b'}
    mappings_ref_R = {'R': 'A', 'r': 'a', 'L': 'B', 'l': 'b'}
    mappings_LUT = {'L': mappings_ref_L, 'R': mappings_ref_R}

    column = f'RL_seq{sequence_length}'
    trials_[f'seq{sequence_length}'] = None
    for i, row in trials_.iterrows():

        if pd.isna(row[column]):
            continue

        reference_direction = row[f'k{sequence_length}'].upper()
        mappings = mappings_LUT[reference_direction]
        sequences = ''.join([mappings.get(s) for s in row[column]])
        trials_.loc[i, f'seq{sequence_length}'] = sequences

    return trials_


@single_session
def get_reward_seq(trials: pd.DataFrame) -> pd.DataFrame:

    '''
    Count sequential rewards and losses, restarting with alternative outcome.

    Args:
        trials (pandas.DataFrame):
            Trial-based dataframe.

    Returns:
        trials_ (pandas.DataFrame):
            Copy of trials with new columns for reward and loss sequences.
    '''

    trials_ = trials.copy()
    cumulative_rewards = trials_.Reward.cumsum()
    cumulative_rewards_reset = (cumulative_rewards
                                .where(trials_.Reward == 0)
                                .ffill().fillna(0).astype(int))
    trials_['rew_seq'] = cumulative_rewards - cumulative_rewards_reset

    cumulative_losses = (1 - trials_.Reward).cumsum()
    cumulative_losses_reset = (cumulative_losses
                               .where(trials_.Reward > 0)
                               .ffill().fillna(0).astype(int))
    trials_['loss_seq'] = cumulative_losses - cumulative_losses_reset

    # If sequence incomplete (doesn't have fully defined history), use NaNs.
    inc_seq = (trials_
               .groupby('Session', observed=True)['nTrial']
               .nth(slice(0, 3)).values)
    trials_.loc[trials_.nTrial.isin(inc_seq), ['rew_seq', 'loss_seq']] = np.nan

    # Create single column with posiive values for reward, negative for loss.
    trials_['outcome_seq'] = trials_['rew_seq'] - trials_['loss_seq']

    return trials_


@single_session
def add_timeseries_clock(timeseries: pd.DataFrame,
                         fs: int = 50):

    '''
    Estimate running clock across a session based on the known sampling rate
    for the analog timeseries.

    Args:
        timeseries:
            Timeseries form of data.
        fs:
            Sampling frequency in Hz.

    Returns:
        session_clock (pd.Series):
            Column of clock times (in seconds) for each row in the timeseries.
    '''

    if timeseries['Session'].dropna().nunique() > 1:
        raise NotImplementedError

    ts_ = timeseries.copy()
    ts_['session_clock'] = 1 / fs  # convert each timestep to seconds
    ts_['session_clock'] = ts_['session_clock'].cumsum()

    return ts_['session_clock']


def count_consumption_licks(timeseries: pd.DataFrame,
                            trials: pd.DataFrame):

    '''
    Count the number of licks within the consumption period for each trial.

    Args:
        timeseries:
            Timeseries form of data.
        trials:
            Trials-based dataframe.

    Returns:
        nLicks (pd.Series):
            Column with number of consumption licks for each trial.
    '''

    trials_ = trials.copy()

    nLicks = (timeseries
              .query('Consumption == 1 & iSpout.isin([1, 2, 3])')
              .groupby(['nTrial'])['iSpout']
              .agg(len)
              )
    trials_['nLicks'] = trials_['nTrial'].map(nLicks)

    return trials_['nLicks']


def label_lick_position(timeseries: pd.DataFrame):

    '''
    Track relative lick number within a trial, including only selection and
    consumption licks. NOTE: this does not necessarily correspond to a single
    lick bout.

    Args:
        timeseries:
            Timeseries form of data.

    Returns:
        iLick (pd.Series):
            Column of lick position labels starting with Select=1.
    '''

    ts_ = timeseries.copy()

    ts_['iLick'] = np.nan
    ts_['iLick'] = (ts_.query('~iSpout.eq(0) & Consumption == 1 | Select == 1')
                       .groupby('nTrial')
                       .cumcount())

    return ts_['iLick']


def label_bout_group(ts, trials, bout_ili_thresh=0.4):

    '''
    Take all licks with ILI over threshold and define them as onset of new
    bout, which each get own numberic ID. Define consumption bout as first
    bout in consumption period.

    Args:
        ts:
            Timeseries form of data.
        trials:
            Trials form of data.
        bout_ili_thresh:
            Threshold above which interlick interval is classified as new lick
            bout (likely based on mean/std of ILI for highly stereotyped lick
            bouts).
    '''
    ts_ = ts.copy()
    trials_ = trials.copy()
    timed_licks = ts_.dropna(subset='ILI').copy()
    assert all(timed_licks.iSpout != 0)

    # Use ILI threshold to separate lick bouts and assign to bout group.
    timed_licks['lick_bout_thresh'] = timed_licks['ILI'] > bout_ili_thresh
    timed_licks['bout_group'] = timed_licks.lick_bout_thresh.cumsum()

    # Bout group ID of first counsumption bout for each trial.
    consumption_bouts = (timed_licks.query('stateConsumption==1')
                                    .groupby('nTrial')
                                    .nth(0)[['nTrial', 'bout_group']]
                                    .set_index('nTrial', drop=True))

    # Add columns for bout group and consumption bout group ID in full
    # timeseries df.
    ts_.loc[timed_licks.index.values, 'bout_group'] = timed_licks['bout_group']
    ts_['cons_bout'] = ts_['nTrial'].map(consumption_bouts['bout_group'])

    # Add column for number of consumption licks in first consumption bout to
    # trial df.
    consumption_nLicks = (ts_.query('bout_group==cons_bout')
                             .groupby('nTrial')['iSpout']
                             .count())
    trials_['cons_nLicks'] = trials_['nTrial'].map(consumption_nLicks)

    return ts_, trials_


def map_sess_variable(trials: pd.DataFrame, ref_df: pd.DataFrame,
                      col: str) -> pd.DataFrame:

    trials_ = trials.copy()
    sess_id = trials_.Session.unique().item()
    trials_[col] = ref_df.query('Session == @sess_id')[col].values.squeeze()

    return trials_


def apply_trial_rep_threshold(trials, col, min_reps=100, by_outcome=False):

    '''
    Filter trial type labels that don't meet occurrence threshold. Leaves trial
    data in place within trial table, but replaces low-replicate trial types
    for a given variable with NaN.
    Note: used for discrete quantitative variables to clip tails (but does not
    check that all intermediates meet threshold).

    Args:
        trials:
            Trial table with row for each trial, column for each feature.
        col:
            Feature to filter low frequency instances from.
        min_reps:
            Minimum number of trials for each instance of a variable.
        by_outcome:
            If true, independently finds bounds for rewarded and unrewarded
            trials.

    Returns:
        trials_:
            Trial table where `col` contains NaNs for values that occur fewer
            than min_reps times (per outcome or not).
    '''

    trials_ = trials.copy()
    if by_outcome:
        outcomes = [[0], [1]]
    else:
        outcomes = [[0, 1]]  # fake it if not splitting counts by outcome

    for outcome in outcomes:

        llim = (trials_
                .query(f'Reward.isin({outcome})')
                .groupby(col)
                .filter(lambda x: len(x) > min_reps)[col]
                .min())
        ulim = (trials_
                .query(f'Reward.isin({outcome})')
                .groupby(col)
                .filter(lambda x: len(x) > min_reps)[col]
                .max())
        print(llim, ulim)

        mask = ((trials_['Reward'].isin(outcome))
                & ((trials_[col] < llim) | (trials_[col] > ulim)))
        trials_.loc[mask, col] = np.nan

    return trials_


def split_penalty_states(ts, penalty='ENLP', cuep_ref_enl=False):

    '''
    Note: Can do this before photometry alignment now that using 23-29 as sync
    pulse.
    '''

    def is_post_final_penalty(trial_ts, pen_state):

        # last enl break is offset, second to last is onset of real enl.
        true_enl_onset = np.where(trial_ts[pen_state].diff() != 0)[0][-2]
        enl_onset_time = trial_ts.iloc[true_enl_onset]['session_clock']

        return trial_ts['session_clock'] < enl_onset_time

    ts_ = ts.copy()

    # Overwrite pre-cue penalty ENL if backtracking from cue penalties.
    pen_state = penalty[:-1] if not cuep_ref_enl else 'ENL'

    pen_trials = ts_.loc[ts_[penalty] == 1].nTrial.dropna().unique()

    if len(pen_trials) > 1:
        # Make mask of the "true" state that runs to completion without any
        # penalties.
        mask = (ts_
                .query('nTrial.isin(@pen_trials)')
                .groupby('nTrial', as_index=False)
                .apply(is_post_final_penalty, pen_state)
                )
    elif len(pen_trials) == 1:
        mask = is_post_final_penalty(ts_.query('nTrial.isin(@pen_trials)'),
                                     pen_state)
    elif len(pen_trials) == 0:
        return ts_

    mask = mask.astype(ts[pen_state].dtype)
    # label pre-penalty states as penalty states
    new_state = f'state_{penalty}' if not cuep_ref_enl else 'state_ENL_preCueP'
    ts_[new_state] = 0
    ts_.loc[ts_.nTrial.isin(pen_trials), new_state] = (mask.values
                                                       * ts_.query('nTrial.isin(@pen_trials)')[pen_state])

    # remove pre-penalty states from true states
    ts_.loc[ts_.nTrial.isin(pen_trials), pen_state] = ((1 - mask.values)
                                                       * ts_.query('nTrial.isin(@pen_trials)')[pen_state])

    return ts_

@single_session
def flag_blocks_for_timeouts(trials, threshold=0.25):

    '''
    Flag any blocks with >threshold proportion of timeout trials (i.e.
    'timeout_blocks'). Also flag blocks, starting with first and last blocks
    as a rule and extending through any consecutive blocks with excessive
    timeouts ('flag_block').

    Args:
        trials:
            Trial-based dataframe.
        threshold:
            Fraction of timeouts within block as (N timeouts / N trials) above
            which a block will be flagged.
    '''
    trials_ = trials.copy()

    # Report on any block with timeouts above threshold.
    trials_['timeout_block'] = False
    for i, block in trials_.groupby('iBlock'):
        above_thresh = np.mean(block.timeout) > threshold
        trials_.loc[trials_.iBlock == i, 'timeout_block'] = above_thresh

    # Record the threshold being used.
    trials_['timeout_thresh'] = threshold

    # Flag first and last block, and recursively flag n-1 continuous blocks
    # of timeouts > threshold.
    trials_['flag_block'] = False
    trials_.loc[trials_.iBlock == trials_.iBlock.min(), 'flag_block'] = True
    trials_.loc[trials_.iBlock == trials_.iBlock.max(), 'flag_block'] = True

    block_search = trials_.iBlock.max() - 1
    continue_search = True
    while continue_search:
        curr_block = trials_.query('iBlock == @block_search')
        # If timeouts exceed threshold for timeouts.
        if np.mean(curr_block.timeout) > threshold:
            trials_.loc[trials_.iBlock == block_search, 'flag_block'] = True
            block_search -= 1
        else:
            continue_search = False

    return trials_


def order_sessions(trials):

    t_ = trials.copy()
    t_['Date'] = pd.to_datetime(t_['Date'], format='%Y_%m_%d')

    for mouse, m_dates in t_.groupby('Mouse', observed=True):
        sorted_dates = np.sort(m_dates.Date.unique())
        sorted_dates = {date: i for i, date in enumerate(sorted_dates)}
        t_.loc[t_['Mouse'] == mouse, 'session_order'] = m_dates['Date'].map(sorted_dates)

    return t_


def add_behavior_cols(trials: pd.DataFrame,
                      timeseries: pd.DataFrame = None,
                      fs: int = None) -> tuple:

    '''
    Add columns defining behavior features to trial data.

    Args:
        trials:
            Trial-based dataframe.
        timeseries:
            Timeseries form of data.
        fs:
            Sampling frequency in Hz.

    Returns:
        trials_:
            Copy of trials dataframe with new behavior columns added.
        timeseries_:
            Copy of timeseries dataframe with new behavior columns added.

    Notes:
        Add columns when loading in data or before dropping any trials.
        Not equipped with multi-session handling.
        +1 or 'next' prefix: indicates back-shifted column to align next trial
        event to current trial and vice versa for -1 prefix
    '''

    history_features = ['seq2', 'seq3', 'tSelection', 'direction', 'Reward']
    future_features = ['seq2', 'seq3', 'tSelection', 'direction', 'Switch']

    trials_ = trials.copy()
    if isinstance(timeseries, pd.DataFrame):
        ts_ = timeseries.copy()
        if 'Mouse' not in ts_.columns:
            ts_['Mouse'] = [sess[:3] for sess in ts_['Session'].values]

    assert trials_.Session.dropna().nunique() == 1  # because of row shifting

    trials_['enlp_trial'] = trials_['n_ENL'] > 1
    trials_ = get_reward_seq(trials_)  # number cumulative rewarded and losses
    trials_ = shift_trial_feature(trials_, col='outcome_seq', n_shift=1,
                                  shift_forward=True,
                                  new_col='outcome_seq_history')

    # Build up columns defining sequential history for each trial and
    # history length. NOTE: here increasing numbers denote advancing
    # back into history from k1=current trial.
    history_lengths = range(2, 4)
    for h_length in history_lengths:
        trials_ = build_history_sequence(trials_, h_length)
        trials_ = convert_to_AB_sequence(trials_, sequence_length=h_length)

    # Some additional columns that can be useful.
    # if 'session_clock' not in ts_.columns:
    #     ts_['session_clock'] = add_timeseries_clock(ts_, fs=fs)
    if isinstance(timeseries, pd.DataFrame):
        trials_['nLicks'] = count_consumption_licks(ts_, trials_)
        history_features.append('nLicks')
        ts_['iLick'] = label_lick_position(ts_)

        # Calculate interlick interval (independent of state lick occurs within).
        ts_['ILI'] = ts_.query('iSpout != 0').session_clock.diff()
        ts_, trials_ = label_bout_group(ts_, trials_)
    # Forward and backward shifts that can be useful (need to shift up front).
    for feature in history_features:
        trials_ = shift_trial_feature(trials_, col=feature, n_shift=1,
                                      shift_forward=True)
    for feature in future_features:
        trials_ = shift_trial_feature(trials_, col=feature, n_shift=1,
                                      shift_forward=False)

    if isinstance(timeseries, pd.DataFrame):
        return trials_, ts_
    else:
        return trials_
