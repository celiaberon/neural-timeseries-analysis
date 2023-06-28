"""
@author: celiaberon
"""

import numpy as np
import pandas as pd


def shift_trial_feature(trials: pd.DataFrame,
                       col_name: str,
                       n_shift: int,
                       shift_forward: bool=True,
                       new_col_name: str=''):
    
    '''
    Shift trial feature forward or backward to relate past or future events to
    current trial, respectively.

    Args:
        trials:
            Trial-based dataframe.
        col_name:
            Column to be shifted.
        n_shift:
            Number of trials to shift the column.
        shift_forward:
            When True, shift column forward. New column will represent history
            of current trial.
        new_col_name:
            Can provide custom name for new column.

    Returns:
        trials_:
            Copy of original data with new column for shifted feature.
    '''

    trials_ = trials.copy()

    if not new_col_name:
        new_col_name = col_name.lower()

    if shift_forward:
        # Use (-) to indicate trial histories (N trials back from current trial).
        trials_[f'-{n_shift}{new_col_name}'] = trials_[f'{col_name}'].shift(n_shift)
    else:
        # Use (+) to indicate subsequent trials (N trials forward from current).
        trials_[f'+{n_shift}{new_col_name}'] = trials_[f'{col_name}'].shift(-n_shift)

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
    trial_mappings = {(1.0, 1.0):'R',
                      (1.0, 0.0):'r',
                      (0.0, 1.0):'L',
                      (0.0, 0.0):'l'}

    # Look up mapping for each trial that has value for both reward and direction.
    seq_clean = trials_.copy().dropna(subset=['Reward','direction'])
    trials_.loc[seq_clean.index, 'k1'] = [trial_mappings[h] for h in 
                                          zip(seq_clean.direction, seq_clean.Reward)]

    return trials_


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
    mappings_ref_L = {'L':'A', 'l':'a', 'R':'B', 'r':'b'}
    mappings_ref_R = {'R':'A', 'r':'a', 'L':'B', 'l':'b'}
    mappings_LUT = {'L':mappings_ref_L, 'R':mappings_ref_R}

    column = f'RL_seq{sequence_length}'

    for i, row in trials_.iterrows():

        if pd.isna(row[column]): continue

        reference_direction = row[f'k{sequence_length}'].upper()
        mappings = mappings_LUT[reference_direction]
        trials_.loc[i, f'seq{sequence_length}'] = ''.join([mappings.get(s) for s in row[column]])

    return trials_


def get_reward_seq(trials: pd.DataFrame) -> pd.DataFrame:

    '''Count sequential rewards and losses, restarting with alternative outcome.
    
    Args:
        trials (pandas.DataFrame):
            Trial-based dataframe.
    
    Returns:
        trials_ (pandas.DataFrame):
            Copy of trials with new columns for reward and loss sequences.
    '''
    
    trials_ = trials.copy()
    cumulative_rewards = trials_.Reward.cumsum()
    cumulative_rewards_reset = cumulative_rewards.where(trials_.Reward==0).ffill().fillna(0).astype(int)
    trials_['reward_seq'] = cumulative_rewards - cumulative_rewards_reset

    cumulative_losses = (1-trials_.Reward).cumsum()
    cumulative_losses_reset = cumulative_losses.where(trials_.Reward>0).ffill().fillna(0).astype(int) 
    trials_['loss_seq'] = cumulative_losses - cumulative_losses_reset
    
    first_trials_to_clear = trials_.groupby('Session')['nTrial'].nth(slice(0,3)).values
    trials_.loc[trials_.nTrial.isin(first_trials_to_clear), ['reward_seq', 'loss_seq']] = np.nan

    # Create single column with posiive values for reward, negative for loss.
    trials_['outcome_seq'] = trials_['reward_seq'] - trials_['loss_seq'] 

    return trials_


def add_timeseries_clock(timeseries: pd.DataFrame,
                         fs: int=50):

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
    
    if timeseries.session.dropna().nunique() > 1:
        raise NotImplementedError
    
    timeseries_ = timeseries.copy()
    timeseries_['session_clock'] = 1/fs # convert each timestep to seconds
    timeseries_['session_clock'] = timeseries_['session_clock'].cumsum()

    return timeseries_['session_clock']


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
             .loc[(timeseries.Consumption==1) & (~timeseries.iSpout.isnull())]
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

    timeseries_ = timeseries.copy()

    timeseries_['iLick'] = np.nan
    timeseries_['iLick'] = (timeseries_.loc[(~timeseries_.iSpout.isna()) 
                             & ((timeseries_.Consumption==1) | (timeseries_.Select==1))]
                          .groupby('nTrial')
                          .cumcount())
    
    return timeseries_['iLick']


def add_behavior_cols(trials: pd.DataFrame,
                      timeseries: pd.DataFrame,
                      fs: int=50) -> tuple:

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

    trials_ = trials.copy()
    timeseries_ = timeseries.copy()

    assert trials_.Session.dropna().nunique()==1 # because of row shifting

    trials_ = get_reward_seq(trials_) # number cumulative rewarded and losses
    trials_ = shift_trial_feature(trials, col_name='outcome_seq', n_shift=1,
                                 shift_forward=True,
                                 new_col_name='outcome_seq_history')

    # Create column with choice-outcome code to construct trial histories.
    trials_ = encode_choice_reward_pairing(trials_)
    
    # Build up columns defining sequential history for each trial and
    # history length. NOTE: here increasing numbers denote advancing
    # back into history from k1=current trial.
    history_lengths = range(2,4)
    for history_length in range(1, max(history_lengths)):
        trials_[f'k{history_length+1}'] = trials_['k1'].shift(history_length)
    for history_length in history_lengths:
        trials_ = build_history_sequence(trials_, history_length)
        trials_ = convert_to_AB_sequence(trials_, sequence_length=history_length)

    # Some additional columns that can be useful.
    timeseries_['session_clock'] = add_timeseries_clock(timeseries_, fs=fs)    
    trials_['nLicks'] = count_consumption_licks(timeseries_, trials_)
    timeseries_['iLick'] = label_lick_position(timeseries)
    
    # Calculate interlick interval (independent of state lick occurs within).
    timeseries_['ILI'] = timeseries_.loc[~timeseries_.iSpout.isna()].session_clock.diff()

    # Forward and backward shifts that can be useful (need to shift up front).
    for col in ['seq2', 'seq3', 'tSelection', 'direction', 'Reward', 'nLicks']:
        trials_ = shift_trial_feature(trials_, col_name=col, n_shift=1, shift_forward=True)
    for col in ['seq2', 'seq3', 'tSelection', 'direction', 'Switch']:
        trials_ = shift_trial_feature(trials_, col_name=col, n_shift=1, shift_forward=False)
    
    return trials_, timeseries_