import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append('/Users/celiaberon/GitHub/neural-timeseries-analysis/')
import nta.features.behavior_features as bf


@pytest.fixture
def sim_trials():
    trials = pd.concat((pd.DataFrame({
        'nTrial': np.arange(10),
        'nTrial_orig': np.arange(10) + 10,
        'Session': np.ones(10),
        'direction': [1, 1, 1, 0, np.nan, np.nan, 0, 1, 0, 0],
        'Reward': [1, 1, 0, 0, np.nan, np.nan, 0, 1, 1, 1],
        'timeout': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    }),
        pd.DataFrame({
            'nTrial': np.arange(10) + 10,
            'nTrial_orig': np.arange(10) + 10,
            'Session': np.ones(10) + 1,
            'direction': [1, np.nan, 0, 0, 1, 1, 0, 1, np.nan, 1],
            'Reward': [1, np.nan, 0, 1, 1, 0, 0, 1, np.nan, 1],
            'timeout': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0]
        })
    )).reset_index(drop=True)

    return trials


def test_combined_trial_ids(sim_trials):
    assert sim_trials.nTrial.nunique() == len(sim_trials), (
        'Each row needs unique trial ID')
    assert np.all(sim_trials.groupby('Session')['nTrial'].diff().dropna() == sim_trials.groupby('Session')['nTrial_orig'].diff().dropna()), 'Original relative sequential trial structure broken'


def test_nans(sim_trials):
    choice_contingent_cols = ['direction', 'Reward', 'selHigh', 'Switch',
                              'rew_seq', 'loss_seq', 'outcome_seq', 'k1',
                              'RL_seq2', 'seq2', 'RL_seq3', 'seq3', '+1seq2',
                              '+1seq3', '+1switch']
    assert np.all(sim_trials.query('timeout == True')[choice_contingent_cols].isna()), 'Cannot have values in choice-contingent columns that are not NaN when choice was timeout'

    sim_trials = bf.shift_trial_feature(sim_trials, col='timeout', n_shift=1,
                                        shift_forward=True)
    sim_trials = bf.shift_trial_feature(sim_trials, col='timeout', n_shift=1,
                                        shift_forward=False)
    sim_trials = sim_trials.rename(columns={'-1timeout': 'prev_timeout',
                                            '+1timeout': 'next_timeout'})

    prev_choice_contingent_cols = ['Switch', '-1outcome_seq_history', 'k2',
                                   'RL_seq2', 'seq2', 'RL_seq3', 'seq3',
                                   '-1seq2', '-1seq3', '-1direction',
                                   '-1reward']

    assert np.all(sim_trials.query('prev_timeout == True')[prev_choice_contingent_cols].isna()), 'Cannot have values in previous choice-contingent columns that are not NaN when previous choice was timeout'

    next_choice_contingent_cols = ['+1seq2', '+1seq3', '+1direction',
                                   '+1switch']

    assert np.all(sim_trials.query('next_timeout == True')[next_choice_contingent_cols].isna()), 'Cannot have values in next choice-contingent columns that are not NaN when next choice was timeout'


def test_binaries(sim_trials):

    binaries_with_nans = ['direction', 'Reward', 'selHigh', 'flag_block',
                          'timeout_block', 'timeout', 'Switch', 'enlp_trial']

    for col in binaries_with_nans:
        assert not set(sim_trials[col].dropna().unique()) - {0.0, 1.0}, (
            'Nonbinary (or NaN) value in binary columns')


def test_encoded_sequences(sim_trials):

    history_lengths = range(2, 4)
    for h_length in history_lengths:
        sim_trials = bf.build_history_sequence(sim_trials, h_length)
        sim_trials = bf.convert_to_AB_sequence(sim_trials,
                                               sequence_length=h_length)

    curr_encoding = ['R', 'R', 'r', 'l', 'l', 'R', 'L', 'L', 'R', 'l', 'L', 'R',
                     'r', 'l', 'R', 'R']
    assert np.all(sim_trials['k1'].dropna().values == curr_encoding), (
        'Initial encoding incorrect')

    rl2_encoding = ['RR', 'Rr', 'rl', 'lR', 'RL', 'LL', 'LR', 'lL', 'LR', 'Rr',
                    'rl', 'lR']
    assert np.all(sim_trials['RL_seq2'].dropna().values == rl2_encoding), (
        'Laterized encoding (length 2) incorrect')

    sym2_encoding = ['AA', 'Aa', 'ab', 'aB', 'AB', 'AA', 'AB', 'aA', 'AB',
                     'Aa', 'ab', 'aB']
    assert np.all(sim_trials['seq2'].dropna().values == sym2_encoding), (
        'Symmetric encoding (length 2) incorrect')
