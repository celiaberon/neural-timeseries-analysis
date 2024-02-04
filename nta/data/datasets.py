import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import nta.preprocessing.quality_control as qc
from nta.features import behavior_features as bf
from nta.features.design_mat import make_design_mat
from nta.utils import load_config_variables


class DataSet(ABC):

    def __init__(self,
                 mice: str | list[str],
                 verbose: bool = False,
                 qc_photo: bool = True,
                 qc_params: dict = {},
                 label: str = '',
                 save: bool = True):

        # if not isinstance(mice, list):
        #     mice = [mice]
        self.mice = mice
        self.verbose = verbose
        self.qc_photo = qc_photo
        self.label = label if label else self.mice

        self.root = self.set_root()
        self.data_path = self.set_data_path()
        self.summary_path = self.set_data_overview_path()
        self.save = save
        if self.save:
            self.save_path = self.set_save_path()
        self.cohort = self.load_cohort_dict()

        self.ts = pd.DataFrame()
        self.trials = pd.DataFrame()
        self.channels = self.set_channels()
        self.sig_channels = set()

        if not isinstance(mice, list):
            self.mouse_ = mice
            multi_sessions = self.read_multi_sessions(qc_params)
            self.ts = multi_sessions['ts']
            self.trials = multi_sessions['trials']
        else:
            self.read_multi_mice(qc_params)

    @abstractmethod
    def set_root(self):
        '''Sets the root path for the dataset'''
        pass

    @abstractmethod
    def set_data_path(self):
        '''Sets the path to the session data'''
        pass

    @abstractmethod
    def set_data_overview_path(self):
        '''Sets the path to the csv containing session summary'''
        pass

    @abstractmethod
    def set_session_path(self):
        '''Sets path to single session data'''
        pass

    def set_save_path(self):
        '''Set save path and create the directory.'''
        save_path = self.root / 'headfixed_DAB_data/figures' / self.label
        if not os.path.exists(os.path.join(save_path, 'metadata')):
            os.makedirs(os.path.join(save_path, 'metadata'))
        return save_path

    def set_channels(self):
        '''Define channels to include as neural signal.'''
        channels = {'z_grnL', 'z_grnR', 'z_redR', 'z_redL'}
        return channels

    def update_columns(self, trials, ts):

        '''
        Column updates (feature definitions, etc.) that should apply to all
        datasets.
        '''
        trials, ts = bf.add_behavior_cols(trials, ts)
        trials = trials.rename(columns={'-1reward': 'prev_rew'})

        # Rectify error in penalty state allocation.
        ts['ENL'] = ts['ENL'] + ts['state_ENLP']  # recover original state
        ts['Cue'] = ts['Cue'] + ts['CueP']  # recover original state
        ts = bf.split_penalty_states(ts, penalty='ENLP')
        ts = bf.split_penalty_states(ts, penalty='CueP')

        return trials, ts

    def custom_update_columns(self):
        '''Column updates that are dataset-specific.'''
        pass

    def set_timeseries_path(self):
        '''Set path to timeseries data file.'''
        file_path = self.set_session_path()
        ts_path = file_path / f'{self.mouse_}_{self.session_}_timeseries.parquet.gzip'
        return ts_path

    def set_trials_path(self):
        '''Set path to trial-level data file.'''
        file_path = self.set_session_path()
        trials_path = file_path / f'{self.mouse_}_trials.csv'
        return trials_path

    def load_session_data(self):
        '''Loads data from single session'''
        trials_path = self.set_trials_path()
        ts_path = self.set_timeseries_path()

        if not (ts_path.exists() & trials_path.exists()):
            if self.verbose: print(f'skipped {self.mouse_} {self.session_}')
            return None, None

        ts = pd.read_parquet(ts_path)
        trials = pd.read_csv(trials_path, index_col=0)

        return ts, trials

    def load_cohort_dict(self):
        '''Load lookup table for sensor expressed in each mouse of cohort.'''
        cohort = load_config_variables(self.root, 'cohort')['cohort']
        return cohort

    def sessions_to_load(self,
                         probs: int = 9010,
                         QC_pass: bool = True):

        '''
        Make list of sessions to include for designated mouse

        Args:
            probs (int):
                Filter bandit data by probability conditions.
            QC_pass (bool):
                Whether to take sessions passing quality control (True) or
                failing (False).

        Returns:
            dates_list (list):
                List of dates to load in for mouse.
        '''
        session_log = pd.read_csv(self.summary_path)
        probs = str(probs) if not isinstance(probs, str) else probs
        session_log = session_log.query('Mouse == @self.mouse_ \
                                        & Condition == @probs \
                                        & Pass == @QC_pass')
        return list(set(session_log.Date.values))

    def get_max_trial(self, full_sessions: dict) -> int:

        '''
        Get maximum trial ID to use for unique trial ID assignment. Importantly,
        also confirm that max trial matches between dataframes.

        Args:
            full_sessions:
                Dictionary containing and trial- and timeseries-based data.

        Returns:
            max_trial:
                Number corresponding to maximum trial value.
        '''

        try:
            max_trial_trials = full_sessions['trials'].nTrial.max()
            max_trial_ts = full_sessions['ts'].nTrial.max()
            assert max_trial_trials == max_trial_ts
            max_trial = max_trial_trials
        except AttributeError:
            max_trial = 0

        return max_trial

    def concat_sessions(self,
                        *,
                        sub_sessions: dict = None,
                        full_sessions: dict = None):

        '''
        Aggregate multiple sessions by renumbering trials to provide unique trial
        ID for each trial. Store original id in separate column.

        Args:
            sub_sessions:
                Smaller unit to be concatenated onto growing aggregate dataframe.
            full_sessions:
                Core larger unit updated with aggregating data.

        Returns:
            full_sessions:
                Original full_sessions data now also containing sub_sessions data.
        '''

        max_trial = self.get_max_trial(full_sessions)

        # Iterate over both trial and timeseries data.
        for key, ss_vals in sub_sessions.items():

            # Store original trial ID before updating with unique value.
            if 'nTrial_orig' not in ss_vals.columns:
                ss_vals['nTrial_orig'] = ss_vals['nTrial'].copy()

            # Create session column to match across dataframes.
            if 'session' not in ss_vals.columns:
                ss_vals['session'] = '_'.join([self.mouse_, self.session_])

            # Add max current trial value to all new trials before concatenation.
            tmp_copy = ss_vals.copy()
            tmp_copy['nTrial'] += max_trial
            full_sessions[key] = pd.concat((full_sessions[key], tmp_copy))
            full_sessions[key] = full_sessions[key].reset_index(drop=True)

        # Use function to assert that new dataframes have matching max trial ID.
        _ = self.get_max_trial(full_sessions)

        return full_sessions

    def read_multi_mice(self,
                        qc_params,
                        **kwargs):

        '''
        Load in sessions by mouse and concatenate into one large dataframe keeping
        every trial id unique.

        Args:
            mice:
                List of mice from which to load data.
            root:
                Path to root directory containing Mouse data folder.

        Returns:
            multi_mice (dict):
                {'trials': trials data, 'timeseries': timeseries data}
        '''

        multi_mice = {key: pd.DataFrame() for key in ['trials', 'ts']}

        for mouse in self.mice:

            self.mouse_ = mouse

            multi_sessions = self.read_multi_sessions(qc_params, **kwargs)

            if len(multi_sessions['trials']) < 1:
                continue  # skip mouse if no sessions returned

            multi_mice = self.concat_sessions(sub_sessions=multi_sessions,
                                              full_sessions=multi_mice)

        self.ts = multi_mice['ts']
        self.trials = multi_mice['trials']
        print(f'{self.trials.Session.nunique()} total sessions loaded in')

    def read_multi_sessions(self,
                            qc_params,
                            **kwargs) -> dict:

        sessions = self.sessions_to_load(**qc_params)
        multi_sessions = {key: pd.DataFrame() for key in ['trials', 'ts']}

        # Loop through files to be processed
        for session_date in tqdm(sessions, self.mouse_, disable=False):

            if self.verbose: print(session_date)
            self.session_ = session_date

            ts, trials = self.load_session_data()
            if ts is None: continue

            trials, ts = self.custom_update_columns(trials, ts)
            trials, ts = self.update_columns(trials, ts)

            ts = self.eval_photo_sig(ts)
            if ts is None: continue

            if kwargs:
                ts = make_design_mat(ts, trials, **kwargs)

            # Trial level quality control needs to come at the end.
            trials_matched = qc.QC_included_trials(ts,
                                                   trials,
                                                   allow_discontinuity=False,
                                                   drop_enlP=False)

            multi_sessions = self.concat_sessions(sub_sessions=trials_matched,
                                                  full_sessions=multi_sessions)

        # TODO: QC all mice sessions by ENL penalty rate set per mouse

        return multi_sessions

    def eval_photo_sig(self, ts):
        '''Run QC on photometry channels, filtering out data from sessions
        where full session lacked real signal.'''
        # If no photometry channels passed QC, move on to next session.
        sensor = self.cohort.get(self.mouse_)
        if self.qc_photo:
            sig_cols = {ch for ch in self.channels
                        if not qc.is_normal(ts.get(ch, None),
                                            sensor=sensor,
                                            verbose=self.verbose)}
        else:
            sig_cols = {ch for ch in self.channels if ch in ts.columns}

        self.sig_channels = self.sig_channels.union(sig_cols)
        if not sig_cols:
            if self.verbose:
                print(f'no sig: {self.mouse_} {self.session_}')
            return None
        # Replace channels without signal with NaNs.
        ts[list(self.channels - sig_cols)] = np.nan

        # Trim ts data to first and last timepoints with photometry signal.
        first_idx = ts[list(sig_cols)].first_valid_index()
        last_idx = ts[list(sig_cols)].last_valid_index()
        ts = ts.loc[first_idx:last_idx]

        return ts


class StandardData(DataSet):

    def __init__(self,
                 mice: str | list[str],
                 **kwargs):
        super().__init__(mice, **kwargs)
        self.dataset = 'celia'
        # self.channels = self.set_channels()

    def set_root(self):
        '''Sets the root path for the dataset'''
        return Path('/Volumes/Neurobio/MICROSCOPE/Celia/data/lickTask/')

    def set_data_path(self):
        '''Sets the path to the session data'''
        return self.root / 'headfixed_DAB_data/preprocessed_data'

    def set_data_overview_path(self):
        '''Sets the path to the csv containing session summary'''
        return self.root / 'data_overviews' / 'session_log_all_cohorts.csv'

    def set_session_path(self):
        '''Sets path to single session data'''
        return self.data_path / self.mouse_ / self.session_

    def set_channels(self):
        channels = {'z_grnL', 'z_grnR'}
        return channels


class DeterministicData(DataSet):

    def __init__(self,
                 mice: str | list[str],
                 **kwargs):
        super().__init__(mice, **kwargs)
        self.dataset = 'ally'
        # self.channels = self.set_channels()

    def set_root(self):
        '''Sets the root path for the dataset'''
        return Path('/Volumes/Neurobio/MICROSCOPE/Celia/data/lickTask/headfixed_DAB_data/Ally_data/rDA')

    def set_data_path(self):
        '''Sets the path to the session data'''
        return self.root / 'output_ally_spect_demod_60sec_rolling_GrabRed'

    def set_data_overview_path(self):
        '''Sets the path to the csv containing session summary'''
        return self.root / 'MasterPhotometrywithBehaviorSpecDemod_Ally_red.xlsx'

    def set_session_path(self):
        '''Sets path to single session data'''
        return self.data_path / self.mouse_ / self.session_ / 'photometry'

    def set_save_path(self):
        save_path = self.root / 'figures' / self.label
        if not os.path.exists(os.path.join(save_path, 'metadata')):
            os.makedirs(os.path.join(save_path, 'metadata'))
        return save_path

    def set_channels(self):
        channels = {'z_redL', 'z_redR'}
        return channels

    def load_cohort_dict(self):
        mice = self.mice if isinstance(self.mice, list) else [self.mice]
        cohort = {mouse: 'rDA' for mouse in mice}
        return cohort

    def set_trials_path(self):

        file_path = self.set_session_path()
        trials_path = file_path / f'{self.mouse_}_{self.session_}_behavior_df_full.csv'
        return trials_path

    def sessions_to_load(self,
                         probs: int = 9010,
                         QC_pass: bool = True):

        '''
        Make list of sessions to include for designated mouse

        Args:
            mouse (str):
                Mouse ID.
            probs (int):
                Filter bandit data by probability conditions.
            QC_pass (bool):
                Whether to take sessions passing quality control (True) or
                failing (False).

        Returns:
            dates_list (list):
                List of dates to load in for mouse.
        '''
        session_log = pd.read_excel(self.summary_path, engine='openpyxl')
        session_log = session_log.query('Mouse == @self.mouse_')
        sessions = list(set(session_log.Date.values))
        sessions = ['20' + '_'.join(a + b for a, b in zip(*[iter(str(s_))] * 2))
                    for s_ in sessions]
        return sessions

    def custom_update_columns(self, trials, ts):

        trials = trials.rename(columns={'Direction': 'direction'})
        trials = bf.flag_blocks_for_timeouts(trials)
        return trials, ts


class SplitConditions(DataSet):

    def __init__(self,
                 mice: str | list[str],
                 **kwargs):
        super().__init__(mice, **kwargs)
        self.dataset = 'dan'
        self.channels = self.set_channels()

    def set_root(self):
        '''Sets the root path for the dataset'''
        return Path('/Volumes/Neurobio/MICROSCOPE/Celia/data/lickTask/')

    def set_data_path(self):
        '''Sets the path to the session data'''
        return self.root / 'headfixed_DAB_data/preprocessed_data/Dan_data'

    def set_data_overview_path(self):
        '''Sets the path to the csv containing session summary'''
        return self.root / 'data_overviews' / 'session_log_dan.csv'

    def set_session_path(self):
        '''Sets path to single session data'''
        return self.data_path / self.mouse_ / self.session_

    def set_channels(self):
        channels = {'z_grnL', 'z_grnR'}
        return channels

    def custom_update_columns(self, trials, ts):
        pass
