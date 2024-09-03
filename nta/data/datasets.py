import gc
import getpass
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ..features import behavior_features as bf
from ..features.design_mat import make_design_mat
from ..preprocessing import quality_control as qc
from ..utils import (cast_object_to_category, downcast_all_numeric,
                     load_config_variables)


class Dataset(ABC):

    def __init__(self,
                 mice: str | list[str],
                 user: str = 'celia',
                 verbose: bool = False,
                 qc_photo: bool = True,
                 qc_params: dict = {},
                 label: str = '',
                 save: bool = True,
                 add_cols: dict[set] = {},
                 session_cap: int = None):

        self.mice = mice
        self.user = user
        self.verbose = verbose
        self.qc_photo = qc_photo
        self.label = label if label else self.mice
        self.ts_add_cols = add_cols.get('ts', set())
        self.trls_add_cols = add_cols.get('trials', set())
        self.session_cap = session_cap  # max number of sessions per mouse

        # Set up paths and standard attributes.
        self.root = self.set_root()
        self.config_path = self.set_config_path()
        self.data_path = self.set_data_path()
        self.summary_path = self.set_data_overview_path()
        self.save = save
        if self.save:
            self.save_path = self.set_save_path()
        self.cohort = self.load_cohort_dict()
        self.palettes = load_config_variables(self.config_path)
        self.add_mouse_palette()

        # Initizalize attributes that will hold data.
        self.ts = pd.DataFrame()
        self.trials = pd.DataFrame()
        self.channels = self.set_channels()
        self.sig_channels = set()

        # Load all data.
        if not isinstance(mice, list):
            self.mouse_ = mice
            multi_sessions = self.read_multi_sessions(qc_params)
            self.ts = multi_sessions.get('ts')
            self.trials = multi_sessions.get('trials')
        else:
            self.read_multi_mice(qc_params)

        self.trials = bf.order_sessions(self.trials)

        # Some validation steps on loaded data.
        self.get_sampling_freq()
        self.check_event_order()

        # Downcast datatypes to make more memory efficient.
        self.downcast_dtypes()

        gc.collect()

    @abstractmethod
    def set_root(self):
        '''Sets the root path for the dataset'''
        pass

    @abstractmethod
    def set_config_path(self):
        '''Sets the path to config file'''
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

    @abstractmethod
    def check_event_order(self):
        '''Expected order of events to ensure data is sequencing correctly.'''
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
        # Check for state labeling consistency.
        trials = bf.match_state_left_right(trials)

        # Add standard set of analysis columns.
        trials, ts = bf.add_behavior_cols(trials, ts)
        trials = trials.rename(columns={'-1reward': 'prev_rew'})

        # Rectify error in penalty state allocation.
        # if self.user == 'celia':
        ts['ENL'] = ts['ENL'] + ts['state_ENLP'] + ts.get('state_ENL_preCueP', 0)  # recover original state
        ts['Cue'] = ts['Cue'] + ts['CueP']  # recover original state
        ts = bf.split_penalty_states(ts, penalty='ENLP')
        ts = bf.split_penalty_states(ts, penalty='CueP')
        ts = bf.split_penalty_states(ts, penalty='CueP', cuep_ref_enl=True)

        if 'trial_clock' not in ts.columns:
            ts['trial_clock'] = 1 / ts['fs']
            ts['trial_clock'] = ts.groupby('nTrial', observed=True)['trial_clock'].cumsum()

        return trials, ts

    def custom_update_columns(self, trials, ts):
        '''Column updates that are dataset-specific.'''
        return trials, ts

    def cleanup_cols(self, df_dict):
        '''Remove unnecessary columns to minimize memory usage.'''
        return df_dict

    def set_timeseries_path(self):
        '''Set path to timeseries data file.'''
        file_path = self.set_session_path()
        fname = f'{self.mouse_}_{self.session_}_timeseries.parquet.gzip'
        ts_path = file_path / fname
        return ts_path

    def set_trials_path(self):
        '''Set path to trial-level data file.'''

        file_path = self.set_session_path()
        if self.user != 'ally':
            trials_path = file_path / f'{self.mouse_}_trials.csv'
        else:
            trials_path = file_path / 'photometry' / f'{self.mouse_}_trials.csv'
        return trials_path

    def define_data_dtypes(self):

        trial_dtypes = {
            'nTrial': np.int32,
            'Mouse': 'object',
            'Date': 'object',
            'Session': 'object',
            'Condition': 'object',
            'tSelection': np.int16,
            'direction': np.float32,
            'Reward': np.float32,
            'T_ENL': np.int16,
            'n_ENL': np.int8,
            'n_Cue': np.int8,
            'State': np.float32,
            'selHigh': np.float32,
            'iBlock': np.int8,
            'blockLength': np.int8,
            'iInBlock': np.int8,
            'flag_block': 'bool',
            'timeout': 'bool',
            'Switch': np.float32
        }

        ts_dtypes = {
            'session_clock': 'float',
            'nTrial': np.int32,
            'iSpout': np.int8,
            'ENLP': np.int8,
            'CueP': np.int8,
            'ENL': np.int8,
            'Cue': np.int8,
            'Select': np.int8,
            'stateConsumption': np.int8,
            'Consumption': np.int8,
            'state_ENLP': np.int8,
            'session': 'object',
            'fs': np.float16
        }

        if self.user != 'celia':  # updated naming of session column
            ts_dtypes['Session'] = ts_dtypes.pop('session')

        return trial_dtypes, ts_dtypes

    def load_session_data(self):
        '''Loads data from single session'''
        trials_path = self.set_trials_path()
        ts_path = self.set_timeseries_path()

        if not (ts_path.exists() & trials_path.exists()):
            if self.verbose: print(f'skipped {self.mouse_} {self.session_}')
            return None, None

        trial_dtypes, ts_dtypes = self.define_data_dtypes()

        usecols = list(trial_dtypes.keys())
        trials = pd.read_csv(trials_path, index_col=None, dtype=trial_dtypes,
                             usecols=usecols)

        usecols = list(ts_dtypes.keys())
        usecols.extend(['z_grnL', 'z_grnR'] + list(self.ts_add_cols))
        usecols = list(set(usecols))

        # Load timeseries data but be forgiving about missing columns.
        while usecols:
            try:
                ts = (pd.read_parquet(ts_path, columns=usecols)
                        .astype(ts_dtypes))
                # Create session column to match across dataframes.
                if 'session' in ts.columns:
                    ts = ts.rename(columns={'session': 'Session'})
                return ts, trials
            except ValueError as e:
                # Extract the missing column name from the error message.
                re_match = [re.search(r'\((.*?)\)|"(.*?)"', str(e)),
                            re.search(r"'(.+)'", str(e))]
                re_match = [s for s in re_match if s is not None]

                for s in re_match:
                    if isinstance(s, re.Match) & (s.group(1) in usecols):
                        missing_col = s.group(1)
                        usecols.remove(missing_col)
                        break
                else:
                    # In the case we can't find missing column.
                    raise e
        raise ValueError('All specified columns missing from parquet file.')

    def load_cohort_dict(self):
        '''Load lookup table for sensor expressed in each mouse of cohort.'''
        cohort = load_config_variables(self.root, 'cohort')['cohort']
        return cohort

    def sessions_to_load(self,
                         probs: int | str = 9010,
                         QC_pass: bool = True,
                         **kwargs) -> list:

        '''
        Make list of sessions to include for designated mouse

        Args:
            probs:
                Filter bandit data by probability conditions.
            QC_pass:
                Whether to take sessions passing quality control (True) or
                failing (False).

        Returns:
            dates_list:
                List of dates to load in for mouse, sorted from earliest to
                latest.
        '''

        session_log = pd.read_csv(self.summary_path)
        if not isinstance(QC_pass, list):
            QC_pass = [QC_pass]
        if isinstance(probs, list):
            probs = [str(p) for p in probs]
        else:
            probs = [str(probs)]

        # Compose query.
        session_log_mouse = session_log.query(f'Mouse == "{self.mouse_}" \
                                              & Condition.isin({probs})')
        q = f'Mouse == "{self.mouse_}" & Condition.isin({probs}) \
            & Pass.isin({QC_pass})' + kwargs.get('query', '')
        session_log = session_log.query(q)
        if self.verbose:
            print(f'{self.mouse_}: {len(session_log)} of',
                  f' {len(session_log_mouse)} sessions meet criteria')
        return sorted(list(set(session_log.Date.values)))

    def get_max_trial(self, full_sessions: dict) -> int:

        '''
        Get maximum trial ID to use for unique trial ID assignment.
        Importantly, also confirm that max trial matches between dataframes.

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
        Aggregate multiple sessions by renumbering trials to provide unique
        trial ID for each trial. Store original id in separate column.

        Args:
            sub_sessions:
                Smaller unit to be concatenated onto growing aggregate df.
            full_sessions:
                Core larger unit updated with aggregating data.

        Returns:
            full_sessions:
                Original full_sessions data now containing sub_sessions data.
        '''

        max_trial = self.get_max_trial(full_sessions)

        # Iterate over both trial and timeseries data.
        for key, ss_vals in sub_sessions.items():

            # Store original trial ID before updating with unique value.
            if 'nTrial_orig' not in ss_vals.columns:
                ss_vals['nTrial_orig'] = ss_vals['nTrial'].copy()

            # Create session column to match across dataframes.
            # if 'session' in ss_vals.columns:
            #     ss_vals = ss_vals.rename(columns={'session': 'Session'})
            if 'Session' not in ss_vals.columns:
                ss_vals['Session'] = '_'.join([self.mouse_, self.session_])

            # Add max current trial value to all new trials before concat.
            tmp_copy = ss_vals.copy()
            tmp_copy['nTrial'] += max_trial
            full_sessions[key] = pd.concat((full_sessions[key], tmp_copy))
            full_sessions[key] = full_sessions[key].reset_index(drop=True)

        # Assert that new dataframes have matching max trial ID.
        _ = self.get_max_trial(full_sessions)

        return full_sessions

    def read_multi_mice(self,
                        qc_params: dict,
                        **kwargs):

        '''
        Load in sessions by mouse and concatenate into one large dataframe
        keeping every trial id unique.

        Stores data from multi_mice dict
        {'trials': trials data,'timeseries': timeseries data}
        as attributes of DataSet.

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
                            qc_params: dict,
                            **kwargs) -> dict:

        '''
        Load in multiple sessions for a single mouse and concatenate into one
        dataframe, keeping every trial id unique.

        Returns:
            multi_sesions:
                {'trials': trials data, 'timeseries': timeseries data}
        '''
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

            trials_matched = self.cleanup_cols(trials_matched)

            multi_sessions = self.concat_sessions(sub_sessions=trials_matched,
                                                  full_sessions=multi_sessions)

            if self.at_session_cap(multi_sessions): break

        # TODO: QC all mice sessions by ENL penalty rate set per mouse

        gc.collect()

        return multi_sessions

    def at_session_cap(self, multi_sessions):

        ''''
        Check whether number of sessions for a given mouse has reached a given
        session cap (max number of sessions per mouse to load). If no session
        cap provided, return false and load all data.
        '''

        if self.session_cap is None:
            return False

        if multi_sessions['trials'].Session.nunique() >= self.session_cap:
            return True

        return False

    def eval_photo_sig(self, ts):

        '''
        Run QC on photometry channels, filtering out data from sessions
        where full session lacked real signal.
        '''

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

    def get_sampling_freq(self, timestamps=None):

        '''
        Calculate a sampling frequency based on the interval between timesteps
        in timeseries.

        Args:
            timestamps:
                Series of timestamps corresponding to sampling rate for data.

        Returns:
            tstep:
                Interval (in seconds) between individual samples in the data.
            fs:
                Sampling frequency (in Hz) of the timeseries.
        '''

        if timestamps is None:
            tstamps = self.ts.session_clock.copy()
        elif not isinstance(timestamps, pd.Series):
            tstamps = pd.Series(timestamps)
        else:
            tstamps = timestamps.copy()

        tsteps = (tstamps
                  .reset_index(drop=True)
                  .diff()
                  .dropna()
                  .astype('float')
                  .round(5))

        tsteps = tsteps.where(abs(tsteps) < 0.2).dropna()
        tsteps_consistency = (tsteps
                              .value_counts(normalize=True)
                              .max())

        if tsteps_consistency < 0.99:
            print('multiple sampling rates detected')
            if any(tsteps == 0) and (np.mean(tsteps == 0) < 1e-5):
                tsteps = tsteps[tsteps > 0]  # presumably due to float limits
            is_close = (tsteps.max() - tsteps.min()) < 0.001
            low_err = tsteps.var() / tsteps.mean()
            if not (is_close and low_err):
                raise ValueError('cannot reconcile multiple sampling rates')
            self.tstep = round(tsteps.mean(), 5)
            self.fs = round(1 / self.tstep, 5)
            print(f'mean fs = {self.fs}')
        else:
            self.tstep = round(tsteps.mode().squeeze(), 5)
            self.fs = round(1 / self.tstep, 5)

    def add_mouse_palette(self):

        '''Set up some consistent mapping to distinguish mice in plots.'''
        pal = sns.color_palette('deep', n_colors=len(self.mice))
        self.palettes['mouse_pal'] = {mouse: color for mouse, color
                                      in zip(self.mice, pal)}

    def downcast_dtypes(self):

        '''
        Downcast columns in trial and timeseries dataframes by datatype if
        possible.
        '''
        self.trials = downcast_all_numeric(self.trials)
        self.ts = downcast_all_numeric(self.ts)
        self.ts = cast_object_to_category(self.ts)
        self.trials = cast_object_to_category(self.trials)


class ProbHFPhotometry(Dataset):

    def __init__(self,
                 mice: str | list[str],
                #  user: str = 'celia',
                 **kwargs):

        assert 'celia' in getpass.getuser().lower(), (
            'Please write your own Dataset class')

        super().__init__(mice, **kwargs)
        # self.user = user

    def set_root(self):
        '''Sets the root path for the dataset'''
        if 'celia' in getpass.getuser().lower():
            root = Path('/Volumes/Neurobio/MICROSCOPE/Celia/data/lickTask/')
        else:
            raise NotImplementedError('Need path for ProbHFPhotometry')
        return root

    def set_data_path(self):
        '''Sets the path to the session data'''
        match self.user:
            case 'celia':
                prefix = self.root / 'headfixed_DAB_data'
            case 'kevin':
                prefix = self.root / 'headfixed_DAB_data/Kevin_data'
        return prefix / 'preprocessed_data'

    def set_data_overview_path(self):
        '''Sets the path to the csv containing session summary'''
        match self.user:
            case 'celia':
                fname = 'session_log_all_cohorts.csv'
            case 'kevin':
                fname = 'session_log_Kevin.csv'
        return self.root / 'data_overviews' / fname

    def set_session_path(self):
        '''Sets path to single session data'''
        return self.data_path / self.mouse_ / self.session_

    def set_channels(self):
        channels = {'z_grnL', 'z_grnR'}
        return channels

    def cleanup_cols(self, df_dict):

        '''Remove unnecessary columns to minimize memory usage.'''

        # Drop columns that aren't typically accessed for analysis but were
        # necessary for preprocessing.
        cols_to_drop = {'state_ENL_preCueP', 'state_CueP', 'state_ENLP',
                        'stateConsumption', 'CueP', 'iLick', 'ILI',
                        'bout_group', 'cons_bout'
                        } & set(df_dict['ts'].columns)
        cols_to_drop = list(cols_to_drop - self.ts_add_cols)
        df_dict['ts'] = df_dict['ts'].drop(columns=cols_to_drop)

        cols_to_drop = {'k1', 'k2', 'k3', '+1seq2', 'RL_seq2', 'RL_seq3',
                        '-1seq3', '+1seq3',
                        } & set(df_dict['trials'].columns)
        col_to_drop = list(cols_to_drop - self.trls_add_cols)
        df_dict['trials'] = df_dict['trials'].drop(columns=col_to_drop)

        gc.collect()

        return df_dict

    def check_event_order(self):

        '''
        Test to ensure no events appear to occur out of task-defined trial
        order. Note, it's expected that a few edge cases may be given the wrong
        trial ID. This seems to only happen for ENLPs that happen at trial time
        of 0 (assigned to preceding trial. They can be  dealt with, but any
        other cases should raise an alarm.
        '''

        trial_event_order = {
            'ENLP': 1,
            'state_ENLP': 1,
            'CueP': 1,
            'state_CueP': 1,
            'ENL': 2,
            'Cue': 3,
            'Select': 4,
            'Consumption': 5
        }

        ts = self.ts.copy()
        ts['event_order'] = np.nan

        for event, val in trial_event_order.items():
            if event not in ts.columns: continue
            ts.loc[ts[event] == 1, 'event_order'] = val

        # Should be monotonic increase. Any events out of order get flagged.
        out_of_order = (ts.dropna(subset='event_order')
                        .groupby('nTrial', observed=True)['event_order']
                        .diff() < 0)

        ooo = ts.loc[out_of_order[out_of_order].index]
        ooo_trials = ooo.nTrial.unique()

        ts['flag_ooo'] = np.nan
        ts.loc[ooo.index, 'flag_ooo'] = 1
        post_ooo = (ts
                    .query('nTrial.isin(@ooo_trials)')
                    .groupby('nTrial')['flag_ooo']
                    .ffill(1)
                    .sum())
        assert (ooo['ENLP'].all()) & (~any(ooo[['Cue', 'Select', 'Consumption', 'ENL']].any())), (
               'events out of order beyond ENLP edge cases')
        assert post_ooo == len(ooo), (
            'rows out of order following ENLP edge cases')

        # Replace mistrialed events with NaNs (because fixing timing tricky
        # and these are very rare).
        self.ts.loc[ooo.index, 'ENLP'] = np.nan


class ProbHFPhotometryTails(ProbHFPhotometry):

    '''
    Bypasses standard session trimming to include all data, even at beginning
    and end of a session.
    '''
    def __init__(self,
                 mice: str | list[str],
                 **kwargs):

        assert 'celia' in getpass.getuser().lower(), (
            'Please write your own Dataset class')

        super().__init__(mice, **kwargs)

    def read_multi_sessions(self,
                            qc_params,
                            **kwargs) -> dict:

        from ..features.select_trials import match_trial_ids

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

            # Trial level quality control needs to come at the end.
            # Instead of qc.included_trials(), just match ids without trimming.
            trials, ts = match_trial_ids(trials, ts, allow_discontinuity=False)
            trials_matched = {'trials': trials, 'ts': ts}

            trials_matched = self.cleanup_cols(trials_matched)

            multi_sessions = self.concat_sessions(sub_sessions=trials_matched,
                                                  full_sessions=multi_sessions)

            if self.at_session_cap(multi_sessions): break

        gc.collect()

        return multi_sessions


class DeterministicData(Dataset):

    def __init__(self,
                 mice: str | list[str],
                 **kwargs):
        super().__init__(mice, **kwargs)
        self.dataset = 'ally'
        # self.channels = self.set_channels()

    def set_root(self):
        '''Sets the root path for the dataset'''

        if 'celia' in getpass.getuser().lower():
            root = Path('/Volumes/Neurobio/MICROSCOPE/Celia/data/lickTask/headfixed_DAB_data/Ally_data/rDA')
        else:
            raise NotImplementedError('Need path DeterministicData')
        return root

    def set_config_path(self):

        return self.root

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
        fname = f'{self.mouse_}_{self.session_}_behavior_df_full.csv'
        trials_path = file_path / fname
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
        sessions = ['20'+'_'.join(a + b for a, b in zip(*[iter(str(s_))] * 2))
                    for s_ in sessions]
        return sessions

    def custom_update_columns(self, trials, ts):

        trials = trials.rename(columns={'Direction': 'direction'})
        trials = bf.flag_blocks_for_timeouts(trials)
        return trials, ts


class SplitConditions(Dataset):

    def __init__(self,
                 mice: str | list[str],
                 **kwargs):
        super().__init__(mice, **kwargs)
        self.dataset = 'dan'
        self.channels = self.set_channels()

    def set_root(self):
        '''Sets the root path for the dataset'''
        if 'celia' in getpass.getuser().lower():
            root = Path('/Volumes/Neurobio/MICROSCOPE/Celia/data/lickTask/')
        else:
            raise NotImplementedError('Need path for SplitConditions')
        return root

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
