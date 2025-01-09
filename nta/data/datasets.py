import gc
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(f'{os.path.expanduser("~")}/GitHub/behavior-helpers/')

from bh.data.datasets import HFDataset

from ..events import align
from ..features import behavior_features as bf
from ..features.design_mat import make_design_mat
from ..features.select_trials import clean_data
from ..preprocessing import quality_control as qc


class PhotometryDataset(HFDataset):

    def __init__(self,
                 mice: str | list[str],
                 qc_photo: bool = True,
                 **kwargs):

        super().__init__(mice, **kwargs)

        self.qc_photo = qc_photo

        self.channels = self.set_channels()
        self.hemi_labels = {'L': 'Left Hemisphere', 'R': 'Right Hemisphere'}
        self.sig_channels = set()

    def load_data(self):

        # Initizalize attributes that will hold data.

        # Load all data.
        if not isinstance(self.mice, list):
            self.mouse_ = self.mice
            multi_sessions = self.read_multi_sessions(self.qc_params)
            self.ts = multi_sessions.get('ts')
            self.trials = multi_sessions.get('trials')
        else:
            multi_mice = self.read_multi_mice(self.qc_params, keys=['trials', 'ts'])
            # Store data from multi_mice as attributes of dataset.
            self.ts = multi_mice.get('ts')
            self.trials = multi_mice.get('trials')

        self.trials = bf.order_sessions(self.trials)

        self.sig_channels = (list(self.sig_channels)
                             if isinstance(self.sig_channels, set) else [])
        self.sig_channels.sort()

        # Some validation steps on loaded data.
        self.get_sampling_freq()
        self.check_event_order()

        # Downcast datatypes to make more memory efficient.
        self.downcast_dtypes()
        print(f'{self.trials.Session.nunique()} total sessions loaded in')

        gc.collect()

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

    def set_timeseries_path(self):
        '''Set path to timeseries data file.'''
        file_path = self.set_session_path()
        fname = f'{self.mouse_}_{self.session_}_timeseries.parquet.gzip'
        ts_path = file_path / fname
        return ts_path

    def define_ts_dtypes(self):

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
            'state_CueP': np.int8,
            'responseTime': np.int8,
            'state_ENL_preCueP': np.int8,
            'session': 'object',
            'fs': np.float16
        }
        if self.user != 'celia':  # updated naming of session column
            ts_dtypes['Session'] = ts_dtypes.pop('session')

        return ts_dtypes

    def define_ts_cols(self):

        ts_dtypes = self.define_ts_dtypes()
        usecols = list(ts_dtypes.keys())
        usecols.extend(list(self.channels) + list(self.ts_add_cols))
        usecols = list(set(usecols))
        return ts_dtypes, usecols

    def custom_dataset_pp(self, trials, ts, **kwargs):
        ts = self.eval_photo_sig(ts)
        if ts is None:
            return trials, None
        # if kwargs:
        #     ts = make_design_mat(ts, trials, **kwargs)
        return trials, ts

    def cleanup_cols(self, df_dict):
        '''Remove unnecessary columns to minimize memory usage.'''
        return df_dict

    def get_sensors(self, prefix='z_'):
        channels = ['grnL', 'redL', 'grnR', 'redR']
        sensors = {f'{prefix}{k}': v
                   for k, v in zip(channels, self.cohort.get(self.mouse_))}
        return sensors

    def eval_photo_sig(self, ts):

        '''
        Run QC on photometry channels, filtering out data from sessions
        where full session lacked real signal.
        '''

        # If no photometry channels passed QC, move on to next session.
        sensors = self.get_sensors()  # only evaluates on z-scored ts
        if self.qc_photo:
            sig_cols = {ch for ch in self.channels
                        if not qc.is_normal(ts.get(ch, None),
                                            sensor=sensors.get(ch),
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

        if np.any(self.ts.get('fs', False)):
            # In case sampling rate was stored explicitly
            assert self.ts['fs'].nunique() == 1, 'multiple sampling rates logged'
            self.fs = self.ts['fs'].unique()[0]
            self.tstep = 1 / self.fs
            return None

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

    def align_to_event(self, events='', cleaning_params={}, **kwargs):

        store_ts = kwargs.pop('store_ts', False)
        align_params = {
            'window': (1, 2),
            'quantify_peaks': True,
            'fs': self.fs,
            'channel': self.sig_channels
        }

        for key, arg in kwargs.items():
            align_params[key] = arg
        print(align_params)

        self.trials_aligned = self.trials.copy()
        for event in events:
            self.trials_aligned = align.align_photometry_to_event(
                self.trials_aligned,
                self.ts,
                aligned_event=event,
                **align_params
            )

        if cleaning_params:
            self.post_alignment_cleaning(cleaning_params, store_ts)

    def post_alignment_cleaning(self, cleaning_params, store_ts=False):

        print('cleaning data post alignment')

        # Now it's ok to start dropping trials because we've stored photometry
        # snippets.

        # Remove trials missing licks (selection or consumption).
        if cleaning_params['drop_timeouts']:
            self.ts_aligned, self.trials_aligned = align.trim_trials_without_licks(
                self.ts,
                self.trials_aligned
            )
        else:
            self.ts_aligned = self.ts.copy()

        # Clean up data (e.g. to remove penalties, timeouts, late blocks, etc).
        self.trials_aligned, self.ts_aligned, self.dropped_trials = clean_data(
            self.trials_aligned,
            self.ts_aligned,
            **cleaning_params
        )
        cleaning_params['dropped_trials'] = self.dropped_trials
        if not store_ts: del self.ts_aligned
        gc.collect()


class ProbHFPhotometryTails(PhotometryDataset):

    '''
    Bypasses standard session trimming to include all data, even at beginning
    and end of a session.
    '''
    def __init__(self,
                 mice: str | list[str],
                 **kwargs):

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


class SplitConditions(PhotometryDataset):

    def __init__(self,
                 mice: str | list[str],
                 **kwargs):
        super().__init__(mice, **kwargs)
        self.dataset = 'dan'
        self.channels = self.set_channels()

    def set_data_path(self):
        '''Sets the path to the session data'''
        return self.root / 'headfixed_DAB_data/preprocessed_data/Dan_data'

    def set_data_overview_path(self):
        '''Sets the path to the csv containing session summary'''
        return self.root / 'data_overviews' / 'session_log_dan.csv'

    def set_session_path(self):
        '''Sets path to single session data'''
        return self.data_path / self.mouse_ / self.session_