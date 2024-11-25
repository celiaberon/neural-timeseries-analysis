import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy.signal as signal
import seaborn as sns


class HeadfixedTask:

    def __init__(self):

        self.cue_duration = 0.08  # in seconds
        self.enl_dist = np.arange(1, 2.1, step=0.25)  # in seconds
        self.max_selection_time = 3  # in seconds
        self.consumption_period = 3  # in seconds
        self.reward_prob = 0.9  # high spout reward probability
        self.nTrial = 0
        self.state = np.random.choice([0, 1])  # initial high spout
        self.timestep = 1/50  # in seconds
        self.ts = pd.DataFrame()
        self.trials = pd.DataFrame()

    # def make_trial(self):

    #     '''
    #     Create timeseries for trial of major task events based on  task
    #     specifications.
    #     '''

    #     tstep = self.timestep  # sec = index_units * tstep

    #     # Draw current ENL duration as index for 50 Hz samples.
    #     penalty = np.random.choice([0, 1], p=[0.8, 0.2])
    #     enl = int(npr.choice(self.enl_dist) // tstep)
    #     penalty_idx = int(np.random.random() * enl) if penalty else 0
    #     cue_dur = self.cue_duration / tstep

    #     # Draw current selection time as index for 50 Hz samples.
    #     sel_time = int(npr.normal(loc=0.3, scale=0.1) // tstep)
    #     timeout = (sel_time * tstep) > self.max_selection_time

    #     # Create first consumption lick as noisy 135 Hz oscillator from
    #     # selection.
    #     second_lick_delay = npr.normal(loc=.135, scale=.02)
    #     second_lick_delay = np.clip(second_lick_delay, 0, 3) // tstep
    #     cons_lick = sel_time + int(second_lick_delay)

    #     # Number of samples comprising the full trial as a timeseries.
    #     n_samples = int(sum((penalty_idx,
    #                          enl,
    #                          cue_dur,
    #                          sel_time,
    #                          (self.consumption_period / tstep))))

    #     # Make dataframe for current trial as a timeseries of evolving states.
    #     states = ['Cue', 'Select', 'Consumption']
    #     curr_trial_ts = {state: np.zeros(n_samples) for state in states}
    #     curr_trial_ts = pd.DataFrame(curr_trial_ts)
    #     curr_trial_ts['step'] = tstep
    #     curr_trial_ts.loc[enl + penalty_idx, 'Cue'] = 1
    #     if penalty:
    #         curr_trial_ts.loc[penalty_idx, 'ENLP'] = 1

    #     # Trial can only contain selection and consumption events if selection
    #     # occurs before max selection time.
    #     if not timeout:
    #         curr_trial_ts.loc[enl + sel_time + cue_dur, 'Select'] = 1
    #         curr_trial_ts.loc[enl + cons_lick + cue_dur, 'Consumption'] = 1
    #     curr_trial_ts['nTrial'] = self.nTrial

    #     # Make dataframe for current trial as a timeseries of evolving states.
    #     current_trial = {'nTrial': self.nTrial,
    #                      'Reward': self.assign_reward(),
    #                      't_cue_to_sel': (sel_time + cue_dur) * tstep,
    #                      't_sel_to_cons': second_lick_delay * tstep,
    #                      't_cue_to_cons': (cons_lick + cue_dur) * tstep,
    #                      't_sel_pre_cons': -second_lick_delay * tstep,
    #                      't_cue_pre_cons': -(cons_lick + cue_dur) * tstep,
    #                      't_cue_pre_sel': -(sel_time + cue_dur) * tstep,
    #                      'enlp_trial': penalty}
    #     current_trial = pd.DataFrame(current_trial, index=[0])

    #     self.nTrial += 1

    #     return curr_trial_ts, current_trial

    def assign_reward(self):

        '''
        Assign reward at reward probability (bypassing any choice mechanic)
        '''

        return npr.choice([-1, 1], p=[1-self.reward_prob, self.reward_prob])

    def assign_state(self):

        '''Cause state transition every 20 trials.'''

        if self.nTrial % 20 == 0:
            self.state = 1 - self.state

    def add_pseudo_columns(self, session_id):

        '''
        Add columns that plotting functions will expect (no current bearing
        on simulated tests.
        '''

        session = f'sim_session_{session_id}'
        self.ts_['Session'] = session
        self.trials_['Session'] = session

    def make_trial(self):

        # Single penalties only, 20% chance of happening.
        penalty = np.random.choice([0, 1], p=[0.8, 0.2])

        # Draw current ENL duration as index for 50 Hz samples.
        enl = npr.choice(self.enl_dist)

        # Time of penalty based on ENL duration.
        t_penalty = int(np.random.random() * enl) if penalty else 0

        # Selection time, with mean of 300 ms.
        sel_time = npr.normal(loc=0.3, scale=0.1)

        # Timeout if greater than max permissble selection time.
        timeout = sel_time > self.max_selection_time

        # Create first consumption lick as noisy 135 Hz oscillator from
        # selection.
        second_lick_delay = npr.normal(loc=.135, scale=.02)
        second_lick_delay = np.clip(second_lick_delay, 0, 3)
        cons_lick = sel_time + second_lick_delay

        # Make dataframe for current trial as a timeseries of evolving states.
        current_trial = {
            'nTrial': self.nTrial,
            'Reward': self.assign_reward(),
            't_cue_to_sel': sel_time + self.cue_duration,
            't_sel_to_cons': second_lick_delay,
            't_cue_to_cons': cons_lick + self.cue_duration,
            't_sel_pre_cons': -second_lick_delay,
            't_cue_pre_cons': -(cons_lick + self.cue_duration),
            't_cue_pre_sel': -(sel_time + self.cue_duration),
            'tSelection': sel_time,
            'enlp_trial': penalty,
            't_penalty': t_penalty,
            'T_ENL': enl,
            'timeout': timeout}
        current_trial = pd.DataFrame(current_trial, index=[0])

        # current_trial_events = pd.DataFrame(trial_events, index=[0])
        self.nTrial += 1

        return current_trial

    def make_trial_timeseries(self, current_trial, tstep=None):

        tstep = tstep or self.timestep  # sec = index_units * tstep

        # ENL penalty occurs at t_penalty
        penalty_idx = int(current_trial['t_penalty'].item() // tstep)

        # ENL idx denotes end of ENL (onset of cue)
        enl_idx = penalty_idx + int(current_trial['T_ENL'].item() // tstep)

        # Cue duration is index at END of cue
        cue_dur_idx = enl_idx + int(self.cue_duration / tstep)

        # Selection lick delayed from end of cue.
        sel_lick_idx = cue_dur_idx + int(current_trial['tSelection'].item() // tstep)

        # Create first consumption lick as noisy 135 Hz oscillator from
        # selection.
        cons_lick_idx = sel_lick_idx + int(current_trial['t_sel_to_cons'].item() // tstep)

        # Number of samples comprising the full trial as a timeseries.
        n_samples = int(sum((sel_lick_idx,
                            (self.consumption_period / tstep))))

        # Make dataframe for current trial as a timeseries of evolving states.
        states = ['Cue', 'Select', 'Consumption', 'ENL', 'ENLP', 'sync_pulse', 'stateConsumption']
        curr_trial_ts = pd.DataFrame({state: np.zeros(n_samples) for state in states})

        curr_trial_ts['nTrial'] = current_trial['nTrial'].item()
        curr_trial_ts['step'] = tstep
        curr_trial_ts['fs'] = 1 / tstep
        curr_trial_ts.loc[0, 'sync_pulse'] = 1

        if current_trial['enlp_trial'].item():
            curr_trial_ts.loc[penalty_idx, 'ENLP'] = 1

        curr_trial_ts.loc[penalty_idx:enl_idx, 'ENL'] = 1
        curr_trial_ts.loc[enl_idx:cue_dur_idx, 'Cue'] = 1  # 'Cue onset

        # Trial can only contain selection and consumption events if selection
        # occurs before max selection time.
        if not current_trial['timeout'].item():
            curr_trial_ts.loc[sel_lick_idx, 'Select'] = 1
            curr_trial_ts.loc[cons_lick_idx, 'Consumption'] = 1
            curr_trial_ts.loc[cons_lick_idx:, 'stateConsumption'] = 1

        curr_trial_ts['iSpout'] = curr_trial_ts['Select'].copy()
        curr_trial_ts['iSpout'] *= (current_trial['nTrial'].item() // 20) % 2

        curr_trial_ts['trial_clock'] = curr_trial_ts['step'].cumsum() - tstep

        return curr_trial_ts

    def generate_session(self, total_trials=300, session_id=0):

        '''
        Generate a session of multiple trials with both timeseries- and trial-
        based representation.
        '''

        session_trials = []
        session_timeseries = []
        for _ in range(total_trials):
            trial = self.make_trial()
            timeseries = self.make_trial_timeseries(trial)
            session_timeseries.append(timeseries)
            session_trials.append(trial)

        session_timeseries = (pd.concat(session_timeseries)
                              .reset_index(drop=True))
        session_timeseries['session_clock'] = session_timeseries.step.cumsum() - self.timestep

        self.ts_ = session_timeseries
        self.trials_ = pd.concat(session_trials).reset_index(drop=True)
        self.add_pseudo_columns(session_id=session_id)

    def generate_multi_sessions(self, n_sessions, trials_per_session=300):

        for session in range(n_sessions):
            self.generate_session(total_trials=trials_per_session,
                                  session_id=session)
            self.ts = pd.concat((self.ts, self.ts_)).reset_index(drop=True)
            self.trials = pd.concat((self.trials, self.trials_)).reset_index(drop=True)

    def generate_timeseries_only(self, tstep=None):

        print(f'generating timeseries with new sampling rate: {round(1 / tstep, 2)}')

        multi_session_timeseries = []
        for session_id, session_data in self.trials.groupby('Session'):
            session_timeseries = []
            for trial_id, trial in session_data.groupby('nTrial'):
                timeseries = self.make_trial_timeseries(trial, tstep)
                session_timeseries.append(timeseries)

            session_timeseries = (pd.concat(session_timeseries)
                                    .reset_index(drop=True))

            # Keep on same overall clock (accounts for slight differences in
            # number of samples resulting from different construction rates).
            session_timeseries['session_clock'] = np.linspace(0, self.ts['session_clock'].iloc[-1], len(session_timeseries))
            session_timeseries['Session'] = session_id
            multi_session_timeseries.append(session_timeseries)
        self.ts_resampled = pd.concat(multi_session_timeseries).reset_index(drop=True)

    def generate_noisy_events(self, mean_amp=1.5, noise=True):

        '''
        Create impulse events as basis for simulated neural data that follow
        a normal distribution around a mean amplitude.
        '''

        for state in ['Cue', 'Consumption']:
            if noise:
                n_events = int(self.ts[state].sum())
                event_amps = npr.normal(mean_amp,
                                        scale=1,
                                        size=n_events)
            else:
                event_amps = np.ones(int(self.ts[state].sum()))
            event_amps = np.clip(event_amps, 0, np.inf)

            if state == 'Consumption':
                event_amps *= self.trials.Reward.values

            col = f'{state}_events'
            self.ts[col] = self.ts[state].copy()
            self.ts.loc[self.ts[state] == 1, col] *= event_amps

        amps = self.ts['Cue_events'] + self.ts['Consumption_events']
        self.ts['amplitudes'] = amps

    def add_gaussian_noise(self, noise=True):

        if not noise:
            return None
        baseline_noise = npr.normal(0,
                                    scale=0.1,
                                    size=len(self.ts))
        self.ts['amplitudes'] += baseline_noise

    def convolve_kernel(self, **kwargs):

        '''
        Convolve event amplitudes with an exponential kernel to mimic slow
        decay of fluorescent sensors.
        '''

        # Generate basis for neural events with noisy amplitudes occurring at
        # behavior/task events.
        self.generate_noisy_events(**kwargs)

        # Add continuous Gaussian noise before convolution to give
        # autoregressive nature to noise.
        self.add_gaussian_noise(**kwargs)

        # Create exponential Gaussian with assymmetric rise and fall kinetics.
        # Std chosen to roughly preserve amplitude of event.
        gauss_filter = signal.windows.gaussian(M=8, std=0.43)
        gauss_filter = np.clip(gauss_filter, 0, np.inf)
        self.gauss_filter = gauss_filter
        convolved_sig = signal.convolve(self.ts.amplitudes.values,
                                        gauss_filter)
        self.ts['amplitudes_gauss'] = convolved_sig[:len(self.ts)]

        exp_filter_pos = signal.windows.exponential(140, 0, tau=15, sym=False)
        self.exp_filter_pos = exp_filter_pos
        pos_signal = self.ts.amplitudes_gauss.values * (self.ts.amplitudes_gauss.values >= 0)
        convolved_sig_pos = signal.convolve(pos_signal,
                                            exp_filter_pos)

        # Separate decay rate for negative levels.
        exp_filter_neg = signal.windows.exponential(140, 0, tau=40,
                                                    sym=False) / 2
        self.exp_filter_neg = exp_filter_neg
        neg_signal = self.ts.amplitudes_gauss.values * (self.ts.amplitudes_gauss.values < 0)
        convolved_sig_neg = signal.convolve(neg_signal,
                                            exp_filter_neg)

        # Combine positive and negative components
        self.ts['grnL'] = convolved_sig_pos[:len(self.ts)] + convolved_sig_neg[:len(self.ts)]

    def plot_event_to_kernel(self):

        '''
        Visualize transoformation of impulse event through convolution with
        Gaussian-exponential kernel
        '''

        # Create single impulse event of magnitude = 3.
        sim_sig = np.zeros(100)
        sim_sig[50] = 3

        # Convolve impulse with gaussian then positive exponential kernel.
        gauss_sig = signal.convolve(sim_sig, self.gauss_filter)
        exp_gauss_sig_pos = signal.convolve(gauss_sig * (gauss_sig >= 0),
                                            self.exp_filter_pos)

        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(sim_sig, lw=1.5, label='simulated event')
        ax.plot(exp_gauss_sig_pos, lw=1.5, alpha=0.8, label='filtered event')

        # Create single impulse event of magnitude = -3.
        sim_sig = np.zeros(100)
        sim_sig[50] = -3

        # Convolve impulse with gaussian then negatvie exponential kernel.
        gauss_sig = signal.convolve(sim_sig, self.gauss_filter)
        exp_gauss_sig_neg = signal.convolve(gauss_sig * (gauss_sig < 0),
                                            self.exp_filter_neg)
        ax.plot(exp_gauss_sig_neg, lw=1.5, alpha=0.8, label='filtered event')

        ax.set(xlabel='n points', ylabel='event magnitude')
        plt.legend()
        sns.despine()
