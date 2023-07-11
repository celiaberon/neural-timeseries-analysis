import numpy as np
import pandas as pd
import scipy.signal as signal

class HeadfixedTask:

    def __init__(self):

        self.cue_duration = 0.08 # in seconds
        self.enl_dist = np.arange(1,2.1, step=0.25) # in seconds
        self.max_selection_time = 3 # in seconds
        self.consumption_period = 3 # in seconds
        self.reward_prob = 0.9 # high spout reward probability
        self.nTrial = 0
        self.state = np.random.choice([0,1]) # initial high spout
        self.timestep = 1/50 # in seconds

    def make_trial(self):

        '''
        Create timeseries for trial of major task events based on  task
        specifications.
        '''

        tstep = self.timestep # sec = index_units * tstep

        # Draw current ENL duration as index for 50 Hz samples.
        enl = int(np.random.choice(self.enl_dist) // tstep)
        cue_dur = self.cue_duration / tstep

        # Draw current selection time as index for 50 Hz samples.
        selection_time = int(np.random.normal(loc=0.3, scale=0.1) // tstep)
        timeout = (selection_time * tstep) > self.max_selection_time
        
        # Create first consumption lick as noisy 135 Hz oscillator from selection.
        second_lick_delay = np.random.normal(loc=.135, scale=.05) // tstep
        first_cons_lick = selection_time + int(second_lick_delay)

        # Number of samples comprising the full trial as a timeseries.
        n_samples = int(sum((enl, 
                             cue_dur,
                             selection_time, 
                             (self.consumption_period / tstep))))
        
        # Make dataframe for current trial as a timeseries of evolving states.
        curr_trial_ts = {state:np.zeros(n_samples) for state in ['Cue', 'Select', 'Consumption']}
        curr_trial_ts = pd.DataFrame(curr_trial_ts)
        curr_trial_ts['sample_interval'] = tstep
        curr_trial_ts.loc[enl, 'Cue'] = 1

        # Trial can only contain selection and consumption events if selection
        # occurs before max selection time.
        if not timeout:
            curr_trial_ts.loc[enl + selection_time + cue_dur, 'Select'] = 1
            curr_trial_ts.loc[enl + first_cons_lick + cue_dur, 'Consumption'] = 1
        curr_trial_ts['nTrial'] = self.nTrial

        # Make dataframe for current trial as a timeseries of evolving states.
        current_trial = {'nTrial': self.nTrial,
                         'Reward': self.assign_reward(),
                         't_cue_to_sel': (selection_time + cue_dur) * tstep,
                         't_sel_to_cons': second_lick_delay * tstep,
                         't_cue_to_cons': (first_cons_lick + cue_dur) * tstep,
                         't_sel_pre_cons': -second_lick_delay * tstep,
                         't_cue_pre_cons': -(first_cons_lick + cue_dur) * tstep,
                         't_cue_pre_sel': -(selection_time + cue_dur) * tstep }
        current_trial = pd.DataFrame(current_trial, index=[0])

        self.nTrial += 1

        return curr_trial_ts, current_trial
        
    def assign_reward(self):

        '''Assign reward at reward probability (bypassing any choice mechanic)'''

        return np.random.choice([-1,1], p=[1-self.reward_prob, self.reward_prob])

    def assign_state(self):

        '''Cause state transition every 20 trials.'''

        if self.nTrial % 20 == 0:
            self.state = 1 - self.state

    def add_pseudo_columns(self):

        '''
        Add columns that plotting functions will expect (no current bearing
        on simulated tests.
        '''

        self.session['session'] = 'sim_session'
        self.session['iSpout'] = self.session['Select'].copy()
        self.session['iSpout'] *= (self.session.nTrial // 20) % 2

    def generate_session(self, total_trials=300):

        '''
        Generate a session of multiple trials with both timeseries- and trial-
        based representation.
        '''

        session_trials = []
        session_timeseries = []
        while self.nTrial < total_trials:

            timeseries, trials = self.make_trial()
            session_timeseries.append(timeseries)
            session_trials.append(trials)

        session_timeseries = pd.concat(session_timeseries).reset_index(drop=True)
        session_timeseries['session_clock'] = session_timeseries.sample_interval.cumsum()
        
        self.session = session_timeseries
        self.add_pseudo_columns()
        self.trials = pd.concat(session_trials).reset_index(drop=True)

    def generate_noisy_events(self, mean_amp=3):

        '''
        Create impulse events as basis for simulated neural data that follow
        a normal distribution around a mean amplitude.
        '''

        for state in ['Cue', 'Consumption']:
            event_amplitudes = np.random.normal(mean_amp,
                                                scale=1,
                                                size=int(self.session[state].sum()))
            
            if state=='Consumption':
                event_amplitudes *= self.trials.Reward.values

            self.session[f'{state}_events'] = self.session[state].copy()
            self.session.loc[self.session[state]==1, f'{state}_events'] *= event_amplitudes

        self.session['amplitudes'] = self.session['Cue_events'] + self.session['Consumption_events']

    def convolve_kernel(self):

        '''
        Convolve event amplitudes with an exponential kernel to mimic slow
        decay of fluorescent sensors.
        '''

        self.generate_noisy_events()
        exp_filter = signal.windows.exponential(20, 0, tau=5, sym=False)
        self.session['z_grnL'] = signal.convolve(self.session.amplitudes.values, exp_filter)[:len(self.session)]
