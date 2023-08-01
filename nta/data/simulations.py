import numpy as np
import pandas as pd
import scipy.signal as signal
import numpy.random as npr

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
        enl = int(npr.choice(self.enl_dist) // tstep)
        cue_dur = self.cue_duration / tstep

        # Draw current selection time as index for 50 Hz samples.
        selection_time = int(npr.normal(loc=0.3, scale=0.1) // tstep)
        timeout = (selection_time * tstep) > self.max_selection_time
        
        # Create first consumption lick as noisy 135 Hz oscillator from selection.
        second_lick_delay = np.clip(npr.normal(loc=.135, scale=.02), 0, 3) // tstep
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

        return npr.choice([-1,1], p=[1-self.reward_prob, self.reward_prob])

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
        self.trials['Session'] = 'sim_session'


    def generate_session(self, total_trials=300):

        '''
        Generate a session of multiple trials with both timeseries- and trial-
        based representation.
        '''

        session_trials = []
        session_timeseries = []
        while self.nTrial < total_trials:

            timeseries, trials = self.make_trial()
            timeseries['trial_clock'] = self.timestep
            timeseries['trial_clock'] = timeseries['trial_clock'].cumsum()
            session_timeseries.append(timeseries)
            session_trials.append(trials)

        session_timeseries = pd.concat(session_timeseries).reset_index(drop=True)
        session_timeseries['session_clock'] = session_timeseries.sample_interval.cumsum()
        
        self.session = session_timeseries
        self.trials = pd.concat(session_trials).reset_index(drop=True)
        self.add_pseudo_columns()

    def generate_noisy_events(self, mean_amp=1.5, noise=True):

        '''
        Create impulse events as basis for simulated neural data that follow
        a normal distribution around a mean amplitude.
        '''

        for state in ['Cue', 'Consumption']:
            if noise:
                event_amplitudes = npr.normal(mean_amp,
                                            scale=1,
                                            size=int(self.session[state].sum()))
            else:
                event_amplitudes = np.ones(int(self.session[state].sum()))
            event_amplitudes = np.clip(event_amplitudes, 0, np.inf)

            if state=='Consumption':
                event_amplitudes *= self.trials.Reward.values

            self.session[f'{state}_events'] = self.session[state].copy()
            self.session.loc[self.session[state]==1, f'{state}_events'] *= event_amplitudes

        self.session['amplitudes'] = self.session['Cue_events'] + self.session['Consumption_events']

    def add_gaussian_noise(self, noise=True):

        if not noise:
            return None
        baseline_noise = npr.normal(0,
                                    scale=0.1,
                                    size=len(self.session))
        self.session['amplitudes'] += baseline_noise

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
        gauss_filter = np.clip(signal.windows.gaussian(M=10, std=1), 0, np.inf)
        convolved_sig = signal.convolve(self.session.amplitudes.values,
                                        gauss_filter)
        self.session['amplitudes_gauss'] = convolved_sig[:len(self.session)]

        exp_filter = signal.windows.exponential(100, 0, tau=20, sym=False)
        convolved_sig = signal.convolve(self.session.amplitudes_gauss.values,
                                        exp_filter)
        self.session['z_grnL'] = convolved_sig[:len(self.session)]
