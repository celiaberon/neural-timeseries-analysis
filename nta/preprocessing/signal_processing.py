import pandas as pd
import numpy as np
from scipy.signal import detrend
from hyphyber.util.sig import gen_sine, downsample, demodulate, fit_reference
from collections import defaultdict

def detrend_signal(raw_signal: pd.Series) -> pd.Series:

    # detrend to correct for bleaching over session
    detrended_signal = pd.Series(detrend(raw_signal.values))

    return detrended_signal


def normalize(x: pd.Series,
              window_length: int,
              adjust_extremes: (float or bool)=False,
              rolling: bool=True) -> pd.Series:

    '''
    Normalize timeseries data between zero and one with/without rolling window:
    norm_x = (x-min(x)) / (max(x) - min(x))

    Args:
        x:
            Timeseries data to be normalized.
        window_length:
            Number of samples to take for each rolling z-score calculation.
            Window is centered within window_length.
        adjust_extremes:
            Percent value by which to cut in from true min/max values for
            more robust estimates.
            If None, use true min/max values.
        rolling:
            Whether (True) or not (False) to use a rolling window or full
            static window.

    Returns:
        normalized_x:
            Normalized timeseries data.
    '''

    if rolling:
        rolling_window = x.rolling(window=window_length, center=True)
    else:
        rolling_window = x.copy() # Mimic rolling window

    lower_bound = rolling_window.min()
    upper_bound = rolling_window.max()

    if adjust_extremes: # instead of .min() and .max()
        lower_bound = rolling_window.quantile(adjust_extremes)
        upper_bound = rolling_window.quantile(1-adjust_extremes)

    normalized_x = (x-lower_bound)/(upper_bound-lower_bound)

    return normalized_x


def deltaF(timeseries: pd.Series,
           detrend: bool=True,
           window_length: int=5000) -> pd.Series:

    '''
    Remove long timescale trend (flatten baseline) from timeseries and
    standardize to z-score. Subtract baseline to compute dF.

    Args:
        timeseries:
            Timeseries data containing column for fluorescence signal.
        fluor_col:
            Column header for fluorescence signal.
        detrend:
            Whether (TRUE) or not (FALSE) to long timescale trend in baseline.
        window_length:
            Number of index points to include in each rolling window, centered
            withing window length.

    Returns:
        deltaF:
            Baseline-subtracted normalized fluorescence timeseries.

    '''

    fluor_data = timeseries.copy()

    if detrend:
        fluor_data = detrend_signal(fluor_data)

    norm_fluor = normalize(fluor_data, window_length=window_length)

    # Rolling median as baseline fluorescence.
    f0 = norm_fluor.rolling(window_length, center=True).median()

    # Delta F (difference in baseline and normalized fluorescence).
    deltaF = norm_fluor - f0

    return deltaF


def rolling_zscore(timeseries: pd.Series,
                   window_length: int=5000,
                   rolling: bool=True) -> pd.Series:

    '''
    Calculate a rolling (or not) z-score on a timeseries as:
    z = (x-mean(x)) / std(x)
    where x is taken by a rolling window over the timeseries.

    Args:
        timeseries:
            Timeseries data to z-score.
        window_length:
            Number of samples to take for each rolling z-score calculation.
            Window is centered within window_length.
        rolling:
            Whether (True) or not (False) to use a rolling window or full
            static window.

    Returns:
        z:
            Z-scored timeseries of ycol.
    '''

    if rolling:
        rolling_window = timeseries.rolling(window=window_length, center=True)
    else:
        rolling_window = timeseries.copy() # Mimic rolling window variable

    rolling_mean = rolling_window.mean()
    rolling_std = rolling_window.std()
    z = (timeseries-rolling_mean)/rolling_std

    return z


def snr_photo_signal(timeseries: pd.DataFrame,
                     signal_col: 'str'=None) -> float:

    '''
    Calculate signal to noise ratio (SNR) using Fast Fourier Transform (FFT).

    Args:
        timeseries:
            DataFrame in timeseries form.
        singal_col:
            Column header in timeseres df containing signal to evaluate.

    Returns:
        ratio:
            Rough approximation of signal to noise ratio.
    '''

    from scipy.fft import fft

    timeseries = timeseries.loc[~timeseries[signal_col].isnull()]

    neural_signal = timeseries[signal_col].values
    yf = fft(neural_signal)
    N = len(neural_signal) # number of sample points
    yf_pos = yf[0:N//2] # positive N points only
    est_signal = 2.0/N * np.abs(yf_pos).max() # power of signal (near zero)
    est_noise = (2.0/N * np.abs(yf_pos)[5000:]).max() # power of noise

    ratio = est_signal / est_noise

    return ratio


def initialize_demod_df(data: pd.DataFrame,
                        metadata: dict,
                        downsample_fs: int,
                        threshold: float=0.5):

    '''
    Create datframe with behavior/neural data sync pulses at sampling rate
    used for demodulated data.

    Args:
        data:
            Timeseries data loaded in from tdt directly.
        metadata:
            Metadata containing sampling frequency of raw timeseries.
        downsample_fs:
            Target frequency after downsampling.
        threshold:
            Threshold for converting downsampled/smoothed signal back into
            binary pulse.

    Returns:
        demod_df:
            Dataframe containing bidirection sync pulses at downsample_fs.

    '''

    if metadata.task_id.values[0].startswith('hf'):
        toBeh = (downsample(data['toBeh'], metadata.sampling_freq.values[0],
                            downsample_fs, method='polyphase') > threshold
                .astype('int'))
        froG = (downsample(data['froG'], metadata.sampling_freq.values[0],
                           downsample_fs, method='polyphase') > threshold)

        demod_df = pd.DataFrame(data={'toBehSys':toBeh,
                                    'fromBehSys':froG})

    else:
       ...

    return demod_df


def extract_metadata_tdt(tdt_file, task_id: str='hf_DAB') -> pd.DataFrame:

    '''
    Extract basic metadata for fiber photometry from tdt file format.

    Args:
        tdt_file:
            Data structure loaded in with tdt software.
        task_id:
            Name referencing behavior task.

    Returns:
        metadata:
            Dataframe containing fibers to analyze and corresponding
            sampling/carrier frequencies.
    '''

    metadata = defaultdict([])

    # Determine fibers that were on from standard deviation on data stream.
    active_fibers = [fiber for fiber in [1, 2] if 
                     np.std(tdt_file.streams[f'Fi{fiber}r'].data[0][5:]>0.05)]
    metadata['active_fibers'] = active_fibers
    metadata['task_id'] = task_id
    # Warning: this comes from user decisions and should probably be set by
    # experimenter!
    fiber_to_side_keys = {1:'R', 2:'L'}

    for fiber in active_fibers:

        # Set left/right label names to correspond to each active fiber.
        metadata['fiber_references'].append(f'fiber_{fiber_to_side_keys[fiber]}_grn')

        # Grab expected carrier frequency for each fiber from tdt.
        metadata['carrier_freq'].append(tdt_file.scalars[f'Fi{fiber}i'].data[1,0])

        # Grab sampling frequency for each fiber from tdt.
        metadata['sampling_freq'].append(tdt_file.streams[f'Fi{fiber}r'].fs)

    return pd.DataFrame(data=metadata).set_index('fiber_references')


def offline_demodulation(data,
                         metadata,
                         tau: int=20,
                         z: bool=True,
                         z_window: int=60,
                         downsample_fs: int=600,
                         bandpass_bw: int=50,
                         **kwargs):

    # Use a short snippet of the signal to fit our offline reference.
    use_points = int(1e4)

    # Create dataframe containing sync pulses at downsampled frequency.
    demod_df = initialize_demod_df(data, metadata, downsample_fs)

    ref = {}
    for fiber in [col for col in data.columns if col.startswith('fiber')]:

        sig = data[fiber].values # photometry signal
        ref_fs = metadata.loc[fiber, 'carrier_freq']
        fs = metadata.loc[fiber, 'sampling_freq']
        win_samples = int(z_window*fs) # num samples for window given in seconds

        # Z-score data using rolling window before demodulation to detrend.
        if z:
            sig = rolling_zscore(sig, window_length=win_samples)
            print('applying first z-score with a 60s rolling window')

        # Convert sample points to timepoints based on sampling frequency.
        tstamps = np.arange(len(sig)) / fs

        # Use snippet to estimate reference sine wave parameters, making sure
        # to bypass z-score window tails. Compare to input reference frequency.
        ref["params_x"], _, _ = fit_reference(sig[win_samples:win_samples+use_points],
                                            tstamps[win_samples:win_samples+use_points],
                                            expected_fs=ref_fs)

        # Remember y (cosine) has a 90 degree phase shift.
        ref["params_y"] = (ref["params_x"][0],
                           ref["params_x"][1],
                           ref["params_x"][2] + np.pi / 2,
                           ref["params_x"][3])

        # Generate new reference sine and cosine waves using empirically fit params.
        ref["ref_x"] = gen_sine(ref["params_x"], tstamps)
        ref["ref_y"] = gen_sine(ref["params_y"], tstamps)

        # Demodulate signal using fit sine/cosine waves for reference signal.
        # Bandpass signal then (conservatively) lowpass filter after downsampling.
        _, _, demod_sig, _ = demodulate(sig,
                                    ref_fs,
                                    ref_x=ref["ref_x"],
                                    ref_y=ref["ref_y"],
                                    demod_tau=tau,
                                    downsample_fs=downsample_fs,
                                    bandpass_bw=bandpass_bw)

        # Ensure NaNs replace any demodulation/z-score beyond rolling window extremities.
        demod_sig[:int(z_window*downsample_fs)] = np.nan
        demod_sig[-int(z_window*downsample_fs):] = np.nan
        demod_data_col = fiber.replace('fiber', 'detrend')
        demod_df[demod_data_col] = demod_sig

    # Trim dataframe to first and last timepoints containing demodulated signal.
    start_idx = demod_df[demod_data_col].first_valid_index()
    end_idx = demod_df[demod_data_col].last_valid_index()
    demod_df = demod_df[start_idx:end_idx].reset_index(drop=True)

    if z:
        # Always include "raw" demodulated signal, even if z-scoring.
        raw = offline_demodulation(data, metadata, tau, z=False,
                                   downsample_fs=downsample_fs, 
                                   bandpass_bw=bandpass_bw, **kwargs)

        # Ensure equivalence/redundancy between sync columns before 
        assert demod_df[['toBehSys','fromBehSys']].equals(raw[['toBehSys','fromBehSys']])

        for fiber in [col for col in data.columns if col.startswith('fiber')]:
            side = fiber[len('fiber_'):]
            demod_df[fiber.replace('fiber', 'raw')] = raw[f'detrend_{side}']

    return demod_df
