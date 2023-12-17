import numpy as np
import pandas as pd
from scipy import signal


def detrend_signal(raw_signal: pd.Series) -> pd.Series:

    '''Detrend to correct for bleaching over session'''

    detrended_signal = pd.Series(signal.detrend(raw_signal.values))

    return detrended_signal


def normalize(x: pd.Series,
              window_length: int,
              adjust_extremes: float | bool = False,
              rolling: bool = True) -> pd.Series:

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
        rolling_window = x.copy()  # Mimic rolling window

    lower_bound = rolling_window.min()
    upper_bound = rolling_window.max()

    if adjust_extremes:  # instead of .min() and .max()
        lower_bound = rolling_window.quantile(adjust_extremes)
        upper_bound = rolling_window.quantile(1 - adjust_extremes)

    normalized_x = (x - lower_bound) / (upper_bound - lower_bound)

    return normalized_x


def deltaF(timeseries: pd.Series,
           detrend: bool = True,
           window_length: int = 5000) -> pd.Series:

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


def rolling_zscore(timeseries: (pd.Series or np.array),
                   window_length: int = 5000,
                   rolling: bool = True,
                   fill_value: float = np.nan) -> pd.Series:

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
        fill_value:
            Value to replace edges of z-scored trace with, where z-score can't
            span complete window.

    Returns:
        z:
            Z-scored timeseries of ycol.
    '''

    if not isinstance(timeseries, pd.Series):
        ts_ = pd.Series(timeseries)
    else:
        ts_ = timeseries.copy()

    if rolling:
        rolling_window = ts_.rolling(window=window_length, center=True)
    else:
        rolling_window = ts_  # Mimic rolling window variable

    rolling_mean = rolling_window.mean()
    rolling_std = rolling_window.std()
    z = (ts_ - rolling_mean) / rolling_std

    # Replace edges where z-score can't be calculated on full window.
    z[:window_length // 2] = fill_value
    z[-window_length // 2:] = fill_value

    return z


def snr_photo_signal(timeseries: pd.DataFrame,
                     signal_col: str = None) -> float:

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
    N = len(neural_signal)  # number of sample points
    yf_pos = yf[0:N // 2]  # positive N points only
    est_signal = 2.0 / N * np.abs(yf_pos).max()  # power of signal (near zero)
    est_noise = (2.0 / N * np.abs(yf_pos)[5000:]).max()  # power of noise

    ratio = est_signal / est_noise

    return ratio


def rolling_demodulation(trace: np.array,
                         carrier_freq: int | float,
                         *,
                         sampling_Hz: int | float = None,
                         nperseg: int = 0,
                         noverlap: int = 0,
                         nnearest: int = 2,
                         **kwargs):

    '''
    Demodulate with a rolling window (spectrogram), effectively downsampling
    by factor of nperseg - noverlap.

    Args:
        trace:
            Frequency-modulated timeseries.
        carrier_freq:
            Frequency of reference signal mixed with true signal.
        sampling_Hz:
            Sampling frequency (in Hertz).
        nperseg:
            Number of samples per rolling window segment for demodulation.
        noverlap:
            Number of samples overlapping between subsequent windows (usually
            set as 50% nperseg).
        nnearest:
            Number of frequency bands to select from the spectrogram for
            demodulation.

    Returns:
        rolling_demod:
            Demodulated signal from `trace`.
        t_:
            Timestamps corresponding to each sample in `rolling_demod`.
    '''

    win = signal.hamming(nperseg, 'periodic')

    f, t_, Zxx = signal.spectrogram(trace,
                                    sampling_Hz,
                                    window=win,
                                    nperseg=nperseg,
                                    noverlap=noverlap)
    power_spectra = np.abs(Zxx)
    if nnearest == 1:
        freq_ind = np.argmin(np.abs(f - carrier_freq))
    else:
        freq_ind = np.argsort(np.abs(f - carrier_freq))[:nnearest]
    # frequency_resolution = np.diff(f)[0]
    # nearest_freq = f[freq_ind]
    rolling_demod = power_spectra[freq_ind, :].mean(axis=0)

    return rolling_demod, t_


def process_trace(raw_photoms: list[np.array],
                  carriers: list[int | float],
                  labels: list[str],
                  detrend_win: int = None,
                  **kwargs,
                  ) -> dict:

    '''
    Wrapper function to detrend raw trace with z-score and  apply rolling
    demodulation. Include additional demodulated trace with no detrending.

    Args:
        raw_photoms:
            Raw photometry timeseries with frequency modulated signal for each
            channel in `labels`.
        carriers:
            Carrier frequencies for reference signal in each raw photom trace.
        labels:
            Fluorophore and hemisphere label for each channel.
        detrend_win:
            Window (in samples) to use for rolling z-score.

    Returns:
        processed_trace:
            Dictionary containing detrended demodulated timeseries and "raw"
            (aka not detrended) demodulated timeseries and corresponding
            timestamps for each channel's photometry stream.
    '''

    processed_trace = {}

    for label, raw_trace, carrier in zip(labels, raw_photoms, carriers):
        detrend_trace = rolling_zscore(raw_trace, detrend_win)

        power_spectra, t = rolling_demodulation(detrend_trace, carrier,
                                                **kwargs)
        processed_trace[f'detrend_{label}'] = power_spectra
        # processed_trace[f'power_spectra_{label}'] = power_spectra
        processed_trace[f't_{label}'] = t

        raw_power_spectra, _ = rolling_demodulation(raw_trace, carrier,
                                                    **kwargs)
        processed_trace[f'raw_{label}'] = raw_power_spectra

    return processed_trace


def get_tdt_streams(tdt_data, sig_thresh: float = 0.05
                    ) -> tuple[list, list, list, list]:

    '''
    Extract standard set of timeseries from TDT object. Filter by channels
    detected to have signal.

    Args:
        tdt_data:
            Object read in from TDT files.
        sig_thresh:
            Threshold for minimum standard deviation on measured carrier
            signal channel to pass for active channel collection.

    Returns:
        labels:
            List of labels identifying fluorophore color and hemisphere for
            corresponding index position in the other returned lists.
        raw_photoms:
            List of raw photometry data from each of the channels in `labels`.
        raw_carriers:
            List of measured input signal (reference signal) as timeseries.
        input_carriers:
            List of carrier frequencies set for each of the channels.
    '''

    # Get trace indices from meta_info
    idcs = {'carrier_g': 0,
            'carrier_r': 1,
            'photom_g': 2,
            'photom_r': 3}

    # fibers = {'right': 1, 'left': 2}  # just as a note
    carrier_g_right = tdt_data.streams.Fi1r.data[idcs.get("carrier_g", None)]
    carrier_r_right = tdt_data.streams.Fi1r.data[idcs.get("carrier_r", None)]
    photom_g_right = tdt_data.streams.Fi1r.data[idcs.get("photom_g", None)]
    photom_r_right = tdt_data.streams.Fi1r.data[idcs.get("photom_r", None)]

    carrier_g_left = tdt_data.streams.Fi2r.data[idcs.get("carrier_g", None)]
    carrier_r_left = tdt_data.streams.Fi2r.data[idcs.get("carrier_r", None)]
    photom_g_left = tdt_data.streams.Fi2r.data[idcs.get("photom_g", None)]
    photom_r_left = tdt_data.streams.Fi2r.data[idcs.get("photom_r", None)]

    carrier_val_g_right = tdt_data.scalars.Fi1i.data[1][0]
    carrier_val_r_right = tdt_data.scalars.Fi1i.data[4][0]
    carrier_val_g_left = tdt_data.scalars.Fi2i.data[1][0]
    carrier_val_r_left = tdt_data.scalars.Fi2i.data[4][0]

    # Get trace names and store in this list for ingestion
    labels = np.array(('grnR', 'redR', 'grnL', 'redL'))

    if len(photom_g_left) != len(photom_g_right):
        use_dtype = 'object'
    else:
        use_dtype = 'float32'
    raw_photoms: list[np.array] = np.array([photom_g_right, photom_r_right,
                                            photom_g_left, photom_r_left],
                                           dtype=use_dtype)
    raw_carriers: list[np.array] = np.array([carrier_g_right, carrier_r_right,
                                             carrier_g_left, carrier_r_left],
                                            dtype=use_dtype)
    carrier_vals: list[float] = np.array([carrier_val_g_right,
                                          carrier_val_r_right,
                                          carrier_val_g_left,
                                          carrier_val_r_left])

    # Determine fibers that were on from standard deviation on data stream.
    active_channels = [i for i, cf in enumerate(raw_carriers) if
                       np.std(cf[100000:]) > sig_thresh]

    return (labels[active_channels], raw_photoms[active_channels],
            raw_carriers[active_channels], carrier_vals[active_channels])


def calc_carrier_freq(raw_carrier_sigs: list[int | float],
                      sampling_Hz: int | float,
                      n_points: int = 2**14) -> list[float]:

    '''
    Source: from DataJoint Pipeline
    Calculate carrier frequencies from carrier signal using FFT.

    Args:
        raw_carrier_sigs:
            List of timeseries containing reference signal (should be sine
            wave).
        sampling_Hz:
            Sampling frequency of timeseries in raw_carrier_freqs (in Hz).

    Returns:
        calc_carrier_freqs:
            List of carrier frequencies (in Hz) best describing input sine
            wave of reference signal.
    '''

    calc_carrier_freqs = []

    for carrier in raw_carrier_sigs:

        start_idx = len(carrier) // 4
        fft_carrier = abs(np.fft.fft(carrier[start_idx:start_idx + n_points]))
        P2 = fft_carrier / n_points
        P1 = abs(P2 / 2 + 1)
        P1[1:-1] = 2 * P1[1:-1]
        f = sampling_Hz * np.arange(n_points // 2) / n_points
        idx = np.argmax(P1, axis=None)
        calc_carrier = round(f[idx])
        calc_carrier_freqs.append(calc_carrier)

    return calc_carrier_freqs
