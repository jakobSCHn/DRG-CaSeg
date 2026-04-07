import logging
import numpy as np
import scipy.ndimage as image
import scipy.signal as signal
import scipy.stats as sts

logger = logging.getLogger(__name__)

def standardize_traces(
    trace_array: np.ndarray
    ):

    stds = np.std(trace_array, axis=1)

    #Invert traces where skew is negative
    #rectified_traces = np.where((skews < 0)[:, None], -trace_array, trace_array)

    #Only scale traces with std higher than 0
    traces_z = np.zeros_like(trace_array, dtype=float)
    valid_stds_mask = stds >= 1e-6

    if np.any(valid_stds_mask):
        traces_z[valid_stds_mask] = sts.zscore(trace_array[valid_stds_mask], axis=1)

    return traces_z


def rectify_traces(
    trace_array: np.ndarray,
    ) -> np.ndarray:
    return np.abs(trace_array)


def filter_traces_butter(
    trace_array: np.ndarray,
    fs: float,
    cutoff_low: float = 0.05,
    cutoff_high: float = 3,
    ):

    sos = signal.butter(
        N=2,
        Wn=[cutoff_low, cutoff_high],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered_traces = signal.sosfiltfilt(
        sos,
        trace_array,
        axis=1,
    )

    return filtered_traces

def filter_traces_savgol(
    trace_array: np.ndarray,
    window: int = 331,
    order: int = 2,
    ) -> np.ndarray:

    if window % 2 == 0:
        window += 1
        logger.warning(f"Window length adjusted to {window}. "
                       f"The window length must be an odd number!")
        
    filtered_traces = signal.savgol_filter(
        trace_array,
        window_length=window,
        polyorder=order,
        axis=1,
    )

    return filtered_traces


def get_dynamic_prominence(
    trace: np.ndarray,
    snr_multiplier: float,
    ):

    median_val = np.median(trace)
    mad = np.median(np.abs(trace - median_val))

    noise_floor_sigma = mad / 0.6745
    if noise_floor_sigma < 0.1:
        noise_floor_sigma = 0.1

    dynamic_prominence = snr_multiplier * noise_floor_sigma

    return dynamic_prominence


def get_rolling_prominence(
    trace: np.ndarray,
    snr_multiplier: float,
    window_size: int = 330,
    ):
    rolling_median = image.median_filter(trace, size=window_size)
    rolling_mad = image.median_filter(np.abs(trace - rolling_median), size=window_size)

    noise_floor_sigma = rolling_mad / 0.6745
    noise_floor_sigma = np.clip(noise_floor_sigma, a_min=0.1, a_max=None)

    dynamic_prominence = snr_multiplier * noise_floor_sigma

    return dynamic_prominence


def edge_crop_traces(
    traces: np.ndarray,
    crop_seconds: float,
    fs: float,
    )-> np.ndarray:
    
    crop_samples = int(crop_seconds * fs)
    if traces.shape[1] <= 2 * crop_samples:
        raise ValueError(f"Trace has shape {traces.shape[0]} in first dimension, "
                         f"which is too short to crop to requested length.")
    
    cropped_traces = traces[:, crop_samples:-crop_samples]

    return cropped_traces


def strict_collapse(series):
    
    unique_vals = series.unique()
    if len(unique_vals) == 0:
        return np.nan
    if len(unique_vals) == 1:
        return unique_vals[0]
    
    raise ValueError(
        f"Data integrity error! Conflicting values found "
        f"for sample: {unique_vals}"
    )