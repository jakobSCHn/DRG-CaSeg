import numpy as np
import scipy.signal as signal
import scipy.stats as sts


def standardize_traces(
    trace_array: np.ndarray
    ):

    stds = np.std(trace_array, axis=1)
    skews = sts.skew(trace_array, axis=1)

    #Invert traces where skew is negative
    rectified_traces = np.where((skews < 0)[:, None], -trace_array, trace_array)

    #Only scale traces with std higher than 0
    traces_z = np.zeros_like(rectified_traces, dtype=float)
    valid_stds_mask = stds >= 1e-6

    if np.any(valid_stds_mask):
        traces_z[valid_stds_mask] = sts.zscore(rectified_traces[valid_stds_mask], axis=1)

    return traces_z


def filter_traces_butter(
    trace_array: np.ndarray,
    fs: float,
    cutoff_low: float = 0.05,
    cutoff_high: float = 1.0,
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

    return np.abs(filtered_traces)


def get_dynamic_prominence(
    trace: np.ndarray,
    snr_multiplier: float,
    ):

    median_val = np.median(trace)
    mad = np.median(np.abs(trace - median_val))

    noise_floor_sigma = mad / 0.6745
    if noise_floor_sigma < 1e-6:
        noise_floor_sigma = 1e-6

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
