import json
import h5py
import numpy as np
import scipy.signal as signal

from pathlib import Path
from skimage import morphology as morph


def mask_background(
    masks: np.ndarray,
    background_img: np.ndarray,
    brightness_threshold: float,
    overlap_threshold: float,
    max_size: int,  
    ):
    #Extract the background from the tissue image
    bg_min = np.min(background_img)
    bg_max = np.max(background_img)
    brightness_range = bg_max - bg_min
    threshold_val = bg_min + (brightness_threshold) * brightness_range

    bg_pixels = background_img <= threshold_val

    bg_denoised = morph.remove_small_objects(
        bg_pixels,
        max_size=max_size,
    )

    #Calculate mask overlap with non-tissue background
    bool_masks = masks > 0
    mask_areas = np.sum(bool_masks, axis=(1,2))

    valid_mask_idx = mask_areas > 0 #discard empty masks

    overlaps = bool_masks & bg_denoised
    overlap_areas = np.sum(overlaps, axis=(1,2))
    
    overlap_pct = np.zeros_like(mask_areas, dtype=float)
    overlap_pct[valid_mask_idx] = overlap_areas[valid_mask_idx] / mask_areas[valid_mask_idx]

    #Logic indexing for deciding which masks to keep and which to drop
    foreground_idx = valid_mask_idx & (overlap_pct <= overlap_threshold)

    return masks[foreground_idx]


def filter_traces(
    traces: np.ndarray,
    cutoff_low: float,
    cutoff_high: float,
    fr: float,  
    ):

    sos = signal.butter(
        2,
        [cutoff_low, cutoff_high],
        btype="bandpass",
        fs=fr,
        output="sos"
    )

    filtered_traces = signal.sosfiltfilt(
        sos=sos,
        x=traces,
        axis=1,
    )

    return filtered_traces


def detect_peaks(
    traces: np.ndarray,
    min_prominence: float,
    min_distance: float | None = None,
    ):

    detected_peaks = []
    for trace in traces:
        peaks, _ = signal.find_peaks(
            trace,
            prominence=min_prominence,
            distance=min_distance,
        )

        detected_peaks.append(peaks)
    
    return detected_peaks
