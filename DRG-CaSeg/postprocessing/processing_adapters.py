"""
Provide wrapper classes to integrate postprocessing utilities into the ML pipeline.

This module adapts the core signal processing functions from the "processing_utils" 
module (such as peak detection and bandpass filtering) into a standard syntax, that can
be used by the ML pipeline to seemless integrate them into the postprocessing
workflow.
"""
import numpy as np

import postprocessing.processing_utils as pu



def filter_background_masks(
    results: dict,
    background_img: np.ndarray,
    brightness_threshold: float = 0.1,
    overlap_threshold: float = 0.8,
    max_size: int = 25,
    ):

    masks = results["masks"]

    filtered_masks = pu.mask_background(
        masks=masks,
        background_img=background_img,
        brightness_threshold=brightness_threshold,
        overlap_threshold=overlap_threshold,
        max_size=max_size,
    )

    results["masks"] = filtered_masks

    return results


def bandpass_filter_traces(
    results: dict,
    fr: float,
    cutoff_low: float = 0.005,
    cutoff_high: float = 7.5,   
    ):

    traces = results["traces"]

    filtered_traces = pu.filter_traces(
        traces=traces,
        cutoff_low=cutoff_low,
        cutoff_high=cutoff_high,
        fr=fr,
    )

    results["traces"] = filtered_traces

    return results