import numpy as np
import scipy.signal as signal

from plotters import visualize_random_peaks

import helpers


def calculate_peak_features(
    time_traces: np.ndarray,
    fs: float,
    height_threshold: float = 0.0,
    peak_snr: float | None = None,
    prominence: float | None = None,
    distance: float | None = None,
    crop: float = 0.0,
    stimuli_regions: list[tuple[float, float]] | None = None,
    debug_plot: bool = False,
    n_debug_samples: int = 5,
    ) -> dict:
    """
    Detects peaks across a 2D matrix of time series and extracts key metrics.
    Expects time_traces shape: (n_traces, time_points)
    """
    #convert the distance from seconds to units of samples
    if distance:
        distance_samples = int(distance * fs)
    else:
        distance_samples = None

    #additional postprocessing of traces for peak detection
    time_traces = helpers.edge_crop_traces(time_traces, crop_seconds=crop, fs=fs)
    time_traces = helpers.standardize_traces(time_traces)
    time_traces = helpers.filter_traces(time_traces, fs=fs)
    

    if debug_plot:
        visualize_random_peaks(
            time_traces=time_traces,
            height_threshold=height_threshold,
            prominence=prominence,
            distance=distance_samples,
            fs=fs,
            n_samples=n_debug_samples
        )

    n_traces = time_traces.shape[0]
    
    peak_counts = np.zeros(n_traces, dtype=int)
    mean_amplitudes = np.zeros(n_traces, dtype=float)
    max_amplitudes = np.zeros(n_traces, dtype=float)

    #prepare stimulus specific peak extraction
    if not stimuli_regions:
        stimuli_regions= [
            (99.9, 130.0),
            (199.9, 230.0),
            (299.9, 330.0),
        ]
    stimuli_features = {}
    for idx in range(len(stimuli_regions)):
        stimuli_features[f"stimulus_{idx + 1}_has_peak"] = np.zeros(n_traces, dtype=bool)
        stimuli_features[f"stimulus_{idx + 1}_max_height"] = np.full(n_traces, np.nan, dtype=float)
    
    #Loop through the n dimension to extract peaks from
    #each individual trace
    for i in range(n_traces):
        trace = time_traces[i]
        
        target_prominence = helpers.get_dynamic_prominence(trace, peak_snr) if peak_snr else prominence
        peaks, properties = signal.find_peaks(
            trace,
            height=height_threshold,
            prominence=target_prominence,
            distance=distance_samples,
        )
        
        peak_counts[i] = len(peaks)
        
        #If the trace actually has peaks, calculate the amplitude metrics
        if len(peaks) > 0:
            peak_heights = properties["peak_heights"]
            mean_amplitudes[i] = np.mean(peak_heights)
            max_amplitudes[i] = np.max(peak_heights)

            #map peaks back to original time scale before cropping
            original_peaks = peaks + int(crop * fs)

            #calculate stimulus specific features
            for idx, (start_t, end_t) in enumerate(stimuli_regions):
                stimulus_mask = (original_peaks >= int(start_t * fs)) & (original_peaks <= int(end_t * fs))
                stimulus_peak_height = peak_heights[stimulus_mask]

                if len(stimulus_peak_height) > 0:
                    stimuli_features[f"stimulus_{idx + 1}_has_peak"][i] = True
                    stimuli_features[f"stimulus_{idx + 1}_max_height"][i] = np.max(stimulus_peak_height)


        else:
            #Safe fallback for flatlines to prevent NaN errors
            mean_amplitudes[i] = 0.0
            max_amplitudes[i] = 0.0
            
    #Pack the results into a pipeline compatible dictionary
    results = {
        "peak_count": peak_counts,
        "mean_peak_amplitude": mean_amplitudes,
        "max_peak_amplitude": max_amplitudes
    }

    results.update(stimuli_features)

    return results


