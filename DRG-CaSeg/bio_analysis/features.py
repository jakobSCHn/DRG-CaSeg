import numpy as np
import scipy.signal as signal
import scipy.stats as sts

from plotters import visualize_random_peaks

import helpers


def calculate_peak_features(
    data: np.ndarray,
    fs: float,
    height_threshold: float = 0.0,
    peak_snr: float | None = None,
    prominence: float | None = None,
    distance: float | None = None,
    width: int | None = 15,
    crop: float = 0.0,
    stimuli_regions: dict[tuple[float, float]] | None = None,
    debug_plot: bool = False,
    n_debug_samples: int = 5,
    ) -> dict:
    """
    Detects peaks across a 2D matrix of time series and extracts key metrics.
    Expects time_traces shape: (n_traces, time_points)
    """
    #don't process region specific features if no ROIs are given
    if stimuli_regions is None:
        stimuli_regions = []

    #convert the distance from seconds to units of samples
    if distance:
        distance_samples = int(distance * fs)
    else:
        distance_samples = None

    #apply preprocessing to the traces
    time_traces = helpers.standardize_traces(data)
    time_traces = helpers.filter_traces_butter(time_traces, fs=fs)
    time_traces = helpers.rectify_traces(time_traces)
    time_traces = helpers.filter_traces_savgol(time_traces)
    time_traces = helpers.edge_crop_traces(time_traces, crop_seconds=crop, fs=fs)
    time_traces = helpers.standardize_traces(time_traces)
    
    #visulaize preprocessing effects and peak detection
    if debug_plot:
        visualize_random_peaks(
            time_traces=time_traces,
            height_threshold=height_threshold,
            peak_snr=peak_snr,
            prominence=prominence,
            distance=distance_samples,
            width=width,
            fs=fs,
            n_samples=n_debug_samples
        )

    n_traces = time_traces.shape[0]
    
    peak_counts = np.zeros(n_traces, dtype=int)
    mean_amplitudes = np.zeros(n_traces, dtype=float)
    max_amplitudes = np.zeros(n_traces, dtype=float)

    #prepare stimulus specific peak extraction
    stimuli_features = {}
    for name in stimuli_regions:
        stimuli_features[f"stimulus_{name}_has_peak"] = np.zeros(n_traces, dtype=bool)
        stimuli_features[f"stimulus_{name}_max_height"] = np.full(n_traces, np.nan, dtype=float)
        stimuli_features[f"stimulus_{name}_peak_time"] = np.full(n_traces, np.nan, dtype=float)
        stimuli_features[f"stimulus_{name}_gradient_time"] = np.full(n_traces, np.nan, dtype=float)
        stimuli_features[f"stimulus_{name}_max_gradient"] = np.full(n_traces, np.nan, dtype=float)
    
    #Loop through the n dimension to extract peaks from
    #each individual trace
    for i in range(n_traces):
        trace = time_traces[i]
        
        #Saftey check for trace "cleanliness"
        if sts.skew(trace) < 1:
            continue
        
        trace_gradient = np.gradient(trace, 1/fs)
        target_prominence = helpers.get_dynamic_prominence(trace, peak_snr) if peak_snr else prominence
        peaks, properties = signal.find_peaks(
            trace,
            height=height_threshold,
            prominence=target_prominence,
            distance=distance_samples,
            width=width,
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
            for stim, (start_t, end_t) in stimuli_regions.items():
                stimulus_mask = (original_peaks >= int(start_t * fs)) & (original_peaks <= int(end_t * fs))

                stimulus_peak_height = peak_heights[stimulus_mask]
                stimulus_peak_indices = original_peaks[stimulus_mask]

                if len(stimulus_peak_height) > 0:
                    max_idx = np.argmax(stimulus_peak_height)
                    peak_time_sec = stimulus_peak_indices[max_idx] / fs

                    grad_start = int(start_t * fs)
                    grad_end = stimulus_peak_indices[max_idx] + 1
                    stim_gradient = trace_gradient[grad_start:grad_end]
                    max_gradient_idx = np.argmax(stim_gradient)
                    max_gradient = stim_gradient[max_gradient_idx]
                    max_gradient_time = (grad_start + max_gradient_idx) / fs
                
                    stimuli_features[f"stimulus_{stim}_has_peak"][i] = True
                    stimuli_features[f"stimulus_{stim}_max_height"][i] = np.max(stimulus_peak_height)
                    stimuli_features[f"stimulus_{stim}_peak_time"][i] = peak_time_sec
                    stimuli_features[f"stimulus_{stim}_gradient_time"][i] = max_gradient_time
                    stimuli_features[f"stimulus_{stim}_max_gradient"][i] = max_gradient

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


def calculate_sample_correlation(
    data: np.ndarray,
    fs: float,
    stimuli_regions: dict[tuple[float, float]] | None = None,
    ):

    n_traces = data.shape[0]
    results = {}

    # skip feature calculations if less than 2 neurons are availble in the sample
    if n_traces < 2:
        results["sample_corr_global"] = [np.nan] * n_traces
        return results

    if stimuli_regions is None:
        stimuli_regions = []

    with np.errstate(divide="ignore", invalid="ignore"):
        #calculate in-sample correlation for full trace
        corr_matrix_global = np.abs(np.corrcoef(data))
        upper_tri_global = corr_matrix_global[np.triu_indices_from(corr_matrix_global, k=1)]
        results["sample_corr_global"] = [np.nanmean(upper_tri_global)] * n_traces

        #calculate in-sample correlation for stimulus regions
        for stim, (start_t, end_t) in stimuli_regions.items():
            start_idx = int(start_t * fs)
            end_idx = int(end_t * fs)

            stimulus_traces = data[:, start_idx:end_idx]

            stimulus_corr = np.abs(np.corrcoef(stimulus_traces))
            stimulus_upper = stimulus_corr[np.triu_indices_from(stimulus_corr, k=1)]

            results[f"stimulus_corr_{stim}"] = [np.nanmean(stimulus_upper)] * n_traces

    return results