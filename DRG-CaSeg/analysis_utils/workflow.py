import logging

logger = logging.getLogger(__name__)

from analysis_utils.pca import cellsort_pca
from analysis_utils.ica import ica_mukamel, extract_rois_and_traces


def run_ica(
    vid_norm,
    n_pcs: int | tuple[int, int] = 20,
    mu=0.5,
    maxrounds=200,
    minsize=15,
    maxsize=200,
    ):
    """
    Orchestrates PCA dimensionality reduction followed by ICA source extraction.
    """
    
    mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm = cellsort_pca(
        vid_norm, 
        n_pcs=n_pcs
    )

    # 2. Run ICA
    # Note: Assuming ica_mukamel is inside the ica module
    ica_sig, ica_filters, ica_A, numiter = ica_mukamel(
        mixedsig=mixedsig,
        mixedfilters=mixedfilters,
        CovEvals=cov_evals,
        mu=mu,
        maxrounds=maxrounds,
    )

    masks, traces = extract_rois_and_traces(
        spatial_filters=ica_filters,
        temporal_signals=ica_sig,
        min_size=minsize,
        max_size=maxsize,
    )


    # 3. Return a dictionary for safe unpacking (matching your loader pattern)
    return {
        "masks": masks,
        "traces": traces,
    }