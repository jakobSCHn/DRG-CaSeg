import logging
import numpy as np
import caiman as cm

from scipy.optimize import linear_sum_assignment

from analysis_utils.pca import cellsort_pca
from analysis_utils.ica import ica_mukamel, extract_rois_and_traces
from analysis_utils.metrics import compute_iou_matrix

logger = logging.getLogger(__name__)

def run_ica(
    mov: cm.movie,
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
        mov.astype(np.float32), 
        pcs=n_pcs
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

    masks, traces, labels = extract_rois_and_traces(
        spatial_filters=ica_filters,
        temporal_signals=ica_sig,
        min_size=minsize,
        max_size=maxsize,
    )


    # 3. Return a dictionary for safe unpacking (matching your loader pattern)
    return {
        "masks": masks,
        "traces": traces,
        "labels": labels,
    }

def evaluate_segmentation(
    pred: np.ndarray, 
    gt: np.ndarray, 
    iou_threshold: float = 0.5
    ) -> dict:
    """
    Matches predicted masks to ground truth using Hungarian algorithm and computes metrics.
    
    Args:
        gt: Boolean array (N_true, H, W)
        pred: Boolean array (N_pred, H, W)
        iou_threshold: Minimum IoU to consider a match valid (True Positive).
        
    Returns:
        dict: {
            "precision": float,
            "recall": float,
            "f1_score": float,
            "mean_iou": float,
            "matches": int,
            "fp": int,
            "fn": int
        }
    """
    if gt.shape[0] == 0 or pred.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "mean_iou": 0.0}

    iou_matrix = compute_iou_matrix(gt, pred)
    
    #Hungarian Algorithm to find best matches
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
    
    #Filter matches by threshold
    matched_ious = iou_matrix[row_ind, col_ind]
    valid_matches = matched_ious > iou_threshold
    
    true_positives = np.sum(valid_matches)
    mean_iou = np.mean(matched_ious[valid_matches]) if true_positives > 0 else 0.0
    
    #False Positives: Predictions that weren't matched to any GT or had low IoU
    false_positives = pred.shape[0] - true_positives
    
    #False Negatives: GT masks that weren't matched to any Pred or had low IoU
    false_negatives = gt.shape[0] - true_positives
    
    #Calculate Metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_iou": mean_iou,
        "matches": int(true_positives),
        "fp": int(false_positives),
        "fn": int(false_negatives)
    }