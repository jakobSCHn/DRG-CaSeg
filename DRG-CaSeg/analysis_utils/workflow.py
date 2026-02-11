import logging
import numpy as np
import caiman as cm

from pathlib import Path
from scipy.optimize import linear_sum_assignment

from analysis_utils.pca import cellsort_pca
from analysis_utils.ica import ica_mukamel, extract_rois_and_traces
from analysis_utils.metrics import compute_iou_matrix, calculate_overlap_correlation
from data_utils.plotter import plot_spatial_filters, plot_ica_components

logger = logging.getLogger(__name__)

def run_ica(
    mov: cm.movie,
    save_filepath: Path,
    n_pcs: int | tuple[int, int] = 20,
    mu=0.5,
    maxrounds=200,
    kurtosis_thres=5.0,
    z_thres=3,
    minsize=25,
    maxsize=25000,
    ):
    """
    Orchestrates PCA dimensionality reduction followed by ICA source extraction.
    """
    
    mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm = cellsort_pca(
        mov.astype(np.float32), 
        pcs=n_pcs
    )


    ica_sig, ica_filters, ica_A, numiter = ica_mukamel(
        mixedsig=mixedsig,
        mixedfilters=mixedfilters,
        CovEvals=cov_evals,
        mu=mu,
        maxrounds=maxrounds,
    )

    plot_spatial_filters(
        spatial_filters=ica_filters,
        save_filepath=save_filepath,
        title="ICA Spatial Components",
        subtitle="IC",
        file_ext="spatial_components.png",
    )
    plot_ica_components(
        spatial_filters=ica_filters,
        time_courses=ica_sig,
        save_filepath=save_filepath,
        sampling_rate=30,
        title="ICA Temporal & Spatial Components",
        subtitle="IC",
        file_ext="full_ica_components.png"
    )

    (
        masks,
        traces,
        labels,
        n_components,
        used_components,
        binary_mask,
        cleaned_mask,
    ) = extract_rois_and_traces(
        spatial_filters=ica_filters,
        temporal_signals=ica_sig,
        min_size=minsize,
        max_size=maxsize,
        kurtosis_thresh=kurtosis_thres,
        z_thresh=z_thres,
    )

    plot_spatial_filters(
        spatial_filters=binary_mask,
        save_filepath=save_filepath,
        title="Binary Masks extracted from Independent Components",
        subtitle="Binary Mask",
        file_ext="binary_masks.png"
    )
    plot_spatial_filters(
        spatial_filters=cleaned_mask,
        save_filepath=save_filepath,
        title="Cleand Masks extracted from Independent Components",
        subtitle="Cleaned Mask",
        file_ext="cleaned_masks.png"
    )


    #Return a dictionary for safe unpacking
    return {
        "masks": masks,
        "traces": traces,
        "labels": labels,
        "analysis_stats": {
            "n_components": n_components,
            "used_components": used_components,
        }
    }


def evaluate_segmentation(
    pred_masks: np.ndarray, 
    gt_masks: np.ndarray, 
    iou_threshold: float,
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

    if gt_masks.shape[0] == 0 or pred_masks.shape[0] == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "mean_iou": 0.0,
            "matches": None,
            "fp": None,
            "fn": None,
        }

    iou_matrix = compute_iou_matrix(gt_masks, pred_masks)
    
    #Hungarian Algorithm to find best matches
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
    
    #Filter matches by threshold
    matched_ious = iou_matrix[row_ind, col_ind]
    valid_matches = matched_ious > iou_threshold

    tp_gt_indices = row_ind[valid_matches]
    tp_pred_indices = col_ind[valid_matches]
    
    true_positives = np.sum(valid_matches)
    if true_positives > 0:
        valid_ious = matched_ious[valid_matches]
        mean_iou = np.mean(valid_ious)
        valid_dices = (2 * valid_ious) / (valid_ious + 1)
        mean_dice = np.mean(valid_dices)

    mean_iou = np.mean(matched_ious[valid_matches]) if true_positives > 0 else 0.0

    all_pred_indices = np.arange(len(pred_masks))
    fp_indices = np.setdiff1d(all_pred_indices, tp_pred_indices)
    
    # False Negatives: GT masks that are NOT in the valid true positive set
    all_gt_indices = np.arange(len(gt_masks))
    fn_indices = np.setdiff1d(all_gt_indices, tp_gt_indices)
    
    #False Positives: Predictions that weren't matched to any GT or had low IoU
    false_positives = pred_masks.shape[0] - true_positives
    
    #False Negatives: GT masks that weren't matched to any Pred or had low IoU
    false_negatives = gt_masks.shape[0] - true_positives

    tp_pairs = list(zip(tp_gt_indices, tp_pred_indices))
    
    #Calculate Metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "ious": valid_ious.tolist(),
        "mean_iou": mean_iou,
        "dices": valid_dices.tolist(),
        "mean_dice": mean_dice,
        "tp": int(true_positives),
        "fp": int(false_positives),
        "fn": int(false_negatives)
    }

    return metrics, tp_pairs, fn_indices, fp_indices  


def evaluate_model_performance(
    res,
    gt,
    save_filepath,
    iou_threshold: float = 0.5,
    gt_binary_threshold: float = 0.25,
    ):
    """
    Wraps the model performance analysis to be compatible with 
    the experiment pipeline.
    
    :param res: Description
    :param gt: Description
    :param save_filepath: Description
    :param coverage_threshold: Description
    :param gt_match_threshold: Description
    :param gt_binary_threshold: Description
    """
    metrics = calculate_overlap_correlation(
        pred_masks=res["masks"],
        gt_masks=gt["spatial"],
        pred_traces=res["traces"],
        gt_traces=gt["temporal"],
        pred_labels=res["labels"],
        save_filepath=save_filepath,
        iou_threshold=iou_threshold,
        gt_binary_threshold=gt_binary_threshold,
        fps=gt["fps"],
    )

    return metrics