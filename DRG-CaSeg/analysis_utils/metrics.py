import numpy as np
import logging

import data_utils.plotter as plotter
import analysis_utils.workflow as wf
import data_utils.plot_adapters as adapt

from pathlib import Path


logger = logging.getLogger(__name__)

def compute_iou_matrix(
    masks_true: np.ndarray,
    masks_pred: np.ndarray,
    ) -> np.ndarray:
    """
    Computes the pairwise Intersection over Union (IoU) matrix.
    
    Args:
        masks_true: Boolean array of shape (N_true, H, W)
        masks_pred: Boolean array of shape (N_pred, H, W)
        
    Returns:
        iou_matrix: Float array of shape (N_true, N_pred)
    """
    num_true = masks_true.shape[0]
    num_pred = masks_pred.shape[0]
    
    # Flatten spatial dimensions for efficient computation
    # shape: (N, H*W)
    flat_true = masks_true.reshape(num_true, -1)
    flat_pred = masks_pred.reshape(num_pred, -1)
    
    # intersection: (N_true, N_pred)
    intersection = np.dot(flat_true, flat_pred.T)
    
    # union = area_true + area_pred - intersection
    area_true = flat_true.sum(axis=1)[:, None]  # (N_true, 1)
    area_pred = flat_pred.sum(axis=1)[None, :]  # (1, N_pred)
    union = area_true + area_pred - intersection
    
    # Avoid division by zero
    iou_matrix = np.divide(intersection, union, out=np.zeros_like(intersection), where=union!=0)
    
    return iou_matrix


def calculate_overlap_correlation(
    pred_masks: np.ndarray, 
    gt_masks: np.ndarray, 
    pred_traces: np.ndarray, 
    gt_traces: np.ndarray,
    pred_labels: np.ndarray,
    save_filepath: str,
    iou_threshold: float,
    gt_binary_threshold: float,
    fps: float,
    ) -> dict:
    """
    Calculates performance. 
    Selection Logic: Prioritizes predictions that cover the largest portion of the GT 
    (Coverage), provided they meet the minimum purity requirement.
    """
    
    # --- Setup & Binarization ---
    n_gt = gt_masks.shape[0]
    n_pred = pred_masks.shape[0]

    n_frames_gt = gt_traces.shape[1] if gt_traces.ndim > 1 else 0
    n_frames_pred = pred_traces.shape[1] if pred_traces.ndim > 1 else 0

    if n_frames_gt != n_frames_pred:
        raise ValueError(
            f"Frame mismatch: gt_traces has {n_frames_gt} frames, "
            f"but pred_traces has {n_frames_pred} frames."
        )
    
    n_frames = n_frames_gt

    # Binarize Ground Truth as it's a Gaussian blob
    gt_binary = (gt_masks > gt_binary_threshold).astype(np.float32)
    
    results = {
        "spatial": {
            "true_positives": 0, "false_negatives": 0, "false_positives": 0,
            "recall": 0.0, "precision": 0.0,
        },
        "temporal": {
            "mean_correlation": 0.0, "matched_correlations": [],
        }
    }

    # --- Calculation Core ---
    if n_gt > 0 and n_pred > 0:
        metrics, tp_pairs, fn_indices, fp_indices = wf.evaluate_segmentation(
            pred_masks=pred_masks,
            gt_masks=gt_binary,
            iou_threshold=iou_threshold,
        )


    # --- Calculate Temporal Correlation ---
    matched_correlations = []
    for gt_i, pred_j in tp_pairs:
        t_gt = gt_traces[gt_i].flatten()
        t_pred = pred_traces[pred_j].flatten()
        
        if np.std(t_gt) == 0 or np.std(t_pred) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(t_gt, t_pred)[0, 1])
        matched_correlations.append(corr)

    
    results["spatial"].update({
        **metrics,
    })
    
    if len(matched_correlations) > 0:
        results["temporal"].update({
            "mean_correlation": float(np.mean(np.abs(matched_correlations))),
            "matched_correlations": matched_correlations,
        })

    # --- Plotting ---
    plotter.plot_segmentation_performance(
        gt_masks=gt_masks,
        pred_masks=pred_masks,
        gt_binary=gt_binary,
        tp_pairs=tp_pairs,
        fn_indices=fn_indices,
        fp_indices=fp_indices,
        results=results,
        save_path=save_filepath,
    )
    plotter.plot_temporal_comparison(
        gt_traces=gt_traces,
        pred_traces=pred_traces,
        tp_pairs=tp_pairs,
        save_path=save_filepath,
        fps=fps,
        title="Temporal Performance Comparison",
        pred_labels=pred_labels,
    )
    adapt.plot_adapter_gt_overlay(
        gt_masks=gt_masks,
        gt_traces=gt_traces,
        pred_masks=pred_masks,
        pred_traces=pred_traces,
        tp_pairs=tp_pairs,
        md={},
        save_path=save_filepath,
        fps=fps,
        n_frames=n_frames,
    )
    plotter.plot_mask_comparison(
        gt_masks=gt_masks,
        gt_traces=gt_traces,
        pred_masks=pred_masks,
        pred_traces=pred_traces,
        tp_pairs=tp_pairs,
        save_filepath=save_filepath,
        fps=fps,
        file_ext="single_mask_comparison.png"
    )
    
    return results