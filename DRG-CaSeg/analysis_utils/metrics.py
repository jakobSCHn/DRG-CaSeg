import numpy as np
import logging

import data_utils.plotter as plotter
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
    intersection = np.dot(flat_true.astype(float), flat_pred.astype(float).T)
    
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
    coverage_threshold: float,
    gt_match_threshold: float,
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
            "recall": 0.0, "precision": 0.0, "ignored": 0
        },
        "temporal": {
            "mean_correlation": 0.0, "matched_correlations": []
        }
    }

    tp_pairs = []
    matched_correlations = []
    matched_gt_set = set()
    matched_pred_set = set()

    # --- Calculation Core ---
    if n_gt > 0 and n_pred > 0:
        # Flatten
        gt_flat = gt_binary.reshape(n_gt, -1)
        pred_flat = pred_masks.reshape(n_pred, -1).astype(np.float32)
        
        # Calculate Areas & Intersection
        gt_areas = gt_flat.sum(axis=1)
        pred_areas = pred_flat.sum(axis=1)
        intersection = np.dot(gt_flat, pred_flat.T) # Shape: (n_gt, n_pred)
        
        # Metric 1: Purity (How much of Pred is inside GT) - USED FOR FILTERING
        purity_matrix = intersection / (pred_areas[np.newaxis, :] + 1e-8)

        # Metric 2: Coverage (How much of GT is covered by Pred) - USED FOR RANKING
        gt_coverage_matrix = intersection / (gt_areas[:, np.newaxis] + 1e-8)

        # --- LOGIC: Combined Coverage Validity ---
        # 1. Identify "Valid Fragments": Predictions that are pure enough (e.g. > 50% inside GT)
        # We still need this filter to prevent matching huge background masks that accidentally cover the GT
        valid_fragment_mask = purity_matrix >= coverage_threshold
        
        # 2. Calculate Combined Coverage per GT (for the 60% requirement)
        combined_intersection = (intersection * valid_fragment_mask).sum(axis=1)
        gt_covered_fraction = combined_intersection / (gt_areas + 1e-8)
        
        # 3. Identify GTs that are "Satisfactorily Covered"
        valid_gt_indices = np.where(gt_covered_fraction >= gt_match_threshold)[0]
        valid_gt_set = set(valid_gt_indices)

        # --- Greedy Matching (UPDATED) ---
        potential_gt, potential_pred = np.where(valid_fragment_mask)
        
        # CHANGE: Use GT Coverage as the score, not Purity.
        # This prioritizes the mask that covers 60% of the GT over the one that covers 20%,
        # even if the 20% one is "purer".
        scores = gt_coverage_matrix[potential_gt, potential_pred]
        
        sorted_indices = np.argsort(scores)[::-1]
        
        for idx in sorted_indices:
            gt_i = potential_gt[idx]
            pred_j = potential_pred[idx]
            
            # Constraints:
            # 1. Neither used yet
            # 2. GT must be in the "Satisfactorily Covered" set
            if (gt_i not in matched_gt_set) and (pred_j not in matched_pred_set):
                if gt_i in valid_gt_set:
                    matched_gt_set.add(gt_i)
                    matched_pred_set.add(pred_j)
                    tp_pairs.append((int(gt_i), int(pred_j)))

    # --- Categorize Remaining Indices ---
    fn_indices = [i for i in range(n_gt) if i not in matched_gt_set]

    fp_indices = []
    ignored_indices = []

    for pred_idx in range(n_pred):
        if pred_idx not in matched_pred_set:
            # Check total overlap with ALL GTs to decide if it's noise
            total_overlap_ratio = intersection[:, pred_idx].sum() / (pred_areas[pred_idx] + 1e-8)
            
            if total_overlap_ratio < coverage_threshold:
                fp_indices.append(int(pred_idx))
            else:
                ignored_indices.append(int(pred_idx))

    # --- Calculate Temporal Correlation ---
    for gt_i, pred_j in tp_pairs:
        t_gt = gt_traces[gt_i].flatten()
        t_pred = pred_traces[pred_j].flatten()
        
        if np.std(t_gt) == 0 or np.std(t_pred) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(t_gt, t_pred)[0, 1])
        matched_correlations.append(corr)

    # --- Update Results ---
    tp_count = len(tp_pairs)
    fp_count = len(fp_indices)
    
    results["spatial"].update({
        "true_positives": int(tp_count),
        "false_negatives": int(len(fn_indices)),
        "false_positives": int(fp_count),
        "ignored": int(len(ignored_indices)),
        "recall": float(tp_count / n_gt) if n_gt > 0 else 0.0,
        "precision": float(tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0
    })
    
    if len(matched_correlations) > 0:
        results["temporal"]["mean_correlation"] = float(np.mean(np.abs(matched_correlations)))
        results["temporal"]["matched_correlations"] = matched_correlations

    # --- Plotting ---
    plotter.plot_segmentation_comparison(
        gt_masks=gt_masks,
        pred_masks=pred_masks,
        gt_binary=gt_binary,
        tp_pairs=tp_pairs,
        fn_indices=fn_indices,
        fp_indices=fp_indices,
        ignored_indices=ignored_indices,
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