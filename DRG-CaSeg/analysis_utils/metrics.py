import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from data_utils.plotter import plot_image

import logging
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
    coverage_threshold: float,
    gt_binary_threshold: float = 0.25,  # New parameter to handle Gaussian GTs
    save_path: str = "segmentation_performance.png"
    ) -> dict:
    """
    Calculates performance with on-the-fly binarization of continuous Ground Truth.
    """
    
    # --- 1. Setup & Binarization ---
    n_gt = gt_masks.shape[0]
    n_pred = pred_masks.shape[0]
    
    # BINARIZE GROUND TRUTH
    # This fixes the "Area vs Intensity" mismatch
    gt_binary = (gt_masks > gt_binary_threshold).astype(np.float32)
    
    # Lists for plotting
    tp_pairs = []       
    fp_indices = []     
    fn_indices = []     
    ignored_indices = [] 
    matched_correlations = []

    results = {
        "spatial": {
            "true_positives": 0, "false_negatives": 0, "false_positives": 0,
            "recall": 0.0, "precision": 0.0, "ignored": 0
        },
        "temporal": {
            "mean_correlation": 0.0, "matched_correlations": []
        }
    }

    # --- 2. Calculation Core ---
    if n_gt == 0:
        fp_indices = list(range(n_pred))
        results["spatial"]["false_positives"] = int(n_pred)
    elif n_pred == 0:
        fn_indices = list(range(n_gt))
        results["spatial"]["false_negatives"] = int(n_gt)
    else:
        # Flatten using the BINARY GT
        gt_flat = gt_binary.reshape(n_gt, -1)
        pred_flat = pred_masks.reshape(n_pred, -1).astype(np.float32)
        
        # Intersection & Purity
        intersection = np.dot(gt_flat, pred_flat.T)
        pred_areas = pred_flat.sum(axis=1)
        
        # Purity: % of Pred j that is inside the BINARIZED GT i
        purity_matrix = intersection / (pred_areas[np.newaxis, :] + 1e-8)

        # Greedy Matching (1-to-1 for TPs)
        potential_gt, potential_pred = np.where(purity_matrix >= coverage_threshold)
        scores = purity_matrix[potential_gt, potential_pred]
        sorted_indices = np.argsort(scores)[::-1]
        
        matched_gt_set = set()
        matched_pred_set = set()
        
        for idx in sorted_indices:
            gt_i = potential_gt[idx]
            pred_j = potential_pred[idx]
            
            if gt_i not in matched_gt_set and pred_j not in matched_pred_set:
                matched_gt_set.add(gt_i)
                matched_pred_set.add(pred_j)
                tp_pairs.append((int(gt_i), int(pred_j)))
        
        # Identify FNs (GTs not in matched set)
        fn_indices = [i for i in range(n_gt) if i not in matched_gt_set]

        # Identify FPs vs Ignored (Preds not in matched set)
        for pred_idx in range(n_pred):
            if pred_idx not in matched_pred_set:
                # Check total overlap with ALL GTs using the binarized versions
                total_overlap = intersection[:, pred_idx].sum() / (pred_areas[pred_idx] + 1e-8)
                
                if total_overlap < coverage_threshold:
                    fp_indices.append(int(pred_idx))
                else:
                    ignored_indices.append(int(pred_idx))

        # Temporal Correlation
        for gt_i, pred_j in tp_pairs:
            t_gt = gt_traces[gt_i].flatten()
            t_pred = pred_traces[pred_j].flatten()
            
            if np.std(t_gt) == 0 or np.std(t_pred) == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(t_gt, t_pred)[0, 1])
            matched_correlations.append(corr)

    # --- 3. Update Results Dictionary ---
    tp_count = len(tp_pairs)
    fp_count = len(fp_indices)
    fn_count = len(fn_indices)
    
    results["spatial"].update({
        "true_positives": int(tp_count),
        "false_negatives": int(fn_count),
        "false_positives": int(fp_count),
        "ignored": int(len(ignored_indices)),
        "recall": float(tp_count / n_gt) if n_gt > 0 else 0.0,
        "precision": float(tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0
    })
    
    if len(matched_correlations) > 0:
        results["temporal"]["mean_correlation"] = float(np.mean(matched_correlations))
        results["temporal"]["matched_correlations"] = matched_correlations

    # --- 4. Plotting & Saving ---
    plt.figure(figsize=(12, 12), dpi=150)
    h, w = gt_masks.shape[1], gt_masks.shape[2]
    plt.imshow(np.zeros((h, w)), cmap="gray", interpolation="nearest")
    
    # Plot False Negatives using BINARIZED masks for consistent visuals
    for idx in fn_indices:
        plt.contour(gt_binary[idx], colors="blue", linewidths=1.0, linestyles="dashed")

    for _, pred_idx in tp_pairs:
        plt.contour(pred_masks[pred_idx], colors="lime", linewidths=1.5)

    for idx in fp_indices:
        plt.contour(pred_masks[idx], colors="red", linewidths=1.5)

    for idx in ignored_indices:
        plt.contour(pred_masks[idx], colors="yellow", linewidths=1.0, alpha=0.7)

    legend_elements = [
        Patch(facecolor='lime', edgecolor='lime', label=f'TP: Match ({tp_count})'),
        Patch(facecolor='blue', edgecolor='blue', linestyle='--', label=f'FN: Missed GT ({fn_count})'),
        Patch(facecolor='red', edgecolor='red', label=f'FP: Noise ({fp_count})'),
        Patch(facecolor='yellow', edgecolor='yellow', label=f'Ignored: Fragments ({len(ignored_indices)})'),
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    title_str = (f"Performance (Thresh={coverage_threshold})\n"
                 f"Recall: {results['spatial']['recall']:.2f} | Precision: {results['spatial']['precision']:.2f}")
    plt.title(title_str)
    plt.axis("off")
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return results