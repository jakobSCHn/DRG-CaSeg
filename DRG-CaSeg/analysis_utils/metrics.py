import numpy as np
from scipy.optimize import linear_sum_assignment

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

def evaluate_segmentation(
    masks_true: np.ndarray, 
    masks_pred: np.ndarray, 
    iou_threshold: float = 0.5
    ) -> dict:
    """
    Matches predicted masks to ground truth using Hungarian algorithm and computes metrics.
    
    Args:
        masks_true: Boolean array (N_true, H, W)
        masks_pred: Boolean array (N_pred, H, W)
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
    if masks_true.shape[0] == 0 or masks_pred.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "mean_iou": 0.0}

    iou_matrix = compute_iou_matrix(masks_true, masks_pred)
    
    #Hungarian Algorithm to find best matches
    row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)
    
    #Filter matches by threshold
    matched_ious = iou_matrix[row_ind, col_ind]
    valid_matches = matched_ious > iou_threshold
    
    true_positives = np.sum(valid_matches)
    mean_iou = np.mean(matched_ious[valid_matches]) if true_positives > 0 else 0.0
    
    #False Positives: Predictions that weren't matched to any GT or had low IoU
    false_positives = masks_pred.shape[0] - true_positives
    
    #False Negatives: GT masks that weren't matched to any Pred or had low IoU
    false_negatives = masks_true.shape[0] - true_positives
    
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