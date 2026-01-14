import numpy as np

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