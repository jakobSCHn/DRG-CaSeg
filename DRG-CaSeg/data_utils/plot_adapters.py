import logging
import numpy as np

from data_utils.plotter import plot_summary_image
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_adapter_prediction_summary(
    results,
    data,
    background_img,
    save_filepath,
    fps=None,
    cmap="viridis",
    dpi=300,
    ):

    md = data.meta_data[0]
    if fps is None:
        fps = getattr(data, "fr", None) or 30

    n_frames = data.shape[0]

    plot_summary_image(
        roi_masks=results["masks"],
        roi_traces=results["traces"],
        roi_labels=results["labels"],
        md=md,
        n_frames=n_frames,
        background_img=background_img,
        save_filepath=save_filepath / "prediction_summary.png",
        fps=fps,
        cmap=cmap,
        dpi=dpi,
        title="Prediction Spatial Layout & Activity Summary"
    )


def plot_adapter_gt_summary(
    gt,
    data,
    background_img,
    save_filepath,
    fps=None,
    cmap="viridis",
    dpi=300,
    ):
    
    if gt:
        md = data.meta_data[0]
        if fps is None:
            fps = getattr(data, "fr", None) or 30

        n_frames = data.shape[0]

        plot_summary_image(
            roi_masks=gt["spatial"],
            roi_traces=gt["temporal"],
            roi_labels=gt["labels"],
            md=md,
            n_frames=n_frames,
            background_img=background_img,
            save_filepath=save_filepath / "gt_summary.png",
            fps=fps,
            cmap=cmap,
            dpi=dpi,
            title="Ground Truth Spatial Layout & Activity Summary"
        )
    else:
        logger.warning(
            "No Ground Truth available! Please check the data" \
            "configuration!") 

def plot_adapter_gt_overlay(
    gt_masks: np.ndarray,
    gt_traces: np.ndarray,
    pred_masks: np.ndarray,
    pred_traces: np.ndarray,
    tp_pairs: list,
    md: dict,
    save_path: Path, 
    fps: int | float, 
    n_frames: int,
    cmap: str = "viridis",
    dpi: int = 200,
    title: str = "Ground Truth vs. Prediction Overlay", 
    ):

    img_h, img_w = gt_masks[0].shape
    gt_composite = np.zeros((img_h, img_w), dtype=float)
    
    for mask in gt_masks:
        blob = mask.astype(float) 
        # Normalize blob so max is 1.0 
        # (Ensures the 0.25 cutoff works consistently for every blob)
        if blob.max() > 0:
            blob /= blob.max()
            
        # Add to composite using max projection
        gt_composite = np.maximum(gt_composite, blob)

    # Apply Cutoff: Pixels < 0.25 intensity become 0 (Black background)
    gt_composite[gt_composite < 0.25] = 0

    roi_masks = []
    roi_traces = []
    ground_truth_traces = []
    pred_indices = []
    gt_indices= []
    for i, (gt_idx, pred_idx) in enumerate(tp_pairs):
        roi_masks.append(pred_masks[pred_idx])
        roi_traces.append(pred_traces[pred_idx])
        ground_truth_traces.append(gt_traces[gt_idx])
        pred_indices.append(pred_idx)
        gt_indices.append(gt_indices)
    roi_masks = np.stack(roi_masks, axis=0)
    roi_traces = np.stack(roi_traces, axis=0)
    ground_truth_traces = np.stack(ground_truth_traces, axis=0)


    plot_summary_image(
        roi_masks=roi_masks,
        roi_traces=roi_traces,
        roi_labels=pred_indices,
        md=md,
        n_frames=n_frames,
        background_img=gt_composite,
        save_filepath=save_path / "gt_overlay.png",
        fps=fps,
        cmap=cmap,
        dpi=dpi,
        title=title,
        gt_traces=ground_truth_traces,
        gt_labels= gt_indices,
    )