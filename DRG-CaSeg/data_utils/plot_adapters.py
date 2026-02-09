import logging
import numpy as np

from data_utils.plotter import plot_summary_image

logger = logging.getLogger(__name__)


def plot_prediction_summary(
    results,
    data,
    save_filepath,
    fps=None,
    trace_stack_offset_std=6.0,
    cmap="viridis",
    dpi=300,
    ):

    md = data.meta_data[0]
    if fps is None:
        fps = getattr(data, "fr", None) or 30

    n_frames = data.shape[0]
    background_img = np.max(data, axis=0)

    plot_summary_image(
        roi_masks=results["masks"],
        roi_traces=results["traces"],
        roi_labels=results["labels"],
        md=md,
        n_frames=n_frames,
        background_img=background_img,
        save_filepath=save_filepath / "prediction_summary.png",
        fps=fps,
        trace_stack_offset_std=trace_stack_offset_std,
        cmap=cmap,
        dpi=dpi,
        title="Prediction Spatial Layout & Activity Summary"
    )


def plot_gt_summary(
    gt,
    data,
    save_filepath,
    fps=None,
    trace_stack_offset_std=6.0,
    cmap="viridis",
    dpi=300,
    ):
    
    if gt:
        md = data.meta_data[0]
        if fps is None:
            fps = getattr(data, "fr", None) or 30

        n_frames = data.shape[0]
        background_img = np.max(data, axis=0)

        plot_summary_image(
            roi_masks=gt["spatial"],
            roi_traces=gt["temporal"],
            roi_labels=gt["labels"],
            md=md,
            n_frames=n_frames,
            background_img=background_img,
            save_filepath=save_filepath / "gt_summary.png",
            fps=fps,
            trace_stack_offset_std=trace_stack_offset_std,
            cmap=cmap,
            dpi=dpi,
            title="Ground Truth Spatial Layout & Activity Summary"
        )
    else:
        logger.warning(
            "No Ground Truth available! Please check the data" \
            "configuration!")  
