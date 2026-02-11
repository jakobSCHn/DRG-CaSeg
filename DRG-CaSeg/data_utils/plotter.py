import PySide6.QtGui
import PySide6.QtWidgets

# Manually map QApplication to QtGui so the debugger finds it
PySide6.QtGui.QApplication = PySide6.QtWidgets.QApplication

import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cv2
import math
import caiman as cm
import matplotlib.gridspec as gridspec

from skimage.measure import find_contours
from matplotlib.patches import Patch
from scipy.stats import zscore, skew
from pathlib import Path
from contextlib import contextmanager
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

logger = logging.getLogger(__name__)

@contextmanager
def suppress_gui():
    """
    Temporarily switches the matplotlib backend to 'Agg' to prevent 
    interactive windows from popping up during figure creation.
    Caiman automatically activates interactive mode when imported.
    """
    #get current backend to restore it later
    original_backend = matplotlib.get_backend()
    try:
        matplotlib.use("Agg", force=True)
        yield
    finally:
        #restore the original backend (e.g., 'Qt5Agg')
        matplotlib.use(original_backend, force=True)


@suppress_gui()
def plot_image(
    image: np.ndarray,
    save_loc: str | Path,
    title: str = "Image",
    frame_id: int | None = None,
    cmap: str = "Greens",
    cbar_label: str = "Pixel Intensity",
    ):

    fig, ax = plt.subplots(figsize=(12,10))
    img = ax.imshow(image, cmap=cmap)
    ax.set_title(
        f"{title}; Frame: {frame_id}" if frame_id else title,
        fontsize=16,
        fontweight="bold",
    )
    ax.axis("off")

    fig.colorbar(img, ax=ax, orientation="vertical", label=cbar_label)
    fig.savefig(
        str(save_loc / f"{title}; Frame: {frame_id}" if frame_id else title),
        bbox_inches="tight",
        dpi=300,
    )
    
    return fig, ax


def plot_function(
    time,
    signal,
    title="Signal Plot",
    xlabel="Time",
    ylabel="Amplitude",
    filename="signal_plot.png"
    ):
    """
    Plots a signal (y-axis) against time (x-axis) using matplotlib
    and saves it to a file.

    Args:
        time (array-like): The x-axis data (e.g., time points).
        signal (array-like): The y-axis data (e.g., signal amplitude).
        title (str): The title for the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        filename (str): The name of the file to save the plot to.
    """
    plt.figure(figsize=(10, 5)) # Set a good figure size
    plt.plot(time, signal, label="Signal") # Plot the data
    
    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a legend
    plt.legend()
    
    # Ensure the layout is tight
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename)


def plot_temporal_patterns(
    ica_signals: np.ndarray,
    fs: float | None = None,
    n_components: int | None = None,
    n_cols: int = 4,
    filename: str | Path = "t_ica.png"
    ):
    n_components = n_components or ica_signals.shape[0]
    time = (
        np.arange(0, 1 / fs * ica_signals.shape[1], 1 / fs)
        if fs
        else np.arange(ica_signals.shape[1])
    )
    n_rows = int(np.ceil(n_components / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols*4, n_rows*2),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for i in range(n_components):
        ax = axes_flat[i]
        ax.plot(time, ica_signals[i, :])
        ax.set_title(f"Time Course of IC {i+1}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (normalized)")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")
    
    fig.tight_layout()
    plt.savefig(filename)


def plot_spatial_patterns(
    ica_filters: np.ndarray,
    n_components: int | None = None,
    n_cols: int = 4,
    filename: str | Path = "s_ica.png"
    ):
    """
    Plots all spatial filters in a grid.
    
    Args:
        mixedfilters (np.ndarray): Shape (pixw, pixh, n_components)
    """
    if ica_filters.ndim != 3:
        raise ValueError("mixedfilters must be a 3D array (pixw, pixh, n_components)")
        
    n_components = ica_filters.shape[0]
    
    n_rows = int(np.ceil(n_components / n_cols))
    
    fig, axes = plt.subplots(
        n_rows,
        n_cols, 
        figsize=(n_cols * 3, n_rows * 3), 
        squeeze=False
    )
    
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()
    
    for i in range(n_components):
        ax = axes_flat[i]
        
        spatial_filter = ica_filters[i, :, :]
        
        ax.imshow(spatial_filter, cmap="Greens", aspect='equal')
        ax.set_title(f"Spatial Filter {i+1}")
        ax.axis("off") # Hide x/y axes for images
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Optional: add colorbar
            
    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    plt.savefig(filename)


def plot_contours(
    masks,
    labels,
    background_img,
    save_filepath, 
    dpi=300, 
    cmap="viridis",
    ):
    """
    Generates a contour plot of all ROIs on a background image,
    where each independent ROI (neuron) has a unique color, 
    and saves it to a file.

    Args:
        master_neuron_mask (np.ndarray): 2D boolean array of detected neurons.
        background_image (np.ndarray): 2D grayscale background image for context.
        save_filepath (str): The full path (including filename) to save the plot.
                               (e.g., "C:/results/my_plot.png" or "./neuron_contours.png")
        dpi (int): Dots per inch for the saved image quality.
        cmap_name (str): The matplotlib colormap to use for unique colors.
                         'gist_rainbow' or 'tab20' are good choices.
    """
    
    num_rois = len(masks)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    try:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, num_rois))
    except AttributeError:
        raise ValueError(f"Colormap {cmap} has not enough colors to plot all ROIs.")
    
    vmin, vmax =  np.percentile(background_img, [1, 99])
    ax.imshow(background_img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")

    h, w = background_img.shape[:2]
    for i in range(num_rois):
        mask = masks[i]
        contours = find_contours(mask, 0.5)
        
        best_point = None
        min_dist = float("inf")

        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.2, color=colors[i])
            
            # Find the top-left-most point on the contour line (minimizing x + y)
            distances = contour[:, 0] + contour[:, 1]
            idx = np.argmin(distances)
            
            if distances[idx] < min_dist:
                min_dist = distances[idx]
                best_point = contour[idx]

        if best_point is not None:
            py, px = best_point
            
            # --- Boundary Safety & Offset Logic ---
            va = "bottom"
            ha = "right"
            offset_x = -2 
            offset_y = -2

            # Logic to "flip" the label inside if it's too close to the edge
            if py < 20: 
                va = "top"
                offset_y = 2
            
            if px < 20:
                ha = "left"
                offset_x = 2
                
            # Clamp coordinates to keep them inside the frame
            final_x = np.clip(px + offset_x, 5, w - 5)
            final_y = np.clip(py + offset_y, 5, h - 5)

            # Use current_color for the text to match the contour
            ax.text(
                final_x, final_y, str(labels[i]), 
                color=colors[i], 
                fontsize=8, 
                fontweight="bold",
                va=va, 
                ha=ha,
                # The black bbox is crucial now to make the colored text readable
                bbox=dict(facecolor="black", alpha=0.7, pad=0.1, edgecolor="none")
            )
    
    #finalize and save
    ax.set_title(f"ROI Contours")
    ax.axis("off")
    
    plt.tight_layout()
    fig.savefig(save_filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Contour Image saved successfully.")


def plot_contour_and_trace(
    roi_masks, 
    roi_traces, 
    background_image, 
    save_filepath, 
    dpi=300, 
    cmap_name="gist_rainbow",
    trace_stack_offset_std=5.0
    ):
    """
    Generates a 2-panel plot:
    1. Left Panel: Colored contours of all ROIs on a background image.
    2. Right Panel: Corresponding temporal traces, Z-scored and stacked.
    
    Each ROI contour and its trace are plotted in the same color.

    Args:
        roi_masks (list): List of 2D boolean masks (output from extract_rois_and_traces).
        roi_traces (np.ndarray): 2D array (n_ROIs, n_timesteps) of traces.
        background_image (np.ndarray): 2D grayscale background image.
        save_filepath (str): Full path to save the plot (e.g., "./rois_and_traces.png").
        dpi (int): Dots per inch for saved image quality.
        cmap_name (str): Colormap for unique ROI colors ('gist_rainbow', 'tab20').
        trace_stack_offset_std (float): How many std devs to offset each trace for stacking.
    """
    
    num_rois = len(roi_masks)
    if num_rois == 0:
        logger.warning("No ROIs provided. No image will be saved.")
        return
    
    if num_rois != roi_traces.shape[0]:
        raise ValueError(
            f"Mismatched inputs: {num_rois} masks and {roi_traces.shape[0]} traces."
        )

    # 1. Set up the figure and axes
    # We create two subplots, side-by-side
    fig, (ax_contour, ax_trace) = plt.subplots(
        1, 2, 
        figsize=(16, 8), 
        gridspec_kw={'width_ratios': [1, 1.2]} # Give traces a bit more space
    )
    
    # 2. Get a colormap
    try:
        colors = plt.colormaps[cmap_name].resampled(num_rois)
    except: # Fallback
        colors = plt.cm.get_cmap(cmap_name, num_rois)

    # --- 3. Left Panel: Plot Contours ---
    ax_contour.set_title(f"Detected ROIs ({num_rois})")
    ax_contour.imshow(background_image, cmap='gray')
    ax_contour.axis('off')

    # --- 4. Right Panel: Plot Traces ---
    ax_trace.set_title("Temporal Traces (Z-scored & Stacked)")
    
    logger.info(f"Plotting {num_rois} ROIs and traces...")
    
    # 5. Loop through each ROI and its trace
    for i in range(num_rois):
        roi_mask = roi_masks[i]
        trace = roi_traces[i]
        
        # Get the unique color for this ROI
        color = colors(i)
        
        # --- Plot Contour (on ax_contour) ---
        contours = find_contours(roi_mask, 0.5)
        for contour in contours:
            ax_contour.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=color)

        # --- Plot Trace (on ax_trace) ---
        # Z-score the trace: (x - mean) / std
        trace_z = zscore(trace)
        
        # Create a vertical offset for stacking
        # We plot the first trace at 0, the next at 5, the next at 10, etc.
        offset = i * trace_stack_offset_std
        
        # Plot the Z-scored trace with its offset
        ax_trace.plot(trace_z + offset, color=color, linewidth=0.5)

    # 6. Clean up trace plot
    ax_trace.set_xlabel("Time (frames)")
    # Hide the Y-axis labels/ticks as they are meaningless (just an offset)
    ax_trace.set_yticks([])
    ax_trace.set_ylabel(f"ROIs (stacked by {trace_stack_offset_std} std)")
    # Invert y-axis so trace 0 is at the top
    ax_trace.invert_yaxis()

    # 7. Finalize and save
    fig.tight_layout()
    try:
        fig.savefig(save_filepath, dpi=dpi, bbox_inches='tight', facecolor='w')
        logger.info(f"Successfully saved plot to: {save_filepath}")
    except Exception as e:
        logger.error(f"Error saving figure: {e}")
        
    plt.close(fig)


@suppress_gui()
def plot_spatial_filters(
    spatial_filters,
    save_filepath,
    cmap="Greens",
    dpi=300,
    title="Spatial Components",
    subtitle="Component",
    file_ext="spatial_components.png"
    ):
    """
    Plots a grid of ICA spatial filters in a roughly square layout.
    
    Parameters:
    - ica_filters: (nIC, X, Y) numpy array of spatial filters.
    - cmap: Colormap to use (default "Greens").
    - title: Overall figure title.
    """
    n_filters = spatial_filters.shape[0]
    if n_filters == 0:
        raise ValueError("No filters found to plot.")

    # Calculate rows and columns for a roughly square grid
    ncols = math.ceil(math.sqrt(n_filters))
    nrows = math.ceil(n_filters / ncols)

    # Create figure with a size appropriate for scientific reports
    # Adjusting the figsize based on the grid dimensions
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(ncols * 2.5, nrows * 2.5),
        constrained_layout=True
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Flatten axes array for easy iteration if it's 2D
    if n_filters > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for i in range(n_filters):
        ax = axes_flat[i]
        
        # Display the filter
        ax.imshow(spatial_filters[i], cmap=cmap, interpolation="nearest")
        ax.set_title(f"{subtitle} #{i+1}", fontsize=10)
        ax.axis("off") # Removes ticks and frame for a cleaner mask-like view

    # Hide any unused subplots in the grid
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.savefig(save_filepath / file_ext, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Display of spatial components saved successfully.")

@suppress_gui()
def plot_ica_components(
    spatial_filters,
    time_courses,
    save_filepath,
    sampling_rate=1.0,
    unit="z-score",
    cmap="Greens",
    dpi=300,
    title="ICA Temporal & Spatial Components",
    subtitle="IC",
    file_ext="ica_components.png"
):
    """
    Plots ICA spatial filters and their corresponding temporal traces.
    Traces are z-scored and grid lines are added for better readability.
    """
    n_filters = spatial_filters.shape[0]
    if n_filters == 0:
        raise ValueError("No filters found to plot.")

    ncols = 2 if n_filters > 1 else 1
    nrows_groups = math.ceil(n_filters / ncols)

    # Figure setup
    fig = plt.figure(figsize=(ncols * 6, nrows_groups * 4.5)) 
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    outer_grid = gridspec.GridSpec(nrows_groups, ncols, figure=fig, hspace=0.4, wspace=0.3)

    n_time_points = time_courses.shape[1]
    time_axis = np.array([t / sampling_rate for t in range(n_time_points)])

    for i in range(n_filters):
        # Create inner grid for Spatial (top) and Temporal (bottom)
        inner_grid = outer_grid[i].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
        
        # --- 1. PRE-PROCESS TRACE (Normalization Logic) ---
        raw_trace = time_courses[i]
        
        # Apply z-score and correct for skewness (flipping if extracting negative spikes)
        if np.std(raw_trace) > 0:
            proc_trace = zscore(raw_trace)
            if skew(proc_trace) < 0:
                proc_trace = -proc_trace
        else:
            proc_trace = raw_trace

        # --- 2. PLOT SPATIAL FILTER ---
        ax_spatial = fig.add_subplot(inner_grid[0])
        ax_spatial.imshow(spatial_filters[i], cmap=cmap, interpolation="nearest", aspect="equal")
        ax_spatial.set_title(f"{subtitle} #{i+1}", fontsize=14, fontweight="bold", pad=10)
        ax_spatial.axis("off")

        # --- 3. PLOT TEMPORAL TRACE ---
        ax_time = fig.add_subplot(inner_grid[1])
        
        # Add Grid Lines (New Feature)
        ax_time.grid(True, linestyle="--", linewidth=0.5, color="#9e9b9b", alpha=0.7)
        ax_time.set_axisbelow(True) # Ensures grid stays behind the plot line

        ax_time.plot(time_axis, proc_trace, color="#1b5e20", linewidth=1.2)
        
        ax_time.set_xlabel("Time (s)", fontsize=10)
        ax_time.set_ylabel(unit, fontsize=10)
        
        # Styling
        ax_time.spines["top"].set_visible(False)
        ax_time.spines["right"].set_visible(False)
        ax_time.tick_params(labelsize=9)

    plt.subplots_adjust(top=0.96)

    # Convert save_filepath to Path object if it isn't one already
    full_path = Path(save_filepath) / file_ext
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Enhanced ICA plot saved to {full_path}")


@suppress_gui()
def plot_summary_image(
    roi_masks: np.ndarray,
    roi_traces: np.ndarray,
    roi_labels: np.ndarray,
    md: dict,
    n_frames: int,
    background_img: np.ndarray,
    save_filepath: Path, 
    fps: int | float, 
    cmap: str,
    dpi: int,
    title: str = "Spatial Layout & Activity Summary",
    gt_traces = None,
    gt_labels = None,
    ):

    num_rois = len(roi_masks)
    img_h, img_w = background_img.shape[:2]
    img_aspect = img_h / img_w

    # --- DYNAMIC HEIGHT CALCULATION ---
    # Constraint 1: Traces must have enough space (e.g., 1.2 inches each)
    height_per_roi = 0.5 
    header_space = 2.0
    trace_min_height = header_space + (num_rois * height_per_roi)

    # Constraint 2: Image must not be crunched
    # We estimate the left column width is ~40% of the figure width.
    # If figure width is 16, left col is ~6.4 inches wide.
    # To maintain aspect ratio, height must be at least: width * aspect_ratio
    est_col_width = 16 * 0.4 
    image_min_height = (est_col_width * img_aspect) + header_space

    # We take the MAXIMUM of these two requirements
    # We also enforce a global minimum of 8 inches to look "scientific"
    fig_height = max(8.0, trace_min_height, image_min_height)

    # Setup Figure
    fig = plt.figure(figsize=(16, fig_height), dpi=dpi)
    
    # GridSpec
    # wspace=0.15, hspace=0.5 (generous vertical space)
    gs = gridspec.GridSpec(num_rois, 2, figure=fig, width_ratios=[1, 1.5], wspace=0.15, hspace=0.5)
    
    # Title positioning
    # We anchor the title to the top edge minus a small margin relative to figure height
    title_y_pos = 1.0 - (0.3 / fig_height) # 0.3 inches from top
    fig.suptitle(title, fontsize=20, fontweight="bold", y=title_y_pos)

    try:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, num_rois))
    except AttributeError:
        raise ValueError(f"Colormap {cmap} has not enough colors to plot all ROIs.")

    # ---------------------------------------------------------
    # LEFT PLOT: Spatial Map
    # ---------------------------------------------------------
    ax_map = fig.add_subplot(gs[:, 0])
    
    # Force alignment to the top if we have extra whitespace
    ax_map.set_anchor('N') 
    
    # Robust contrast stretching
    vmin, vmax = np.percentile(background_img, [1, 99])
    ax_map.imshow(background_img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")

    for i in range(num_rois):
        contours = find_contours(roi_masks[i], 0.5)
        largest_contour_idx = -1
        max_len = 0
        for c_idx, c in enumerate(contours):
            if len(c) > max_len:
                max_len = len(c)
                largest_contour_idx = c_idx

        for c_idx, contour in enumerate(contours):
            ax_map.plot(contour[:, 1], contour[:, 0], linewidth=1.2, color=colors[i])
            
            if c_idx == largest_contour_idx:
                ys, xs = contour[:, 0], contour[:, 1]
                min_y, max_y = np.argmin(ys), np.argmax(ys)
                min_x, max_x = np.argmin(xs), np.argmax(xs)
                candidates = [
                    (ys[min_y], xs[min_y], 'bottom', 'center'),
                    (ys[max_y], xs[max_y], 'top', 'center'),   
                    (ys[min_x], xs[min_x], 'center', 'right'), 
                    (ys[max_x], xs[max_x], 'center', 'left')   
                ]
                np.random.shuffle(candidates)
                final_pos = candidates[0]
                margin = img_h * 0.05    

                for y, x, va, ha in candidates:
                    if not (va=='bottom' and y<margin) and \
                       not (va=='top' and y>(img_h-margin)) and \
                       not (ha=='right' and x<margin) and \
                       not (ha=='left' and x>(img_w-margin)):
                        final_pos = (y, x, va, ha)
                        break

                txt = ax_map.text(
                    final_pos[1], final_pos[0], 
                    roi_labels[i], 
                    color=colors[i], 
                    fontsize=9, 
                    ha=final_pos[3], 
                    va=final_pos[2], 
                    fontweight="bold"
                )
                txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    if md.get("scale", []):
        bar_px = (100 / md["scale"])
        scalebar = AnchoredSizeBar(ax_map.transData, bar_px, "100 \u03bcm", "lower right", pad=0.5, color="white", frameon=False, size_vertical=2)
        ax_map.add_artist(scalebar)
    ax_map.axis("off")

    # ---------------------------------------------------------
    # RIGHT PLOTS: Stacked Traces
    # ---------------------------------------------------------
    time_seconds = np.arange(n_frames) / fps
    axes_traces = []
    
    ax_first = fig.add_subplot(gs[0, 1])
    axes_traces.append(ax_first)

    for i in range(num_rois):
        if i == 0:
            ax_trace = ax_first
        else:
            ax_trace = fig.add_subplot(gs[i, 1], sharex=ax_first)
            axes_traces.append(ax_trace)
        
        raw_trace = roi_traces[i]
        if np.std(raw_trace) == 0:
            trace_z = np.zeros_like(raw_trace)
        else:
            if skew(raw_trace) < 0: raw_trace = -raw_trace
            trace_z = zscore(raw_trace)
        if gt_traces is not None:
            raw_trace = gt_traces[i]
            if np.std(raw_trace) == 0:
                trace_z_gt = np.zeros_like(raw_trace)
            else:
                if skew(raw_trace) < 0: raw_trace = -raw_trace
                trace_z_gt = zscore(raw_trace)
        else:
            trace_z_gt= None
        
        if trace_z_gt is not None:
            ax_trace.plot(time_seconds, trace_z_gt, color="#000000", linewidth=1.2)
        ax_trace.plot(time_seconds, trace_z, color=colors[i], linewidth=1.2)
        ax_trace.grid(True, linestyle="--", linewidth=0.5, color="#9e9b9b", alpha=0.5)
        ax_trace.set_axisbelow(True)

        ax_trace.text(0.01, 0.85, f"ROI {roi_labels[i]}", transform=ax_trace.transAxes, fontsize=10, fontweight="bold", color="black")

        ax_trace.set_ylabel("z-score", fontsize=8, labelpad=10)
        ax_trace.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune='both'))
        ax_trace.tick_params(axis='y', labelsize=7)

        ax_trace.spines["top"].set_visible(False)
        ax_trace.spines["right"].set_visible(False)
        
        if i < num_rois - 1:
            ax_trace.tick_params(labelbottom=False, bottom=True) 
            ax_trace.spines["bottom"].set_visible(True) 
        else:
            ax_trace.set_xlabel("Time (s)", fontsize=10)
            ax_trace.tick_params(axis='x', labelsize=9)

    ax_first.set_xlim(time_seconds[0], time_seconds[-1])
    fig.align_ylabels(axes_traces)

    # TITLES - Anchored relative to top of figure (safe for dynamic height)
    # We calculate the Y position to be just below the suptitle
    header_y = 1.0 - (0.8 / fig_height)
    fig.text(0.28, header_y, "Neuron ROIs", fontsize=14, fontweight="bold", ha="center", va="top")
    fig.text(0.72, header_y, "Activity Traces (Z-scored)", fontsize=14, fontweight="bold", ha="center", va="top")

    # Margin adjust
    plt.subplots_adjust(top=1.0 - (1.2 / fig_height), bottom=0.05, left=0.05, right=0.95)

    full_path = Path(save_filepath)
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Summary Image saved to {full_path}")


@suppress_gui()
def plot_segmentation_comparison(
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    gt_binary: np.ndarray,
    tp_pairs: list,
    fn_indices: list,
    fp_indices: list,
    ignored_indices: list,
    results: dict,
    save_path: Path,
    ):
    """
    Handles the visualization of segmentation performance.
    """
    h, w = gt_masks.shape[1], gt_masks.shape[2]
    
    plt.figure(figsize=(12, 12), dpi=150)
    plt.imshow(np.zeros((h, w)), cmap="gray", interpolation="nearest")
    
    # 1. Plot False Negatives (Missed GTs)
    # Using the passed gt_binary for consistency with the calculation
    for idx in fn_indices:
        plt.contour(gt_binary[idx], colors="blue", linewidths=1.0, linestyles="dashed")

    # 2. Plot True Positives (Matched Pairs)
    for _, pred_idx in tp_pairs:
        plt.contour(pred_masks[pred_idx], colors="lime", linewidths=1.5)

    # 3. Plot False Positives (Noise)
    for idx in fp_indices:
        plt.contour(pred_masks[idx], colors="red", linewidths=1.5)

    # 4. Plot Ignored (Fragments/partial overlaps)
    for idx in ignored_indices:
        plt.contour(pred_masks[idx], colors="yellow", linewidths=1.0, alpha=0.7)

    # Legend & Title
    tp_count = results["spatial"]["true_positives"]
    fn_count = results["spatial"]["false_negatives"]
    fp_count = results["spatial"]["false_positives"]
    
    legend_elements = [
        Patch(facecolor="lime", edgecolor="lime", label=f"TP: Match ({tp_count})"),
        Patch(facecolor="blue", edgecolor="blue", linestyle="--", label=f"FN: Missed GT ({fn_count})"),
        Patch(facecolor="red", edgecolor="red", label=f"FP: Noise ({fp_count})"),
        Patch(facecolor="yellow", edgecolor="yellow", label=f"Ignored: Fragments ({len(ignored_indices)})"),
    ]
    
    plt.legend(handles=legend_elements, loc="upper right")
    
    title_str = (f"Performance\n"
                 f"Recall: {results['spatial']['recall']:.2f} | Precision: {results['spatial']['precision']:.2f}")
    plt.title(title_str)
    plt.axis("off")
    
    plt.savefig(save_path / "segmentation_performance.png", bbox_inches="tight", dpi=300)
    plt.close()


@suppress_gui()
def plot_temporal_comparison(
    gt_traces: np.ndarray,
    pred_traces: np.ndarray,
    tp_pairs: list,
    save_path: Path,
    fps: float,
    title: str = "Temporal Comparison",
    pred_labels: list = None
    ):
    """
    Plots a stacked comparison of ALL True Positive calcium traces.
    Dynamically adjusts figure height. Flips negative correlations.
    Includes grid lines and visible Y-axis.
    """
    n_pairs = len(tp_pairs)
    
    if n_pairs == 0:
        logger.warning("No True Positives found. Skipping temporal plot.")
        return

    # Dynamic Figure Height: 2 inches per subplot
    fig_height = max(4, 2.0 * n_pairs)
    
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, fig_height), dpi=150, sharex=True)
    
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    if n_pairs == 1:
        axes = [axes]

    gt_color = "#333333"
    pred_color = "#FF5733"
    
    for i, (gt_idx, pred_idx) in enumerate(tp_pairs):
        ax = axes[i]
        
        t_gt = gt_traces[gt_idx].flatten()
        t_pred = pred_traces[pred_idx].flatten()
        
        def process_trace(tr, fl):
            if np.std(tr) > 0:
                tr_proc = zscore(tr)
                if skew(tr_proc) < 0:
                    tr_proc = -tr_proc
                    fl = True
                return tr_proc, fl
            return tr, fl

        is_flipped = False

        t_gt_norm, is_flipped = process_trace(t_gt, is_flipped)
        t_pred_norm, _ = process_trace(t_pred, is_flipped)
        
        # Calculate Correlation
        if np.std(t_gt) == 0 or np.std(t_pred) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(t_gt, t_pred)[0, 1]

        
        # Plotting
        time_axis = np.arange(len(t_gt)) / fps
        ax.plot(time_axis, t_gt_norm, color=gt_color, linewidth=1.5, alpha=0.8, label="Ground Truth")
        ax.plot(time_axis, t_pred_norm, color=pred_color, linewidth=1.2, linestyle="--", label="Prediction")
        
        # --- Custom Labelling Logic ---
        p_name = pred_labels[pred_idx] if pred_labels else pred_idx
        flip_text = " (trace inverted)" if is_flipped else ""
        label_text = f"Pair #{i+1} (GT {gt_idx} | Pred {p_name}) - Pearson r: {corr:.3f}{flip_text}"
        
        ax.text(0.01, 0.85, label_text, 
                transform=ax.transAxes, fontsize=10, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))
        
        # --- STYLING UPDATES ---
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 1. Make Left Spine Visible (The Y-axis line)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # 2. Add Subtle Grid
        # alpha=0.3 makes it very faint; linestyle=':' makes it dotted
        ax.grid(True, which='major', axis='both', linestyle=':', color='gray', alpha=0.3)

        # Y-Axis labelling
        ax.set_ylabel("zscore", fontsize=8, rotation=90, labelpad=20, va="center")

        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=9)
            
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path / "temporal_performance.png", bbox_inches="tight")
    plt.close()


@suppress_gui()
def plot_mask_comparison(
    gt_masks,
    gt_traces,
    pred_masks,
    pred_traces,
    tp_pairs,
    save_filepath,
    fps=1.0,
    unit="z-score",
    cmap="viridis",
    dpi=300,
    title="Prediction vs. Ground Truth Component Comparison",
    file_ext="gt_pred_comparison.png"
):
    """
    Generates a grid plot where each pair is visually separated by a 
    centered row header (Title -> Plots order).
    """
    n_pairs = len(tp_pairs)
    if n_pairs == 0:
        raise ValueError("No matched pairs found to plot.")

    ncols = 2
    nrows = n_pairs

    # Increase height multiplier to ensure space for headers
    fig = plt.figure(figsize=(14, nrows * 5.0)) 
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.98)

    # hspace set to 0.6 to provide distinct gaps for the row headers
    outer_grid = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.6, wspace=0.25)

    n_frames = pred_traces.shape[1]
    time_axis = np.arange(n_frames) / fps

    for i, (gt_idx, pred_idx) in enumerate(tp_pairs):

        # --- PRE-PROCESS TRACES ---
        raw_p = pred_traces[pred_idx]
        raw_g = gt_traces[gt_idx]

        def process_trace(tr):
            if np.std(tr) > 0:
                tr_proc = zscore(tr)
                if skew(tr_proc) < 0:
                    tr_proc = -tr_proc
                return tr_proc
            return tr

        proc_p = process_trace(raw_p)
        proc_g = process_trace(raw_g)

        y_min = min(np.min(proc_p), np.min(proc_g))
        y_max = max(np.max(proc_p), np.max(proc_g))
        padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        shared_ylim = (y_min - padding, y_max + padding)

        pairs_data = [
            {"type": "Prediction", "idx": pred_idx, "mask": pred_masks[pred_idx], "trace": proc_p, "color": "#1b5e20"},
            {"type": "Ground Truth", "idx": gt_idx, "mask": gt_masks[gt_idx], "trace": proc_g, "color": "#444444"}
        ]

        for col_idx, data in enumerate(pairs_data):
            inner_grid = outer_grid[i, col_idx].subgridspec(2, 1, height_ratios=[1.8, 1], hspace=0.1)
            
            # --- Spatial Mask Plot ---
            ax_spatial = fig.add_subplot(inner_grid[0])
            ax_spatial.imshow(data["mask"], cmap=cmap, interpolation="nearest")
            ax_spatial.set_title(f"Pair: {i}; {data["type"]} (ID: {data["idx"]})", fontsize=12, fontweight="bold", pad=8)
            ax_spatial.axis("off")

            # --- Temporal Trace Plot ---
            ax_time = fig.add_subplot(inner_grid[1])
            ax_time.grid(True, linestyle="--", linewidth=0.5, color="#9e9b9b", alpha=0.7)
            ax_time.set_axisbelow(True)
            
            ax_time.plot(time_axis, data["trace"], color=data["color"], linewidth=1.2)
            ax_time.set_ylim(shared_ylim)
            
            if i == nrows - 1:
                ax_time.set_xlabel("Time (s)", fontsize=10)
            
            ax_time.set_ylabel(unit, fontsize=8)
            ax_time.spines["top"].set_visible(False)
            ax_time.spines["right"].set_visible(False)
            ax_time.tick_params(labelsize=8)

    plt.subplots_adjust(top=0.96, bottom=0.05, left=0.07, right=0.93)
    full_path = Path(save_filepath) / file_ext
    fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_inference_video(
    roi_masks, 
    roi_traces, 
    video_data, 
    save_filepath, 
    fps=30, 
    dpi=100,  
    cmap_name="gist_rainbow",
    trace_stack_offset_std=5.0
    ):
    
    num_rois = len(roi_masks)
    n_frames, height, width = video_data.shape
    
    
    fig, (ax_video, ax_trace) = plt.subplots(
        1, 2, 
        figsize=(16, 8), 
        gridspec_kw={"width_ratios": [1, 1.2]},
        dpi=dpi
    )
    fig.suptitle("Isolated Spatial and Temporal Signatures", fontsize=20, fontweight="bold", y=0.95)

    
    try:
        colors = plt.colormaps[cmap_name].resampled(num_rois)
    except AttributeError:
        colors = plt.cm.get_cmap(cmap_name, num_rois)

    ax_trace.set_title("Temporal Traces (Z-scored & Stacked)")
    
    for i in range(num_rois):
        trace_z = zscore(roi_traces[i])
        offset = i * trace_stack_offset_std
        ax_trace.plot(trace_z + offset, color=colors(i), linewidth=0.5)

    ax_trace.set_yticks([])
    ax_trace.set_xlabel("Time (frames)")
    ax_trace.set_ylabel(f"ROIs (stacked)")
    ax_trace.invert_yaxis()
    
    # The moving vertical line
    time_indicator = ax_trace.axvline(x=0, color="black", linestyle='--', alpha=0.7)

    ax_video.set_title("Detected ROIs")
    ax_video.axis("off")
    
    # Pre-calculate contours to avoid doing it in the loop
    for i in range(num_rois):
        contours = find_contours(roi_masks[i], 0.5)
        for contour in contours:
            ax_video.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=colors(i))

    # Initial Image
    vmin, vmax = np.percentile(video_data, 1), np.percentile(video_data, 99)
    img_display = ax_video.imshow(
        video_data[0], 
        cmap="gray", 
        vmin=vmin, 
        vmax=vmax
    )
    
    fig.canvas.draw()
    mat_img = np.array(fig.canvas.renderer.buffer_rgba())
    h_plot, w_plot, _ = mat_img.shape

    # Define the codec (mp4v is standard for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(save_filepath, fourcc, fps, (w_plot, h_plot))

    if not out.isOpened():
        raise IOError(f"Could not open video writer for {save_filepath}")

    logger.info(f"Rendering {n_frames} frames to {save_filepath}.")


    for frame_idx in range(n_frames):
        img_display.set_data(video_data[frame_idx])
        time_indicator.set_xdata([frame_idx, frame_idx])
        ax_video.set_title(f"Frame {frame_idx}")
        
        fig.canvas.draw()
        img_rgba = np.array(fig.canvas.renderer.buffer_rgba())
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        
        out.write(img_bgr)
        
        # Optional: Print progress every 10%
        if frame_idx % (n_frames // 10) == 0:
            logger.info(f"Processed {frame_idx}/{n_frames} frames...")

    # --- 6. Cleanup ---
    out.release()
    plt.close(fig)
    logger.info("Video saved successfully.")
