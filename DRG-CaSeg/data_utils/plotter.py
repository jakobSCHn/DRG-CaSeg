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
    unit="ΔF/F",
    cmap="Greens",
    dpi=300,
    title="ICA Temporal & Spatial Components",
    subtitle="IC",
    file_ext="ica_components_combined.png"
):
    n_filters = spatial_filters.shape[0]
    if n_filters == 0:
        raise ValueError("No filters found to plot.")

    ncols = 2 if n_filters > 1 else 1
    nrows_groups = math.ceil(n_filters / ncols)

    # ADJUSTMENT 1: Reduce the height multiplier if the whitespace is still too much
    fig = plt.figure(figsize=(ncols * 6, nrows_groups * 4.5)) 
    
    # ADJUSTMENT 2: Move the title closer to the top edge (y=0.95 or 0.98)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    # outer_grid remains mostly the same
    outer_grid = gridspec.GridSpec(nrows_groups, ncols, figure=fig, hspace=0.4, wspace=0.3)

    n_time_points = time_courses.shape[1]
    time_axis = [t / sampling_rate for t in range(n_time_points)]

    for i in range(n_filters):
        inner_grid = outer_grid[i].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
        
        ax_spatial = fig.add_subplot(inner_grid[0])
        ax_spatial.imshow(spatial_filters[i], cmap=cmap, interpolation="nearest", aspect="equal")
        ax_spatial.set_title(f"{subtitle} #{i+1}", fontsize=14, fontweight="bold", pad=10)
        ax_spatial.axis("off")

        ax_time = fig.add_subplot(inner_grid[1])
        ax_time.plot(time_axis, time_courses[i], color="#1b5e20", linewidth=1.2)
        ax_time.set_xlabel("Time (s)", fontsize=10)
        ax_time.set_ylabel(unit, fontsize=10)
        ax_time.spines["top"].set_visible(False)
        ax_time.spines["right"].set_visible(False)
        ax_time.tick_params(labelsize=9)

    # ADJUSTMENT 3: Explicitly define the top margin of the subplots
    # top=0.92 tells Matplotlib to start the plots at 92% of the figure height
    plt.subplots_adjust(top=0.96)

    full_path = save_filepath / file_ext
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
    trace_stack_offset_std: float,
    cmap: str,
    dpi: int,
    title: str = "Spatial Layout & Activity Summary", 
    ):
    """
    Renders a static summary plot using a Maximum Intensity Projection (MIP)
    and stacked activity traces.
    Features: 
      - Colored labels matching ROIs
      - Randomized label positioning (Top/Bottom/Left/Right)
      - Boundary checks to keep labels inside image
    """

    num_rois = len(roi_masks)
    img_h, img_w = background_img.shape[:2] # Get image dimensions for boundary checks

    fig, (ax_map, ax_trace) = plt.subplots(
        1, 2, 
        figsize=(16, 9), 
        gridspec_kw={"width_ratios": [1, 1.2]},
        dpi=dpi
    )
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.95)

    try:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, num_rois))
    except AttributeError:
        raise ValueError(f"Colormap {cmap} has not enough colors to plot all ROIs.")

    # LEFT PLOT: Spatial Map
    vmin, vmax =  np.percentile(background_img, [1, 99])
    ax_map.imshow(background_img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")

    for i in range(num_rois):
        contours = find_contours(roi_masks[i], 0.5)
        
        # Find the largest blob to label
        largest_contour_idx = -1
        max_len = 0
        for c_idx, c in enumerate(contours):
            if len(c) > max_len:
                max_len = len(c)
                largest_contour_idx = c_idx

        for c_idx, contour in enumerate(contours):
            ax_map.plot(contour[:, 1], contour[:, 0], linewidth=1.2, color=colors[i])
            
            # --- New Labelling Logic ---
            if c_idx == largest_contour_idx:
                # Extract coordinates (y=row, x=col)
                ys = contour[:, 0]
                xs = contour[:, 1]
                
                # Identify 4 cardinal points on the blob
                min_y_idx, max_y_idx = np.argmin(ys), np.argmax(ys)
                min_x_idx, max_x_idx = np.argmin(xs), np.argmax(xs)

                # Create candidates: (y, x, vertical_align, horizontal_align)
                # 'va=bottom' means text sits ON TOP of the anchor point
                # 'va=top' means text hangs BELOW the anchor point
                candidates = [
                    (ys[min_y_idx], xs[min_y_idx], 'bottom', 'center'), # Top Edge
                    (ys[max_y_idx], xs[max_y_idx], 'top', 'center'),    # Bottom Edge
                    (ys[min_x_idx], xs[min_x_idx], 'center', 'right'),  # Left Edge
                    (ys[max_x_idx], xs[max_x_idx], 'center', 'left')    # Right Edge
                ]
                
                # Randomize the candidates to avoid clustering
                np.random.shuffle(candidates)
                
                # Select the first valid candidate that fits within boundaries
                final_pos = candidates[0] # Fallback
                margin = img_h * 0.05     # 5% buffer from edge (approx 20-50px)

                for y, x, va, ha in candidates:
                    is_safe = True
                    
                    # Check Top Boundary
                    if va == 'bottom' and y < margin: is_safe = False
                    # Check Bottom Boundary
                    if va == 'top' and y > (img_h - margin): is_safe = False
                    # Check Left Boundary
                    if ha == 'right' and x < margin: is_safe = False
                    # Check Right Boundary
                    if ha == 'left' and x > (img_w - margin): is_safe = False
                    
                    if is_safe:
                        final_pos = (y, x, va, ha)
                        break

                # Plot Text
                # 1. Use colors[i] for text color
                # 2. Add white stroke for readability on dark background
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

    bar_length_um = 100
    bar_length_px = bar_length_um / md["scale"]
    scalebar = AnchoredSizeBar(ax_map.transData, bar_length_px, f"{bar_length_um} \u03bcm", 
                                "lower right", pad=0.5, color="white", frameon=False, size_vertical=2)
    ax_map.add_artist(scalebar)
    ax_map.set_title("Neuron ROIs", fontsize=14)
    ax_map.axis("off")

    # RIGHT PLOT: Stacked Traces
    time_seconds = np.arange(n_frames) / fps
    ax_trace.set_title("Activity Traces (Z-scored)")
    
    for i in range(num_rois):
        raw_trace = roi_traces[i]
        offset = i * trace_stack_offset_std

        # Handle Zero Traces
        if np.std(raw_trace) == 0:
            trace_z = np.zeros_like(raw_trace)
        else:
            # Auto-Flip based on Skewness
            if skew(raw_trace) < 0:
                raw_trace = -raw_trace
            trace_z = zscore(raw_trace)
        
        # Plot (inverted y-axis logic: offset - trace)
        ax_trace.plot(time_seconds, offset - trace_z, color=colors[i], linewidth=0.8)
        
        # Label the trace
        ax_trace.text(
            -0.01 * time_seconds[-1], 
            offset, 
            roi_labels[i], 
            color=colors[i], 
            fontweight="bold", 
            ha="right", 
            va="center", 
            fontsize=9
        )
        
        ax_trace.axhline(y=offset, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)

    ax_trace.plot([time_seconds[-1] * 1.02, time_seconds[-1] * 1.02], [0, 2], color="black", lw=1.5)
    ax_trace.text(time_seconds[-1] * 1.03, 1, "2 SD", rotation=270, va="center")

    ax_trace.set_xlabel("Time (s)", fontsize=12)
    ax_trace.set_xlim(-0.1 * time_seconds[-1], time_seconds[-1])
    ax_trace.set_yticks([])
    ax_trace.invert_yaxis()
    
    for spine in ["top", "right", "left"]:
        ax_trace.spines[spine].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary Image saved successfully.")


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
        
        # Normalize (0-1)
        t_gt_norm = (t_gt - np.min(t_gt)) / (np.ptp(t_gt) + 1e-8)
        t_pred_norm = (t_pred - np.min(t_pred)) / (np.ptp(t_pred) + 1e-8)
        
        # Calculate Correlation
        if np.std(t_gt) == 0 or np.std(t_pred) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(t_gt, t_pred)[0, 1]

        # --- FLIP LOGIC ---
        is_flipped = False
        if corr < 0:
            t_pred_norm = 1 - t_pred_norm
            is_flipped = True

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

        # Y-Axis Scale
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=8)
        ax.set_ylabel("Norm.\nAmp.", fontsize=8, rotation=0, labelpad=20, va="center")

        if i == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=9)
            
    axes[-1].set_xlabel("Time (s)", fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(save_path / "temporal_performance.png", bbox_inches="tight")
    plt.close()


@suppress_gui()
def plot_gt_overlay(
    gt_masks: np.ndarray,
    gt_traces: np.ndarray,
    pred_masks: np.ndarray,
    pred_traces: np.ndarray,
    tp_pairs: list,
    md: dict,
    save_path: Path, 
    fps: int | float, 
    n_frames: int,
    trace_stack_offset_std: float = 5.0,
    cmap: str = "viridis",
    dpi: int = 200,
    title: str = "Ground Truth vs. Prediction Overlay", 
    ):
    """
    Renders a hybrid summary plot:
    - Left: Existing GT Gaussian blobs (grayscale) with Prediction contours overlaid.
    - Right: Stacked GT traces (grey) overlaid with Prediction traces (colored, dotted).
    """

    n_pairs = len(tp_pairs)
    if n_pairs == 0:
        logger.warning("No matches to plot.")
        return

    # Image Dimensions from the first mask
    img_h, img_w = gt_masks[0].shape

    fig, (ax_map, ax_trace) = plt.subplots(
        1, 2, 
        figsize=(18, 10), 
        gridspec_kw={"width_ratios": [1, 1.2]},
        dpi=dpi
    )
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.96)

    try:
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_pairs))
    except AttributeError:
        raise ValueError(f"Colormap {cmap} not found.")

    # ==========================================
    # 1. LEFT PLOT: GT Background + Pred Overlay
    # ==========================================
    
    # Composite the pre-existing Gaussian blobs
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
    
    # Plot GT Background
    ax_map.imshow(gt_composite, cmap="gray", vmin=0, vmax=1, origin='upper')

    # Plot Prediction Contours (Matched Only)
    for i, (gt_idx, pred_idx) in enumerate(tp_pairs):
        # We find contours on the binary prediction mask
        contours = find_contours(pred_masks[pred_idx], 0.5)
        
        largest_contour_idx = -1
        max_len = 0
        for c_idx, c in enumerate(contours):
            if len(c) > max_len:
                max_len = len(c)
                largest_contour_idx = c_idx

        for c_idx, contour in enumerate(contours):
            ax_map.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=colors[i])
            
            # --- Smart Labelling ---
            if c_idx == largest_contour_idx:
                ys, xs = contour[:, 0], contour[:, 1]
                min_y, max_y = np.argmin(ys), np.argmax(ys)
                min_x, max_x = np.argmin(xs), np.argmax(xs)

                # Anchor Candidates
                candidates = [
                    (ys[min_y], xs[min_y], 'bottom', 'center'),
                    (ys[max_y], xs[max_y], 'top', 'center'),
                    (ys[min_x], xs[min_x], 'center', 'right'),
                    (ys[max_x], xs[max_x], 'center', 'left')
                ]
                np.random.shuffle(candidates)
                
                # Boundary Checks
                final_pos = candidates[0]
                margin = img_h * 0.05
                for y, x, va, ha in candidates:
                    is_safe = True
                    if va == "bottom" and y < margin: is_safe = False
                    if va == "top" and y > (img_h - margin): is_safe = False
                    if ha == "right" and x < margin: is_safe = False
                    if ha == "left" and x > (img_w - margin): is_safe = False
                    if is_safe:
                        final_pos = (y, x, va, ha)
                        break

                label_txt = f"P{pred_idx}"
                txt = ax_map.text(
                    final_pos[1], final_pos[0], 
                    label_txt, 
                    color=colors[i], 
                    fontsize=9, 
                    ha=final_pos[3], 
                    va=final_pos[2], 
                    fontweight="bold"
                )
                txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    if "scale" in md:
        bar_length_um = 100
        bar_length_px = bar_length_um / md["scale"]
        scalebar = AnchoredSizeBar(ax_map.transData, bar_length_px, f"{bar_length_um} \u03bcm", 
                                    "lower right", pad=0.5, color="white", frameon=False, size_vertical=2)
        ax_map.add_artist(scalebar)
    
    ax_map.set_title("Background: GT (Gray) | Overlay: Prediction (Color)", fontsize=14)
    ax_map.axis("off")

    # ==========================================
    # 2. RIGHT PLOT: Stacked Hybrid Traces
    # ==========================================
    time_seconds = np.arange(n_frames) / fps
    ax_trace.set_title("Trace Comparison (Solid: GT | Dotted: Pred)")
    
    for i, (gt_idx, pred_idx) in enumerate(tp_pairs):
        offset = i * trace_stack_offset_std
        
        # --- Prepare GT Trace (Baseline) ---
        raw_gt = gt_traces[gt_idx]
        if np.std(raw_gt) == 0:
            gt_z = np.zeros_like(raw_gt)
        else:
            if skew(raw_gt) < 0: raw_gt = -raw_gt
            gt_z = zscore(raw_gt)

        # --- Prepare Pred Trace (Overlay) ---
        raw_pred = pred_traces[pred_idx]
        if np.std(raw_pred) == 0:
            pred_z = np.zeros_like(raw_pred)
        else:
            # Match flip to GT
            corr = np.corrcoef(raw_gt, raw_pred)[0, 1] if np.std(raw_gt) > 0 else 0
            if corr < 0:
                raw_pred = -raw_pred
            pred_z = zscore(raw_pred)

        # Plot GT (Solid Gray)
        ax_trace.plot(time_seconds, offset - gt_z, color="#444444", linewidth=1.5, alpha=0.6, zorder=1)
        
        # Plot Prediction (Dotted Color)
        ax_trace.plot(time_seconds, offset - pred_z, color=colors[i], linewidth=1.2, linestyle=":", zorder=2)
        
        # Label
        label_text = f"Pair {i+1} (G{gt_idx}/P{pred_idx})"
        ax_trace.text(
            -0.01 * time_seconds[-1], 
            offset, 
            label_text, 
            color=colors[i], 
            fontweight="bold", 
            ha="right", 
            va="center", 
            fontsize=8
        )
        
        ax_trace.axhline(y=offset, color="gray", linestyle="-", linewidth=0.3, alpha=0.3)

    ax_trace.plot([time_seconds[-1] * 1.02, time_seconds[-1] * 1.02], [0, 2], color="black", lw=1.5)
    ax_trace.text(time_seconds[-1] * 1.03, 1, "2 SD", rotation=270, va="center")

    ax_trace.set_xlabel("Time (s)", fontsize=12)
    ax_trace.set_xlim(-0.15 * time_seconds[-1], time_seconds[-1])
    ax_trace.set_yticks([])
    ax_trace.invert_yaxis()
    
    for spine in ["top", "right", "left"]:
        ax_trace.spines[spine].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path / "gt_overlay.png", dpi=dpi, bbox_inches="tight")
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
