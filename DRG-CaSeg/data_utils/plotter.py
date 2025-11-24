import numpy as np
import logging
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import label
from skimage.measure import find_contours
from scipy.stats import zscore
from pathlib import Path

logger = logging.getLogger(__name__)

def plot_frame(
    vid: np.ndarray,
    frame_id: int,
    save_loc: str | Path,
    title: str = "Video Frame"
    ):

    img = vid[frame_id]    
    return plot_image(
        frame=img,
        save_loc=save_loc,
        title=title,
        frame_id=frame_id)


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


def save_colored_contour_plot(master_neuron_mask, 
                              background_image, 
                              save_filepath, 
                              dpi=300, 
                              cmap_name='gist_rainbow'):
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
    
    # 1. Find and label all independent blobs
    # labeled_array will have 0 for bg, 1 for blob 1, 2 for blob 2, etc.
    labeled_array, num_features = label(master_neuron_mask)
    
    if num_features == 0:
        print("No neurons found in the mask. No image will be saved.")
        return

    # 2. Set up the plot
    # We create a figure and axis object to have full control
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 3. Get a colormap to generate unique colors
    # Get `num_features` distinct colors from the specified colormap
    try:
        colors = plt.colormaps[cmap_name].resampled(num_features)
    except: # Fallback for older matplotlib versions
        colors = plt.cm.get_cmap(cmap_name, num_features)
    
    # 4. Plot the background image
    ax.imshow(background_image, cmap='gray')
    
    # 5. Loop through each blob, find its contour, and plot it
    logger.info(f"Plotting {num_features} unique neuron contours...")
    for i in range(1, num_features + 1): # Loop from 1 to num_features
        # Create a mask for *only* this blob
        blob_mask = (labeled_array == i)
        
        # Find contours for this single blob
        # find_contours returns a list of contour arrays
        contours = find_contours(blob_mask, 0.5)
        
        # Get the unique color for this blob (index i-1)
        blob_color = colors(i - 1)
        
        # Plot all contours found for this blob (usually just one)
        for contour in contours:
            # contour[:, 1] is x (col), contour[:, 0] is y (row)
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1.5, color=blob_color)
    
    # 6. Finalize and save
    ax.set_title(f"Unique ROI Contours ({num_features} found)")
    ax.axis('off')
    
    # Adjust layout to prevent title/labels from being cut off
    fig.tight_layout()
    
    # 7. Save the figure
    try:
        # bbox_inches='tight' crops the saved image to the plot contents
        fig.savefig(save_filepath, dpi=dpi, bbox_inches='tight', facecolor='w')
        logger.info(f"Successfully saved contour plot to: {save_filepath}")
    except Exception as e:
        logger.error(f"Error saving figure: {e}")
        
    # 8. Clean up: Close the figure to free up memory
    # This is critical if you call this function in a loop!
    plt.close(fig)


def save_contour_and_trace_plot(
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

def save_roi_video(
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

    logger.info(f"Rendering {n_frames} frames to {save_filepath} using OpenCV...")


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