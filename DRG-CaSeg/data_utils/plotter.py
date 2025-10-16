import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


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