import sys
from pathlib import Path
import numpy as np
import cv2

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from data_utils.synthesizer import DRGtissueModel
from utils import seed_everything

SEED = 42
seed_everything(SEED)


def save_blue_timelapse(
    video_array: np.ndarray, 
    save_filepath: str | Path, 
    original_duration_sec: float = 30.0, 
    target_duration_sec: float = 4.0,
    output_fps: int = 30
):
    """
    Saves a 3D numpy array (Frames, Height, Width) as a time-lapsed .mp4 video 
    using a Black (0) to Blue (255) colormap.
    """
    # 1. Apply the Black-to-Blue colormap
    # OpenCV uses BGR (Blue, Green, Red) channel ordering.
    num_frames, height, width = video_array.shape
    
    # Initialize an empty array with 3 color channels
    bgr_video = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    
    # Map the original grayscale values strictly to the Blue channel (index 0)
    bgr_video[..., 0] = video_array 
    
    # 2. Calculate Time-lapse parameters
    # Determine how many total frames we need for a 4-second video at 30 fps
    target_frame_count = int(target_duration_sec * output_fps)
    
    # Calculate the step size needed to skip frames 
    skip_factor = max(1, num_frames // target_frame_count)
    
    # Subsample the video array
    timelapse_video = bgr_video[::skip_factor]
    
    # 3. Save as .mp4
    save_path_str = str(save_filepath)
    
    # "mp4v" is the standard codec for .mp4 in OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    
    out = cv2.VideoWriter(
        save_path_str, 
        fourcc, 
        output_fps, 
        (width, height)
    )
    
    for frame in timelapse_video:
        out.write(frame)
        
    out.release()
    print(f"Time-lapse saved successfully to: {save_path_str}")


"""
model = DRGtissueModel(
    duration_s=30,
    num_small_neurons=4,
    num_large_neurons=4,
    full_well_capacity=15000,
)

model.render_video()

model.plot_ground_truth(
    save_loc="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/base_plot.png",
)

model.perturb_positions(
    target_indices=[7],
    angle_deg=[315],
    shift_px=[100],
)
model.render_video()
model.plot_ground_truth(
    save_loc="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/shifted_neuron.png",
)
"""

"""
model = DRGtissueModel(
    duration_s=30,
    num_small_neurons=15,
    num_large_neurons=15,
    full_well_capacity=15000,
)
model.render_video()

model.plot_ground_truth(
    save_loc="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/many_neurons.png",
)
"""

model = DRGtissueModel(
    duration_s=8,
    num_small_neurons=10,
    num_large_neurons=30,
    background_brightness=80,
    full_well_capacity=15000,
    snr=3.0,
)

vid = model.render_video(
    store_traces=True,
)
vid_8bit = (vid * 255).astype(np.uint8)

save_blue_timelapse(
        video_array=vid_8bit, 
        save_filepath="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/sim_time_lapse.mp4",
        original_duration_sec=8.0,
        target_duration_sec=4.0
    )


model.plot_statistics(
    save_loc="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/skew_viz_single.png",
    n_footprints=1,
)