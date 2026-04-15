import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def overlay_masks(image_path, mask_path, alpha=0.5):
    """
    Loads an image and an .npz mask file, overlaying all masks with a single contrasting color.
    """
    # Load the base image and normalize to [0, 1] for matplotlib
    base_img = Image.open(image_path).convert("RGB")
    img_array = np.array(base_img) / 255.0

    # Load the .npz file
    npz_file = np.load(mask_path)
    
    # Extract the mask array
    masks = npz_file["masks"] 
    
    n_masks = masks.shape[0]

    # Define a single bright color contrasting to green, blue, and purple (Red)
    # Format: RGB values between 0.0 and 1.0
    single_color = [1.0, 0.0, 0.0]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_array)

    # Iterate through masks and apply them
    for i in range(n_masks):
        current_mask = masks[i]
        
        # Create an RGBA array for the current mask overlay
        rgba_overlay = np.zeros((*current_mask.shape, 4))
        rgba_overlay[..., :3] = single_color
        
        # Set alpha channel only where the mask is active (>0)
        rgba_overlay[..., 3] = (current_mask > 0) * alpha
        
        # Overlay this specific mask on the plot
        ax.imshow(rgba_overlay)

    ax.axis("off")
    plt.show()

# Example usage:
overlay_masks(
    "/home/jaschneider/projects/DRG-CaSeg/5thB1_CytDD2-TD3-1_z2c1+2+3.png",
    "/home/jaschneider/projects/DRG-CaSeg/results/real_data_final/run_real_data_v2_20260409_105104_real_5th_B3_ica_mukamel_n_40/postprocessed_matrices.npz",
)