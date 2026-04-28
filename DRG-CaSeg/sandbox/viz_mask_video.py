import numpy as np
import imageio
import cv2

# --- Configuration ---
# Specify your file paths here
input_gif_path = "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/16658 - 2nd A1.czi #1.gif"
input_npz_path = "/home/jaschneider/projects/DRG-CaSeg/results/run_real_data_20260318_174848/run_real_data_20260318_174848_real_2nd_A1_ica_mukamel_n_120/postprocessed_matrices.npz"
output_gif_path = "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/mask_overlay.gif"
output_no_mask_gif_path = "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/no_mask_overlay.gif"

# Specify the index of the mask you want to outline
mask_index = 5

def main():
    print("Loading data...")
    # Load the spatial masks from the .npz file
    npz_data = np.load(input_npz_path)
    masks_array = npz_data["masks"]
    
    # Extract the specific mask. 
    # This assumes the shape is (N, 292, 384). 
    # If your array shape is (292, 384, N), change the line below to:
    # selected_mask = masks_array[:, :, mask_index]
    selected_mask = masks_array[mask_index]

    # Convert the mask to a binary uint8 format, which OpenCV requires for contour finding
    # We assume the mask is boolean or contains 0s and 1s
    binary_mask = (selected_mask > 0).astype(np.uint8) * 255

    # Find the contours (outlines) of the mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Shift the contours 10 pixels right (x) and 10 pixels down (y) for a diagonal bottom-right shift
    # This is done to correct the visualization for the motion correction algorithm
    shifted_contours = [contour + np.array([10, 3]) for contour in contours]

    # Load the GIF
    print("Processing GIF frames...")
    gif_reader = imageio.get_reader(input_gif_path)
    
    # Try to grab the original FPS to keep the timing the same; default to 10 if missing
    fps = gif_reader.get_meta_data().get("fps", 10) 
    
    processed_frames = []
    clean_frames = []

    for frame in gif_reader:
        # Convert the frame to standard 3-channel RGB so we can draw a red line
        if frame.ndim == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        else:
            # If it is already RGB, just make a copy so we don't modify the original in memory
            frame_rgb = frame.copy()
            
        # Store a copy of the clean frame before drawing the mask on it
        clean_frames.append(frame_rgb.copy())
        
        # Draw the outline in bright red using the shifted contours. 
        # Color format is (R, G, B) for imageio standard, so we use (255, 0, 0).
        # The final parameter "1" is the thickness of the line in pixels.
        cv2.drawContours(frame_rgb, shifted_contours, -1, (255, 0, 0), 1)
        
        processed_frames.append(frame_rgb)

    # Save the compiled frames to the new destination
    # loop=0 ensures the GIF loops indefinitely
    print("Saving output GIF with mask...")
    imageio.mimsave(output_gif_path, processed_frames[::5], fps=fps, loop=0)
    print("Done! Saved successfully to:", output_gif_path)

    print("Saving output GIF without mask...")
    imageio.mimsave(output_no_mask_gif_path, clean_frames[::5], fps=fps, loop=0)
    print("Done! Saved successfully to:", output_no_mask_gif_path)

if __name__ == "__main__":
    main()