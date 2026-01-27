import numpy as np
import czifile
import tifffile
import cv2
from skimage import transform as tf
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import os
import glob
from tkinter import Tk, filedialog

# --- 1. FILE LOADING ---
def load_image(prompt):
    """Opens a file dialog to load an image/video."""
    root = Tk()
    root.withdraw()
    print(f"Please select the {prompt}...")
    file_path = filedialog.askopenfilename(title=f"Select {prompt}")
    root.destroy()
    
    if not file_path:
        raise ValueError("No file selected!")
    
    # Handle CZI vs Tiff
    if file_path.lower().endswith('.czi'):
        print(f"Loading CZI: {os.path.basename(file_path)}...")
        # czifile usually returns (Time, Z, Channel, Y, X, 1). We squeeze unnecessary dims.
        data = czifile.imread(file_path)
        return np.squeeze(data), file_path
    else:
        print(f"Loading TIFF: {os.path.basename(file_path)}...")
        return tifffile.imread(file_path), file_path

def preprocess_histology(hist_data):
    """
    Flattens a stack into a single 2D RGB image.
    Assumes standard CZI ordering.
    """
    print("Preprocessing Histology...")
    
    # Handle dimensions. Common CZI: (Channels, Z, Y, X) or (Z, Channels, Y, X)
    if hist_data.ndim == 4: 
        # Heuristic: We want to Max Project along Z.
        # If shape is (3, Z, Y, X) -> Project axis 1
        # If shape is (Z, 3, Y, X) -> Project axis 0
        if hist_data.shape[0] < 10 and hist_data.shape[1] > 10: # Likely (C, Z, Y, X)
             proj = np.max(hist_data, axis=1)
             proj = np.transpose(proj, (1, 2, 0)) # To (Y, X, C)
        else:
             proj = np.max(hist_data, axis=0)
             if proj.shape[0] == 3: # If (C, Y, X)
                 proj = np.transpose(proj, (1, 2, 0))
            
    elif hist_data.ndim == 3: # (Z, Y, X) or (C, Y, X)
         # If dim 0 is small (3), it's RGB
         if hist_data.shape[0] == 3:
             proj = np.transpose(hist_data, (1, 2, 0))
         else:
             # It's a Z-stack grayscale
             proj = np.max(hist_data, axis=0)
             proj = np.stack([proj]*3, axis=-1) # Fake RGB
    else:
        proj = hist_data

    # Normalize to 8-bit for display
    if proj.dtype != np.uint8:
        proj = cv2.normalize(proj, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    return proj

# --- 2. INTERACTIVE LANDMARKS (New Zoom/Pan) ---
def select_landmarks(image, title):
    """
    Advanced interactive window with Mouse Wheel Zoom and Pan.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"{title}\nScroll: Zoom | Middle Drag (or Shift+Click): Pan | Left Click: Add Point")
    
    # 'nearest' interpolation makes pixels sharp when zooming
    img_obj = ax.imshow(image, cmap='gray' if image.ndim==2 else None, interpolation='nearest')
    
    points = []
    
    # State storage for panning
    class ViewState:
        def __init__(self):
            self.press = None
            self.cur_xlim = None
            self.cur_ylim = None
            self.is_panning = False

    state = ViewState()

    def update_display():
        """Redraws points without reloading the image."""
        [p.remove() for p in ax.lines] 
        [p.remove() for p in ax.texts]
        
        for i, p in enumerate(points):
            # Red dot with black outline
            ax.plot(p[0], p[1], 'ro', markersize=6, markeredgecolor='black')
            # Yellow text with black outline
            txt = ax.text(p[0], p[1], str(i+1), color='yellow', fontsize=12, fontweight='bold')
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
            
        fig.canvas.draw_idle()

    # --- ZOOM (Scroll) ---
    def on_scroll(event):
        if event.inaxes != ax: return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        
        xdata, ydata = event.xdata, event.ydata
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
        ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])
        fig.canvas.draw_idle()

    # --- PAN (Middle Drag) ---
    def on_press(event):
        if event.button == 2 or (event.button == 1 and event.key == 'shift'):
            state.is_panning = True
            state.press = event.x, event.y
            state.cur_xlim = ax.get_xlim()
            state.cur_ylim = ax.get_ylim()

    def on_release(event):
        state.is_panning = False
        state.press = None
        
    def on_motion(event):
        if state.is_panning and state.press:
            # Calculate pixel delta and shift axis limits
            inv = ax.transData.inverted()
            start_xy = inv.transform(state.press)
            curr_xy = inv.transform((event.x, event.y))
            
            delta_x = start_xy[0] - curr_xy[0]
            delta_y = start_xy[1] - curr_xy[1]
            
            ax.set_xlim(state.cur_xlim[0] + delta_x, state.cur_xlim[1] + delta_x)
            ax.set_ylim(state.cur_ylim[0] + delta_y, state.cur_ylim[1] + delta_y)
            fig.canvas.draw_idle()

    # --- ADD/REMOVE POINTS ---
    def on_click(event):
        if state.is_panning or fig.canvas.toolbar.mode != '': return
        if event.button == 2 or event.key == 'shift': return 
        
        if event.button == 1: # Left Click
            if event.xdata is not None:
                points.append([event.xdata, event.ydata])
                update_display()
                print(f"Point {len(points)}: {int(event.xdata)}, {int(event.ydata)}")

        elif event.button == 3: # Right Click
            if points:
                points.pop()
                update_display()
                print(f"Removed Point {len(points)+1}")

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_press_event', on_click)

    print(f"--- {title} ---")
    print("SCROLL to Zoom.")
    print("MIDDLE-CLICK (or SHIFT+CLICK) and DRAG to Pan.")
    print("LEFT-CLICK to place a point.")
    print("Close window when done.")
    
    plt.show()
    return np.array(points)


# --- 3. MAIN LOGIC ---

# A. Load Data
vid_data, vid_path = load_image("Calcium Video")
hist_data, hist_path = load_image("Histology Image")

# B. Prep Images
if vid_data.ndim == 3: # (Time, Y, X)
    first_frame = vid_data[0]
else:
    raise ValueError(f"Video should be (Time, Y, X). Got {vid_data.shape}")

hist_flat = preprocess_histology(hist_data)

# C. Select Points
pts_vid = select_landmarks(first_frame, "Calcium Video (Source)")
print(f"Selected {len(pts_vid)} points on video.")

pts_hist = select_landmarks(hist_flat, "Histology (Target)")
print(f"Selected {len(pts_hist)} points on histology.")

if len(pts_vid) != len(pts_hist) or len(pts_vid) < 3:
    raise ValueError("Error: Point mismatch or too few points (min 3).")

# D. Calculate Transform (Smart Scaling)
print("\nCalculating Transformation...")

# 1. Estimate Raw Transform (Video -> Big Histology)
tform_raw = tf.AffineTransform()
tform_raw.estimate(pts_vid, pts_hist)

# 2. Project Video Corners to find physical bounds
h, w = first_frame.shape
corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])
corners_trans = tform_raw(corners)

min_x, min_y = np.min(corners_trans, axis=0)
max_x, max_y = np.max(corners_trans, axis=0)

real_width = max_x - min_x
real_height = max_y - min_y

print(f"Projected Physical Size: {int(real_width)} x {int(real_height)}")

# 3. Calculate Scale to maintain original width
scale_factor = w / real_width
print(f"Auto-Scaling by factor: {scale_factor:.4f}")

# 4. Construct Final Matrix (Shift -> Scale)
shift_matrix = tf.AffineTransform(translation=(-min_x, -min_y))
scale_matrix = tf.AffineTransform(scale=(scale_factor, scale_factor))

final_transform = tform_raw + shift_matrix + scale_matrix

final_w = int(w)
final_h = int(real_height * scale_factor)
print(f"Final Video Size: {final_w} x {final_h}")

# E. Process Frames (Stream to Disk)
output_folder = os.path.join(os.path.dirname(vid_path), "Registered_Frames")
os.makedirs(output_folder, exist_ok=True)

print(f"\nProcessing frames to: {output_folder}")
num_frames = vid_data.shape[0]

for i in range(num_frames):
    if i % 50 == 0: print(f"Processing frame {i}/{num_frames}...")
    
    frame = vid_data[i]
    
    # Warp
    warped = tf.warp(frame, final_transform.inverse, output_shape=(final_h, final_w), 
                     preserve_range=True, order=1)
    
    warped = warped.astype(vid_data.dtype)
    
    tifffile.imwrite(os.path.join(output_folder, f"frame_{i:05d}.tif"), warped)

# F. Reassemble
print("\nReassembling stack...")
frames = glob.glob(os.path.join(output_folder, "*.tif"))
frames.sort()
full_stack = tifffile.imread(frames)

final_path = os.path.join(os.path.dirname(vid_path), "Registered_Calcium_Video.tif")
tifffile.imwrite(final_path, full_stack)

print("------------------------------------------------")
print(f"DONE! Saved to:\n{final_path}")
print("ROI Matching Info:")
print(f"Offset X: {min_x}")
print(f"Offset Y: {min_y}")
print(f"Scale Factor: {scale_factor}")
print("------------------------------------------------")