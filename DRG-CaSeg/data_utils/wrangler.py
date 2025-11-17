import czifile
import numpy as np
import logging
import cv2 as cv

from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform, warp
from pathlib import Path

logger = logging.getLogger(__name__)


def load_video(
    filename: str | Path,
    ):

    vid = czifile.imread(filename=filename)
    vid = np.squeeze(vid)
    
    return vid


def flatten_video(
    video: np.ndarray
    ):

    flat_video = video.reshape(video.shape[0], -1)

    return flat_video


def scale_video(
    video: np.ndarray,
    min_intensity: int = 0,
    max_intensity: int = 255
    ):

    normalized_video = video.astype(np.float32).copy()
    normalized_video = np.clip(normalized_video, a_min=min_intensity, a_max=max_intensity)

    normalized_video = (normalized_video - min_intensity) / (max_intensity - min_intensity)
    return normalized_video


def save_video(
    video: np.ndarray,
    fps: float,
    output_filename: str | Path = "video.mp4",
    ):
    
    height, width = video.shape[1:]
    frame_size = (width, height) 
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(str(output_filename), fourcc, fps, frame_size, isColor=True)
    logger.info("Saving video with OpenCV...")
    for frame in video:
        color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        color_frame[:, :, 1] = frame
        out.write(color_frame)
    logger.info(f"Video saved as {output_filename}")
    out.release()


def correct_motion(
    video: np.ndarray,
    ):
    
    num_frames, h, w = video.shape

    feature_detector_ref = ORB()
    ref_frame = video[0]
    feature_detector_ref.detect_and_extract(ref_frame)
    if feature_detector_ref.descriptors is None:
        logger.warning("Error: No descriptors found in reference frame. Aborting.")
        return video
    
    corrected_video = np.zeros_like(video)
    corrected_video[0] = ref_frame
    for i in range(1, num_frames):
        current_frame = video[i]
        feature_detector_cur = ORB()
        feature_detector_cur.detect_and_extract(current_frame)
        if feature_detector_cur.descriptors is None:
            logger.warning(f"No descriptors found in frame {i}. Appending original frame.")
            corrected_video[i] = current_frame
            continue
        
        matches = match_descriptors(
            feature_detector_ref.descriptors,
            feature_detector_cur.descriptors,
            metric="hamming",
            cross_check=True,
        )

        if matches.shape[0] < 3:
            logger.warning(f"Not enough matches in frame {i} ({matches.shape[0]}). Appending original frame.")
            corrected_video[i] = current_frame
            continue

        src = feature_detector_cur.keypoints[matches[:, 1]]
        dst = feature_detector_ref.keypoints[matches[:, 0]]
        try:
            model_robust, inliers = ransac(
                (dst, src),
                AffineTransform,
                min_samples=3,
                residual_threshold=5,
                max_trials=100
            )
            
            if model_robust is None:
                logger.warning(f"RANSAC failed to find a model for frame {i}. Appending original frame.")
                corrected_video[i] = current_frame
                continue
                
        except Exception as e:
            logger.warning(f"RANSAC error on frame {i}: {e}. Appending original frame.")
            corrected_video[i] = current_frame
            continue

    
        warped_frame = warp(
            current_frame,
            model_robust.inverse,
            output_shape=(h, w),
            mode="constant",
            cval=0,
            preserve_range=True  
        )

        corrected_video[i] = warped_frame
        
        if (i+1) % 50 == 0 or i == num_frames - 1:
             logger.info(f"Processed frame {i+1}/{num_frames}")

    return corrected_video



def scale_video_to_memmap(
    czi_filename: Path, 
    memmap_filename: Path, 
    chunk_size: int = 100,
    min_intensity: int = 0, 
    max_intensity: int = 255
    ):
    """
    Streams a CZI file, scales it chunk by chunk, and saves to a numpy memmap file.
    """
    logger.info(f"Opening CZI file: {czi_filename}")
    with czifile.CziFile(czi_filename) as czi:
        # Get shape info
        # CZI shape is complex (e.g., [1, 1, T, C, Z, Y, X, 0])
        # We'll rely on np.squeeze to get it to (T, H, W) as in your code
        temp_arr = czi.asarray(slice(0, 1)) # Load one frame to get squeezed shape
        squeezed_shape = np.squeeze(temp_arr).shape
        
        if len(squeezed_shape) != 3:
             raise ValueError(f"Expected squeezed video to be 3D (T, H, W), but got {squeezed_shape}")
        
        # Get full video shape
        # We assume the 'T' dimension is czi.shape[2] based on CZI standards
        total_frames = czi.shape[2] 
        H, W = squeezed_shape[1], squeezed_shape[2]
        final_shape = (total_frames, H, W)
        logger.info(f"Total frames: {total_frames}, H: {H}, W: {W}")

    # Create the output memmap file
    # This file is created on disk and backed by memory
    logger.info(f"Creating memmap file: {memmap_filename} with shape {final_shape}")
    scaled_vid_memmap = np.memmap(
        memmap_filename, 
        dtype=np.float32, 
        mode='w+', 
        shape=final_shape
    )
    
    # Process in chunks
    for start_frame in range(0, total_frames, chunk_size):
        end_frame = min(start_frame + chunk_size, total_frames)
        logger.info(f"Processing frames {start_frame} to {end_frame}...")
        
        # Load one chunk from CZI file
        # We need to construct the correct slice object for czi.asarray
        # e.g., (slice(None), slice(None), slice(start, end), ...)
        czi_slice = (slice(None),) * 2 + (slice(start_frame, end_frame),) + (slice(None),) * (czi.ndim - 3)

        with czifile.CziFile(czi_filename) as czi:
             chunk = czi.asarray(czi_slice)
             chunk = np.squeeze(chunk).astype(np.float32)
        
        # Apply your scaling logic from scale_video
        np.clip(chunk, a_min=min_intensity, a_max=max_intensity, out=chunk)
        chunk -= min_intensity
        chunk /= (max_intensity - min_intensity)
        
        # Write the scaled chunk to the memmap file
        scaled_vid_memmap[start_frame:end_frame, :, :] = chunk
    
    # Flush changes to disk
    scaled_vid_memmap.flush()
    logger.info("Memmap file created and scaled.")
    return scaled_vid_memmap, final_shape