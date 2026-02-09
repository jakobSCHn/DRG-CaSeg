import czifile
import os
import numpy as np
import logging
import cv2 as cv
import random
import xmltodict
import caiman as cm
import uuid

from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform, warp
from pathlib import Path

from data_utils.synthesizer import DRGtissueModel
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params

from analysis_utils.caiman import get_default_params

logger = logging.getLogger(__name__)


def load_czi(
    id: str,
    filename: str | Path,
    ):

    vid = czifile.imread(filename=filename)
    vid = np.squeeze(vid)
    
    return {
        "id": id,
        "data": vid
    }


def flatten_video(
    video: np.ndarray
    ):

    flat_video = video.reshape(video.shape[0], -1)

    return flat_video


def scale_video(
    mov: cm.movie,
    min_intensity: int = 0,
    max_intensity: int = 255
    ):
    
    #clip the pixel brightness to the allowed values
    np.clip(mov, a_min=min_intensity, a_max=max_intensity, out=mov)

    #perform actual normalization (modify in place to save RAM)
    mov -= min_intensity
    mov /= (max_intensity - min_intensity)

    #throw a type error if it's not a cm.movie, as metadata will be lost
    if not isinstance(mov, cm.movie):
        raise TypeError(f"cm.movie objects has been cast to {type(mov)} during preprocessing.")

    return mov

def norm_video(
    mov: cm.movie,
    ):
    time_means = np.mean(mov, axis=0)
    mov = (mov / (time_means + 1e-6)) - 1

    frame_means = np.mean(mov, axis=(1,2), keepdims=True)
    mov = mov - frame_means

    if not isinstance(mov, cm.movie):
        raise TypeError(f"cm.movie objects has been cast to {type(mov)} during preprocessing.")

    return mov


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


def correct_motion_legacy(
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


def load_drg_model_video(
    **params
    ):
    """
    Wrapper to bridge the YAML configuration with the DRGtissueModel class.
    Handles seeding, initialization, and optional position perturbation.
    """
    
    seed = params.get("seed", None)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"Initialized synthetic generator with Seed: {seed}")

    #Separate Model Params from Wrapper Params and remove what's not meant for the constructor
    wrapper_specific_keys = ["id", "seed", "perturbation", "cluster", "n_processes"]
    model_params = {k: v for k, v in params.items() if k not in wrapper_specific_keys}
    
    #Initialize the model
    model = DRGtissueModel(**model_params)

    #Generate the default neurons using the seed provided
    model.build_image()

    #Apply Modifications to neuronal positions
    if "perturbation" in params:
        perturb_conf = params["perturbation"]
        
        #Extract arguments with safe defaults
        target_indices = perturb_conf.get("target_indices", [])
        angle_deg = perturb_conf.get("angle_deg", None)
        shift_px = perturb_conf.get("shift_px", None)

        if target_indices:
            logger.info(f"Applying perturbation to indices: {target_indices}")
            model.perturb_positions(
                target_indices=target_indices,
                angle_deg=angle_deg,
                shift_px=shift_px
            )
        else:
            logger.warning("Perturbation instructions found, but 'target_indices' was empty.")

    #Render the final video
    logger.info("Rendering video frames...")
    
    md_dict = {
        "seed": seed,
        "fps": model.fps,
        "scale": model.um_per_pixel,
        "width": model.width_px,
        "height": model.height_px,
    }
    mov = cm.movie(
        model.render_video(),
        fr=model.fps,
        start_time=0,
        file_name=params["id"],
        meta_data=md_dict
    )

    #Return standardized dataset structure
    dataset_id = params.get("id", f"synthetic_{seed}")
    
    return {
        "id": dataset_id,
        "data": mov,
        "gt": {
            "spatial": model.footprints,
            "temporal": model.activities,
            "labels": model.labels,
        },
        "meta": {
            "cell_metadata": model.cell_metadata
        }
    }


def load_czi_to_caiman(
    id,
    filename,
    fr=None,
    start_time=0,
    meta_data=None,
    ):
    """
    Loads a .czi file, massages the dimensions, and returns a fully initialized
    caiman.movie object.
    
    Args:
        file_name (str): Path to the .czi file.
        fr (float, optional): Frame rate. If None, attempts to read from CZI metadata.
        start_time (float): Start time in seconds.
        meta_data (dict): Optional manual metadata.

    Returns:
        caiman.movie: The initialized movie object.
    """
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found!")

    with czifile.CziFile(filename) as czi:
        raw_image_data = czi.asarray()
        
        if meta_data is None:
            md_xml = czi.metadata()
            md = xmltodict.parse(md_xml, attr_prefix="")


        if fr is None and meta_data is None:
            #Extract the offsets the frames were taken at to calculate the frame rate
            try:
                img_timepoints = md["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["T"]["Positions"]["List"]["Offsets"]
                num_img_timepoints = np.fromstring(img_timepoints, sep=" ") #Cast the timestamps to be numerical instead of being formatted as a string
                median_diff = np.median(np.diff(num_img_timepoints))
                fr = 1 / median_diff
            except:
                increment = md["ImageDocument"]["Metadata"]["Information"]["Image"]["Dimensions"]["T"]["Positions"]["Interval"]["Increment"]
                fr = 1 / float(increment)
        else:
            fr = meta_data["fr"]

    
    image_data = raw_image_data.squeeze()
    # Handling Multichannel or Multi-Z stacks
    # If the data is still > 3 dimensions after squeezing (e.g. Time, Z, Y, X),
    # you might need to select a specific plane or channel.
    # This logic assumes the standard CaImAn input of (Time, Y, X) 
    if image_data.ndim > 3:
        raise ValueError(
            f"Input data has {image_data.ndim} dimensions with shape {image_data.shape}. "
            "CaImAn movie objects expect 3 dimensions (Time, Height, Width). "
        )


    movie_obj = cm.movie(
        image_data.astype(np.float32), #Convert to float32 as CaImAn prefers floats for processing
        fr=fr,
        start_time=start_time,
        file_name=os.path.basename(filename),
        meta_data=meta_data
    )

    return {
        "id": id,
        "data": movie_obj,
    }


def get_default_params_motion():
    #motion correction parameters
    pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = (6, 6)       # sigma for high pass spatial filter applied before motion correction, used in 1p data
    max_shifts = (5, 5)      # maximum allowed rigid shift
    strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)      # overlap between patches (size of patch = strides + overlaps)
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = "copy"      # replicate values along the boundaries

    mc_dict = {
        "pw_rigid": pw_rigid,
        "max_shifts": max_shifts,
        "gSig_filt": gSig_filt,
        "strides": strides,
        "overlaps": overlaps,
        "max_deviation_rigid": max_deviation_rigid,
        "border_nan": border_nan
    }

    parameters = params.CNMFParams(params_dict=mc_dict)
    return parameters.get_group("motion")


def correct_motion(
    mov: cm.movie,
    cluster = None,
    **params,
    ):
    unique_id = uuid.uuid4().hex
    temp_filename = f"temp_mc_{unique_id}.mmap"
    
    fname_path = mov.save(temp_filename, order="C")

    #remove the "id" parameter as it's not used in caiman but should be present at function call
    params.pop("id")
    parameters = get_default_params()
    #get the framerate directly form the video to always be accurate
    try:
        fr_params = {"fr": mov.fr}
    except AttributeError:
        fr_params = {"fr": 30}
        logger.warning(f"Movie frame rate hasn't been specified. Using default of {fr_params['fr']} Hz")

    parameters.change_params(fr_params)
    if params is not None:
        parameters.change_params(params_dict=params)

    try:
        corrector = MotionCorrect(fname=fname_path, dview=cluster, **parameters.get_group("motion"))
        corrector.motion_correct(save_movie=False)
        corrected_mov = corrector.apply_shifts_movie(fname_path)
        
        return corrected_mov
    
    finally:
        #Since we created the file, we are responsible for deleting it
        if os.path.exists(fname_path):
            try:
                os.remove(fname_path)
            except PermissionError:
                logger.warning(f"Could not delete temp file {fname_path} (still in use).")
