import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import config
import numpy as np
import tifffile as tiff
import data_utils.wrangler as wrangler

from pathlib import Path

from utils import seed_everything
#from data_utils.synthesizer import DRGtissueModel
from analysis_utils.pca import cellsort_pca
from analysis_utils.ica import ica_mukamel, extract_rois_and_traces
from data_utils.plotter import  plot_spatial_patterns, plot_temporal_patterns,save_roi_video
#from analysis_utils.caiman import CaimanPipeline
from data_utils.synthesizer import DRGtissueModel

config.setup_logging()
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    SEED = int(os.getenv("GLOBAL_SEED", 42))
    seed_everything(SEED)

    
    p = Path("/mnt/c/Users/jakschneid0621/OneDrive - AO Foundation/Documents/Data/DRG slice calcium-B1.czi")
    vid = wrangler.load_video(filename=p)
    logger.info("Video loaded")
    vid_sec = vid[10000:10855]
    del vid
    vid_norm = wrangler.scale_video(vid_sec, max_intensity=64)
    logger.info("Video normalized")
    
    """
    model = DRGtissueModel(
        num_large_neurons=5,
        num_small_neurons=10,
        snr=2.5,
        duration_s=30,
        background_brightness=50,
        background_noise_lvl=25,
        full_well_capacity=15000,
    )
    model.generate_static_background()
    vid_norm = model.render_video()
    """
    video_uint8 = (np.clip(vid_norm, 0, 1) * 255).round().astype(np.uint8)

    tiff.imwrite("/home/jaschneider/caiman_data/example_movies/example_section_real.tif", video_uint8)
    tif_video = tiff.TiffFile("/home/jaschneider/caiman_data/example_movies/example_section_real.tif")
    shape = tif_video.shape

    mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm = cellsort_pca(vid_norm, n_pcs=20)

    ica_sig, ica_filters, ica_A, numiter = ica_mukamel(
        mixedsig=mixedsig,
        mixedfilters=mixedfilters,
        CovEvals=cov_evals,
        mu=0.5,
        maxrounds=200,
    )
    
    masks, traces = extract_rois_and_traces(
        spatial_filters=ica_filters,
        temporal_signals=ica_sig,
        min_size=15,
        max_size=10000,
    )

    plot_temporal_patterns(
        ica_signals=ica_sig,
        fs=28.5,
        n_cols=5,
    )
    plot_spatial_patterns(
        ica_filters=ica_filters,
        n_cols=5,
    )


    save_roi_video(
        roi_masks=masks, 
        roi_traces=traces, 
        video_data=vid_norm, 
        save_filepath="/home/jaschneider/projects/DRG-CaSeg/Plots/plots_to_keep/roi_video_ica_art.mp4", 
    )
    """
    save_roi_video(
        roi_masks=model.footprints,
        roi_traces=model.activities,
        video_data=vid_norm,
        save_filepath="/home/jaschneider/projects/DRG-CaSeg/Plots/plots_to_keep/roi_video_ica_gt.mp4",
    )
    """


    """
    model = DRGtissueModel(
        num_large_neurons=5,
        num_small_neurons=10,
        snr=2.5,
        duration_s=30,
        background_brightness=50,
        background_noise_lvl=25,
        full_well_capacity=15000,
    )

    fp, ac = model.build_image()
    bg = model.generate_static_background()
    plt = model.plot_ground_truth("/home/jaschneider/projects/DRG-CaSeg/Plots/GT_plot.png")
    vid = model.render_video()
    logger.info("Video generated")
    """