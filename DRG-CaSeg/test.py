import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import config
import numpy as np
import matplotlib.pyplot as plt
import data_utils.wrangler as wrangler
import tifffile as tiff

from pathlib import Path

from utils import seed_everything
#from data_utils.synthesizer import DRGtissueModel
from analysis_utils.pca import cellsort_pca
from analysis_utils.ica import ica_mukamel, extract_rois_and_traces
from data_utils.plotter import  plot_spatial_patterns, plot_temporal_patterns, save_colored_contour_plot, save_contour_and_trace_plot
#from analysis_utils.caiman import CaimanPipeline
from data_utils.synthesizer import DRGtissueModel
from analysis_utils.pca_dask import cellsort_pca_dask

config.setup_logging()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    SEED = int(os.getenv("GLOBAL_SEED", 42))
    seed_everything(SEED)
    

    p = Path("/mnt/c/Users/jakschneid0621/OneDrive - AO Foundation/Documents/Data/DRG slice calcium-B1.czi")

    vid = wrangler.load_video_dask(filename=p)
    shape = vid.shape
    vid = wrangler.scale_video_dask(vid, max_intensity=64)
    logger.info("Scaling Complete")

    mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm = cellsort_pca_dask(vid, n_pcs=15)
    logger.info("Graph building complete")
    ica_sig, ica_filters, ica_A, numiter = ica_mukamel(
        mixedsig=mixedsig.compute(),
        mixedfilters=mixedfilters.compute(),
        CovEvals=cov_evals,
        mu=0.5,
        maxrounds=200,
    )

    vid_memmap_path, shape = wrangler.scale_video_to_memmap(
        czi_filename=p,
        memmap_filename=Path("/home/jaschneider/projects/DRG-CaSeg/DRG-CaSeg/data/interim/czi_vid.mmap"), 
        max_intensity=64,
    )

    """
    vid = wrangler.load_video(filename=p)
    logger.info("Video loaded")
    vid_sec = vid[10000:11500]
    del vid
    
    vid_norm = wrangler.scale_video(vid_sec, max_intensity=64)
    logger.info("Video normalized")
    
    mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm = cellsort_pca(vid_norm, n_pcs=50)

    ica_sig, ica_filters, ica_A, numiter = ica_mukamel(
        mixedsig=mixedsig,
        mixedfilters=mixedfilters,
        CovEvals=cov_evals,
        mu=0.5,
        maxrounds=200,
    )
    
    master_mask, good_comp = segment_neurons_from_ica(
        spatial_filters=ica_filters
    )

    save_colored_contour_plot(
        master_neuron_mask=master_mask,
        background_image=vid_norm.mean(axis=0),
        save_filepath="/home/jaschneider/projects/DRG-CaSeg/Plots/MasterMask.png",
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
    mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm = cellsort_pca(vid_norm, n_pcs=50)

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
        max_size=1500,
    )

    save_contour_and_trace_plot(
        roi_masks=masks,
        roi_traces=traces,
        background_image=vid_norm.mean(axis=0),
        save_filepath="/home/jaschneider/projects/DRG-CaSeg/Plots/MasterMask.png",
    )

    plot_temporal_patterns(
        ica_signals=ica_sig,
        fs=model.fps,
        n_cols=5,
    )
    plot_spatial_patterns(
        ica_filters=ica_filters,
        n_cols=5,
    )





    logger.info("Saving video")
    wrangler.save_video(
        video=(vid * 255).astype(np.uint8),
        fps=28.5,
        output_filename="test_sim.mp4"
    )
    logger.info("Exiting Script")
