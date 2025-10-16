import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import config
import numpy as np
import matplotlib.pyplot as plt
import data_utils.wrangler as wrangler

from pathlib import Path

from utils import seed_everything
#from data_utils.synthesizer import DRGtissueModel
from analysis_utils.pca import cellsort_pca
from analysis_utils.ica import ica_mukamel
from data_utils.plotter import  plot_spatial_patterns, plot_temporal_patterns
from analysis_utils.caiman import CaimanPipeline

config.setup_logging()
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    
    SEED = int(os.getenv("GLOBAL_SEED", 42))
    seed_everything(SEED)
    
    """
    p = Path("/mnt/c/Users/jakschneid0621/OneDrive - AO Foundation/Documents/Data/DRG slice calcium-B1.czi")
    vid = wrangler.load_video(filename=p)
    logger.info("Video loaded")
    vid_sec = vid[10000:12000]
    del vid
    vid_norm = wrangler.scale_video(vid_sec, max_intensity=64)
    logger.info("Video normalized")
    """

    pipeline = CaimanPipeline()

    p = pipeline._cast_motion_memmap(
        filepath="/home/jaschneider/caiman_data/temp/tmp_mov_mot_corr.hdf5"
    )
    pipeline.set_video_from_memmap(
        memmap=p
    )
    pipeline.start_cluster()
    pipeline.setup_cnmfe()
    logger.info("CNMF-E model set.")
    pipeline.fit_model()

    md = pipeline.cnmfe_model
    md.estimates.nb_view_components(
        img=pipeline.correlation_image,
        idx=md.estimates.idx_components
    )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    md.estimates.plot_contours(
        img=pipeline.correlation_image,
    )
    plt.savefig("/home/jaschneider/projects/DRG-CaSeg/Caiman_test.png")

    mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm = cellsort_pca(vid_norm, n_pcs=15)

    ica_sig, ica_filters, ica_A, numiter = ica_mukamel(
        mixedsig=mixedsig,
        mixedfilters=mixedfilters,
        CovEvals=cov_evals,
        mu=0.5,
        maxrounds=200,
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
        duration_s=10,
        background_brightness=50,
        background_noise_lvl=25,
        full_well_capacity=15000,
    )

    fp, ac = model.build_image()
    bg = model.generate_static_background()
    plt = model.plot_ground_truth("/home/jschneider/projects/msc_thesis/plots/GT_plot.png")
    vid = model.render_video()
    logger.info("Video generated")

    mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm = cellsort_pca(vid, n_pcs=15)

    ica_sig, ica_filters, ica_A, numiter = ica_mukamel(
        mixedsig=mixedsig,
        mixedfilters=mixedfilters,
        CovEvals=cov_evals,
        mu=0.5,
        maxrounds=200,
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
