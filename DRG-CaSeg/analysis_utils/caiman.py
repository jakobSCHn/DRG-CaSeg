import logging
import attrs
import numpy as np
import psutil

import caiman as cm
import caiman.source_extraction.cnmf as secnmf

from pathlib import Path
from attrs.validators import instance_of
from multiprocessing.pool import Pool
from ipyparallel.client.view import DirectView

from caiman.motion_correction import MotionCorrect
from data_utils.plotter import plot_image

logger = logging.getLogger(__name__)



@attrs.define
class CaimanPipeline:

    params: dict[str, any] = attrs.field(factory=dict, validator=instance_of(dict))

    caiman_params: secnmf.params.CNMFParams = attrs.field(init=False)

    video: cm.movie | None = attrs.field(init=False, default=None)
    cluster: Pool | DirectView | None = attrs.field(init=False, default=None)
    n_processes: int | None = attrs.field(init=False, default=None)
    corrector: MotionCorrect | None = attrs.field(init=False, default=None)
    cnmfe_model: secnmf.CNMF | None = attrs.field(init=False, default=None)
    estimates: secnmf.cnmf.Estimates | None = attrs.field(init=False, default=None)
    correlation_image: np.ndarray | None = attrs.field(init=False, default=None)
    dims: tuple[int, int] | None = attrs.field(init=False, default=None)



    def __attrs_post_init__(
        self,
        ):

        default_params = self._get_default_params()

        params_dict = default_params | self.params
        self.caiman_params = secnmf.params.CNMFParams(params_dict=params_dict)


    def _get_default_params(
        self,
        ):

        fps = 28.5                      # imaging rate in frames per second
        decay_time = 0.4                # length of a typical transient in seconds
        dims = (384, 292)
        
        # motion correction parameters
        strides = (48, 48)
        overlaps = (24, 24)
        max_shifts = (5, 5)
        max_deviation_rigid = 3
        pw_rigid = False
        gSig_filt = (3, 3)
        border_nan = "copy"
        
        # CNMF parameters
        p = 1
        gnb = 0
        merge_thr = 0.7
        rf = 40
        stride_cnmf = 20
        k = None
        gSig = np.array([3, 3])
        gSiz = 2*gSig + 1
        method_init = "corr_pnr"
        only_init = True
        ssub = 1
        tsub = 2
        low_rank_background = None
        update_background_components = True
        normalize_init = False
        center_psf = True
        del_duplicates = True
        nb_patch = 0
        min_corr = 0.8
        min_pnr = 10
        ssub_B = 2
        ring_size_factor = 1.4
        method_deconvolution = "oasis"
        
        
        parameter_dict = {
            # General
            "fr": fps, "decay_time": decay_time, "dims": dims,
            
            # Motion Correction
            "strides": strides, "overlaps": overlaps, "max_shifts": max_shifts,
            "max_deviation_rigid": max_deviation_rigid, "pw_rigid": pw_rigid,
            "gSig_filt": gSig_filt, "border_nan": border_nan,

            # CNMF
            "p": p, "nb": gnb, "rf": rf, "K": k, "gSig": gSig, "gSiz": gSiz,
            "stride": stride_cnmf, "method_init": method_init, "ssub": ssub,
            "tsub": tsub, "merge_thr": merge_thr,
            "only_init": only_init, "low_rank_background": low_rank_background,
            "update_background_components": update_background_components,
            "normalize_init": normalize_init, "center_psf": center_psf,
            "del_duplicates": del_duplicates, "nb_patch": nb_patch,
            "min_corr": min_corr, "min_pnr": min_pnr, "ssub_B": ssub_B,
            "ring_size_factor": ring_size_factor,
            "method_deconvolution": method_deconvolution,
        }
        return parameter_dict


    def __getattr__(
        self,
        name: str
        ):
        try:
            return getattr(self.video, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute {name}"
            )


    def start_cluster(
        self,
        ignore_preexisting: bool = True,
        single_thread: bool = True,
        ):

        logger.info(f"{psutil.cpu_count()} CPUs available in current environment")

        if "cluster" in locals():
            logger.info("Closing previous cluster")
            cm.stop_server(dview=locals()["cluster"])

        logger.info("Setting up new cluster")
        _, cluster, n_processes = cm.cluster.setup_cluster(
            backend="sequential",
            n_processes=None,
            ignore_preexisting=ignore_preexisting,
            single_thread=single_thread,
        )
        logger.info(f"Successfully initilialized multicore processing with a pool of {n_processes} CPU cores")
        self.cluster = cluster
        self. n_processes = n_processes


    def stop_cluster(
        self,
        ):
        if self.cluster:
            cm.stop_server(dview=self.cluster)
            self.cluster = None
        else:
            logger.warning("No cluster available. Skipping cluster shutdown")


    def set_video_from_array(
        self,
        video: np.ndarray,
        ):
        self.video = cm.movie(video)


    def set_video_from_path(
        self,
        path: str | Path,
        ):
        self.video = cm.load(str(path))


    def set_video_from_memmap(
        self,
        memmap: str | Path,
        ):
        self.video = cm.load_memmap(str(memmap))


    def setup_cnmfe(
        self,
        ):
        self.cnmfe_model = secnmf.CNMF(
            n_processes=self.n_processes,
            dview=self.cluster,
            params=self.caiman_params,
        )


    def correct_motion(
        self,
        ):
        if not self.cluster:
            logger.warning("Cluster is initialized to reduce run-time")
            self.start_cluster(
                ignore_preexisting=False,
            )
        
        if self.corrector:
            self.corrector.motion_correct(
                save_movie=True
            )
            new_filepath = self._cast_motion_memmap(
                self.corrector.mmap_file
            )
            self.video = self.set_video_from_memmap(
                new_filepath
            )
            self.corrector.mmap_file = new_filepath
        elif not self.video:
            raise AttributeError(
                "Cannot perform motion correction: No video available"
            )
        else:
            self.corrector = MotionCorrect(
                self.video,
                dview=self.cluster,
                **self.parameters.motion
            )
            self.corrector.motion_correct(
                save_movie=True
            )
            new_filepath = self._cast_motion_memmap(
                self.corrector.fname
            )
            self.video = self.set_video_from_memmap(
                new_filepath
            )
            self.corrector.mmap_file = new_filepath

    def _cast_motion_memmap(
        self,
        filepath,
        ):
        new_filepath = cm.save_memmap(
            [filepath],
            base_name="tmp_motion_corrected",
            order="C",
        )
        return new_filepath


    def fit_model(
        self,
        ):
        
        if not self.cnmfe_model:
            raise AttributeError(
                "No CNMF model has been initialized."
            )
        
        images, _ = self.extract_images_from_memmap(self.video)
        self.cnmfe_model.fit(images)

    def compute_correlation_image(
        self,
        ):
        images, T = self.extract_images_from_memmap(self.video)
        gsig_tmp = (3, 3)
        self.correlation_image, _ = cm.summary_images.correlation_pnr(
            images[::max(T//1000, 1)],
            gSig=gsig_tmp[0],
            swap_dim=False,
        )

    def extract_images_from_memmap(
        self,
        video: tuple[np.memmap, tuple[int, int], int],
        ):
        if isinstance(video, tuple):
            yr, dims, num_frames = video
            images = np.reshape(yr.T, [num_frames] + list(dims), order="F")
            return images, num_frames
        else:
            raise AttributeError(
                "Video has not been converted to a memory mapped file."
            )


    def plot_correlation_image(
        self,
        save_loc: str | Path,
        swap_dim: bool = False,
        title: str = "Correlation Image",
        cmap: str = "viridis",
        cbar_label: str = "Correlation Strength",
        motion_corrected: bool = False, 
        ):

        correlation_image_og = cm.local_correlations(
            self.mc_video if motion_corrected else self.video,
            swap_dim=swap_dim
        )
        correlation_image_og[np.isnan(correlation_image_og)] = 0
        _, _ = plot_image(
            correlation_image_og,
            save_loc=save_loc,
            title=title,
            cmap=cmap,
            cbar_label=cbar_label,
        )



