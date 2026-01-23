import logging
import os
import gc
import numpy as np
import caiman as cm

from caiman.source_extraction.cnmf import params
from caiman.source_extraction import cnmf

logger = logging.getLogger(__name__)


def extract_cnmfe_results(
    model: cnmf.CNMF,
    dims: tuple,          
    ):
    #TODO: extract dims in run and pass it to this function
    traces = model.estimates.C
    A_sparse = model.estimates.A

    A_dense = A_sparse.toarray()
    
    masks = A_dense.reshape(dims + (-1,), order="F").transpose(2, 0, 1)
    bin_masks = masks > 0

    return bin_masks, traces


def get_default_params(
    ):

    #motion correction parameters
    pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
    gSig_filt = (6, 6)       # sigma for high pass spatial filter applied before motion correction, used in 1p data
    max_shifts = (5, 5)      # maximum allowed rigid shift
    strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)      # overlap between patches (size of patch = strides + overlaps)
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = "copy"      # replicate values along the boundaries

    # parameters for source extraction and deconvolution
    decay_time = 10     #length of a typical transient in seconds
    p = 1               #order of the autoregressive system
    K = None            #upper bound on number of components per patch, in general None for CNMFE
    gSig = np.array([6, 6])  #expected half-width of neurons in pixels 
    gSiz = 2*gSig + 1     #half-width of bounding box created around neurons during initialization
    merge_thr = .9      #merging threshold, max correlation allowed
    rf = 40             #half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
    stride_cnmf = 15    #amount of overlap between the patches in pixels 
    tsub = 2            #downsampling factor in time for initialization, increase if you have memory problems
    ssub = 1            #downsampling factor in space for initialization, increase if you have memory problems
    gnb = 0             #number of background components (rank) if positive, set to 0 for CNMFE
    low_rank_background = None  #None leaves background of each patch intact (use True if gnb>0)
    nb_patch = 0        #number of background components (rank) per patch (0 for CNMFE)
    min_corr = .8       #min peak value from correlation image
    min_pnr = 10        #min peak to noise ration from PNR image
    ssub_B = 2          #additional downsampling factor in space for background (increase to 2 if slow)
    ring_size_factor = 1.4  #radius of ring is gSiz*ring_size_factor
    bord_px = 0         #should be zero if the motion correction border_nan is set to copy (default
    use_cnn = False     #1p caiman should not use the CNN classifier to sort neuron shape since it was trained on 2p data

    parameters = params.CNMFParams(params_dict={
    "data": {
        'decay_time': decay_time,
    },
    "init": {
        'K': K,
        'gSig': gSig,
        'gSiz': gSiz,
        'method_init': "corr_pnr",  # use this for 1 photon
        'min_corr': min_corr,
        'min_pnr': min_pnr,
        'nb': gnb,  # number of global background components
        'normalize_init': False,  # just leave as is
        'center_psf': True,  # True for 1p
        'ring_size_factor': ring_size_factor,
        'ssub': ssub,
        'ssub_B': ssub_B,
        'tsub': tsub,
    },
    "motion":{
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan,
    },
    "patch": {
        'rf': rf,
        'stride': stride_cnmf,
        'only_init': True,  # set it to True to run CNMF-E
        'nb_patch': nb_patch,
        'low_rank_background': low_rank_background,
        'del_duplicates': True,  # whether to remove duplicates from initialization
        'border_pix': bord_px,  # number of pixels to not consider in the borders
    },
    "spatial": {
        'update_background_components': True,  # sometimes setting to False improve the results
    },
    "temporal": {
        'p': p,
        'method_deconvolution': "oasis",  # could use "cvxpy" alternatively
    },
    "merging": {
        'merge_thr': merge_thr,
    },
    "quality": {
        'use_cnn': use_cnn,
    },
    "preprocess": {
        'p': p, # order of AR indicator dynamics
    }
    })

    return parameters


def run_cnmfe(
    mov: cm.movie,
    temp_path = None,
    cluster = None,
    n_processes: int = 1,
    **params,   
    ):

    if temp_path is None:
        temp_folder = os.path.join(os.getcwd(), "temp")
    else:
        temp_folder = temp_path
    os.makedirs(temp_folder, exist_ok=True)
    base_path = os.path.join(temp_folder, "_memmap.mmap")

    fname_new = None
    parameters = get_default_params()
    if params is not None:
        parameters.change_params(params_dict=params)
    try:
        fname_new = mov.save(base_path, order="C")
            
        Yr, dims, T = cm.load_memmap(fname_new)
        images = Yr.T.reshape((T,) + dims, order="F")

        cnmfe_model = cnmf.CNMF(
            n_processes=n_processes,
            dview=cluster,
            params=parameters
        )
        
        cnmfe_model.fit(images)

        masks, traces = extract_cnmfe_results(
            model=cnmfe_model,
            dims=dims,
            )

        return {
            "masks": masks,
            "traces": traces,
        }


    finally:
        #delete variables to unlock the memmap file
        if "Yr" in locals(): del Yr
        if "images" in locals(): del images
        gc.collect() #ensure variables are deleted right now before any further code is executed
        
        if fname_new and os.path.exists(fname_new):
            os.remove(fname_new)
            logger.info("Temp file deleted successfully")

