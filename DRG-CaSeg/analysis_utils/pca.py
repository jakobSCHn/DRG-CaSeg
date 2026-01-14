import numpy as np
import logging

from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, inv

logger = logging.getLogger(__name__)

def cellsort_pca(
    video: np.ndarray,
    pcs: int | tuple[int, int] | list[int, int] = None,
    ):
    """
    Reads TIFF movie data and performs SVD/PCA dimensional reduction.
    
    This is a Python port of the MATLAB CellsortPCA function.

    Args:
        fn (str): Movie file name (TIFF format).
        flims (tuple, optional): 2-element tuple (start, end) specifying the range of
                                 frames to analyze (0-indexed, end-exclusive). 
                                 If None, all frames are used.
        n_pcs (int, optional): Number of principal components to return.
                               If None, defaults to min(150, n_frames).
        dsamp (tuple, optional): Downsampling factor. (temporal, spatial).
                                 If None, defaults to (1, 1).
        output_dir (str, optional): Directory to store output .mat files.
                                    If None, defaults to './cellsort_preprocessed_data/'.
        bad_frames (list, optional): List of 0-indexed frame indices to exclude.
                                     If None, no frames are excluded.

    Returns:
        tuple:
            - mixedsig (np.ndarray): N x T matrix of temporal signals.
            - mixedfilters (np.ndarray): N x X x Y array of spatial filters.
            - cov_evals (np.ndarray): Eigenvalues of the covariance matrix.
            - cov_trace (float): Trace of the covariance matrix.
            - movm (np.ndarray): Average movie frame.
            - movtm (np.ndarray): Time-averaged movie trace.
    """
    
    logger.info("Starting PCA - Mukamel et al. 2009")

    
    nt, pixh, pixw = video.shape
    npix = pixw * pixh

    # Handle default n_pcs
    if pcs is None:
        n_pcs = min(150, nt)
        pc_slice = slice(0, n_pcs)
    elif isinstance(pcs, (tuple, list)):
        n_pcs = pcs[1] #Set the pcs to be calculated to the higher number
        pc_slice = slice(pcs[0], pcs[1])
    else:
        n_pcs = pcs
        pc_slice = slice(0, pcs)


    logger.info(f"Processing {npix} pixels x {nt} time frames")
    if nt < npix:
        logger.info("Using temporal covariance matrix.")
        covmat, mov, movm, movtm = _create_covmat(video, pixw, pixh, nt, mode="temporal")
    else:
        logger.info("Using spatial covariance matrix.")
        covmat, mov, movm, movtm = _create_covmat(video, pixw, pixh, nt, mode="spatial")

    cov_trace = np.trace(covmat) / npix
    movm = movm.reshape((pixw, pixh), order="F") # Use Fortran order

    if nt < npix:
        # Perform SVD on temporal covariance
        mixedsig, cov_evals, percentvar = _cellsort_svd(covmat, n_pcs, pc_slice, nt, npix)
        # Load the other set of principal components
        mixedfilters = _reload_moviedata(pixw * pixh, mov, mixedsig, cov_evals)
    else:
        # Perform SVD on spatial components
        mixedfilters, cov_evals, percentvar = _cellsort_svd(covmat, n_pcs, pc_slice, nt, npix)
        # Load the other set of principal components
        mixedsig = _reload_moviedata(nt, mov.T, mixedfilters, cov_evals)

    # Infer the actual number of PCs returned (length of the slice) for reshaping
    n_pcs_out = pcs[1] - pcs[0] if isinstance(pcs, (list, tuple)) else n_pcs
    
    # Reshape spatial filters
    mixedfilters = mixedfilters.reshape((pixh, pixw, n_pcs_out), order="F")
    
    logger.info(f"CellsortPCA: saving data and exiting.")

    return mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm


def _create_covmat(
    mov,
    pixw,
    pixh,
    nt,
    mode="spatial"
    ):

    npix = pixw * pixh
    # Reshape to (pixels, time) using Fortran order
    mov = mov.transpose((1, 2, 0))
    mov = mov.reshape((npix, nt), order="F")

    # --- Common DFoF normalization ---
    movm = np.mean(mov, axis=1) # Average over time (F0 for each pixel)
    movm_zero = (movm == 0)
    movm[movm_zero] = 1
    
    # Use broadcasting for (mov / movm) - 1
    mov = mov / movm[:, np.newaxis] - 1
    mov[movm_zero, :] = 0

    # --- Mode-specific calculations ---
    if mode == "temporal":
        # Average over space (pixels)
        movtm = np.mean(mov, axis=0) 
        # Covariance between time points (variables=rows, so use T)
        covmat = np.cov(mov.T, bias=True)
        
    elif mode == "spatial":
        # Average over time
        movtm = np.mean(mov, axis=1) 
        # Covariance between pixels (variables=rows, use mov directly)
        covmat = np.cov(mov, bias=True)
        
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'temporal' or 'spatial'.")

    return covmat, mov, movm, movtm


def _cellsort_svd(
    covmat,
    n_pcs,
    pc_slice,
    nt,
    npix
    ):
    """Perform SVD"""
    cov_trace = np.trace(covmat) / npix

    if n_pcs < covmat.shape[0]:
        # Use sparse eigs
        cov_evals, mixedsig = eigsh(covmat, k=n_pcs)
        cov_evals = np.diag(cov_evals)
    else:
        # Use dense eig
        cov_evals, mixedsig = eigh(covmat)
        # Sort eigenvalues and eigenvectors
        idx = np.diag(cov_evals).argsort()[::-1]
        cov_evals = cov_evals[idx]
        mixedsig = mixedsig[:, idx]
        cov_evals = np.diag(cov_evals)

    # Get real parts and ensure they are positive
    cov_evals = np.real(np.diag(cov_evals))
    mixedsig = np.real(mixedsig)

    positive_evals = cov_evals > 0
    if np.sum(positive_evals) < n_pcs:
        n_pcs = np.sum(positive_evals)
        logger.info(f"Throwing out {np.sum(~positive_evals)} negative eigenvalues; new # of PCs = {n_pcs}.")
        mixedsig = mixedsig[:, positive_evals]
        cov_evals = cov_evals[positive_evals]

    #Extract the proper slice of eigenvectors to be processed
    mixedsig = mixedsig[:, pc_slice]
    cov_evals = cov_evals[pc_slice]
    #Retrieve the number of PCs actually used
    used_pcs = cov_evals.shape[0]

    mixedsig = mixedsig.T * nt
    cov_evals = cov_evals / npix
    percentvar = 100 * np.sum(cov_evals) / cov_trace
    logger.info(f"Selected {used_pcs} PCs containing {percentvar:.3f}% of the variance.")
    
    return mixedsig, cov_evals, percentvar

def _reload_moviedata(npix, mov, mixedsig, cov_evals):
    """Re-load movie data to get the other set of components"""
    n_pcs = mixedsig.shape[0]
    
    # Create S_inv
    s_inv = inv(np.diag(cov_evals**0.5))

    # Calculate spatial average
    movtm = np.mean(mov, axis=0) # Average over space
    
    # Use broadcasting
    mov_use = mov - movtm[np.newaxis, :] 
    
    # Calculate filters
    mixedfilters = (mov_use @ mixedsig.T) @ s_inv
    mixedfilters.reshape((npix, n_pcs), order="F")
    
    return mixedfilters