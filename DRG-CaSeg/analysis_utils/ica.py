import numpy as np
import logging

import skimage.measure as ms
import cv2

from scipy.stats import skew, kurtosis, median_abs_deviation
from scipy.linalg import inv, sqrtm

from scipy.ndimage import label, binary_opening, binary_closing
from skimage.measure import regionprops
from skimage.morphology import disk
from scipy.ndimage import sum as ndi_sum

from data_utils.plotter import plot_image

logger = logging.getLogger(__name__)


def ica_mukamel(
    mixedsig,
    mixedfilters,
    CovEvals,
    PCuse=None,
    mu=None, 
    nIC=None,
    w_init=None,
    termtol: float = 1e-6,
    maxrounds: int = 100
    ):
    """
    Performs ICA with a standard set of parameters, including skewness as the
    objective function.

    This is a Python translation of the MATLAB function CellsortICA by
    Eran Mukamel, Axel Nimmerjahn, and Mark Schnitzer, 2009.
    
    Based on the fastICA package (http://www.cis.hut.fi/projects/ica/fastica).

    Inputs:
    - mixedsig: (N, T) numpy array of N temporal signal mixtures sampled at T points.
    - mixedfilters: (X, Y, N) numpy array of N spatial signal mixtures sampled at
                    X x Y spatial points. Note the (X, Y, N) dimension order.
    - CovEvals: 1D array of eigenvalues of the covariance matrix.
    - PCuse: 1D array or list of 0-based indices of the components to be included. 
             If None, use all components.
    - mu: Parameter (between 0 and 1) specifying weight of temporal
          information in spatio-temporal ICA.
    - nIC: Number of ICs to derive. If None, defaults to len(PCuse).
    - w_init: Initial guess for the unmixing matrix A. 
                   Shape (len(PCuse), nIC). If None, random guess is used.
    - termtol: Termination tolerance for the fixed-point algorithm.
    - maxrounds: Maximum number of rounds of iterations.

    Outputs:
    - ica_sig: (nIC, T) numpy array of ICA temporal signals.
    - ica_filters: (nIC, X, Y) numpy array of ICA spatial filters.
    - ica_A: (len(PCuse), nIC) orthogonal unmixing matrix.
    - numiter: Number of rounds of iteration before termination.
    """

    
    logger.info(f"Starting ICA - Mukamel et al. 2009")

    # 1. Set default values
    if PCuse is None:
        PCuse = np.arange(mixedsig.shape[0])
    
    if nIC is None:
        nIC = len(PCuse)
    
    if w_init is None:
        w_init = np.random.randn(len(PCuse), nIC)
    

    # 2. Input validation
    if mu is None or (mu > 1) or (mu < 0):
        raise ValueError("Spatio-temporal parameter, mu, must be between 0 and 1.")
    
    if w_init.shape[0] != len(PCuse) or w_init.shape[1] != nIC:
        raise ValueError("Initial guess for ica_A is the wrong size. "
                         f"Expected ({len(PCuse)}, {nIC}), got {w_init.shape}")
    
    if nIC > len(PCuse):
        raise ValueError("Cannot estimate more ICs than the number of PCs.")

    # 3. Data preparation
    pixw, pixh = mixedfilters.shape[0], mixedfilters.shape[1]
    npix = pixw * pixh
    
    # Select PCs
    if mu > 0:
        mixedsig = mixedsig[PCuse, :]
        
    if mu < 1:
        # Reshape mixedfilters, using Fortran ('F') order to match MATLAB's
        # column-major reshape.
        # Original shape (pixw, pixh, N) -> select PCuse on 3rd dim
        temp_filters = mixedfilters[:, :, PCuse] # Shape (pixw, pixh, len(PCuse))
        # Reshape to (npix, len(PCuse))
        mixedfilters = temp_filters.reshape((npix, len(PCuse)), order="F")
        
    CovEvals = CovEvals[PCuse]
    
    # Center the data by removing the mean of each PC
    mixedmean = np.mean(mixedsig, axis=1, keepdims=True)
    mixedsig = mixedsig - mixedmean  # Broadcasting handles the subtraction

    # 4. Create concatenated data for spatio-temporal ICA
    nx = mixedfilters.shape[0]
    nt = mixedsig.shape[1]
    
    if mu == 1:
        # Pure temporal ICA
        sig_use = mixedsig
    elif mu == 0:
        # Pure spatial ICA
        sig_use = mixedfilters.T
    else:
        # WHITENING/NORMALIZATION IS MANDATORY HERE
        # Center and scale to unit variance so mu actually works
        f_part = mixedfilters.T - np.mean(mixedfilters.T, axis=1, keepdims=True)
        f_part /= (np.std(f_part) + 1e-10)
        
        s_part = mixedsig - np.mean(mixedsig, axis=1, keepdims=True)
        s_part /= (np.std(s_part) + 1e-10)
        
        # Now mu acts on standardized data
        sig_use = np.hstack([(1 - mu) * f_part, mu * s_part])
        
        # Final re-standardization of the joint matrix
        sig_use /= np.sqrt((1 - mu)**2 + mu**2)
    
    # Final whitening: Ensure sig_use has identity covariance
    # This is crucial for symmetric orthogonalization to work
    sig_use = sig_use - np.mean(sig_use, axis=1, keepdims=True)
    sig_use = sig_use / np.std(sig_use)

    # 5. Perform ICA
    ica_A, numiter = _fpica_standardica(sig_use, nIC, w_init, termtol, maxrounds)

    # 6. Post-processing and sorting
    ica_sig = ica_A.T @ mixedsig
    
    # Calculate ica_filters
    # (mixedfilters @ diag(CovEvals**(-1/2)) @ ica_A).T
    diag_inv_sqrt_cov = np.diag(CovEvals**(-0.5))
    ica_filters = (mixedfilters @ diag_inv_sqrt_cov @ ica_A).T
    ica_filters.reshape((nIC, nx), order="F")
    
    # Normalization
    ica_filters = ica_filters / (npix**2)
    
    # Sort ICs according to skewness of the temporal component
    # We want the skewness for each row (IC)
    icskew = skew(ica_sig, axis=1)
    
    # Sort in descending order
    ICord = np.argsort(icskew)[::-1]
    
    # Re-order outputs
    ica_A = ica_A[:, ICord]
    ica_sig = ica_sig[ICord, :]
    ica_filters = ica_filters[ICord, :]
    
    # Final reshape of filters
    # Reshape (nIC, npix) -> (nIC, pixw, pixh)
    # Use 'F' order to match MATLAB's column-major reshape
    ica_filters = ica_filters.reshape((nIC, pixw, pixh), order="F")

    return ica_sig, ica_filters, ica_A, numiter


def _fpica_standardica(x, nIC, w_init, termtol, maxrounds):
        """
        Nested function for the FastICA fixed-point algorithm.
        """
        numSamples = x.shape[1]
        
        b = w_init
        #initial orthogonalization of the random guess
        b = b @ np.real(inv(sqrtm(b.T @ b)))
        bOld = np.zeros_like(b)
        
        iternum = 0
        minAbsCos = 0
        
        while (iternum < maxrounds) and ((1 - minAbsCos) > termtol):
            iternum += 1
            
            # Symmetric orthogonalization.
            # B = (X * ((X' * B) .^ 2)) / numSamples;
            b = (x @ ((x.T @ b) ** 2)) / numSamples
            
            # B = B * real(inv(B' * B)^(1/2));
            # This is B = B @ (B.T @ B)^(-1/2)
            cov_b = b.T @ b
            eps = 1e-10 * np.eye(cov_b.shape[0]) #add a small epsilon for numerical stability
            b = b @ np.real(inv(sqrtm(cov_b + eps)))
            
            # Test for termination condition.
            minAbsCos = np.min(np.abs(np.diag(b.T @ bOld)))
            
            bOld = b.copy()
        
        if iternum < maxrounds:
            logger.info(f"Convergence in {iternum} rounds.")
        else:
            logger.warning(f"Failed to converge; terminating after {iternum} rounds, "
                           f"current change in estimate {1-minAbsCos:3.3g}.")
            
        return b, iternum


def _extract_cells(
    mask,
    min_size,
    max_size,
    eccentricity_thresh=0.8,
    solidity_thresh=0.6,
    ):
    """
    Filters a binary mask to keep only roundish, cell-like blobs within size limits.
    """

    # 2. Label connected components
    closed_mask = binary_closing(mask, disk(2))
    labeled_image = label(closed_mask)
    
    # 3. Create a blank mask for the result
    cleaned_mask = np.zeros_like(mask, dtype=bool)

    # 4. Iterate through every blob found
    for region in regionprops(labeled_image):
        
        # --- CRITERIA TUNING ---
        # Size Filtering
        if region.area < min_size or region.area > max_size:
            continue

        # Eccentricity (Roundness)
        # 0 = circle, 1 = line. 
        if region.eccentricity > eccentricity_thresh:
            continue
            
        # Solidity (Regularity)
        if region.solidity < solidity_thresh: 
            continue

        # 5. If it passes, add it to the clean mask
        # We use the coordinates of the region to set pixels to True
        for coords in region.coords:
            cleaned_mask[coords[0], coords[1]] = True
            
    return cleaned_mask


def extract_rois_and_traces(
    spatial_filters, 
    temporal_signals, 
    min_size, 
    max_size,
    kurtosis_thresh,
    z_thresh,
    ):
    """
    Selects neuron-like ICA components and extracts a list of individual 
    ROI masks and their corresponding temporal traces.

    Args:
        spatial_filters (np.ndarray): 3D array (n_components, height, width).
        temporal_signals (np.ndarray): 2D array (n_components, n_timesteps).
        kurtosis_thresh (float): Min kurtosis for a "good" component.
        z_thresh (float): Z-score for pixel thresholding.
        min_size (int): Minimum pixel area for a neuron blob.
        max_size (int): Maximum pixel area for a neuron blob.

    Returns:
        list: A list of 2D (height, width) boolean masks. 
              Each entry is a single ROI.
        np.ndarray: A 2D (n_ROIs, n_timesteps) array of temporal traces.
                    The trace at index `i` corresponds to the mask at index `i`.
    """
    
    n_components = spatial_filters.shape[0]
    if temporal_signals.shape[0] != n_components:
        raise ValueError("Shape mismatch: spatial_filters and temporal_signals")
    
    final_roi_masks = []
    final_roi_traces = []
    roi_labels = []
    used_components = []
    bin_masks = []
    cl_masks = []

    logger.info(f"Processing {n_components} spatial filters...")

    counter = 0
    for i in range(n_components):
        component_img = spatial_filters[i]
        component_trace = temporal_signals[i]

        k = kurtosis(component_img.ravel()) 
        
        if k > kurtosis_thresh:
            logger.info(f"  [Component {i:2d}]: SELECTED (Kurtosis = {k:.2f})")
            

            # --- Mask Generation ---
            mean_val = np.mean(component_img)
            std_val = np.std(component_img)
            binary_mask = np.abs(component_img - mean_val) > (z_thresh * std_val)

            structure = np.ones((3, 3))
            cleaned_mask = binary_opening(binary_mask, structure=structure)
            cleaned_mask = binary_closing(cleaned_mask, structure=structure)

            labeled_array, num_features = label(cleaned_mask)
            
            if num_features == 0:
                continue

            blob_labels = np.arange(1, num_features + 1)
            blob_sizes = ndi_sum(cleaned_mask, labeled_array, index=blob_labels)
            
            good_blob_labels = blob_labels[(blob_sizes >= min_size) & 
                                           (blob_sizes <= max_size)]
            
            if good_blob_labels.size == 0:
                continue
            component_mask = np.isin(labeled_array, good_blob_labels)
            labeled_rois, num_rois = label(component_mask)

            bin_masks.append(binary_mask)
            cl_masks.append(component_mask)

            logger.info(f"  [Component {i:2d}]: Found {good_blob_labels.size} good blobs.")
            used_components.append(i)

            use_suffix = num_rois > 1
            
            # Add each individual blob as its own ROI
            counter += 1
            for j in range(1, num_rois + 1):
                roi_mask = (labeled_rois == j)
                final_roi_masks.append(roi_mask)
                final_roi_traces.append(-component_trace) # All blobs from this component get the same trace

                suffix = chr(96 + j) if use_suffix else ""
                roi_labels.append(f"{counter}{suffix}")
                
        else:
            logger.info(f"  [Component {i:2d}]: REJECTED (Kurtosis = {k:.2f})")
            
    logger.info(f"Extraction complete: Found {len(final_roi_masks)} ROIs.")
    
    # Convert lists of results to a 2D numpy array
    final_roi_masks_array = np.array(final_roi_masks)
    final_roi_traces_array = np.array(final_roi_traces)

    binary_masks = np.stack(bin_masks, axis=0)
    cleaned_masks = np.stack(cl_masks, axis=0)
    
    return (
        final_roi_masks_array,
        final_roi_traces_array,
        roi_labels,
        n_components,
        used_components,
        binary_masks,
        cleaned_masks,
    )