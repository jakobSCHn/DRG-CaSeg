import numpy as np
import logging

from scipy.stats import skew, kurtosis
from scipy.linalg import inv, sqrtm

from scipy.ndimage import label, binary_opening, binary_closing
from scipy.ndimage import sum as ndi_sum

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
        # Spatial-temporal ICA
        sig_use = np.hstack([(1 - mu) * mixedfilters.T, mu * mixedsig])
        # Normalization
        sig_use = sig_use / np.sqrt(1 - 2 * mu + 2 * mu**2)

    # 5. Perform ICA
    ica_A, numiter = _fpica_standardica(sig_use, nIC, w_init, termtol, maxrounds)

    # 6. Post-processing and sorting
    ica_W = ica_A.T
    ica_sig = ica_W @ mixedsig
    
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
            b = b @ np.real(inv(sqrtm(b.T @ b)))
            
            # Test for termination condition.
            minAbsCos = np.min(np.abs(np.diag(b.T @ bOld)))
            
            bOld = b
        
        if iternum < maxrounds:
            logger.info(f"Convergence in {iternum} rounds.")
        else:
            logger.warning(f"Failed to converge; terminating after {iternum} rounds, "
                           f"current change in estimate {1-minAbsCos:3.3g}.")
            
        return b, iternum


def extract_rois_and_traces(
    spatial_filters, 
    temporal_signals, 
    kurtosis_thresh=5.0, 
    z_thresh=3.0, 
    min_size=10, 
    max_size=500
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

    logger.info(f"Processing {n_components} spatial filters...")

    for i in range(n_components):
        component_img = spatial_filters[i]
        component_trace = temporal_signals[i]
        
        k = kurtosis(component_img.ravel()) 
        
        if k > kurtosis_thresh:
            logger.info(f"  [Component {i:2d}]: SELECTED (Kurtosis = {k:.2f})")
            

            # --- Mask Generation (same as before) ---
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
                
            logger.info(f"  [Component {i:2d}]: Found {good_blob_labels.size} good blobs.")
            
            # Find all pixels belonging to any good blob from this component
            component_mask = np.isin(labeled_array, good_blob_labels)
            
            # Re-label the component_mask to separate any non-contiguous blobs
            labeled_rois, num_rois = label(component_mask)
            
            # Add each individual blob as its own ROI
            for j in range(1, num_rois + 1):
                roi_mask = (labeled_rois == j)
                final_roi_masks.append(roi_mask)
                final_roi_traces.append(-component_trace) # All blobs from this component get the same trace
                
        else:
            logger.info(f"  [Component {i:2d}]: REJECTED (Kurtosis = {k:.2f})")
            
    logger.info(f"Extraction complete: Found {len(final_roi_masks)} ROIs.")
    
    # Convert list of traces to a 2D numpy array
    final_roi_traces_array = np.array(final_roi_traces)
    
    return final_roi_masks, final_roi_traces_array