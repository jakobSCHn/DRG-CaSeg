import numpy as np
import logging
import dask.array as da

from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, inv
from analysis_utils.pca import _cellsort_svd, _reload_moviedata
from dask_ml.decomposition import PCA

logger = logging.getLogger(__name__)

def cellsort_pca_dask(
    video,
    n_pcs=None,
    ):
    """
    Dask implementation of CellsortPCA using dask_ml.decomposition.PCA.
    
    Args:
        video (dask.array.core.Array or np.ndarray): Input video (Time x Height x Width).
        n_pcs (int, optional): Number of components. Defaults to min(150, n_frames).
        chunks: Chunk size for dask array creation if numpy array is passed.
    
    Returns:
        tuple: (mixedsig, mixedfilters, cov_evals, cov_trace, movm, movtm)
        Note: Returns dask arrays. Call .compute() on results to get numpy arrays.
    """
    
        
    nt, pixh, pixw = video.shape
    npix = pixh * pixw
    
    if n_pcs is None:
        n_pcs = min(150, nt)

    logger.info(f"Processing {npix} pixels x {nt} time frames (Dask)")

    # 2. Flatten and Transpose to (Pixels, Time)
    # Original code flattens Fortran style: (Height, Width) -> (Width, Height)
    # We permute (T, H, W) -> (H, W, T) then reshape to (Pixels, T)
    video_flat = video.transpose(1, 2, 0).reshape((npix, nt))

    # 3. Df/F Normalization (Strict adherence to original logic)
    # Calculate Mean per pixel (over time)
    video_mean = video_flat.mean(axis=1) 
    
    # Handle zeros to avoid division by zero (preserve original logic)
    video_safe = da.where(video_mean == 0, 1, video_mean)
    
    # Normalize: (Frame - Mean) / Mean  =>  Frame/Mean - 1
    # We keep it as (Pixels, Time) for now
    video_norm = (video_flat / video_safe[:, None]) - 1
    
    # Apply the zero-mask back (where mean was 0, result should be 0)
    # We use map_blocks or simple multiplication if broadcasting allows
    mask = (video_mean != 0)
    video_norm = video_norm * mask[:, None]
    video_norm = da.rechunk(video_norm, (10000, 2500))
    # 4. Setup PCA based on dimensions (Mukamel logic)
    # dask_ml PCA expects (Samples, Features). It computes covariance of Features.
    
    if nt < npix:
        logger.info("Using temporal covariance matrix (Features=Time).")
        # Input: (Pixels, Time). 
        # Samples = Pixels, Features = Time. 
        # PCA computes covariance of Time (T x T).
        
        # We perform PCA
        pca = PCA(n_components=n_pcs)
        logger.info("PCA built")
        pca.fit(video_norm)
        logger.info("PCA fitted")
        # 5. Extract Components (Temporal Mode)
        
        # pca.components_ is (n_components, n_features) -> (k, Time)
        # This corresponds to the TEMPORAL signals (eigenvectors of covmat)
        mixedsig_t = pca.components_
        
        # pca.transform gives (n_samples, n_components) -> (Pixels, k)
        # This corresponds to the SPATIAL filters (projections)
        mixedfilters_flat = pca.transform(video_norm)
        
        # Replicate original scaling: mixedsig = eigenvectors * nt
        mixedsig = mixedsig_t.T * nt
        
        # Reshape Spatial filters to (Width, Height, k) 
        # Note: Dask reshape order might differ, we ensure (H, W, k) then transpose if needed
        mixedfilters = mixedfilters_flat.reshape((pixh, pixw, n_pcs))
        # Original code returns (Height, Width, k) (actually pixw/pixh order varies in Matlab ports)
        # Assuming standard image layout:
        mixedfilters = mixedfilters.transpose(1, 0, 2) # Adjust to match Fortran/Matlab conventions if needed

    else:
        logger.info("Using spatial covariance matrix (Features=Pixels).")
        # Input: (Time, Pixels).
        # Samples = Time, Features = Pixels.
        # PCA computes covariance of Pixels (P x P).        
        pca = PCA(n_components=n_pcs) # Randomized is better for huge Spatial dims
        pca.fit(video_norm.T)
        
        # pca.components_ is (k, Pixels) -> SPATIAL filters
        mixedfilters_flat = pca.components_.T
        
        # pca.transform is (Time, k) -> TEMPORAL signals
        mixedsig = pca.transform(video_norm.T)
        
        # Reshape Spatial filters
        mixedfilters = mixedfilters_flat.reshape((pixh, pixw, n_pcs))
        mixedfilters = mixedfilters.transpose(1, 0, 2)

    # 6. Metrics and Averages
    
    # Eigenvalues of covariance
    # dask_ml explained_variance_ is exactly the eigenvalues of the covariance matrix
    cov_evals = pca.explained_variance_
    
    # Trace of covariance
    # The sum of all eigenvalues approximates the trace
    cov_trace = cov_evals.sum() # Or pca.noise_variance_ correction if using randomized solver
    
    # Mean Image (reshaped)
    movm_img = movm.reshape((pixh, pixw)).transpose(1, 0)

    # Mean Trace (over space)
    if nt < npix:
        movtm = video_norm.mean(axis=0)
    else:
        movtm = video_norm.mean(axis=1)

    return mixedsig, mixedfilters, cov_evals, cov_trace, movm_img, movtm