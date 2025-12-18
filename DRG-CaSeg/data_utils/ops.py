import numpy as np

from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.ndimage import distance_transform_edt


import logging
logger = logging.getLogger(__name__)


def generate_gaussian_blob(
    x: int,
    y: int,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    """Generates a 2D Gaussian function (or blob) value for a given (x, y) coordinate.

    This function calculates the value of an unnormalized 2D Gaussian centered 
    at (0, 0) with independent standard deviations along the x and y axes.
    A small epsilon (1e-9) is added to the denominator to prevent 
    division by zero in case a sigma is exactly zero.

    Args:
        x: The x-coordinate at which to evaluate the Gaussian.
        y: The y-coordinate at which to evaluate the Gaussian.
        sigma_x: The standard deviation (width) of the Gaussian along the x-axis.
        sigma_y: The standard deviation (width) of the Gaussian along the y-axis.

    Returns:
        A float or np.ndarray representing the value of the 2D Gaussian 
        at the specified coordinate (x, y).
    """
    exponent_neuron = -(((y)**2 / (2 * (sigma_y**2 + 1e-9))) + 
                            ((x)**2 / (2 * (sigma_x**2 + 1e-9))))
    
    return np.exp(exponent_neuron)


def generate_gaussian_ring(
    boundary_mask: np.ndarray[np.bool],
    thickness: float,
    thickness_amplitude: float
    ) -> np.ndarray:
    """Generates a 2D Gaussian ring structure around a boundary mask.

    This function first calculates the distance of every pixel to the boundary 
    center line. It then uses Perlin noise to create a spatially varying thickness 
    map, which is used as the standard deviation `sigma` in a Gaussian 
    function to model a ring-like structure with potentially non-uniform width. 
    The resulting ring is "thicker" where the calculated `sigma` is larger.

    Args:
        boundary_mask: A boolean numpy array (mask) where True/1 
                       indicates the pixels forming the center line of the ring.
        thickness: The base mean thickness (standard deviation, `sigma`) of the 
                   Gaussian ring.
        thickness_amplitude: The maximum amplitude for the Perlin noise, which 
                             introduces variability to the base thickness across 
                             the ring.

    Returns:
        A float or np.ndarray of the same shape as `boundary_mask`, containing 
        the values of the 2D Gaussian ring. The values will be highest (near 1) 
        along the center line and decay outward according to the local Gaussian 
        standard deviation.
    """

    inverted_boundary = ~boundary_mask
    
    # Comput the distance from each pixel to the center line
    distance_map, indices = distance_transform_edt(
        inverted_boundary,
        return_indices=True
    )
    
    # Generate Perlin noise to model varying thickness of the glia Cell
    thickness_map = perlin_noise_1D(
        constant=thickness,
        amplitude=thickness_amplitude,
        length=np.sum(boundary_mask),
        nodes=5
    )
    
    full_zeta_map = create_thickness_map(
        boundary_mask=boundary_mask,
        thickness_1d=thickness_map
    )

    sigma_map = full_zeta_map[indices[0], indices[1]]
    

    gaussian_ring = np.exp(-distance_map**2 / (2 * sigma_map**2 + 1e-9))
    
    return gaussian_ring


def generate_gaussian_noise(
    video: np.ndarray,  
    snr: float = 4.0,
    ) -> np.ndarray:
    """Adds Gaussian white noise to a video array based on a target Signal-to-Noise Ratio (SNR).

    The noise power (standard deviation) is calculated by dividing the 
    signal power (standard deviation of the video array) by the target SNR. 
    The resulting noise is then added to the original video array.

    Args:
        video: A NumPy array representing the video data (the signal). Shape has to be frames*width*height. 
               The noise will have the same shape.
        snr: The desired linear (not dB) Signal-to-Noise Ratio. 
             Defaults to 4.0.

    Returns:
        A NumPy array of the same shape as the input `video`, 
        now corrupted with Gaussian noise.
    """
                
    if snr <= 0:
        raise ValueError("SNR must be positive.")
    signal_power = np.std(video)
    noise_power = signal_power / snr

    noise = np.random.normal(loc=0.0, scale=noise_power, size=video.shape)

    return video + noise


def perlin_noise_1D(
    constant: float,
    amplitude: float,
    nodes: int,
    length: int,
    ) -> np.ndarray:
    """Generates a 1D noise array using cubic spline interpolation over random nodes.

    This function simulates a 1D Perlin-like noise by first generating a set of 
    random control points (nodes) and then smoothly interpolating between them 
    using a cubic spline to create the final noise array.

    Args:
        constant: The baseline (mean) value around which the noise is centered.
        amplitude: The maximum distance the random node values can deviate 
                   from the "constant" (i.e., noise range is [constant-amplitude, constant+amplitude]).
        nodes: The number of control points (nodes) to use for the interpolation. 
               More nodes result in higher frequency noise.
        length: The desired total length of the final 1D noise array.

    Returns:
        A NumPy array of shape (length,) containing the 1D interpolated noise signal.
        The values will be within the range defined by 'constant' and 'amplitude'.
    """
    nodes_x = np.linspace(0, length-1, nodes)
    nodes_y = np.random.uniform(
        constant-amplitude,
        constant+amplitude,
        nodes
        )
    
    f = CubicSpline(nodes_x, nodes_y)


    x_new = np.arange(length)
    perlin_noise = f(x_new)

    return perlin_noise


def perlin_noise_2D(
    constant: float,
    amplitude: float,
    nodes_x: int,
    nodes_y: int,
    width: int,
    height: int,
    ):
    """Generates a 2D noise array (landscape) using bicubic spline interpolation over random control nodes.

    This simulates a 2D Perlin-like noise texture by setting up a grid of 
    random control points (nodes) and then using a RectBivariateSpline with 
    cubic interpolation (kx=3, ky=3) to smoothly fill the space between them. 
    It ensures all output values are non-negative.

    Args:
        constant: The baseline (mean) value around which the random noise nodes are centered.
        amplitude: The maximum distance the random node values can deviate 
                   from the `constant` (noise range is [constant-amplitude, constant+amplitude]).
        nodes_x: The number of control points (nodes) along the x-axis.
        nodes_y: The number of control points (nodes) along the y-axis.
        width: The desired final width (number of columns) of the 2D array.
        height: The desired final height (number of rows) of the 2D array.

    Returns:
        A NumPy array of shape (height, width) containing the 2D interpolated noise map. 
        All values in the returned array are clipped to be non-negative (>= 0).
    """
    x_nodes = np.linspace(0, width - 1, nodes_x)
    y_nodes = np.linspace(0, height - 1, nodes_y)
    
    z_nodes = np.random.uniform(
        constant - amplitude,
        constant + amplitude,
        (nodes_x, nodes_y)
    )
    
    interpolator = RectBivariateSpline(x_nodes, y_nodes, z_nodes, kx=3, ky=3)
    
    x_fine = np.arange(width)
    y_fine = np.arange(height)
    
    landscape = interpolator(x_fine, y_fine)
    landscape[landscape < 0] = 0
    
    return landscape.T


def generate_photon_shot_noise(
    image: np.ndarray,
    full_well_capacity: float,
    ) -> np.ndarray:
    """
    Models photon shot noise and sensor saturation using a Poisson 
    distribution based on a physical full well capacity.

    Args:
        image: The noiseless input image (or video frame), 
                          assumed to be normalized to the [0, 1] range. 
                          This represents the *relative* mean light intensity.
        full_well_capacity: The maximum number of electrons (e-) a single 
                            pixel can store. This defines the "signal ceiling".

    Returns:
        The noisy image, normalized back to the [0, 1] range.
    """
    lambda_image = image * full_well_capacity
    del image
    noisy_image_electrons = np.random.poisson(lam=lambda_image)
    del lambda_image

    saturated_image_electrons = np.clip(
        noisy_image_electrons, 0, full_well_capacity
    )
    
    noisy_image = saturated_image_electrons / full_well_capacity

    return noisy_image


def create_thickness_map(
    boundary_mask: np.ndarray, 
    thickness_1d: np.ndarray
    ) -> np.ndarray:
    """Maps a 1D array of thickness values onto a 2D boundary mask.

    This function takes a binary mask defining the location of a boundary and a 
    1D array of corresponding thickness values (zeta). It creates a 2D array 
    where the thickness values are placed at the locations specified by the 
    boundary mask, and all other locations are zero.

    Args:
        boundary_mask: A 2D NumPy array (typically boolean or integer) where 
                       True/non-zero values mark the location of the boundary.
        thickness_1d: A 1D NumPy array of float values representing the thickness 
                      at each corresponding pixel in the boundary_mask, in row-major order.

    Returns:
        A 2D NumPy array of the same shape as `boundary_mask` containing the 
        `thickness_1d` values placed only at the boundary locations.
    """
    boundary_coords_u, boundary_coords_v = np.where(boundary_mask)
    if len(thickness_1d) != len(boundary_coords_u):
        raise ValueError(
            f"The 1D zeta array length ({len(thickness_1d)}) does not match "
            f"the number of boundary pixels ({len(boundary_coords_u)})."
        )
    zeta_on_boundary = np.zeros_like(boundary_mask, dtype=float)
    zeta_on_boundary[boundary_coords_u, boundary_coords_v] = thickness_1d
    return zeta_on_boundary