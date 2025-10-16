import attrs
import logging
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy.ndimage import binary_erosion, distance_transform_edt, affine_transform
from scipy.interpolate import CubicSpline, RectBivariateSpline
from pathlib import Path


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
    noisy_image_electrons = np.random.poisson(lam=lambda_image)

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




@attrs.define
class DRGtissueModel:
    """A configuration and generation model for simulating synthetic Dorsal Root Ganglia (DRG) tissue calcium imaging video data.

    This class serves as a blueprint for generating a realistic video dataset by defining 
    all necessary physical, temporal, and noise parameters. It integrates spatial 
    components (neurons, glia, vessels) with temporal activity traces (spiking and decay) 
    and applies realistic noise sources (Gaussian and background).

    Attributes:
        # --- Basic Dimensions and Scale ---
        width_px (int): The width of the video frames in pixels.
        height_px (int): The height of the video frames in pixels.
        fps (float): The frames per second (Hz) of the video.
        duration_s (int): The total duration of the video in seconds.
        um_per_pixel (float): The spatial scale factor, converting micrometers (µm) to pixels.

        # --- Spatial Component Parameters ---
        num_small_neurons (int): The number of small neurons to include in the simulation.
        num_large_neurons (int): The number of large neurons to include in the simulation.
        small_neuron_size_um (tuple): The (min, max) range for the diameter of small neurons in µm.
        large_neuron_size_um (tuple): The (min, max) range for the diameter of large neurons in µm.
        glia_thickness_um (tuple): The (min, max) range for the thickness of the surrounding glia (satellite cells) in µm.
        glia_variance_um (float): The variance applied to the glia thickness, used for generating noise ring thickness in µm.
        vessel_area (float): The fractional area (e.g., 0.05 for 5%) of the image to be covered by the capillary network mask.

        # --- Brightness and Noise Parameters ---
        neuron_base_brightness (int): The base brightness (0-255) for neuron footprints.
        glia_base_brightness (int): The base brightness (0-255) for glia footprints.
        snr (float): The desired linear Signal-to-Noise Ratio for the final Gaussian noise addition.
        background_brightness (int): The mean static brightness (0-255) value of the background.
        background_noise_lvl (int): The standard deviation (0-255) of the static background intensity.

        # --- Temporal Activity Parameters ---
        spike_rate_neuron (float): The mean event (spike) frequency for neurons in Hz.
        tau_neuron_s (float): The decay time constant (tau) in seconds for the neuron activity trace kernel.
        spike_rate_glia (float): The mean event frequency for glia in Hz.
        tau_glia_s (float): The decay time constant (tau) in seconds for the glia activity trace kernel.

        # --- Calculated Attributes (Read-Only) ---
        width_um (float): Calculated total width of the image in µm (width_px * um_per_pixel).
        height_um (float): Calculated total height of the image in µm (height_px * um_per_pixel).
        num_frames (int): Calculated total number of frames (duration_s * fps).
        s_neuron_px (tuple): The (min, max) diameter range for small neurons in pixels.
        l_neuron_px (tuple): The (min, max) diameter range for large neurons in pixels.
    """
    width_px: int = attrs.field(default=384)
    height_px: int = attrs.field(default=292)
    fps: float = attrs.field(default=28.5)
    duration_s: int = attrs.field(default=60)
    um_per_pixel: float = attrs.field(default=7.206)
    
    num_small_neurons: int = attrs.field(default=20)
    num_large_neurons: int = attrs.field(default=15)
    small_neuron_size_um: tuple = attrs.field(default=(20, 50))
    large_neuron_size_um: tuple = attrs.field(default=(50, 200))
    glia_thickness_um: tuple = attrs.field(default=(5, 15))
    glia_variance_um: float = attrs.field(default=3)
    vessel_area: float = attrs.field(default=0.05)
    
    neuron_base_brightness: int = attrs.field(default=75)
    glia_base_brightness: int = attrs.field(default=50)
    snr: float = attrs.field(default=3.0)
    background_brightness: int = attrs.field(default=25)
    background_noise_lvl: int = attrs.field(default=5)

    spike_rate_neuron: float = attrs.field(default=0.7)
    tau_neuron_s: float = attrs.field(default=0.2)
    spike_rate_glia: float = attrs.field(default=0.06)
    tau_glia_s: float = attrs.field(default=1.6)

    full_well_capacity: int = attrs.field(default=0)
    movement_artifact: bool = attrs.field(default=False)
    shrink_rate: float = attrs.field(default=0.01)

    width_um: float = attrs.field(init=False)
    height_um: float = attrs.field(init=False)
    num_frames: int = attrs.field(init=False)
    s_neuron_px: tuple = attrs.field(init=False)
    l_neuron_px: tuple = attrs.field(init=False)

    footprints: np.ndarray = attrs.field(init=False)
    activities: np.ndarray = attrs.field(init=False)
    background: np.ndarray = attrs.field(init=False)


    def __attrs_post_init__(self):
        self.width_um = self.width_px * self.um_per_pixel
        self.height_um = self.height_px * self.um_per_pixel
        self.num_frames = int(self.duration_s * self.fps)

        self.s_neuron_px = tuple(size / self.um_per_pixel for size in self.small_neuron_size_um)
        self.l_neuron_px = tuple(size / self.um_per_pixel for size in self.large_neuron_size_um)
        
        logger.info(f"DRGtissueModel instance created with dimension {self.width_px}x{self.height_px}.")


    def render_video(
        self,
        ) -> np.ndarray:
        """Renders the final synthetic video by combining spatial footprints, 
        temporal activities, and background noise.

        The method performs a matrix multiplication-like process: 
        For each frame, it scales the spatial footprints by their corresponding 
        temporal activities (brightness), sums them up, adds a static background, 
        clips the result, and then adds Gaussian noise based on the instance's SNR. 
        The final video is normalized to the range [0, 1].

        The resulting video tensor has a shape of (num_frames, width, height).

        Args:
            self: The instance of the class containing properties like:
                - num_frames (int): Total number of frames to generate.
                - snr (float): Signal-to-Noise Ratio for Gaussian noise addition.
                - build_image() (method): Returns (footprints, activities).
                - generate_static_background() (method): Returns the background image.

        Returns:
            A NumPy array representing the final rendered video, with pixel values 
            normalized to the range [0.0, 1.0].
        """
        if hasattr(self, "footprints") and hasattr(self, "activities"):
            footprints = self.footprints
            activites = self.activities
        else:
            footprints, activites = self.build_image()
        video_frames = []
        for t in range(self.num_frames):
            if self.movement_artifact:
                current_footprints = self._apply_motion_artifact(footprints, t)
            else:
                current_footprints = footprints
            brightness = activites[:,t]
            brightness_vector = brightness[:, np.newaxis, np.newaxis]

            scaled_footprints = current_footprints * brightness_vector
            frame = np.sum(scaled_footprints, axis=0)
            video_frames.append(frame)
        
        video = np.stack(video_frames, axis=0) + self.background
        video = np.clip(video, 0, 255)
        video_noisy = generate_gaussian_noise(
            video,
            snr=self.snr,
        )
        video_norm = np.clip(video_noisy, 0, 255) / 255

        if self.full_well_capacity > 0.5:
            logger.info("Adding Photon Shot Noise")
            video_norm = generate_photon_shot_noise(
                video_norm,
                self.full_well_capacity
            )


        return video_norm


    def build_image(
        self
        ) -> tuple[np.ndarray, np.ndarray]:
        """Constructs the spatial footprint and temporal activity (trace) matrices 
        for all neurons and their associated glia.

        This method generates a set of small and large neurons with randomly chosen 
        diameters and glial thickness. It then iteratively calls helper methods to 
        generate the 2D spatial map and 1D temporal trace for each component (neuron 
        and glia), finally stacking them into large matrices used for video rendering.

        Args:
            self: The instance of the class containing configuration properties like:
                - num_small_neurons, num_large_neurons (int): Counts of each neuron type.
                - s_neuron_px, l_neuron_px (tuple): Diameter ranges for neurons.
                - glia_thickness_um (tuple): Glia thickness range.
                - height_px, width_px (int): Dimensions of the video frame.
                - num_frames, fps (int): Temporal properties for traces.
                - generate_neuron_with_glia() (method): Generates spatial maps.
                - build_timeline_neuron(), build_timeline_glia() (methods): Generate traces.

        Returns:
            A tuple containing two NumPy arrays:
            - footprints (np.ndarray): The 3D array of stacked spatial maps for all 
            components (shape: [num_components, height_px, width_px]).
            - activities (np.ndarray): The 2D array of stacked temporal traces for all 
            components (shape: [num_components, num_frames]).
        """
        spatial_maps = []
        traces = []
        diameters = np.concatenate(
            (
            np.random.uniform(self.s_neuron_px[0], self.s_neuron_px[1], self.num_small_neurons),
            np.random.uniform(self.l_neuron_px[0], self.l_neuron_px[1], self.num_large_neurons)
            )
        )
        glia_thickness = np.random.uniform(self.glia_thickness_um[0], self.glia_thickness_um[1], len(diameters))
        for i in range(len(diameters)):
            s_neuron, s_glia = self.generate_neuron_with_glia(
                h=self.height_px,
                w=self.width_px,
                center_y=np.random.randint(0, self.height_px),
                center_x=np.random.randint(0, self.width_px),
                sig_y_neuron=diameters[i],
                sig_x_neuron=np.random.uniform(0.8, 1.2)*diameters[i],
                cutoff_percentage=15,
                glia_thickness_um=glia_thickness[i],
                glia_variance_um=self.glia_variance_um,
                angle_deg=np.random.uniform(0, 90),
            )
            trace_neuron = self.build_timeline_neuron(
                num_frames=self.num_frames,
                frame_rate_hz=self.fps,
                spike_rate_hz=self.spike_rate_neuron,
                tau_s=self.tau_neuron_s
            )
            trace_glia = self.build_timeline_glia(
                num_frames=self.num_frames,
                frame_rate_hz=self.fps,
                spike_rate_hz=self.spike_rate_glia,
                tau_s=self.tau_glia_s
            )
            spatial_maps.extend([s_neuron, s_glia])
            traces.extend([trace_neuron, trace_glia])


        
        footprints = np.stack(spatial_maps, axis=0)
        activities = np.stack(traces, axis=0)

        self.footprints = footprints
        self.activities = activities

        return footprints, activities


    def _apply_motion_artifact(
            self, 
            base_footprints: np.ndarray, 
            t: int
            ) -> np.ndarray:
            """Applies a centered shrinkage transformation to all footprints for a given time "t".

            Args:
                base_footprints: The original, non-transformed spatial footprints.
                t: The current frame index.

            Returns:
                A new array of footprints, scaled towards the image center.
            """
            # Calculate time in seconds
            t_sec = t / self.fps
            
            # Calculate the scale factor. Starts at 1.0 and decreases.
            scale_factor = 1.0 + (t_sec * self.shrink_rate) 
            
            # Get image center coordinates
            center_y = self.height_px / 2.0
            center_x = self.width_px / 2.0
            
            matrix = np.diag([scale_factor, scale_factor])
            offset = [center_y * (1 - scale_factor), 
                    center_x * (1 - scale_factor)]
            
            transformed_footprints = []
            for fp in base_footprints:
                moved_fp = affine_transform(
                    fp,
                    matrix,
                    offset=offset,
                    order=1,
                    mode='constant',
                    cval=0.0
                )
                transformed_footprints.append(moved_fp)
            
            return np.stack(transformed_footprints, axis=0)


    def generate_neuron_with_glia(
        self,
        h: int,
        w: int,
        center_y: float,
        center_x: float,
        sig_y_neuron: float,
        sig_x_neuron: float,
        cutoff_percentage: float,
        glia_thickness_um: float,
        glia_variance_um: float,
        angle_deg: float = 0.0,
        ) -> tuple[np.ndarray, np.ndarray]:
        """Generates the 2D spatial footprint (masks) for a single elliptical neuron 
        and its surrounding glial component.

        It creates a rotated elliptical Gaussian representing the neuron, thresholds 
        it to create a sharp boundary, and then uses that boundary to generate a 
        Gaussian ring representing the glia (e.g., satellite cell).

        Args:
            self: The instance of the class, expected to contain `um_per_pixel`.
            h: The height (rows) of the image/map in pixels.
            w: The width (columns) of the image/map in pixels.
            center_y: The y-coordinate (row index) of the neuron's center.
            center_x: The x-coordinate (column index) of the neuron's center.
            sig_y_neuron: The standard deviation (size) of the Gaussian along the y-axis.
            sig_x_neuron: The standard deviation (size) of the Gaussian along the x-axis.
            cutoff_percentage: The percentage of the peak neuron value used as 
                            the threshold to define the sharp boundary of the neuron.
            glia_thickness_um: The mean thickness of the glial component in micrometers (µm).
            glia_variance_um: The amplitude/variance of the thickness in micrometers (µm).
            angle_deg: The rotation angle of the elliptical neuron in degrees. Defaults to 0.0.

        Returns:
            A tuple containing two 2D NumPy arrays:
            - mask_neuron (np.ndarray): The 2D spatial footprint of the neuron.
            - mask_glia (np.ndarray): The 2D spatial footprint of the surrounding glia.
        """

        # Generate an index grid with specified dimensions
        y, x = np.indices((h, w))
        # Generate a relative coordinate system with respect to the Neuron center
        x_rel = x - center_x
        y_rel = y - center_y
        
        # Rotate the system to generate rotated neurons
        theta = np.deg2rad(angle_deg)
        x_rot = x_rel * np.cos(theta) - y_rel * np.sin(theta)
        y_rot = x_rel * np.sin(theta) + y_rel * np.cos(theta)
        
        
        # Create an elliptical Gaussian Blob representing the neuron
        mask_neuron = generate_gaussian_blob(
            x=x_rot,
            y=y_rot,
            sigma_x=sig_x_neuron,
            sigma_y=sig_y_neuron
        )
        
        # Threshold the mask to create a sharp boundary of the neuron
        threshold = (cutoff_percentage/100) * np.max(mask_neuron)
        mask_neuron[mask_neuron < threshold] = 0

        bool_mask_neuron = mask_neuron.astype(bool)
        # Extract the boundary of the neuron
        eroded_mask = binary_erosion(bool_mask_neuron)
        boundary_mask = bool_mask_neuron &~ eroded_mask

        # Clean up the mask to remove image border detection
        interior_mask = np.zeros_like(boundary_mask, dtype=bool)
        interior_mask[1:-1, 1:-1] = True
        boundary_mask = boundary_mask & interior_mask
        
        # Create the mask for the glia glia cell
        glia_thickness_px = glia_thickness_um / self.um_per_pixel
        glia_variance_px = glia_variance_um / self.um_per_pixel
        mask_glia = generate_gaussian_ring(
            boundary_mask=boundary_mask,
            thickness=glia_thickness_px,
            thickness_amplitude=glia_variance_px
        )
        
        return mask_neuron, mask_glia


    def build_timeline_neuron(
        self,
        num_frames: int, 
        frame_rate_hz: float, 
        spike_rate_hz: float, 
        tau_s: float,
        ) -> np.ndarray:
        """Generates a synthetic temporal trace (activity timeline) for a neuron 
        using an exponential decay kernel convolved with a random event train.

        The method simulates neuronal "spikes" occurring with a probability 
        determined by the spike rate and frame rate. It then models the decay 
        of the resulting fluorescence using an exponential kernel defined by tau, 
        and finally scales the trace by a random brightness factor.

        Args:
            self: The instance of the class, expected to contain 
                `self.neuron_base_brightness`.
            num_frames: The desired length of the resulting trace in frames.
            frame_rate_hz: The frames per second (Hz) of the video.
            spike_rate_hz: The mean frequency of spontaneous events (spikes) in Hz.
            tau_s: The exponential decay time constant (tau) in seconds.

        Returns:
            A 1D NumPy array of length `num_frames` representing the time-varying 
            activity (brightness) of the neuron. The values are scaled by a 
            random brightness factor.
        """       
        p_event_per_frame = spike_rate_hz / frame_rate_hz

        event_train = np.random.rand(num_frames) < p_event_per_frame

        # Calculate number of frames tau is covering 
        tau_frames = tau_s * frame_rate_hz
        # Calcualte kernel length in frames (limit to 3x tau due to 95% rule)
        kernel_len_frames = int(3 * tau_frames)
        t_kernel_sec = np.arange(kernel_len_frames) / frame_rate_hz
        
        kernel = np.exp(-t_kernel_sec / tau_s)
        
        # Normalize kernel so its peak equals 1 
        peak = np.max(kernel)
        norm_kernel = kernel / peak

        # Convolve the event train with the exponential decay and discard excess frames
        full_trace = np.convolve(event_train, norm_kernel, mode="full")

        # Adjust brightness of the signal
        brightness_factor = np.clip(
            np.random.normal(
                self.neuron_base_brightness*0.75,
                self.neuron_base_brightness*1.25),
            0,
            255
        )
        full_trace *= brightness_factor
        full_trace += brightness_factor
        return full_trace[:num_frames]


    def build_timeline_glia(
        self,
        num_frames: int, 
        frame_rate_hz: float, 
        spike_rate_hz: float, 
        tau_s: float,
        )-> np.ndarray:
        """Generates a synthetic temporal trace for a glial component (e.g., satellite cell) 
        using a bi-exponential (alpha) kernel convolved with a spike train that includes 
        a hard refractory period.

        The method simulates events using an **Exponential Inter-Spike Interval (ISI) distribution** (Poisson process),
        constrained by a **refractory period** to model realistic glia dynamics. 
        It then convolves this event train with an **alpha function kernel** (t * exp(-t/tau)) 
        to generate the fluorescence trace, which is scaled by a random brightness factor.

        Args:
            self: The instance of the class, expected to contain 
                `self.glia_base_brightness`.
            num_frames: The desired length of the resulting trace in frames.
            frame_rate_hz: The frames per second (Hz) of the video.
            spike_rate_hz: The mean frequency of events (spikes) in Hz.
            tau_s: The decay time constant (tau) in seconds, used for both the refractory 
                period calculation and the alpha function kernel.

        Returns:
            A 1D NumPy array of length `num_frames` representing the time-varying 
            activity (brightness) of the glial component. The values are scaled by 
            a random brightness factor.
        """     
        refractory_period_s = 5.7 * tau_s
    
        # Calculate the desired mean time *between* spikes
        mean_isi_s = 1.0 / spike_rate_hz

        # Check for impossible spike rates ---
        if mean_isi_s < refractory_period_s:
            logger.warning(
                f"Requested mean interval ({mean_isi_s:.2f}s) is less than the "
                f"refractory period ({refractory_period_s:.2f}s). "
                f"Spikes will occur deterministically at the refractory period."
            )
            # Set wait time to (near) zero. Spikes will be periodic.
            mean_wait_time_s = 1e-9 
        else:
            mean_wait_time_s = mean_isi_s - refractory_period_s

        # Generate spike times
        event_train = np.zeros(num_frames, dtype=bool)
        current_time_s = 0.0

        i = 0
        while True:
            random_wait_s = np.random.exponential(scale=mean_wait_time_s)
            isi_s = random_wait_s if i == 0 else (refractory_period_s + random_wait_s)
            
            # Calculate the absolute time of the next spike
            current_time_s += isi_s
            frame_index = int(round(current_time_s * frame_rate_hz))
            
            # Place spike and check bounds
            if frame_index < num_frames:
                if not event_train[frame_index]:
                    event_train[frame_index] = True
            else:
                break
            i += 1

        # Calculate number of frames tau is covering 
        tau_frames = tau_s * frame_rate_hz
        # Calcualte kernel length in frames (limit to 3x tau due to 95% rule)
        kernel_len_frames = int(5.7 * tau_frames)
        t_kernel_sec = np.arange(kernel_len_frames) / frame_rate_hz
        
        kernel = t_kernel_sec * np.exp(-t_kernel_sec / tau_s)
        
        # Normalize kernel so its AUC equals 1 
        peak = np.max(kernel)
        norm_kernel = kernel / peak

        # Convolve the event train with the exponential decay and discard excess frames
        full_trace = np.convolve(event_train, norm_kernel, mode="full")

        # Adjust brightness of the signal
        brightness_factor = np.clip(
            np.random.normal(
                self.glia_base_brightness*0.75,
                self.glia_base_brightness*1.25),
            0,
            255
        )
        full_trace *= brightness_factor
        #full_trace += brightness_factor
        return full_trace[:num_frames]

    
    def generate_static_background(
        self,    
        ) -> np.ndarray:
        """Generates the static background image by combining Perlin noise and a capillary network.

        The method first creates a low-frequency, large-scale background texture 
        using 2D Perlin noise (to simulate uneven illumination or large-scale tissue structure).
        It then multiplies this landscape with a separate map representing the capillary 
        network to introduce local variations and structure into the background signal.

        Args:
            self: The instance of the class containing properties required for generation:
                - background_brightness (float): Mean brightness level of the background.
                - background_noise_lvl (float): Amplitude of the Perlin noise variation.
                - width_px (int): Width of the final background image.
                - height_px (int): Height of the final background image.
                - generate_capillary_network() (method): Returns the 2D capillary network map.

        Returns:
            A 2D NumPy array of shape (height_px, width_px) representing the 
            final static background image.
        """
        landscape = perlin_noise_2D(
            constant=self.background_brightness,
            amplitude=self.background_noise_lvl,
            nodes_x=10,
            nodes_y=10,
            width=self.width_px,
            height=self.height_px,
        )
        capillary_network = self.generate_capillary_network()
        static_background = landscape * capillary_network
        self.background = static_background
    
        return static_background


    def generate_capillary_network(
        self,
        ) -> np.ndarray:
        """Generates a static, binary mask simulating a capillary or vessel network 
        using 2D Perlin noise and quantile-based thresholding.

        The method first generates a smooth 2D Perlin noise map (a "landscape"). 
        It then determines a threshold value that ensures a specific percentage of 
        the total area (`self.vessel_area`) is covered by "vessels". The final 
        mask is binary, where True represents the vessel network.

        The technique uses a high threshold to select the "hilltops" of the Perlin 
        noise, and then inverts this selection so that the **low-value regions 
        (the "valleys")** represent the network.

        Args:
            self: The instance of the class, expected to contain configuration 
                properties:
                - width_px, height_px (int): Dimensions of the output map.
                - vessel_area (float): The target fraction (e.g., 0.05 for 5%) 
                    of the map that should be covered by vessels.

        Returns:
            A 2D NumPy array (boolean mask) of shape (height_px, width_px), 
            where `True` indicates the location of the capillary network.
        """
        landscape = perlin_noise_2D(
            constant=128,
            amplitude=128,
            nodes_x=20,
            nodes_y=20,
            width=self.width_px,
            height=self.height_px,
        )
        
        threshold_value = np.quantile(landscape, 1 - self.vessel_area)

        hilltop_mask = landscape > threshold_value
    
        return ~hilltop_mask
    
    
    def plot_ground_truth(
        self,
        save_loc: Path | str, 
        ):
        """Saves a 2D visualization of all cellular structures.

        This method generates a single RGB image by rendering each spatial
        footprint from `self.footprints` with a unique color. The components
        are drawn sequentially on top of a base image.

        The base image is determined by `self.background`. If `self.background`
        is not a valid 2D numpy array a solid gray background is created using
        the `self.background_brightness` attribute.

        This method relies on the following instance attributes being set:
            - self.footprints (np.ndarray): A 3D array of shape
                (n_components, height, width).
            - self.height_px (int): The image height.
            - self.width_px (int): The image width.
            - self.background (np.ndarray, optional): A 2D array for the
                background.
            - self.background_brightness (float, fallback): Used if
                `self.background` is absent or invalid.

        Args:
            save_loc (Path | str): The file path (e.g., "path/to/image.png")
                where the resulting plot will be saved.

        Returns:
            matplotlib.figure.Figure: The Figure object for the plot. Note that
                the figure is closed (plt.close(fig)) immediately after saving.
        """
        
        if not hasattr(self, "footprints") or self.footprints is None:
            logger.warning("Footprints not found. Please create cellular structures first.")
            return

        num_components = self.footprints.shape[0]

        if hasattr(self, "background") and isinstance(self.background, np.ndarray) and self.background.shape == (self.height_px, self.width_px):
            base_image = self.background.astype(np.float32) / 255.0
        else:
            logger.warning(f"`self.background` attribute not found or invalid. "
                           f"Creating a new background with base brightness  "
                           f"{self.background_brightness}.")
            base_image = self.generate_static_background()

        colored_image = np.stack([base_image, base_image, base_image], axis=-1)

        if num_components <= 10:
            colors = cm.get_cmap("tab10", num_components)
        elif num_components <= 20:
            colors = cm.get_cmap("tab20", num_components)
        else:
            colors = cm.get_cmap("nipy_spectral", num_components) 
            

        for i in range(num_components):
            footprint = self.footprints[i]
            max_val = footprint.max()
            if max_val > 0:
                normalized_footprint = footprint / max_val
            else:
                normalized_footprint = footprint
            
            component_color = np.array(colors(i))[0:3]
            footprint_3d = normalized_footprint[:, :, np.newaxis]
            colored_image = colored_image * (1 - footprint_3d) + component_color * footprint_3d

        colored_image = np.clip(colored_image, 0, 1)

        aspect_ratio = self.height_px / self.width_px
        fig_width = 10 
        fig_height = fig_width * aspect_ratio

        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(colored_image)
        ax.set_title("Ground Truth Footprints")

        fig.tight_layout()
        fig.savefig(save_loc, dpi=300)
        plt.close(fig) 

        return fig