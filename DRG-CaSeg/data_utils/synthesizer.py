import attrs
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec

from scipy.ndimage import binary_erosion, affine_transform
from skimage.measure import find_contours
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from pathlib import Path

import data_utils.ops as ops
from data_utils.plotter import plot_image


import logging
logger = logging.getLogger(__name__)




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
    # --- Basic Dimensions and Scale ---
    width_px: int = attrs.field(default=384)
    height_px: int = attrs.field(default=292)
    fps: float = attrs.field(default=28.5)
    duration_s: int = attrs.field(default=60)
    um_per_pixel: float = attrs.field(default=7.206)
    
    # --- Spatial Component Parameters ---
    num_small_neurons: int = attrs.field(default=8)
    num_large_neurons: int = attrs.field(default=7)
    small_neuron_size_um: tuple[float, float] = attrs.field(default=(20, 50))
    large_neuron_size_um: tuple[float, float] = attrs.field(default=(50, 200))
    glia_thickness_um: tuple[float, float] = attrs.field(default=(5, 15))
    glia_variance_um: float = attrs.field(default=3)
    vessel_area: float = attrs.field(default=0.05)
    
    # --- Brightness and Noise Parameters ---
    neuron_base_brightness: int = attrs.field(default=75)
    glia_base_brightness: int = attrs.field(default=50)
    snr: float = attrs.field(default=3.0)
    background_brightness: int = attrs.field(default=25)
    background_noise_lvl: int = attrs.field(default=5)

    # --- Temporal Activity Parameters ---
    spike_rate_neuron: float = attrs.field(default=0.7)
    tau_neuron_s: float = attrs.field(default=0.2)
    spike_rate_glia: float = attrs.field(default=0.06)
    tau_glia_s: float = attrs.field(default=1.6)

    # --- Artifact Simulation ---
    full_well_capacity: int = attrs.field(default=0)
    movement_artifact: bool = attrs.field(default=False)
    shrink_rate: float = attrs.field(default=0.01)

    # --- Internal State (Initialized automatically) ---
    width_um: float = attrs.field(init=False)
    height_um: float = attrs.field(init=False)
    num_frames: int = attrs.field(init=False)
    s_neuron_px: tuple[float, float] = attrs.field(init=False)
    l_neuron_px: tuple[float, float] = attrs.field(init=False)

    # Storage for the rendering matrices
    footprints = attrs.field(init=False, default=None)
    activities = attrs.field(init=False, default=None)
    labels = attrs.field(init=False, default=None)
    background = attrs.field(init=False, default=None)
    noisy_activities = attrs.field(init=False, default=None)
    summary_image = attrs.field(init=False, default=None)
    
    # Storage for the abstract definition of cells (allows moving them later)
    cell_metadata: list[dict[str, any]] = attrs.field(init=False, factory=list)

    def __attrs_post_init__(self):
        # Calculate derived physical properties
        self.width_um = self.width_px * self.um_per_pixel
        self.height_um = self.height_px * self.um_per_pixel
        self.num_frames = int(self.duration_s * self.fps)

        self.s_neuron_px = tuple(s / self.um_per_pixel for s in self.small_neuron_size_um)
        self.l_neuron_px = tuple(s / self.um_per_pixel for s in self.large_neuron_size_um)
        
        logger.info(f"DRGtissueModel initialized: {self.width_px}x{self.height_px} px ({self.num_frames} frames).")


    def render_video(
        self,
        store_traces: bool,
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
        if self.footprints is None or self.activities is None:
            self.build_image()
            
        if self.background is None:
            self.generate_static_background()

        video = np.zeros((self.num_frames, self.height_px, self.width_px), dtype=np.float32)        
        for t in range(self.num_frames):
            if self.movement_artifact:
                current_footprints = self._apply_motion_artifact(self.footprints, t)
            else:
                current_footprints = self.footprints
            brightness_values = self.activities[:, t]

            frame = np.einsum('nhw,n->hw', current_footprints, brightness_values)       
            video[t] = frame

        video += self.background
        video = np.clip(video, 0, 255)
        video_noisy = ops.generate_gaussian_noise(
            video,
            snr=self.snr,
        )
        video_norm = np.clip(video_noisy, 0, 255) / 255

        if self.full_well_capacity > 0.5:
            logger.info("Adding Photon Shot Noise")
            video_norm = ops.generate_photon_shot_noise(
                video_norm,
                self.full_well_capacity
            )

        self.summary_image = np.percentile(video_norm, 98, axis=0)
        if store_traces:
            self._extract_noisy_traces(video_norm)

        return video_norm


    def build_image(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs footprints and traces based on `cell_metadata`. 
        If metadata doesn't exist, it generates it.
        """
        
        if not self.cell_metadata:
            self._generate_cell_definitions()

        spatial_maps = []
        traces = []

        for cell in self.cell_metadata:
            s_neuron, s_glia = self.generate_neuron_with_glia(
                h=self.height_px,
                w=self.width_px,
                center_y=cell["y"],
                center_x=cell["x"],
                sig_y_neuron=cell["size"],
                sig_x_neuron=cell["size"] * cell["aspect_ratio"],
                cutoff_percentage=25,
                glia_thickness_um=cell["glia_thick"],
                glia_variance_um=self.glia_variance_um,
                angle_deg=cell["angle"],
            )

            if "trace_neuron" not in cell:
                cell["trace_neuron"] = self.build_timeline_neuron(
                    self.num_frames, self.fps, self.spike_rate_neuron, self.tau_neuron_s
                )
                cell["trace_glia"] = self.build_timeline_glia(
                    self.num_frames, self.fps, self.spike_rate_glia, self.tau_glia_s
                )

            spatial_maps.extend([s_neuron, s_glia])
            traces.extend([cell["trace_neuron"], cell["trace_glia"]])

        self.footprints = np.stack(spatial_maps, axis=0)
        self.activities = np.stack(traces, axis=0)
        self.labels = np.arange(self.footprints.shape[0])
        
        return self.footprints, self.activities

    def _generate_cell_definitions(self):
        """Generates random parameters for all cells once."""
        self.cell_metadata = []
        
        # Create parameters for Small and Large neurons
        sizes_small = np.random.uniform(self.s_neuron_px[0], self.s_neuron_px[1], self.num_small_neurons)
        sizes_large = np.random.uniform(self.l_neuron_px[0], self.l_neuron_px[1], self.num_large_neurons)
        all_sizes = np.concatenate((sizes_small, sizes_large))
        
        
        glia_thicks = np.random.uniform(self.glia_thickness_um[0], self.glia_thickness_um[1], len(all_sizes))

        for i in range(len(all_sizes)):
            self.cell_metadata.append({
                "y": np.random.randint(0, self.height_px),
                "x": np.random.randint(0, self.width_px),
                "size": all_sizes[i],
                "aspect_ratio": np.random.uniform(0.8, 1.2),
                "angle": np.random.uniform(0, 90),
                "glia_thick": glia_thicks[i]
            })

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
        mask_neuron = ops.generate_gaussian_blob(
            x=x_rot,
            y=y_rot,
            sigma_x=sig_x_neuron,
            sigma_y=sig_y_neuron
        )
        
        # Threshold the mask to create a sharp boundary of the neuron
        threshold_neuron = (cutoff_percentage/100) * np.max(mask_neuron)
        mask_neuron[mask_neuron < threshold_neuron] = 0

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
        mask_glia = ops.generate_gaussian_ring(
            boundary_mask=boundary_mask,
            thickness=glia_thickness_px,
            thickness_amplitude=glia_variance_px
        )
        
        threshold_glia = (cutoff_percentage/100) * np.max(mask_glia)
        mask_glia[mask_glia < threshold_glia] = 0
        
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
        #Force at least one event, if the generator failed to produce any
        if not np.any(event_train):
            #Add a buffer to avoid putting the event into the very last frame
            buffer = int(frame_rate_hz * tau_s) 
            max_idx = max(1, num_frames - buffer)
            
            forced_idx = np.random.randint(0, max_idx)
            event_train[forced_idx] = True

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
            np.random.uniform(
                self.neuron_base_brightness*0.75,
                self.neuron_base_brightness*1.25),
            0,
            255
        )
        full_trace *= brightness_factor
        full_trace += self.neuron_base_brightness
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

        if not np.any(event_train):
            #Add a buffer to avoid putting the event into the very last frame
            buffer = int(frame_rate_hz * tau_s) 
            max_idx = max(1, num_frames - buffer)
            
            forced_idx = np.random.randint(0, max_idx)
            event_train[forced_idx] = True
    
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
            np.random.uniform(
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
        landscape = ops.perlin_noise_2D(
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
        plot_image(
            image=static_background,
            save_loc="static_background_image",
            cmap="gray"
        )
    
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
        landscape = ops.perlin_noise_2D(
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

        This method generates a spatial map by rendering the contours of each 
        spatial footprint from `self.footprints` with a unique color over a 
        contrast-stretched base image. It also renders ground truth cells in 
        grayscale if the `gt_footprints` attribute is present.

        Args:
            save_loc (Path | str): The file path where the plot will be saved.
            dpi (int): Resolution of the saved figure.

        Returns:
            matplotlib.figure.Figure: The closed Figure object for the plot.
        """
        if not hasattr(self, "footprints") or self.footprints is None:
            logger.warning("Footprints not found. Please create cellular structures first.")
            return

        num_components = self.footprints.shape[0]
        img_h = self.height_px
        img_w = self.width_px
        img_aspect = img_h / img_w

        if hasattr(self, "summary_image") and isinstance(self.summary_image, np.ndarray) and self.summary_image.shape == (img_h, img_w):
            background_img = self.summary_image
        else:
            logger.warning(
                f"`self.summary` attribute not found or invalid. "
                f"Creating a new rendering "
                f"{self.background_brightness}."
            )
            self.render_video()
            background_img = self.summary_image

        if num_components <= 10:
            colors = cm.get_cmap("tab10")(np.linspace(0, 1, num_components))
        elif num_components <= 20:
            colors = cm.get_cmap("tab20")(np.linspace(0, 1, num_components))
        else:
            colors = cm.get_cmap("nipy_spectral")(np.linspace(0, 1, num_components))

        fig_spatial = plt.figure(figsize=(10, 10 * img_aspect), dpi=300)
        ax_map = fig_spatial.add_subplot(111)
        
        fig_spatial.suptitle("Ground Truth Spatial Layout", fontsize=20, fontweight="bold")

        # Robust contrast stretching
        vmin, vmax = np.percentile(background_img, [1, 99])
        ax_map.imshow(background_img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bilinear")

        for i in range(num_components):
            footprint = self.footprints[i]
            
            contours = find_contours(footprint, 0.5)
            if not contours:
                continue

            largest_contour_idx = -1
            max_len = 0
            for c_idx, c in enumerate(contours):
                if len(c) > max_len:
                    max_len = len(c)
                    largest_contour_idx = c_idx

            for c_idx, contour in enumerate(contours):
                ax_map.plot(contour[:, 1], contour[:, 0], linewidth=1.2, color=colors[i])
                
                if c_idx == largest_contour_idx:
                    ys, xs = contour[:, 0], contour[:, 1]
                    min_y, max_y = np.argmin(ys), np.argmax(ys)
                    min_x, max_x = np.argmin(xs), np.argmax(xs)
                    candidates = [
                        (ys[min_y], xs[min_y], "bottom", "center"),
                        (ys[max_y], xs[max_y], "top", "center"),   
                        (ys[min_x], xs[min_x], "center", "right"), 
                        (ys[max_x], xs[max_x], "center", "left")   
                    ]
                    np.random.shuffle(candidates)
                    final_pos = candidates[0]
                    margin = img_h * 0.05    

                    for y, x, va, ha in candidates:
                        if not (va == "bottom" and y < margin) and \
                           not (va == "top" and y > (img_h - margin)) and \
                           not (ha == "right" and x < margin) and \
                           not (ha == "left" and x > (img_w - margin)):
                            final_pos = (y, x, va, ha)
                            break

                    txt = ax_map.text(
                        final_pos[1], final_pos[0], 
                        str(i), 
                        color=colors[i], 
                        fontsize=9, 
                        ha=final_pos[3], 
                        va=final_pos[2], 
                        fontweight="bold"
                    )
                    txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

        # Add scale bar to the plot if metadata is available on the instance
        if hasattr(self, "um_per_pixel"):
            bar_px = (100 / self.um_per_pixel)
            scalebar = AnchoredSizeBar(
                ax_map.transData, 
                bar_px, 
                "100 \u03bcm", 
                "lower right", 
                pad=0.5, 
                color="white", 
                frameon=False, 
                size_vertical=2
            )
            ax_map.add_artist(scalebar)
        
        # Finish layout and save plot
        ax_map.axis("off")
        fig_spatial.tight_layout()
        fig_spatial.savefig(save_loc, dpi=300, bbox_inches="tight")
        plt.close(fig_spatial)

        return fig_spatial
    

    def perturb_positions(
        self, 
        target_indices: list[int], 
        angle_deg: list|  tuple | None = None, 
        shift_px: list | tuple | None = None,
        ):
        """
        Shifts specific neurons by a defined or random vector, then triggers a rebuild.

        Args:
            target_indices (list[int]): A list of indices corresponding to the neurons 
                                        in `self.cell_metadata` that should be moved.
            angle_deg (float | None): The direction of movement in degrees (0-360). 
                                      0 is East (Right), 90 is South (Down).
                                      If None, a random angle is chosen for each neuron.
            shift_px (float | None): The exact distance to move in pixels.
                                     If None, a random distance is chosen between 1px 
                                     and the maximum distance possible before hitting 
                                     the image edge along the chosen angle.
        """
        if not self.cell_metadata:
            self._generate_cell_definitions()

        logger.info(f"Perturbing positions of {len(target_indices)} neurons...")

        for i, idx in enumerate(target_indices):
            # Validate index
            if idx < 0 or idx >= len(self.cell_metadata):
                logger.warning(f"Index {idx} is out of bounds. Skipping.")
                continue

            cell = self.cell_metadata[idx]
            curr_x, curr_y = cell["x"], cell["y"]

            # 1. Determine Angle (Theta)
            if angle_deg[i] is not None:
                theta = np.deg2rad(angle_deg[i])
            else:
                theta = np.random.uniform(0, 2 * np.pi)

            # Calculate vector components (Unit vector)
            dx_unit = np.cos(theta)
            dy_unit = np.sin(theta)

            # 2. Determine Shift Magnitude (Distance)
            if shift_px[i] is not None:
                dist = shift_px[i]
            else:
                # Calculate maximum distance to the edge along this specific vector
                candidates = []
                
                # Check distance to X boundaries (0 and width)
                # If moving right (dx > 0), limit is width. If moving left (dx < 0), limit is 0.
                if dx_unit > 0:
                    candidates.append((self.width_px - curr_x) / dx_unit)
                elif dx_unit < 0:
                    candidates.append((0 - curr_x) / dx_unit)
                
                # Check distance to Y boundaries (0 and height)
                # If moving down (dy > 0), limit is height. If moving up (dy < 0), limit is 0.
                if dy_unit > 0:
                    candidates.append((self.height_px - curr_y) / dy_unit)
                elif dy_unit < 0:
                    candidates.append((0 - curr_y) / dy_unit)

                # The valid max distance is the smallest positive distance to any wall
                max_dist = min(candidates) if candidates else 0
                
                # Handle edge case where neuron is already on the edge
                if max_dist < 1.0:
                    dist = 0.0
                else:
                    dist = np.random.uniform(1.0, max_dist)

            # 3. Apply the shift
            new_x = curr_x + (dist * dx_unit)
            new_y = curr_y + (dist * dy_unit)

            # Final safety clip to ensure floating point math didn't push it slightly over
            self.cell_metadata[idx]["x"] = np.clip(new_x, 0, self.width_px)
            self.cell_metadata[idx]["y"] = np.clip(new_y, 0, self.height_px)

        # FORCE REBUILD of footprints with new coordinates
        self.build_image()


    def _extract_noisy_traces(self, video: np.ndarray):
        """
        Extracts temporal traces from the fully rendered video using the 
        spatial footprints as extraction weights.
        
        Calculates the weighted average fluorescence for each ROI across all frames.
        """
        
        # Get dimensions
        T, H, W = video.shape
        N = self.footprints.shape[0]
        
        # Reshape video and footprints for fast matrix multiplication
        # video goes from (T, H, W) -> (T, H*W)
        # footprints go from (N, H, W) -> (N, H*W)
        flat_video = video.reshape(T, -1)
        flat_footprints = self.footprints.reshape(N, -1)
        
        # Calculate the sum of weights for each footprint to normalize the output
        # Shape becomes (N, 1)
        weight_sums = flat_footprints.sum(axis=1, keepdims=True)
        
        # Safety check: avoid division by zero if a footprint is completely empty
        weight_sums[weight_sums == 0] = 1.0 
        
        # Matrix multiplication: (N, H*W) @ (H*W, T) -> (N, T)
        # Then divide by weight_sums to get the weighted average
        raw_traces = (flat_footprints @ flat_video.T) / weight_sums
        
        # Store in the new attribute
        self.noisy_activities = raw_traces

    def plot_traces(
        self,
        save_loc: Path | str,
        ):
        """Plots the extracted noisy traces against the ground truth activities."""
        if not hasattr(self, "noisy_activities") or self.noisy_activities is None:
            logger.warning("Noisy traces not found. Run render_video with extract_noisy_traces=True.")
            return

        num_components = self.noisy_activities.shape[0]
        max_rois = 100
        num_trace_rois = min(num_components, max_rois)
        
        trace_subset = self.noisy_activities[:num_trace_rois]
        gt_trace_subset = self.activities[:num_trace_rois] if self.activities is not None else None
        
        if num_components <= 10:
            colors = cm.get_cmap("tab10")(np.linspace(0, 1, num_components))
        elif num_components <= 20:
            colors = cm.get_cmap("tab20")(np.linspace(0, 1, num_components))
        else:
            colors = cm.get_cmap("nipy_spectral")(np.linspace(0, 1, num_components))

        height_per_roi = 0.5 
        header_space = 2.0
        trace_min_height = header_space + (num_trace_rois * height_per_roi)
        fig_height = max(8.0, trace_min_height)

        fig_temporal = plt.figure(figsize=(10, fig_height), dpi=300)
        gs = gridspec.GridSpec(max(1, num_trace_rois), 1, figure=fig_temporal, hspace=0.5)
        
        title_y_pos = 1.0 - (0.3 / fig_height)
        fig_temporal.suptitle("Temporal Activity", fontsize=20, fontweight="bold", y=title_y_pos)

        time_seconds = np.arange(self.num_frames) / self.fps
        axes_traces = []
        
        ax_first = fig_temporal.add_subplot(gs[0, 0])
        axes_traces.append(ax_first)

        for i in range(num_trace_rois):
            if i == 0:
                ax_trace = ax_first
            else:
                ax_trace = fig_temporal.add_subplot(gs[i, 0], sharex=ax_first)
                axes_traces.append(ax_trace)
            
            trace_plot = trace_subset[i]
            
            if gt_trace_subset is not None:
                # Scale the [0, 255] ground truth down to [0.0, 1.0] to match the video space
                trace_plot_gt = gt_trace_subset[i] / 255.0
            else:
                trace_plot_gt = None
            
            if trace_plot_gt is not None:
                ax_trace.plot(time_seconds, trace_plot_gt, color="#000000", linewidth=1.2)
                
            ax_trace.plot(time_seconds, trace_plot, color=colors[i], linewidth=1.2)
            ax_trace.grid(True, linestyle="--", linewidth=0.5, color="#9e9b9b", alpha=0.5)
            ax_trace.set_axisbelow(True)

            ax_trace.text(0.01, 0.85, f"ROI {i}", transform=ax_trace.transAxes, fontsize=10, fontweight="bold", color="black")

            ax_trace.yaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))
            ax_trace.tick_params(axis="y", labelsize=7)

            ax_trace.spines["top"].set_visible(False)
            ax_trace.spines["right"].set_visible(False)
            
            if i < num_trace_rois - 1:
                ax_trace.tick_params(labelbottom=False, bottom=True) 
                ax_trace.spines["bottom"].set_visible(True) 
            else:
                ax_trace.set_xlabel("Time (s)", fontsize=10)
                ax_trace.tick_params(axis="x", labelsize=9)

        ax_first.set_xlim(time_seconds[0], time_seconds[-1])
        fig_temporal.align_ylabels(axes_traces)

        # Apply the single unified Y-axis label here, explicitly setting x position to avoid overlap
        fig_temporal.supylabel("Intensity [a.u.]", fontsize=12, fontweight="bold", x=0.02)

        header_y = 1.0 - (0.8 / fig_height)
        fig_temporal.text(0.5, header_y, "Raw Extracted vs Ground Truth Traces", fontsize=14, fontweight="bold", ha="center", va="top")

        # Increased left margin from 0.05 to 0.08 to give the numbers more space
        plt.subplots_adjust(top=1.0 - (1.2 / fig_height), bottom=0.05, left=0.08, right=0.95)

        fig_temporal.savefig(save_loc, dpi=300, bbox_inches="tight")
        plt.close(fig_temporal)
        
        return fig_temporal