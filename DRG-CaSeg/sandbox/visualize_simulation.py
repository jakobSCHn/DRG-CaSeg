import sys
from pathlib import Path
import numpy as np

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from data_utils.synthesizer import DRGtissueModel
from utils import seed_everything

SEED = 42
seed_everything(SEED)

"""
model = DRGtissueModel(
    duration_s=30,
    num_small_neurons=4,
    num_large_neurons=4,
    full_well_capacity=15000,
)

model.render_video()

model.plot_ground_truth(
    save_loc="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/base_plot.png",
)

model.perturb_positions(
    target_indices=[7],
    angle_deg=[315],
    shift_px=[100],
)
model.render_video()
model.plot_ground_truth(
    save_loc="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/shifted_neuron.png",
)
"""

"""
model = DRGtissueModel(
    duration_s=30,
    num_small_neurons=15,
    num_large_neurons=15,
    full_well_capacity=15000,
)
model.render_video()

model.plot_ground_truth(
    save_loc="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/many_neurons.png",
)
"""

model = DRGtissueModel(
    duration_s=30,
    num_small_neurons=4,
    num_large_neurons=4,
    full_well_capacity=15000,
    snr=3.0,
)

model.render_video(
    store_traces=True,
)

model.plot_traces(
    save_loc="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/base_plot_traces.png",
)