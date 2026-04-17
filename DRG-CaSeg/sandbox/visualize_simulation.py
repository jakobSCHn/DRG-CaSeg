import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


from data_utils.synthesizer import DRGtissueModel

model = DRGtissueModel(
    duration_s=60,
    num_small_neurons=4,
    num_large_neurons=4,
)

model.build_image()

model.plot_ground_truth(
    save_loc="/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/tissue_simulation_plots/base_plot.png",
)
