# DRG-CaSeg
A specialized computational framework designed for automated signal extraction and biological analysis of calcium imaging data from Dorsal Root Ganglion (DRG) neurons. This pipeline integrates ICA- and CaImAn-based source separation with Mixed Effects Modelling statistical analysis to model the degree of synchronization in the neuronal signals extracted from the tissue. A simulation framework is provided additionally to get started easily and generated augmented datasets with exact specifications meeting the composition of the tissue of interest.

### Project Structure

```
DRG-CaSeg/
├── .gitignore                              # Defines files/folders ignored by Git (e.g., .venv, .pixi)
├── pyproject.toml                          # The unified dependency manifest for Python and R
├── README.md                               # You are here!
├── LICENSE                                 # Licensing agreement for further use
├── legacy                                  # Initial code used to start off the project developed by Dr. Junxuan Ma
├── notebooks                               # Example code on how to use the Caiman framework developed by Giovannucci et al., 2019
└── DRG-CaSeg/                              # Framework repo for automated signal extraction and analysis
    ├── main.py                             # Entry point for source separation pipeline. Use CL arguments to add configs to a run
    │                                       # and determine hardware usage
    ├── config.py                           # Configure event logging
    ├── utils.py                            # System helper functions
    ├── analysis_utils/                     # Core functionalities for performing source separation
    │   ├── caiman.py                       # Functions for default caiman param definition and running Caiman analysis
    │   ├── ica.py                          # Functions for running ICA analysis, including the custom unmixing matrix optimization
    │   ├── metrics.py                      # Functions for calculating algorithm performance metrics when compared to ground truth
    │   ├── pca.py                          # PCA implementation to whiten data before feeding it to ICA
    │   └── workflow.py                     # Wrapper functions to make ICA, Caiman and metrics calculation compatible with the
    │                                       # modular experiment class setup
    ├── bio_analysis/                       # Code package to analyse extracted neuronal signal based on their features and perform
    │   │                                   # statistical testing
    │   ├── bio_features.yaml               # Configuration file to determine how windowing should be performed, which features
    │   │                                   # should be calculated and how they should be plotted
    │   ├── core.py                         # Entry point for the bio_analysis toolbox. CL arguments can be used to determine config
    │   │                                   # files and storage location of the data
    │   ├── features.py                     # Functions for calculation of signal features. New functions can be added to
    │   │                                   # characterize the signal differently as long as the have the same output format
    │   ├── helpers.py                      # Standard signal processing operations such as filtering
    │   ├── plotters.py                     # Bio analysis specific visualization functions
    │   ├── sample_metadata.csv             # Metadata on tissue samples to sort them in treatment and control group
    │   ├── trace_featurs.csv               # Storage file for features calculated from core.py
    │   ├── stats.r                         # Statistical analysis and visualization of calculated features
    │   └── sample_metadata.csv             # Metadata on tissue samples to sort them in treatment and control group
    ├── data_utils/                         # Toolbox for loading and manipulating data
    │   ├── ops.py                          # Functions for generating geometrical shapes in 1D, 2D and 3D
    │   ├── plot_adapters.py                # Wrapper functions to make plotting functions compatible with the modular experiment
    │   │                                   # class setup
    │   ├── plotter.py                      # Plotting functions for the ML pipline
    │   ├── synthesizer.py                  # Class for simulating DRG data samples
    │   └── wrangler.py                     # Functions for loading and preprocessing data. New loaders can be added to includ
    │                                       # new datatypes if the cast the loaded 3D array to a Caiman movie object as output
    ├── exps/                               # Experiment configuration files for the ML pipeline
    ├── infra/                              # Infrastructure scripts to run source separation ML experiments
    │   ├── experiment.py                   # Class to store an experiment configuration and run through the distinct experiment
    │   │                                   # steps
    │   └── experiment_utils.py             # Wrapper functions to dynamically configure objects and setup storage for experiment
    │                                       # resulsts
    ├── postprocessing/                     # Toolbox for postprocessing source separation signals
    │   ├── processing_adapters.py          # Wrapper functions to make postprocessing compatible with modular experiments
    │   └── processing_utils.py             # Functions for postprocessing extracted signals such as filtering or peak detection
    └── sandbox/                            # Collection of small standalone functionalities
        ├── calculate_overall_metrics.py    # Calculate overall scores over full experiment
        ├── patch_framerates.py             # Add framerates to already computed .npz files
        ├── performance_analyzer.py         # Visualize performance for hyperparameter optimization
        ├── sourceseparation_viz.py         # Visualize source separation algorithm concept
        ├── visualize_simulation.py         # Plots simulated samples and their returned calcium imaging data
        └── viz_mask_video.py               # Plots computed ROI masks on the original calcium imaging data
```


This work is part of a Master Thesis in partial fulfillment of the Biomedical Engineering degree at ETH Zürich and was supported by the AO Foundation and the ETH Zürich Foundation.