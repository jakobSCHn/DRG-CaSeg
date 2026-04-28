DRG-CaSeg/
├── .gitignore              # Defines files/folders ignored by Git (e.g., .venv, .pixi)
├── pyproject.toml          # The unified dependency manifest for Python and R
├── README.md               # You are here!
├── create_toml.py          # Script used to generate/filter the pyproject.toml file
│
├── data/                   # [TODO: Describe input data types/formats (e.g., .tif, .hdf5)]
│   └── raw/                # [TODO: Placeholder for raw imaging files]
│
├── scripts/                # Functional scripts for the analysis pipeline
│   ├── python/             # Python-based segmentation and extraction
│   │   └── segmentation.py # [TODO: Describe the CaImAn parameters or workflow used here]
│   │
│   └── r/                  # R-based statistical analysis and plotting
│       └── analysis.R      # [TODO: Describe the ggplot2/lmerTest analysis workflow]
│
└── results/                # [TODO: Describe expected output (e.g., CSVs, PNG plots)]
    ├── figures/            # [TODO: Export directory for publication-ready plots]
    └── logs/               # [TODO: Processing logs and quality control metrics]