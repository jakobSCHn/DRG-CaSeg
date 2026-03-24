"""
Post-Processing and Data Augmentation Entry Point.

This script serves as the primary driver for all post-processing tasks
within the project. It coordinates the data augmentation pipeline,
including image transformations, noise injection, and dataset balancing,
ensuring that processed outputs are ready for model training.

Usage:
    python -m postprocessing.post_main --input_dir "./data/raw" --output_dir "./data/aug"
"""

import logging
import argparse
import yaml

import prep

from pathlib import Path
from config import setup_logging
from infra.infra_utils import configure_callable

setup_logging()
logger = logging.getLogger(__name__)


def parse_yaml_config():
    parser = argparse.ArgumentParser(
        description="Run analysis of extracted ROIs from Calcium Imaging data."
    )

    parser.add_argument(
        "-c",
        "--config", 
        required=True,
        type=Path,
        help="Yaml file containing the analysis configuration.",
    )

    parser.add_argument(
        "-r",
        "--results_folder",
        required=True,
        type=Path,
        help="Directory containint the target .npz files to be analysed.",
    )

    parser.add_argument(
        "-t",
        "--target_string",
        required=True,
        type=str,
        help="Identifier string to fetch the right .npz files from the results directory."
    )

    parser.add_argument(
        "-a",
        "--analysis_pattern",
        required=False,
        default=None,
        type=str,
        help="Identifier string to fetch the right .npz files from the results directory."
    )

    args = parser.parse_args()

    return args


def run_bioanalysis(
    args: argparse.Namespace
    ):
    if not args.results_folder.exists():
        raise FileNotFoundError(
            f"The results folder does not exist: {args.results_folder}"
        )

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    try:
        matrix_oi = config["matrix_oi"]
        feature_configs = config["features"]
        plotting_configs = config["plotters"]
    except KeyError as e:
        raise KeyError(f"Missing mandatory key in YAML config file: {e}")

    #Configure the functions for calculating features form
    #the definitions in the yaml config file
    feature_functions = []
    for feature in feature_configs:
        func = configure_callable(
            id=feature["id"],
            import_path=feature["function"],
            params=feature.get("params", {}),
            context={}
        )
        feature_functions.append(func)

    feature_df = prep.process_files(
        root_dir=args.results_folder,
        target_string=args.target_string,
        analysis_pattern=args.analysis_pattern,
        processing_functions=feature_functions,
        matrix_oi=matrix_oi,
    )

    #Configure the functions for plotting extracted features
    #from the definitions in the yaml config file
    plotting_functions = []
    for plotter in plotting_configs:
        func = configure_callable(
            id=plotter["id"],
            import_path=plotter["function"],
            params=plotter.get("params", {}),
            context={}
        )
        plotting_functions.append(func)

    prep.plot_results(
        data=feature_df,
        plotting_functions=plotting_functions,
    )

    print(Hello)

    

if __name__ == "__main__":
    pipeline_args = parse_yaml_config()
    run_bioanalysis(pipeline_args)