import argparse
import os

from infra.utils import get_config_files
from infra.experiment import Experiment

import logging
from config import setup_logging
from utils import seed_everything

setup_logging()
logger = logging.getLogger(__name__)


def main():
    SEED = int(os.getenv("GLOBAL_SEED", 42))
    seed_everything(SEED)

    parser = argparse.ArgumentParser(description="Run batch experiments.")
    
    # allow passing multiple files or folders
    parser.add_argument(
        "--configs", 
        nargs="+",
        required=True,
        help="List of yaml files or folders containing yaml files."
    )
    
    # Allow controlling parallelism
    parser.add_argument(
        "--workers", 
        type=int, 
        default=2, 
        help="Number of parallel processes. Use 1 for serial execution."
    )

    args = parser.parse_args()
    
    #Extract all of the experiment files
    configs_to_run = get_config_files(args.configs)
    logger.info(f"Found {len(configs_to_run)} experiments to run.")
    
    #Set up and run the experiments
    for cfg in configs_to_run:
        exp = Experiment.from_yaml(cfg)
        exp.run()

    


if __name__ == "__main__":
    main()