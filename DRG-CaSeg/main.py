import argparse
import os
import logging

from config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
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
        default=1, 
        help="Number of workers. Set to -1 to use all available cores."
    )

    args = parser.parse_args()

    # 2. Handle the logic after parsing
    if args.workers == -1:
        n_processes = max(1, (os.cpu_count() or 1) - 1)
    else:
        # User requested specific number
        n_processes = args.workers
    
    if n_processes < 2:
        backend = "single"
    else:
        backend = "multiprocessing"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"

    logger.info(f"Running Experiments with {n_processes} parallel processes")

    #Lazy imports to avoid automatic initialization of environment variables
    from infra.utils import get_config_files, setup_cluster
    from utils import seed_everything
    from infra.experiment import Experiment



    #Start a cluster with a pool of worker processes
    
    runtime_context = setup_cluster(
        backend=backend,
        n_processes=n_processes,
        ignore_preexisting=False
    )

    SEED = int(os.getenv("GLOBAL_SEED", 42))
    seed_everything(SEED)

    #Extract all of the experiment files
    configs_to_run = get_config_files(args.configs)
    logger.info(f"Found {len(configs_to_run)} experiments to run.")
    
    #Set up and run the experiments
    for cfg in configs_to_run:
        exp = Experiment.from_yaml(cfg)
        exp.run(runtime_context)

    


if __name__ == "__main__":
    main()