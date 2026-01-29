import importlib
import os
import glob
import inspect
import functools
import shutil
import multiprocessing

from pathlib import Path

import logging
logger = logging.getLogger(__name__)


def setup_cluster(
    backend: str = "multiprocessing",
    n_processes: int = None,
    ignore_preexisting: bool = False,
    maxtasksperchild: int = None
    ):

    if n_processes is None:
        n_processes = max(int(psutil.cpu_count() - 1), 1) 

    if backend == "multiprocessing":
        if len(multiprocessing.active_children()) > 0:
            if ignore_preexisting:
                logger.warning("Found an existing multiprocessing pool. "
                               "This is often indicative of an already-running CaImAn cluster. "
                               "You have configured the cluster setup to not raise an exception.")
            else:
                raise Exception(
                    "A cluster is already running. Terminate with dview.terminate() if you want to restart.")
        
        dview = multiprocessing.Pool(n_processes, maxtasksperchild=maxtasksperchild)

    elif backend == "single":
        dview = None
        n_processes = 1

    else:
        raise Exception("Unknown Backend")

    return {
        "cluster": dview,
        "n_processes": n_processes
    }


def get_object_from_path(
    import_path
    ):
    
    try:
        module_path, object_name = import_path.rsplit(".", 1)
        
        module = importlib.import_module(module_path)
        obj = getattr(module, object_name)

        return obj
    
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load object '{import_path}'. Error: {e}")


def configure_callable(
    id,
    import_path,
    params,
    context: dict = None,
    ):
    obj_def = get_object_from_path(import_path)
    
    params["id"] = id
    context = context or {}

    sig = inspect.signature(obj_def)
    has_kwargs = any(param.kind ==param.VAR_KEYWORD for param in sig.parameters.values())

    if has_kwargs:
        valid_params = params
    else:
        valid_params = {
            k: v for k, v in params.items()
            if k in sig.parameters
        }
        dropped_keys = set(params.keys()) - set(valid_params.keys())
        if dropped_keys:
            logger.warning(f"Dropped params for {import_path}: {dropped_keys}")

    if has_kwargs:
        valid_context = context
    else:
        #Only keep context variables (cluster, n_processes) if the function explicitly asks for them
        valid_context = {
            k: v for k, v in context.items()
            if k in sig.parameters
        }

    kwargs = {**valid_params, **valid_context}

    return functools.partial(obj_def, **kwargs)


def get_config_files(
    inputs: list[str] | set[str]
    ):
    """
    Parses the input arguments to resolve all YAML files.
    'inputs' can be a list containing file paths or directory paths.
    """
    yaml_files = []
    for path in inputs:
        if os.path.isdir(path):
            #If it's a folder, grab all .yaml files inside
            found = glob.glob(os.path.join(path, "*.yaml"))
            yaml_files.extend(found)
        elif os.path.isfile(path) and path.endswith(".yaml"):
            #If it's a direct file, add it
            yaml_files.append(path)
        else:
            logger.warning(f"Warning: '{path}' is not a valid YAML file or directory.")
            
    #Remove duplicates
    return sorted(list(set(yaml_files)))


def setup_experiment_folder(
    experiment_name: str,
    run_id: str,
    config_path: str,
    data_id: str,
    ana_id: str | None = None,
    ):

    p = Path(config_path)
    project_root = Path(__file__).resolve().parents[1]
    results_base_dir = project_root / "results"

    if ana_id:
        output_dir_name = f"{p.stem}_{run_id}_{data_id}_{ana_id}"
    else:
        output_dir_name = f"{p.stem}_{run_id}_{data_id}"
    output_dir = results_base_dir / f"{experiment_name}_{run_id}" / output_dir_name

    output_dir.mkdir(parents=True)
    shutil.copy(p, output_dir / "config.yaml")

    logger.info(f"Initialized Experiment folder: {output_dir}")

    return output_dir