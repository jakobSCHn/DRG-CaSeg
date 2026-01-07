import importlib
import os
import glob
import inspect
import functools
import shutil

from pathlib import Path

import logging
logger = logging.getLogger(__name__)

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
    import_path,
    params,
    ):
    obj_def = get_object_from_path(import_path)
        
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

    return functools.partial(obj_def, **valid_params)



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
            # If it's a folder, grab all .yaml files inside
            found = glob.glob(os.path.join(path, "*.yaml"))
            yaml_files.extend(found)
        elif os.path.isfile(path) and path.endswith(".yaml"):
            # If it's a direct file, add it
            yaml_files.append(path)
        else:
            logger.warning(f"Warning: '{path}' is not a valid YAML file or directory.")
            
    # Remove duplicates
    return sorted(list(set(yaml_files)))


def setup_experiment_folder(
    run_id: str,
    config_path: str,
    ana_id: str | None = None,
    ):

    p = Path(config_path)
    if ana_id:
        output_dir_name = f"{p.stem}_{run_id}_{ana_id}"
    else:
        output_dir_name = f"{p.stem}_{run_id}_{ana_id}"
    output_dir = Path("results") / output_dir_name

    output_dir.mkdir(parents=True)
    shutil.copy(p, output_dir / "config.yaml")

    logger.info(f"Initialized Experiment folder: {p.stem}")
