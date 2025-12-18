import importlib
import os
import glob

import logging
logger = logging.getLogger(__name__)

def get_object_from_path(path):
    
    try:
        module_path, object_name = path.rsplit(".", 1)
        
        module = importlib.import_module(module_path)
        obj = getattr(module, object_name)

        return obj
    
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load object '{path}'. Error: {e}")
    

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