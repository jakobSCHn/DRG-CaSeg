import os
import random
import numpy as np
import yaml
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


def seed_everything(
    seed: int,
    ):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Global random seed set to {seed}")


def save_dict_to_yaml(
    data: dict,
    save_path: Path,
    ):
    """
    Saves a dictionary to a YAML file, handling pathlib.Path objects
    and converting common Numpy types to native Python types.
    """
    #Helper to convert numpy types for YAML safety
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    #Recursively clean the dictionary
    def clean_dict(d):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, dict):
                new_d[k] = clean_dict(v)
            else:
                new_d[k] = convert_numpy(v)
        return new_d

    sanitized_data = clean_dict(data)

    with open(save_path, "w") as f:
        yaml.dump(sanitized_data, f, sort_keys=False, default_flow_style=False)