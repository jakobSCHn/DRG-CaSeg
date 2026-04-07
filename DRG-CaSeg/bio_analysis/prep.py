import logging
import numpy as np
import pandas as pd
import re
import sys

from pathlib import Path
from typing import Iterator, Callable, Any

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from infra.infra_utils import configure_callable

logger = logging.getLogger(__name__)


def get_metadata_df(
    name: str,
    ):
    
    metadata_df = pd.read_csv(name, sep=",")
    metadata_df["sample_id"] = metadata_df["Calcium_czi_name"].str.replace(".czi", "", regex=False)

    return metadata_df


def find_target_files(
    root_dir: str | Path,
    target_string: str,
    analysis_pattern: str | None,
    ) -> Iterator[Path]:
    """
    Recursively yields .npz file paths that contain the target string.
    """
    root_path = Path(root_dir)
    
    # rglob is highly optimized for recursive directory searching
    search_pattern = f"*{target_string}*.npz"
    
    for file_path in root_path.rglob(search_pattern):
        if analysis_pattern and analysis_pattern not in file_path.parent.name:
            continue
        yield file_path


def extract_data(
    file_path: Path,
    matrix_key: str | list,
    ) -> np.ndarray:
    """
    Safely loads the .npz file and extracts the target matrix.
    """
    matrix_key = [matrix_key] if isinstance(matrix_key, str) else list(matrix_key)

    with np.load(file_path) as data:
        
        extracted_data = []
        for key in matrix_key:
            if key not in data:
                raise KeyError(f"Key {matrix_key} not found in {file_path.name}")
            
            extracted_data.append(data[key].copy())

        return tuple(extracted_data)



def process_files(
    root_dir: str | Path, 
    target_string: str,
    analysis_pattern,
    processing_functions: list,
    matrix_oi: str,
    stimuli_regions: list[tuple],
    ):
    """
    The main coordinator function that ties the workflow together.
    """
    target_files = find_target_files(
        root_dir=root_dir,
        target_string=target_string,
        analysis_pattern=analysis_pattern,
    )

    all_file_dfs = []
    
    for tf in target_files:
        try:
            time_traces, fs = extract_data(tf, [matrix_oi, "sampling_frequency"])
            folder_name = tf.parent.name
            #Extract the sample name from the folder name by looking for its
            #specific pattern
            match = re.search(r"_real_(\d+[a-z]+_[A-Z]\d+)", folder_name)

            if match:
                sample_id = match.group(1)
            else:
                logger.warning(
                    f"Target file path {tf} could not be aligned with a "
                    f"sample ID for biological analysis. Skipping the file."
                )
                continue

            n_traces = time_traces.shape[0]

            file_features = {
                "trace_id": [f"{sample_id}_{i}" for i in range(n_traces)],
                "sample_id": [sample_id] * n_traces,
                "parent_folder": [folder_name] * n_traces,
            }

            for func in processing_functions:
                batch_results = func(
                    data=time_traces,
                    fs=fs.item(),
                    stimuli_regions=stimuli_regions,
                ) 
                file_features.update(batch_results)

            #Create a DataFrame for this specific file and store it
            file_df = pd.DataFrame(file_features)
            all_file_dfs.append(file_df)

        except Exception as e:
            logger.error(f"Error processing {tf.name}: {e}")

    #Concatenate all file-level DataFrames efficiently at the end
    if all_file_dfs:
        global_df = pd.concat(all_file_dfs, ignore_index=True)

        metadata_df = get_metadata_df(name="/home/jaschneider/projects/DRG-CaSeg/DRG-CaSeg/DRG-CaSeg/bio_analysis/sample_metadata.csv")
        group_dict = dict(zip(metadata_df["sample_id"], metadata_df["Group"]))
        global_df["group"] = global_df["sample_id"].map(group_dict)

        global_df.set_index(["sample_id", "trace_id"], inplace=True)
    else:
        global_df = pd.DataFrame()


    return global_df


def plot_results(
    data: pd.DataFrame,
    plotting_functions: list,
    ):

    for func in plotting_functions:
        func(data)