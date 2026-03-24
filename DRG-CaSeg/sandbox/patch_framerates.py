import numpy as np
import random
from pathlib import Path

#Define the root directory containing all the sample subfolders
TARGET_DIRECTORY = "/home/jaschneider/projects/DRG-CaSeg/results/run_real_data_20260318_174848_patched"

#Map the sampling IDs to their sampling frequencies
FREQUENCY_MAP = {
    "1st_A2": 33.329,
    "1st_A3": 33.329,
    "1st_B1": 33.334,
    "1st_B2": 33.334,
    "1st_B3": 33.322,
    "2nd_A1": 33.329,
    "2nd_A2": 33.334,
    "2nd_B1": 33.331,
    "2nd_B2": 33.334,
    "3rd_A1": 33.322,
    "3rd_A2": 33.326,
    "3rd_B1": 33.322,
    "3rd_B2": 33.329,
    "4th_A3": 33.329,
    "4th_B3": 33.329,
    "5th_A3": 33.326,
    "5th_B3": 33.334,
}

def patch_npz_files(
    base_path: str,
    freq_map: dict
    ) -> None:
    """
    Iterates through subdirectories, finds .npz files, and appends 
    'sampling_frequency' if it is missing, based on the folder name.
    """
    base_dir = Path(base_path)
    
    # rglob("*.npz") recursively finds all .npz files in all subdirectories
    for npz_file in base_dir.rglob("*.npz"):
        
        # Check if any of our known Sample IDs are in the file's directory path
        matched_sample_id = None
        for sample_id in freq_map.keys():
            if sample_id in npz_file.parent.name:
                matched_sample_id = sample_id
                break
                
        # If the file isn't in a folder we recognize, skip it safely
        if not matched_sample_id:
            print(f"Skipping {npz_file.name}: No matching Sample ID found in its path."
                    f"Parent folder is {npz_file.parent.name}"
            )
            continue
            
        expected_frequency = freq_map[matched_sample_id]
        
        # Load the archive
        with np.load(npz_file) as data:
            # If the key is already there, don't waste time rewriting the file
            if "sampling_frequency" in data.files:
                print(f"Already patched: {npz_file.relative_to(base_dir)}")
                continue
                
            # You cannot directly append to an existing .npz file in place.
            # We must unpack all existing arrays into a new dictionary first.
            patched_data = {key: data[key] for key in data.files}
            
        # Add the missing frequency to our unpacked dictionary
        patched_data["sampling_frequency"] = expected_frequency
        
        # Overwrite the original file with the updated, complete data dictionary
        # The ** unpacks the dictionary into keyword arguments for savez_compressed
        np.savez_compressed(npz_file, **patched_data)
        
        print(f"Successfully patched {npz_file.name} with {expected_frequency} Hz")


def verify_patched_frequencies(
    base_path: str,
    n_samples: int = 10,
    ) -> None:
    """
    Randomly selects .npz files from the directory and prints their
    sampling_frequency to verify the patching process worked.
    """
    base_dir = Path(base_path)
    
    #Gather every single .npz file in the directory tree
    all_npz_files = list(base_dir.rglob("*.npz"))
    
    if not all_npz_files:
        print("No .npz files found to verify.")
        return
        
    #Safety check: If you have fewer than 10 files total, just check them all
    n_samples = min(n_samples, len(all_npz_files))
    
    #Randomly grab our target number of files
    selected_files = random.sample(all_npz_files, n_samples)
    
    print(f"\n--- Verifying {n_samples} Randomly Selected Files ---")
    
    #Open each one and check the frequency
    for npz_file in selected_files:
        try:
            with np.load(npz_file) as data:
                sample_name = npz_file.parent.name
                
                if "sampling_frequency" in data.files:
                    # Remember to use .item() to pull the float out of the 0-D array!
                    freq = data["sampling_frequency"].item()
                    print(f"[OK] {sample_name}/{npz_file.name} -> {freq} Hz")
                else:
                    print(f"[MISSING] {sample_name}/{npz_file.name} -> Key not found!")
                    
        except Exception as e:
            print(f"[ERROR] Could not read {npz_file.name}: {e}")
            
    print("---------------------------------------------------")



if __name__ == "__main__":
    print("Starting NPZ patching routine...")
    patch_npz_files(TARGET_DIRECTORY, FREQUENCY_MAP)
    print("Patching complete!")

    verify_patched_frequencies(TARGET_DIRECTORY, n_samples=10)