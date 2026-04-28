import os
import yaml
import numpy as np

def extract_metrics_recursively(data_dict, target_keys, collected_dict):
    """
    Recursively searches a nested dictionary for specific keys 
    and appends their values to the collected_dict.
    """
    found_data = False
    
    if isinstance(data_dict, dict):
        for key, value in data_dict.items():
            # If we find one of our target metrics, append it
            if key in target_keys:
                # Ensure the value is a number and not another nested dictionary
                if isinstance(value, (int, float)):
                    collected_dict[key].append(value)
                    found_data = True
            # If the value is another dictionary, search inside it recursively
            elif isinstance(value, dict):
                if extract_metrics_recursively(value, target_keys, collected_dict):
                    found_data = True
                    
    return found_data

def calculate_overall_metrics(base_folder, target_string, metrics_folder_name):
    # Dictionary to collect all values from the subfolders (kept flat!)
    collected_metrics = {
        "recall": [],
        "precision": [],
        "f1_score": [],
        "mean_iou": [],
        "mean_correlation": []
    }
    
    # Initialize a counter for the number of processed files
    processed_files_count = 0
    
    # List to store the filepaths to be processed
    yaml_files_to_process = []

    # Phase 1: Iterate through all items in the base folder and collect filepaths
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        
        # Check if it is a directory and contains the target string
        if os.path.isdir(folder_path) and target_string in folder_name:
            yaml_path = os.path.join(folder_path, f"{metrics_folder_name}.yaml")
            
            # Check if the metrics file exists in this subfolder
            if os.path.isfile(yaml_path):
                yaml_files_to_process.append(yaml_path)

    # Phase 2: Iterate through the collected filepaths and extract data
    for yaml_path in yaml_files_to_process:
        with open(yaml_path, "r") as file:
            try:
                data = yaml.safe_load(file)
                
                # If the file is not empty, extract the requested values using recursion
                if data:
                    file_had_data = extract_metrics_recursively(data, collected_metrics.keys(), collected_metrics)
                            
                    # Only increment if we actually found and extracted metrics from this file
                    if file_had_data:
                        processed_files_count += 1
                        
            except yaml.YAMLError as error:
                print(f"Error parsing {yaml_path}: {error}")

    # Calculate mean and standard deviation for each metric using numpy
    final_results = {}
    for metric_name, values in collected_metrics.items():
        if values:
            # We cast to float() because PyYAML cannot cleanly serialize numpy data types
            final_results[f"{metric_name}_mean"] = float(np.mean(values))
            
            # ddof=1 calculates the sample standard deviation (matching the old statistics behavior)
            if len(values) > 1:
                final_results[f"{metric_name}_std"] = float(np.std(values, ddof=1))
            else:
                final_results[f"{metric_name}_std"] = 0.0
        else:
            final_results[f"{metric_name}_mean"] = None
            final_results[f"{metric_name}_std"] = None

    # Add the requested N= value indicating the number of processed files
    final_results["N="] = processed_files_count

    # Define the output path in the original base folder
    output_path = os.path.join(base_folder, f"{target_string}_overall_metrics.yaml")
    
    # Save the aggregated results to the new YAML file
    with open(output_path, "w") as output_file:
        yaml.dump(final_results, output_file, default_flow_style=False)
        
    print(f"Successfully processed {processed_files_count} folders and saved results to {output_path}")

if __name__ == "__main__":
    # You can change these variables to match your specific paths and strings
    target_folder_path = "/home/jaschneider/projects/DRG-CaSeg/results/hyperparam_opt_mu"
    search_string = "ica_mukamel_mu_1.0"
    metrics_folder_name = "metrics"
    
    calculate_overall_metrics(target_folder_path, search_string, metrics_folder_name)