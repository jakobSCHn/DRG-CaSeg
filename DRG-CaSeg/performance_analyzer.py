import re
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import filedialog
from pathlib import Path

def extract_sample_id(folder_name):
    """Extracts the sample ID based on the defined folder structure."""
    syn_match = re.search(r"(syn_\d+)", folder_name)
    
    if not syn_match:
        return "Unknown_ID"
        
    syn_part = syn_match.group(1)
    end_part = folder_name[-3:]
    
    return f"{syn_part}_{end_part}"

def flatten_metrics(item, parent_key="", sep="_"):
    """
    Recursively flattens nested dictionaries and lists of dictionaries.
    Lists of scalars are kept intact as standard lists.
    """
    items = {}
    
    if isinstance(item, dict):
        for k, v in item.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            # Recursive call for dictionary values
            items.update(flatten_metrics(v, new_key, sep=sep))
            
    elif isinstance(item, list):
        # Inspect the list: does it contain any dictionaries?
        contains_dicts = any(isinstance(v, dict) for v in item)
        
        if contains_dicts:
            for i, v in enumerate(item):
                if isinstance(v, dict):
                    # Unpack the dictionary inside the list
                    for k_sub, v_sub in v.items():
                        new_key = f"{parent_key}{sep}{k_sub}" if parent_key else f"{i}{sep}{k_sub}"
                        items.update(flatten_metrics(v_sub, new_key, sep=sep))
                else:
                    # If there is a random scalar mixed in with dicts
                    new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                    items.update({new_key: v})
        else:
            # It is a list of scalars; keep it as a single list
            if parent_key:
                items[parent_key] = item
            else:
                # Fallback if a scalar list is the very first thing in the file
                items["unnamed_list"] = item
                
    else:
        # Base case: we hit a raw value (number, string, boolean)
        items[parent_key] = item
        
    return items

def plot_violin(
    data: pd.DataFrame,
    x,
    y,
    save_path,
    title=None,
    x_label=None,
    y_label=None,
    ):

    data[x] = pd.to_numeric(data[x])
    
    #sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, rc={"grid.linestyle": "--"})
    sns.set_theme(style="darkgrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.violinplot(
        data=data, 
        x=x, 
        y=y,
        hue=x,
        ax=ax,
        palette="colorblind", 
        inner="quart",
        legend=False,
        linewidth=1.5
    )
    
    #sns.despine()
    ax.grid(visible=True, axis="x", color="white")

    ax.set_xlabel(x_label if x_label else"Sample ID", fontweight="bold")
    ax.set_ylabel(y_label if y_label else f"{y}", fontweight="bold")
    ax.set_title(title if title else f"Distribution of {y}", pad=15, fontsize=18, fontweight="bold")
    ax.set_ylim(bottom=0.0, top=1.0)

    first_group = data[x].unique()[0]
    n_count = (data[x] == first_group).sum()
    ax.text(
        0.95, 
        0.05, 
        f"n={n_count}", 
        transform=ax.transAxes, 
        horizontalalignment="right", 
        verticalalignment="bottom", 
        fontsize=10, 
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.4"}
    )
    
    save_path = Path(save_path) / f"{x}_violin_plot.png"

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)

def plot_metric_stability(
        data,
        save_path,
        metric_name="mean_iou",
        group_col="Float_Group",
        y_label=None,
    ):
    """
    Calculates and plots the cumulative mean of a given metric to visualize 
    stability over an increasing sample size.
    """
    if metric_name not in data.columns:
        print(f"Cannot plot stability: The column '{metric_name}' is missing.")
        return

    # 1. Prepare the Data safely
    plot_df = data.copy().sort_index()
    
    # Create a counter (1 to N) for the x-axis, grouping by your parameter
    plot_df["Number_of_Samples"] = plot_df.groupby(group_col).cumcount() + 1
    
    # Calculate the cumulative average for each group
    plot_df["Cumulative_Mean"] = plot_df.groupby(group_col)[metric_name].transform(lambda x: x.expanding().mean())

    # 2. Generate the Plot
    sns.set_theme(style="darkgrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.lineplot(
        data=plot_df,
        x="Number_of_Samples",
        y="Cumulative_Mean",
        hue=group_col,
        palette="colorblind",
        linewidth=1.0,
        ax=ax
    )
    
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color("black")
    ax.tick_params(bottom=True, left=True, color="black", width=1.2)

    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_xlabel("Number of Samples Included", fontweight="bold")
    ax.set_ylabel(f"Cumulative Mean of {y_label if y_label else metric_name}", fontweight="bold")
    ax.set_title(f"Stability of Mean {y_label if y_label else metric_name} across Sample Size", pad=15, fontsize=18, fontweight="bold")
    
    # Format the legend 
    ax.legend(title="\u03bc", loc="best", frameon=True, facecolor="white", framealpha=1.0)
    
    # A bit of path manipulation so it works in batches
    prefix = save_path.name
    new_filename = f"{prefix}_{y_label if y_label else metric_name}_metric_progress_plot.png"
    final_path = save_path.with_name(new_filename)

    plt.tight_layout()
    plt.savefig(final_path, dpi=300)

def main():
    root = tk.Tk()
    root.withdraw()
    
    target_dir = filedialog.askdirectory(title="Select the Parent Directory")
    if not target_dir:
        print("No directory selected. Exiting script.")
        return

    base_path = Path(target_dir)
    folder_name = base_path.name

    extracted_data = []

    print(f"Starting deep search in: {target_dir}")
    files_checked = 0 

    for yaml_file in base_path.rglob("metrics.y*ml"):
        files_checked += 1
        folder_name = yaml_file.parent.name
        sample_id = extract_sample_id(folder_name)
        
        try:
            with open(yaml_file, "r") as file:
                # Using UnsafeLoader to bypass the NumPy tag error
                metrics_raw = yaml.load(file, Loader=yaml.UnsafeLoader)
            
            if metrics_raw:
                # Flatten the entire YAML structure
                flat_metrics = flatten_metrics(metrics_raw)
                
                # Add the Sample ID to the flattened dictionary
                flat_metrics["Sample_ID"] = sample_id
                
                # Append the entire flat dictionary as a single row
                extracted_data.append(flat_metrics)
                
        except Exception as e:
            print(f"Error reading {yaml_file}: {e}")

    print(f"Total YAML files found and checked: {files_checked}")

    if not extracted_data:
        print("No valid data found.")
        return

    # Create DataFrame: Pandas will automatically align matching keys into columns
    # and fill missing metrics for specific samples with NaN
    df = pd.DataFrame(extracted_data)
    df.set_index("Sample_ID", inplace=True)
    df["Param_Group"] = df.index.str.split("_").str[-1]
    df["Data_Sample"] = df.index.str.rsplit("_", n=1).str[0]
    print("\nExtracted Data Preview:")
    print(df.head())

    # Example Plotting Logic (You will need to adjust the 'y' parameter)
    # We will plot the 'spatial_fp' metric as an example
    target_plot_metric = "spatial_fp"
    
    plot_violin(
        data=df,
        x="Param_Group",
        y="spatial_f1_score",
        x_label=r"$\mu$ Group",
        y_label="F1 Score",
        title=r"Effect of $\mu$ on F1 Score Distribution",
        save_path="/home/jaschneider/projects/DRG-CaSeg/thesis_plots/"
    )
    plot_metric_stability(
        data=df,
        metric_name="spatial_f1_score",
        group_col="Param_Group",
        y_label="F1 Score",
        save_path=f"/home/jaschneider/projects/DRG-CaSeg/thesis_plots/{folder_name}",
    )

    fullstop = "Fullstop"


if __name__ == "__main__":
    main()