import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from scipy.signal import find_peaks

import helpers


def plot_feature_boxplot(
    df: pd.DataFrame, 
    feature_column: str, 
    group_column: str = "group",
    title: str = "",
    ):
    """
    Plots a boxplot comparing a specific biological feature across groups.
    """
    # 1. Set the visual style for publication-ready plots
    sns.set_theme(style="whitegrid")
    
    # 2. Create the figure explicitly so we can control the size
    plt.figure(figsize=(8, 6))
    
    #The Seaborn magic: tell it the dataframe, the x-axis, and the y-axis
    ax = sns.boxplot(
        data=df, 
        x=group_column, 
        y=feature_column,
        palette="Set2"  # A nice, colorblind-friendly color palette
    )
    
    # 4. Add points on top of the boxplot to show the actual data distribution
    # This is highly recommended in biological sciences!
    sns.stripplot(
        data=df, 
        x=group_column, 
        y=feature_column,
        color="black", 
        alpha=0.5, 
        size=4,
        jitter=True
    )
    
    # 5. Clean up the labels
    if not title:
        title = f"Distribution of {feature_column} by {group_column}"
        
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel(group_column, fontsize=12)
    plt.ylabel(feature_column, fontsize=12)
    
    # 6. Adjust layout and display
    plt.tight_layout()
    plt.savefig("/home/jaschneider/projects/DRG-CaSeg/thesis_plots/peak_boxplot.png", dpi=300)


def plot_discrete_peak_counts(
    df: pd.DataFrame, 
    feature_column: str, 
    group_column: str = "group",
    title: str = "Proportion of Traces with >1 Peak"
    ):
    """
    Visualizes the proportion of active traces that have more than 1 peak.
    """
    # 1. Create the new binned categories
    plot_df = df[df[feature_column] > 0].copy()
    conditions = [
        plot_df[feature_column] == 1,
        plot_df[feature_column] > 1
    ]
    choices = ["1", ">1"]
    
    plot_df["peak_category"] = np.select(conditions, choices)
    
    # 2. Calculate the exact proportions for each group
    counts = plot_df.groupby([group_column, "peak_category"], observed=False).size().reset_index(name="count")
    totals = plot_df.groupby(group_column).size().reset_index(name="total")
    
    merged_df = pd.merge(counts, totals, on=group_column)
    merged_df["proportion"] = merged_df["count"] / merged_df["total"]

    # Filter to only keep the ">1" category
    final_plot_df = merged_df[merged_df["peak_category"] == ">1"].copy()
    
    # 3. Set the visual style and figure
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    # 4. Plot using barplot: Put the GROUP on the X-axis!
    ax = sns.barplot(
        data=final_plot_df,
        x=group_column,
        y="proportion",
        hue=group_column,
        palette="Set2"
    )
    
    # 5. Lock Y-axis between 0 and 1
    ax.set_ylim(0, 0.5)
    
    # 6. Add the percentage numbers on top of the bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)
        
    # 7. Add the sample size text box
    total_samples = len(plot_df)
    box_style = dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.9)
    ax.text(
        0.95, 0.95, 
        f"Active Samples: {total_samples}",
        transform=ax.transAxes, 
        ha="right", 
        va="top",
        bbox=box_style,
        fontsize=10
    )
    
    # 8. Clean up labels
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Treatment Group", fontsize=12)
    plt.ylabel("Proportion of Neurons", fontsize=12)
    
    # 9. Adjust layout and display/save
    plt.tight_layout()
    plt.savefig("/home/jaschneider/projects/DRG-CaSeg/thesis_plots/peak_barchart_ica.png", dpi=300)


def plot_all_peak_counts(
    df: pd.DataFrame,
    feature_column: str,
    group_column: str = "group",
    title: str = "Peak Count Distribution"
    ):
    """
    Visualizes the distribution of discrete integer counts across groups,
    explicitly excluding any traces that have exactly 0 peaks.
    """
    # Create a new DataFrame containing only the active traces
    active_df = df[df[feature_column] > 0]


    # 1. Set the visual style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))   

    # 2. Plot the discrete histogram on the filtered data
    ax = sns.histplot(
        data=active_df,
        x=feature_column,
        hue=group_column,
        multiple="dodge",    
        discrete=True,        
        stat="percent",      
        common_norm=False,    
        palette="Set2",
        shrink=0.8            
    )

    # 3. Clean up the labels to reflect the filtered data
    if not title:
        title = f"Distribution of {feature_column} by {group_column} (Excluding 0s)"

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Number of Peaks (Active Traces Only)", fontsize=12)
    plt.ylabel("Percentage of Traces (%)", fontsize=12)

    # 4. Force the X-axis to only show clean integers present in the filtered data
    plt.xticks(sorted(active_df[feature_column].unique()))

    # 5. Adjust layout and display
    plt.tight_layout()
    plt.savefig("/home/jaschneider/projects/DRG-CaSeg/thesis_plots/peak_barchart_all.png", dpi=300)


def plot_peak_heights_per_stimulus(
    df: pd.DataFrame
    ):
    df_long = pd.melt(
        df,
        id_vars=["sample_id", "group"], 
        value_vars=["stimulus_1_max_height", "stimulus_2_max_height", "stimulus_3_max_height"],
        var_name="peak_group",
        value_name="peak_height"
    )

    custom_labels = {
        "stimulus_1_max_height": "Capsaicin 100 nM",
        "stimulus_2_max_height": "KCl 50 nM",
        "stimulus_3_max_height": "KCl 75 nM"
    }
    df_long["peak_group"] = df_long["peak_group"].map(custom_labels)

    df_clean = df_long.dropna(subset=["peak_height"])

    # Initialize a figure to ensure enough space
    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="lightgray", zorder=0)

    dodge_width = 0.4
    sns.pointplot(
        data=df_clean, 
        x="peak_group", 
        y="peak_height", 
        hue="group", 
        errorbar=("ci", 95), 
        capsize=0.1, 
        dodge=dodge_width,
        linestyles="", 
        err_kws={"linewidth": 1.0, "color": "black"},
        ax=ax,
        zorder=2,
    )
    sns.stripplot(
        data=df_clean, 
        x="peak_group", 
        y="peak_height", 
        hue="group", 
        alpha=0.6,
        dodge=dodge_width,
        ax=ax,
        zorder=1
    )

    ax.set_xlabel("Stimulus Condition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Maximum Peak Height", fontsize=12, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    n_groups = len(df_clean["group"].unique())
    ax.legend(handles[:n_groups], labels[:n_groups], title="Group")

    plt.tight_layout()


    plt.show()


def visualize_random_peaks(
    time_traces: np.ndarray,
    height_threshold: float,
    prominence: float,
    distance: int,
    fs: float,
    n_samples: int = 12
    ) -> None:
    """
    Randomly samples time traces and plots them with detected peaks.
    Displays the detection parameters in the figure title.
    """
    n_traces = time_traces.shape[0]
    sample_indices = np.random.choice(n_traces, size=n_samples, replace=False)
    
    cols = 4
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
    axes = axes.flatten()
    
    param_text = (
        f"Peak Detection Parameters | "
        f"Height: {height_threshold} | "
        f"Prominence: {prominence} | "
        f"Distance: {distance}"
    )
    fig.suptitle(param_text, fontsize=16, fontweight="bold", y=1.02)
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        trace = time_traces[idx]
        time_axis = np.arange(len(trace)) / fs
        
        peaks, _ = find_peaks(
            trace,
            height=height_threshold,
            prominence=helpers.get_dynamic_prominence(trace, 7.5),
            distance=distance
        )
        
        ax.plot(time_axis, trace, color="steelblue", label="Raw Trace")
        if len(peaks) > 0:
            ax.plot(time_axis[peaks], trace[peaks], "X", color="red", markersize=8, label="Detected Peaks")
            
        ax.set_title(f"Trace Index: {idx} | Peaks: {len(peaks)}", fontsize=10)
        ax.grid(True, alpha=0.3)
        
    for j in range(len(sample_indices), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()