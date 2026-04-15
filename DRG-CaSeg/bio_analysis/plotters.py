import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as sts
import scipy.signal as signal

import helpers

from matplotlib.ticker import PercentFormatter


#define a global color palette to use for all plots
cb_colors = sns.color_palette("colorblind")
HUE_COLORS = {
    "Control": cb_colors[0],
    "Cytokine": cb_colors[1],
}
PLOTTING_ORDER = [
    "Control",
    "Cytokine",
]

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
    plt.show()


def plot_responder_proportions(
    df: pd.DataFrame,
    ):
    # 1. Melt the dataframe to 'long' format
    # Adjust 'id_vars' to include any other identifier columns you need
    df_reset = df.reset_index(level="sample_id")

    df_long = pd.melt(
        df_reset,
        id_vars=["sample_id", "group"], 
        value_vars=["stimulus_capsaicin_100nM_has_peak", "stimulus_KCl_50mM_has_peak", "stimulus_KCl_75mM_has_peak"],
        var_name="stimulus",
        value_name="responded"
    )

    # 3. Ensure the boolean column is treated as numeric (True=1.0, False=0.0)
    df_long["responded"] = df_long["responded"].astype(float)

    # 4. Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="lightgray", zorder=0)

    # Seaborn barplot automatically calculates the mean (which equals the % of responders)
    sns.barplot(
        data=df_long,
        x="stimulus",
        y="responded",
        hue="group",
        errorbar=("se"),
        capsize=0.1,
        err_kws={"linewidth": 1.5, "color": "black"},
        ax=ax,
        zorder=2,
        palette=HUE_COLORS,
    )

    # 5. Formatting aesthetics
    ax.set_xlabel("Stimulus Condition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Proportion of Responders", fontsize=12, fontweight="bold")
    
    # Format the y-axis to show percentages rather than decimals
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.05) # Add a little headroom above 100% for error bars

    plt.legend(title="Experimental Group")
    plt.tight_layout()
    plt.show()



def plot_peak_heights_per_stimulus(
    df: pd.DataFrame
    ):
    df_reset = df.reset_index(level="sample_id")

    df_long = pd.melt(
        df_reset,
        id_vars=["sample_id", "group"], 
        value_vars=["stimulus_capsaicin_100nM_max_gradient",
                    "stimulus_KCl_50mM_max_gradient", 
                    "stimulus_KCl_75mM_max_gradient"
                    ],
        var_name="peak_group",
        value_name="peak_height"
    )

    df_clean = df_long.dropna(subset=["peak_height"])

    # Initialize a figure to ensure enough space
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="silver", zorder=0)

    # 1. The Violin Plot (replaces pointplot)
    sns.violinplot(
        data=df_clean, 
        x="peak_group", 
        y="peak_height", 
        hue="group", 
        dodge=True,          # Side-by-side groups
        inner=None,          # Removes quartiles for the clean look
        fill=False,          # Outline only
        linewidth=1.0,
        ax=ax,
        zorder=1,
        palette=HUE_COLORS,
        cut=0.0              # Don't extend density past data limits
    )

    # 2. The Individual Data Points
    sns.stripplot(
        data=df_clean,
        x="peak_group",
        y="peak_height",
        hue="group",
        dodge=True,          # Aligns points with their specific violin
        alpha=0.5,
        jitter=True,
        ax=ax,
        zorder=2,            # On top of violin
        palette=HUE_COLORS,
        legend=False         # No duplicate legends
    )

    # 3. Highlight the Mean
    sns.pointplot(
        data=df_clean,
        x="peak_group",
        y="peak_height",
        hue="group",
        dodge=0.54,          # Manual tweak if diamonds aren't centered; use True first
        estimator="mean",
        errorbar=None,
        color="black",       # Contrast mean marker
        markers="D",         # Diamond shape
        linestyle="none",    # No connecting lines
        markersize=6,
        ax=ax,
        zorder=3,            # On top of points
        legend=False
    )
    

    ax.set_xlabel("Stimulus Condition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Maximum Peak Height", fontsize=12, fontweight="bold")

    # Clean up the legend (stripplot + violinplot duplicates legend entries)
    handles, labels = ax.get_legend_handles_labels()
    n_groups = len(df_clean["group"].unique())
    ax.legend(handles[:n_groups], labels[:n_groups], title="Group")

    plt.tight_layout()
    plt.show()


def plot_peak_heights_per_selection(
    df: pd.DataFrame
    ):
    
    # 1. Filter out rows where "has_stimulus_1" is True
    df_filtered = df[~df["stimulus_1_has_peak"]]

    # 2. Update value_vars to only include stimulus 2 and 3
    df_long = pd.melt(
        df_filtered,
        id_vars=["sample_id", "group"], 
        value_vars=["stimulus_2_max_height", "stimulus_3_max_height"],
        var_name="peak_group",
        value_name="peak_height"
    )

    custom_labels = {
        "stimulus_2_max_height": "KCl 50 nM",
        "stimulus_3_max_height": "KCl 75 nM"
    }
    df_long["peak_group"] = df_long["peak_group"].map(custom_labels)

    df_clean = df_long.dropna(subset=["peak_height"])

    # Initialize a figure to ensure enough space
    fig, ax = plt.subplots(figsize=(10, 6))
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
        palette="colorblind",
    )
    sns.stripplot(
        data=df_clean, 
        x="peak_group", 
        y="peak_height", 
        hue="group", 
        alpha=0.6,
        dodge=HUE_COLORS,
        ax=ax,
        zorder=1,
        palette=HUE_COLORS,
    )

    ax.set_xlabel("Stimulus Condition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Maximum Peak Height", fontsize=12, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    n_groups = len(df_clean["group"].unique())
    ax.legend(handles[:n_groups], labels[:n_groups], title="Group")

    plt.tight_layout()
    plt.show()


def plot_max_gradient(
    df: pd.DataFrame
    ):
    df_reset = df.reset_index(level="sample_id")

    df_long = pd.melt(
        df_reset,
        id_vars=["sample_id", "group"], 
        value_vars=["stimulus_capsaicin_100nM_max_gradient", "stimulus_KCl_50mM_max_gradient", "stimulus_KCl_75mM_max_gradient"],
        var_name="peak_group",
        value_name="max_gradient"
    )

    df_clean = df_long.dropna(subset=["max_gradient"])

    # Initialize a figure to ensure enough space
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="silver", zorder=0)

    # 1. The Violin Plot (replaces pointplot)
    sns.violinplot(
        data=df_clean, 
        x="peak_group", 
        y="max_gradient", 
        hue="group", 
        dodge=True,          # Ensures control/treatment violins sit side-by-side
        inner="quartile",    # Draws dashed lines at the 25th, 50th, and 75th percentiles
        linewidth=1.0,
        ax=ax,
        zorder=1,
        palette=HUE_COLORS,
        cut=0.0,
    )
    

    ax.set_xlabel("Stimulus Condition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Maximum Gradient", fontsize=12, fontweight="bold")

    # Clean up the legend (stripplot + violinplot duplicates legend entries)
    handles, labels = ax.get_legend_handles_labels()
    n_groups = len(df_clean["group"].unique())
    ax.legend(handles[:n_groups], labels[:n_groups], title="Group")

    plt.tight_layout()
    plt.show()


def plot_sample_correlations(df: pd.DataFrame):
    """
    Collapses MultiIndex trace data to the sample level, melts it,
    and plots the sample correlations as violins with standard error bars.
    Uses the dynamically generated dictionary keys for the x-axis.
    """
    # "1. Identify the sample-level correlation columns dynamically"
    # "Matches the 'sample_corr_' prefix we set in the calculation function"
    sync_cols = [col for col in df.columns if "corr" in col]
    cols_to_collapse = ["group"] + sync_cols

    # "2. Collapse MultiIndex down to Sample-Level"
    df_sample = df.groupby(level="sample_id")[cols_to_collapse].agg(helpers.strict_collapse).reset_index()

    # "3. Melt the dataframe into long format for Seaborn"
    df_long = pd.melt(
        df_sample,
        id_vars=["sample_id", "group"], 
        value_vars=sync_cols,
        var_name="stimulus_name",
        value_name="correlation_value"
    )

    # "4. Clean up the labels to show just your custom dictionary key"
    # "Removes the prefix so only the key (e.g., 'capsaicin' or 'global') remains"
    df_long["stimulus_name"] = df_long["stimulus_name"].str.replace("sample_corr_", "", regex=False)
    df_long["stimulus_name"] = df_long["stimulus_name"].str.replace("stimulus_corr_", "", regex=False)
    
    # "Format the keys for a cleaner plot (e.g., 'stimulus_1' -> 'Stimulus 1')"
    df_long["stimulus_name"] = df_long["stimulus_name"].str.replace("_", " ", regex=False).str.title()

    df_clean = df_long.dropna(subset=["correlation_value"])

    # "5. Initialize figure"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="silver", zorder=0)

    # "6. The Violin Plot (Distributions and Quartiles)"
    sns.violinplot(
        data=df_clean, 
        x="stimulus_name", 
        y="correlation_value", 
        hue="group", 
        dodge=True,          
        inner="quartile",    
        linewidth=1.0,
        cut=0,
        ax=ax,
        zorder=1,
        palette=HUE_COLORS,
    )

    ax.set_xlabel("Stimulus Region", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Sample Correlation", fontsize=12, fontweight="bold")

    # "8. Clean up the legend"
    handles, labels = ax.get_legend_handles_labels()
    n_groups = len(df_clean["group"].dropna().unique())
    ax.legend(handles[:n_groups], labels[:n_groups], title="Group")

    plt.tight_layout()
    plt.show()


def plot_peak_jitter(
    df: pd.DataFrame,
    time_point: str,
    ):
    
    time_cols = [
        f"stimulus_capsaicin_100nM_{time_point}_time",
        f"stimulus_KCl_50mM_{time_point}_time",
        f"stimulus_KCl_75mM_{time_point}_time"
    ]

    # "1. Group by Sample and calculate the standard deviation of peak times"
    # "Calculates the jitter per sample. If a sample has 1 or 0 peaks, std() safely returns NaN"
    df_jitter = df.groupby(level="sample_id")[time_cols].std()
    
    # "2. Re-attach the group labels (Control vs Treatment)"
    df_jitter["group"] = df.groupby(level="sample_id")["group"].first()
    
    # "Reset index so sample_id is a standard column for melting"
    df_jitter = df_jitter.reset_index()

    # "3. Melt the dataframe to 'long' format"
    df_long = pd.melt(
        df_jitter,
        id_vars=["sample_id", "group"], 
        value_vars=time_cols,
        var_name="stimulus",
        value_name="peak_jitter_std"
    )

    # "Clean up the x-axis labels dynamically"
    df_long["stimulus"] = (
        df_long["stimulus"]
        .str.replace("stimulus_", "")
        .str.replace("_peak_time", "")
        .str.replace("_", " ")
    )

    # "Drop NaNs to avoid plotting errors for samples lacking enough peaks"
    df_clean = df_long.dropna(subset=["peak_jitter_std"])

    # "4. Create the plot"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="lightgray", zorder=0)

    # "Extract unique categories to control plotting order and text placement"
    stimuli_order = df_clean["stimulus"].unique()
    groups_order = df_clean["group"].unique()

    # "5. The Violin Plot"
    sns.violinplot(
        data=df_clean,
        x="stimulus",
        y="peak_jitter_std",
        hue="group",
        order=stimuli_order,
        hue_order=groups_order,
        dodge=True,
        inner="quartile",
        linewidth=1.0,
        ax=ax,
        zorder=1,
        palette=HUE_COLORS,
    )

    # "6. Add sample size (n) annotations"
    n_groups = len(groups_order)
    
    # "Calculate x-axis dodge offsets based on Seaborn's default geometry (width=0.8)"
    dodge_width = 0.8
    offsets = np.linspace(0, dodge_width - dodge_width / n_groups, n_groups)
    offsets -= offsets.mean()

    # "Determine a y-position slightly above the maximum data point for the labels"
    y_pos = df_clean["peak_jitter_std"].max() * 1.05

    for i, stim in enumerate(stimuli_order):
        for j, grp in enumerate(groups_order):
            # "Count the number of non-NaN datapoints for this specific violin"
            subset = df_clean[(df_clean["stimulus"] == stim) & (df_clean["group"] == grp)]
            count = len(subset)
            
            # "Calculate exact x-coordinate"
            x_pos = i + offsets[j]
            
            # "Place the text annotation"
            ax.text(
                x=x_pos,
                y=y_pos,
                s=f"n={count}",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=10,
                color="black"
            )

    # "7. Formatting aesthetics"
    ax.set_xlabel("Stimulus Condition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Peak Jitter (Standard Deviation in Seconds)", fontsize=12, fontweight="bold")
    
    # "Clean up the legend to prevent duplication from the violinplot"
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:n_groups], labels[:n_groups], title="Experimental Group")

    # "Expand the upper y-limit slightly so the text doesn't get cut off"
    ax.set_ylim(bottom=ax.get_ylim()[0], top=y_pos * 1.1)
    ax.set_title(f"Standard deviation of {time_point}")

    plt.tight_layout()
    plt.show()


def plot_latency(
    df: pd.DataFrame,
    time_point: str,
    ):
    
    time_cols = [
        f"stimulus_capsaicin_100nM_{time_point}_time",
        f"stimulus_KCl_50mM_{time_point}_time",
        f"stimulus_KCl_75mM_{time_point}_time"
    ]
    
    # "1. Extract the raw data without aggregating"
    # "Copy only the necessary columns to avoid melting a massive dataframe"
    cols_to_keep = ["group"] + time_cols
    df_raw = df[cols_to_keep].copy()
    
    # "Subtract the specific stimulus onset times to calculate lag/latency"
    df_raw[time_cols[0]] -= 100
    df_raw[time_cols[1]] -= 200
    df_raw[time_cols[2]] -= 300
    
    # "Reset index so sample_id is a standard column for melting"
    df_raw = df_raw.reset_index()

    # "2. Melt the dataframe to long format"
    df_long = pd.melt(
        df_raw,
        id_vars=["sample_id", "group"], 
        value_vars=time_cols,
        var_name="stimulus",
        value_name="gradient_time"
    )

    # "Clean up the x-axis labels dynamically"
    df_long["stimulus"] = (
        df_long["stimulus"]
        .str.replace("stimulus_", "")
        .str.replace("_gradient_time", "")
        .str.replace("_", " ")
    )

    # "Drop NaNs to avoid plotting errors for missing peaks/gradients"
    df_clean = df_long.dropna(subset=["gradient_time"])

    # "3. Create the plot"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color="lightgray", zorder=0)

    # "Extract unique categories to control plotting order and text placement"
    stimuli_order = df_clean["stimulus"].unique()
    groups_order = df_clean["group"].unique()

    # "4. The Violin Plot"
    sns.violinplot(
        data=df_clean,
        x="stimulus",
        y="gradient_time",
        hue="group",
        order=stimuli_order,
        hue_order=groups_order,
        dodge=True,
        inner=None,          # Removes the inner box/quartiles for a clean look
        fill=False,          # Creates the outline
        linewidth=1.0,
        ax=ax,
        zorder=1,
        palette=HUE_COLORS,
        cut=0.0,
    )

# 2. The Individual Data Points
    sns.stripplot(
        data=df_clean,
        x="stimulus",
        y="gradient_time",
        hue="group",
        order=stimuli_order,
        hue_order=groups_order,
        dodge=True,          # Aligns the scattered points with the dodged violins
        alpha=0.5,
        jitter=True,
        ax=ax,
        zorder=2,            # Draws on top of the violin outline
        palette=HUE_COLORS,
        legend=False,        # Prevents adding a duplicate legend
    )

# 3. Highlight the Mean
    sns.pointplot(
        data=df_clean,
        x="stimulus",
        y="gradient_time",
        hue="group",
        order=stimuli_order,
        hue_order=groups_order,
        dodge=True,          # Aligns the mean marker with the respective group
        estimator="mean",
        errorbar=None,
        color="black",       # Forces the mean marker to be black to stand out
        markers="D",         # Diamond shaped marker
        linestyle="none",    # Prevents drawing connecting lines between categories
        markersize=6,
        ax=ax,
        zorder=3,            # Draws on top of the individual points
        legend=False,        # Prevents adding a duplicate legend
    )

    # "5. Add sample size (n) annotations"
    n_groups = len(groups_order)
    
    # "Calculate x-axis dodge offsets based on Seaborn's default geometry (width=0.8)"
    dodge_width = 0.8
    offsets = np.linspace(0, dodge_width - dodge_width / n_groups, n_groups)
    offsets -= offsets.mean()

    # "Determine a y-position slightly above the maximum data point for the labels"
    y_pos = df_clean["gradient_time"].max() * 1.05

    for i, stim in enumerate(stimuli_order):
        for j, grp in enumerate(groups_order):
            # "Count the number of non-NaN datapoints for this specific violin"
            subset = df_clean[(df_clean["stimulus"] == stim) & (df_clean["group"] == grp)]
            count = len(subset)
            
            # "Calculate exact x-coordinate"
            x_pos = i + offsets[j]
            
            # "Place the text annotation"
            ax.text(
                x=x_pos,
                y=y_pos,
                s=f"n={count}",
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=10,
                color="black"
            )

    # "6. Formatting aesthetics"
    ax.set_xlabel("Stimulus Condition", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latency after Stimulus onset (s)", fontsize=12, fontweight="bold")
    
    # "Clean up the legend to prevent duplication from the violinplot"
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:n_groups], labels[:n_groups], title="Experimental Group")

    # "Expand the upper y-limit slightly so the text doesn't get cut off"
    ax.set_ylim(bottom=ax.get_ylim()[0], top=y_pos * 1.1)

    ax.set_title(f"Latency of {time_point}")

    plt.tight_layout()
    plt.show()


def visualize_random_peaks(
    time_traces: np.ndarray,
    height_threshold: float,
    prominence: float | None,
    peak_snr: float | None,
    distance: int | None,
    width: int | None,
    fs: float,
    n_samples: int = 12
    ) -> None:
    """
    Randomly samples time traces and plots them with detected peaks.
    Displays the detection parameters in the figure title.
    """
    n_traces = time_traces.shape[0]
    sample_indices = np.random.choice(n_traces, size=min(n_samples, n_traces), replace=False)
    
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
        
        target_prominence = helpers.get_rolling_prominence(trace, peak_snr) if peak_snr else prominence
        peaks, _ = signal.find_peaks(
            trace,
            height=height_threshold,
            prominence=target_prominence,
            distance=distance,
            width=width,
        )
        
        ax.plot(time_axis, trace, color="steelblue", label="Raw Trace")
        if len(peaks) > 0:
            ax.plot(time_axis[peaks], trace[peaks], "X", color="red", markersize=8, label="Detected Peaks")

        #see whether statistical metrics ran refine the predicitons
        kurt = sts.kurtosis(trace)
        skew = sts.skew(trace)

        ax.set_title(f"Trace Index: {idx} | Kurtosis: {kurt:.2f} | Skewness: {skew:.2f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        
    for j in range(len(sample_indices), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()