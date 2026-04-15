library(ggplot2)
library(tidyr)
library(dplyr)
library(lmerTest)
library(ggpubr)
library(stringr)

# Define the reusable function
generate_feature_plot <- function(df, feature_name, save_path = NULL) {
  
  # --- 1. Dynamically Construct Column Names ---
  col_cap <- paste0("stimulus_capsaicin_100nM_", feature_name)
  col_kcl_low <- paste0("stimulus_KCl_50mM_", feature_name)
  col_kcl_high <- paste0("stimulus_KCl_75mM_", feature_name)
  
  # Check if columns exist in the dataframe to prevent confusing errors
  req_cols <- c(col_cap, col_kcl_low, col_kcl_high)
  if(!all(req_cols %in% names(df))) {
    stop("One or more generated column names were not found in the dataframe.")
  }
  
  # --- 1.5 CONDITIONAL TIME SHIFT ---
  # If "time" is anywhere in the feature name (ignoring uppercase/lowercase), 
  # subtract the specific baselines from the raw dataframe columns.
  if (grepl("time", feature_name, ignore.case = TRUE)) {
    df[[col_cap]]      <- df[[col_cap]] - 100
    df[[col_kcl_low]]  <- df[[col_kcl_low]] - 200
    df[[col_kcl_high]] <- df[[col_kcl_high]] - 300
  }
  
  # --- 2. Dynamically Run LMMs ---
  model_cap <- lmer(as.formula(paste0(col_cap, " ~ group + (1 | sample_id)")), data = df)
  model_kcl_low <- lmer(as.formula(paste0(col_kcl_low, " ~ group + (1 | sample_id)")), data = df)
  model_kcl_high <- lmer(as.formula(paste0(col_kcl_high, " ~ group + (1 | sample_id)")), data = df)
  
  # --- 3. Prepare the Data ---
  df_long <- df %>%
    pivot_longer(
      cols = all_of(req_cols),
      names_to = "stimulus", values_to = "value"
    ) %>%
    mutate(stimulus = factor(stimulus, 
                             levels = req_cols,
                             labels = c("Capsaicin 100nM", "KCl 50mM", "KCl 75mM"))) %>%
    drop_na(value)
  
  # --- 4. Calculate Sample Sizes (n) ---
  n_data <- df_long %>%
    group_by(stimulus, group) %>%
    summarize(n_count = n(), .groups = "drop") %>%
    mutate(
      y_pos = min(df_long$value, na.rm = TRUE) - 
        (max(df_long$value, na.rm = TRUE) - min(df_long$value, na.rm = TRUE)) * 0.05
    )
  
  # --- 5. Extract P-Values and Calculate Bracket Positions ---
  get_p <- function(model) {
    summary(model)$coefficients[2, "Pr(>|t|)"]
  }
  
  p_values <- data.frame(
    stimulus = c("Capsaicin 100nM", "KCl 50mM", "KCl 75mM"),
    x_min = c(1 - 0.2, 2 - 0.2, 3 - 0.2), 
    x_max = c(1 + 0.2, 2 + 0.2, 3 + 0.2), 
    p_label = c(
      paste0("p = ", round(get_p(model_cap), 3)),
      paste0("p = ", round(get_p(model_kcl_low), 3)),
      paste0("p = ", round(get_p(model_kcl_high), 3))
    ),
    y_pos = max(df_long$value, na.rm = TRUE) * 1.05 
  )
  
  # --- 6. Build the Plot ---
  cb_palette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
  
  clean_title <- str_to_title(gsub("_", " ", feature_name))
  
  p <- ggplot(df_long, aes(x = stimulus, y = value, fill = group)) +
    
    geom_point(aes(color = group), 
               position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8), 
               alpha = 0.6, size = 1.5) +
    
    # Thin dashed black line indicating the mean (with show.legend = FALSE)
    stat_summary(fun = mean, geom = "errorbar", 
                 aes(ymax = after_stat(y), ymin = after_stat(y)), 
                 width = 0.6, linewidth = 0.5, color = "black", linetype = "dashed",
                 position = position_dodge(0.8), show.legend = FALSE) +
    
    # Small black diamond exactly in the middle of the mean line (with show.legend = FALSE)
    stat_summary(fun = mean, geom = "point", 
                 shape = 18, size = 3, color = "black", 
                 position = position_dodge(0.8), show.legend = FALSE) +
                 
    # NEW: Faint filled violin with NO border outlines (inherits fill=group from main ggplot)
    geom_violin(color = NA, position = position_dodge(0.8), alpha = 0.2) +
    
    geom_bracket(
      data = p_values, 
      aes(xmin = x_min, xmax = x_max, label = p_label, y.position = y_pos), 
      inherit.aes = FALSE, tip.length = 0.02, linewidth = 0.6, fontface = "bold"
    ) +
    
    geom_text(
      data = n_data, 
      aes(x = stimulus, y = y_pos, label = paste0("n=", n_count), group = group), 
      inherit.aes = FALSE, position = position_dodge(0.8), size = 3.5, color = "grey30"
    ) +
    
    theme_classic() +
    scale_fill_manual(values = cb_palette) +
    scale_color_manual(values = cb_palette) +
    
    labs(title = "Latency of Peak in Transients in Stimuli-Responsive Neurons",
         y = "Latency [s]", x = "") +
    
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),  
      plot.subtitle = element_text(hjust = 0.5),              
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5), 
      axis.line = element_line(linewidth = 0.8, color = "black"), 
      axis.ticks = element_line(linewidth = 0.8, color = "black") 
    ) +
    coord_cartesian(clip = "off")
  
  # --- 7. Save and Return ---
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = 8, height = 6, dpi = 300)
  }
  
  return(p)
}

df <- read.csv("/home/jaschneider/projects/DRG-CaSeg/DRG-CaSeg/DRG-CaSeg/bio_analysis/trace_features.csv")
df$group <- as.factor(df$group)
df$sample_id <- as.factor(df$sample_id)

feature_name <- "peak_time"
default_path <- "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/r_plots/xxfeaturexx_plot.png"
plot_gradient <- generate_feature_plot(
  df = df, 
  feature_name = feature_name, 
  save_path = str_replace(default_path, "xxfeaturexx", feature_name)
)