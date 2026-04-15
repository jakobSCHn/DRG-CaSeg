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
                 
    # MODIFIED: Faint filled violin that calculates its shape by ignoring outliers (1.5x IQR)
    geom_violin(
      data = function(d) {
        d %>%
          group_by(stimulus, group) %>%
          mutate(
            q1 = quantile(value, 0.25, na.rm = TRUE),
            q3 = quantile(value, 0.75, na.rm = TRUE),
            iqr = q3 - q1
          ) %>%
          filter(value >= (q1 - 1.5 * iqr) & value <= (q3 + 1.5 * iqr)) %>%
          ungroup()
      },
      color = NA, position = position_dodge(0.8), alpha = 0.2
    ) +
    
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
    
    labs(title = str_wrap("Maximum Ascending Gradient in Stimulus-Responsive Neurons", width = 60),
         y = "Maximum Gradient [z-score/s]", x = "") +
    
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


plot_sample_correlations <- function(df, save_path = NULL) {
  
  # --- 1. Dynamically Identify Correlation Columns ---
  sync_cols <- grep("corr", names(df), value = TRUE)
  if(length(sync_cols) == 0) {
    stop("No columns containing 'corr' were found in the dataframe.")
  }
  
  # --- 2. Collapse to Sample-Level ---
  # Assuming you want the mean correlation per sample per group
  df_sample <- df %>%
    group_by(sample_id, group) %>%
    summarize(across(all_of(sync_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
  
  # --- 3. Melt to Long Format and Clean Labels ---
  df_long <- df_sample %>%
    pivot_longer(
      cols = all_of(sync_cols),
      names_to = "stimulus_name", 
      values_to = "correlation_value"
    ) %>%
    mutate(
      # Clean prefixes
      stimulus_name = str_replace(stimulus_name, "sample_corr_", ""),
      stimulus_name = str_replace(stimulus_name, "stimulus_corr_", ""),
      # Replace underscores and convert to Title Case
      stimulus_name = str_to_title(str_replace_all(stimulus_name, "_", " "))
    ) %>%
    drop_na(correlation_value) %>%
    # Convert to factor to lock the plotting order
    mutate(stimulus_name = as.factor(stimulus_name))
  
  # --- 4. Calculate Sample Sizes (n) ---
  n_data <- df_long %>%
    group_by(stimulus_name, group) %>%
    summarize(n_count = n(), .groups = "drop") %>%
    mutate(
      y_pos = min(df_long$correlation_value, na.rm = TRUE) - 
        (max(df_long$correlation_value, na.rm = TRUE) - min(df_long$correlation_value, na.rm = TRUE)) * 0.05
    )
  
  # --- 5. Extract P-Values via Standard Linear Models ---
  p_values <- df_long %>%
    group_by(stimulus_name) %>%
    summarize(
      p_val = {
        # Fit a standard linear model for each stimulus
        model <- lm(correlation_value ~ group)
        summary(model)$coefficients[2, "Pr(>|t|)"]
      },
      .groups = "drop"
    ) %>%
    mutate(
      # Dynamically calculate bracket x-coordinates based on the factor levels
      x_min = as.numeric(stimulus_name) - 0.2,
      x_max = as.numeric(stimulus_name) + 0.2,
      p_label = paste0("p = ", round(p_val, 3)),
      y_pos = max(df_long$correlation_value, na.rm = TRUE) * 1.05
    )
  
  # --- 6. Build the Plot ---
  cb_palette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
  
  p <- ggplot(df_long, aes(x = stimulus_name, y = correlation_value, fill = group)) +
    
    geom_point(aes(color = group), 
               position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8), 
               alpha = 0.6, size = 1.5) +
    
    # Thin dashed black line indicating the mean
    stat_summary(fun = mean, geom = "errorbar", 
                 aes(ymax = after_stat(y), ymin = after_stat(y)), 
                 width = 0.6, linewidth = 0.5, color = "black", linetype = "dashed",
                 position = position_dodge(0.8), show.legend = FALSE) +
    
    # Small black diamond exactly in the middle of the mean line
    stat_summary(fun = mean, geom = "point", 
                 shape = 18, size = 3, color = "black", 
                 position = position_dodge(0.8), show.legend = FALSE) +
                 
    # Faint filled violin calculating its shape by ignoring outliers (1.5x IQR)
    geom_violin(
      data = function(d) {
        d %>%
          group_by(stimulus_name, group) %>%
          mutate(
            q1 = quantile(correlation_value, 0.25, na.rm = TRUE),
            q3 = quantile(correlation_value, 0.75, na.rm = TRUE),
            iqr = q3 - q1
          ) %>%
          filter(correlation_value >= (q1 - 1.5 * iqr) & correlation_value <= (q3 + 1.5 * iqr)) %>%
          ungroup()
      },
      color = NA, position = position_dodge(0.8), alpha = 0.2
    ) +
    
    geom_bracket(
      data = p_values, 
      aes(xmin = x_min, xmax = x_max, label = p_label, y.position = y_pos), 
      inherit.aes = FALSE, tip.length = 0.02, linewidth = 0.6, fontface = "bold"
    ) +
    
    geom_text(
      data = n_data, 
      aes(x = stimulus_name, y = y_pos, label = paste0("n=", n_count), group = group), 
      inherit.aes = FALSE, position = position_dodge(0.8), size = 3.5, color = "grey30"
    ) +
    
    theme_classic() +
    scale_fill_manual(values = cb_palette) +
    scale_color_manual(values = cb_palette) +
    
    labs(title = str_wrap("Mean Sample Correlation by Stimulus Region", width = 60),
         subtitle = "P-values calculated via standard Linear Models",
         y = "Correlation Value", x = "Stimulus Region") +
    
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


feature_name <- "max_gradient"
default_path <- "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/r_plots/xxfeaturexx_plot.png"

corr_plot <- plot_sample_correlations(
  df = df, 
  save_path = "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/r_plots/sample_correlation_plot.png"
)

#plot_gradient <- generate_feature_plot(
#  df = df, 
#  feature_name = feature_name, 
#  save_path = str_replace(default_path, "xxfeaturexx", feature_name)
#)