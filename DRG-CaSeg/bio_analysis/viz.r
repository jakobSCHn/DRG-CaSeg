library(ggplot2)
library(tidyr)
library(dplyr)
library(lmerTest)
library(ggpubr)
library(stringr)

# Define the reusable function
generate_feature_plot <- function(df, feature_name, plot_title, y_axis_label, p_within_group = FALSE, save_path = NULL) {
  
  # --- 1. Dynamically Construct Column Names ---
  col_cap <- paste0("stimulus_capsaicin_100nM_", feature_name)
  col_kcl_low <- paste0("stimulus_KCl_50mM_", feature_name)
  col_kcl_high <- paste0("stimulus_KCl_75mM_", feature_name)
  
  # Check if columns exist in the dataframe to prevent confusing errors
  req_cols <- c(col_cap, col_kcl_low, col_kcl_high)
  if(!all(req_cols %in% names(df))) {
    stop("One or more generated column names were not found in the dataframe.")
  }
  
  # Replace "Cytokine" with "Treatment" in the group column
  df <- df %>%
    mutate(group = str_replace(as.character(group), "Cytokine", "Treatment"))
  
  # --- 1.5 CONDITIONAL TIME SHIFT ---
  if (grepl("time", feature_name, ignore.case = TRUE)) {
    df[[col_cap]]      <- df[[col_cap]] - 100
    df[[col_kcl_low]]  <- df[[col_kcl_low]] - 200
    df[[col_kcl_high]] <- df[[col_kcl_high]] - 300
  }
  
  # --- 2. Dynamically Run LMMs (Between Groups) ---
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
                             labels = c("Capsaicin 100 nM", "KCl 50 mM", "KCl 75 mM"))) %>%
    drop_na(value)
  
  # --- 4. Calculate Sample Sizes (N) ---
  max_val <- max(df_long$value, na.rm = TRUE)
  min_val <- min(df_long$value, na.rm = TRUE)
  y_range <- max_val - min_val
  
  n_data <- df_long %>%
    group_by(stimulus, group) %>%
    summarize(n_count = n(), .groups = "drop") %>%
    mutate(y_pos = min_val - (y_range * 0.05))
  
  # --- 5. Extract P-Values and Calculate Bracket Positions ---
  get_p <- function(model) {
    summary(model)$coefficients[2, "Pr(>|t|)"]
  }
  
  # Formatting helper for p-values
  format_p <- function(p_val) {
    if (p_val < 0.001) {
      return("p < 0.001")
    } else {
      return(paste0("p = ", round(p_val, 3)))
    }
  }
  
  # Standard Between-Group P-values
  p_values <- data.frame(
    stimulus = c("Capsaicin 100 nM", "KCl 50 mM", "KCl 75 mM"),
    x_min = c(1 - 0.2, 2 - 0.2, 3 - 0.2), 
    x_max = c(1 + 0.2, 2 + 0.2, 3 + 0.2), 
    p_label = c(
      format_p(get_p(model_cap)),
      format_p(get_p(model_kcl_low)),
      format_p(get_p(model_kcl_high))
    ),
    y_pos = max_val + (y_range * 0.08) 
  )
  
  # --- 5.5 Within-Group P-Values ---
  if (p_within_group) {
    get_within_p <- function(grp, stim1, stim2) {
      sub_df <- df_long %>% filter(group == grp, stimulus %in% c(stim1, stim2))
      m <- lmer(value ~ stimulus + (1 | sample_id), data = sub_df)
      summary(m)$coefficients[2, "Pr(>|t|)"]
    }
    
    # Identify the two groups dynamically 
    grp_levels <- levels(factor(df_long$group))
    grp1 <- grp_levels[1] # Usually Control
    grp2 <- grp_levels[2] # Usually Treatment
    
    p_within <- data.frame(
      # Group 1 (Left dodge: integers - 0.2) | Group 2 (Right dodge: integers + 0.2)
      # Coordinates map exactly to the center of the dodged points
      x_min = c(0.8, 1.2, 1.8, 2.2),
      x_max = c(1.8, 2.2, 2.8, 3.2),
      # Stagger the heights significantly so the horizontal lines never intersect
      y_pos = c(
        max_val + (y_range * 0.16), 
        max_val + (y_range * 0.24), 
        max_val + (y_range * 0.32), 
        max_val + (y_range * 0.40)
      )
    )
    
    p_within$p_label <- c(
      format_p(get_within_p(grp1, "Capsaicin 100 nM", "KCl 50 mM")),
      format_p(get_within_p(grp2, "Capsaicin 100 nM", "KCl 50 mM")),
      format_p(get_within_p(grp1, "KCl 50 mM", "KCl 75 mM")),
      format_p(get_within_p(grp2, "KCl 50 mM", "KCl 75 mM"))
    )
  }
  
  # --- 6. Build the Plot ---
  cb_palette <- c("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
  
  p <- ggplot(df_long, aes(x = stimulus, y = value, fill = group)) +
    
    geom_point(aes(color = group), 
               position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8), 
               alpha = 0.6, size = 1.5) +
    
    stat_summary(fun = mean, geom = "errorbar", 
                 aes(ymax = after_stat(y), ymin = after_stat(y)), 
                 width = 0.6, linewidth = 0.5, color = "black", linetype = "dashed",
                 position = position_dodge(0.8), show.legend = FALSE) +
    
    stat_summary(fun = mean, geom = "point", 
                 shape = 18, size = 3, color = "black", 
                 position = position_dodge(0.8), show.legend = FALSE) +
                 
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
    
    # Standard Between-Group Brackets
    geom_bracket(
      data = p_values, 
      aes(xmin = x_min, xmax = x_max, label = p_label, y.position = y_pos), 
      inherit.aes = FALSE, tip.length = 0.02, linewidth = 0.6, fontface = "bold"
    )
    
  # Add Within-Group Brackets if toggled
  if (p_within_group) {
    p <- p + geom_bracket(
      data = p_within, 
      aes(xmin = x_min, xmax = x_max, label = p_label, y.position = y_pos), 
      inherit.aes = FALSE, tip.length = 0.02, linewidth = 0.6, fontface = "bold"
    )
  }
  
  p <- p + 
    geom_text(
      data = n_data, 
      aes(x = stimulus, y = y_pos, label = paste0("N=", n_count), group = group), 
      inherit.aes = FALSE, position = position_dodge(0.8), size = 4.5, color = "grey30"
    ) +
    
    theme_classic(base_size = 14) +
    scale_fill_manual(values = cb_palette) +
    scale_color_manual(values = cb_palette) +
    
    # Applied dynamic title mapping and explicitly assigned the X and Y axes 
    labs(
      title = str_wrap(plot_title, width = 45),
      y = y_axis_label, 
      x = "Stimulus Region"
    ) +
    
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 22, margin = margin(t = 15, b = 15, l = 30)),  
      plot.subtitle = element_text(hjust = 0.5),
      
      # Adjusted t = 5 to exactly match the reference function
      plot.margin = margin(t = 5, r = 10, b = 5, l = 5),
      
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5), 
      
      axis.title.y = element_text(face = "bold", size = 16, margin = margin(r = 15)),
      axis.text.y = element_text(size = 12, color = "black"),
      
      axis.title.x = element_text(face = "bold", size = 16, margin = margin(t = 15)),
      axis.text.x = element_text(size = rel(1.2), color = "black"),
      
      axis.line = element_line(linewidth = 0.8, color = "black"), 
      axis.ticks = element_line(linewidth = 0.8, color = "black"),
      
      legend.text = element_text(size = 12),
      legend.title = element_text(size = 14)
    ) +
    coord_cartesian(clip = "off")
  
  # --- 7. Save and Return ---
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = 10, height = 8, dpi = 300)
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
    # TWEAK 3: Changed "treatment" to "Treatment"
    mutate(group = str_replace(as.character(group), "Cytokine", "Treatment")) %>%
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
      stimulus_name = str_replace(stimulus_name, "stimulus_", ""), # Catch extra prefixes if present
      
      # Replace underscores with spaces
      stimulus_name = str_replace_all(stimulus_name, "_", " "),
      
      # Capitalize ONLY the first letter so scientific units like "nM" and "mM" are preserved
      stimulus_name = paste0(toupper(substr(stimulus_name, 1, 1)), substr(stimulus_name, 2, nchar(stimulus_name))),
      
      # Added spaces before units
      stimulus_name = str_replace(stimulus_name, "100nM", "100 nM"),
      stimulus_name = str_replace(stimulus_name, "50mM", "50 mM"),
      stimulus_name = str_replace(stimulus_name, "75mM", "75 mM")
    ) %>%
    drop_na(correlation_value)
  
  # --- 3.5 Explicitly Define Plotting Order ---
  # Identify levels to force Global first, then Capsaicin, then KCl
  all_levels <- unique(df_long$stimulus_name)
  lvl_global <- grep("Global", all_levels, ignore.case = TRUE, value = TRUE)
  lvl_cap    <- grep("Capsaicin", all_levels, ignore.case = TRUE, value = TRUE)
  lvl_kcl    <- grep("KCl", all_levels, ignore.case = TRUE, value = TRUE)
  lvl_other  <- setdiff(all_levels, c(lvl_global, lvl_cap, lvl_kcl))
  
  # Apply ordered factor
  df_long <- df_long %>%
    mutate(stimulus_name = factor(stimulus_name, levels = c(lvl_global, lvl_cap, lvl_kcl, lvl_other)))
  
  # --- 4. Calculate Sample Sizes (N) ---
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
    
    # TWEAK 4: Enforced fontface = "bold" for the p-value labels
    geom_bracket(
      data = p_values, 
      aes(xmin = x_min, xmax = x_max, label = p_label, y.position = y_pos), 
      inherit.aes = FALSE, tip.length = 0.02, linewidth = 0.6, fontface = "bold"
    ) +
    
    geom_text(
      data = n_data, 
      aes(x = stimulus_name, y = y_pos, label = paste0("N=", n_count), group = group), 
      inherit.aes = FALSE, position = position_dodge(0.8), size = 4.5, color = "grey30"
    ) +
    
    theme_classic(base_size = 14) +
    scale_fill_manual(values = cb_palette) +
    scale_color_manual(values = cb_palette) +
    
    labs(title = str_wrap("Mean Neuronal Activity Correlation by Stimulus Region", width = 60),
         y = "Pearson Correlation Coefficient", x = "Stimulus Region") +
    
    theme(
      # TWEAK 5: Added left margin (l = 20) to nudge the title to the right
      plot.title = element_text(hjust = 0.5, face = "bold", size = 22, margin = margin(t = 15, b = 15, l = 20)),  
      plot.subtitle = element_text(hjust = 0.5),
      
      plot.margin = margin(t = 5, r = 10, b = 5, l = 5),
      
      panel.grid.major.y = element_line(color = "grey90", linewidth = 0.5), 
      
      axis.title.y = element_text(face = "bold", size = 16, margin = margin(r = 15)),
      
      # TWEAK 1: Removed face = "bold" so y-axis ticks and labels are normal weight
      axis.text.y = element_text(size = 12, color = "black"),
      
      axis.title.x = element_text(face = "bold", size = 16, margin = margin(t = 15)),
      axis.text.x = element_text(size = rel(1.2), color = "black"),
      
      axis.line = element_line(linewidth = 0.8, color = "black"), 
      axis.ticks = element_line(linewidth = 0.8, color = "black"),
      
      legend.text = element_text(size = 12),
      
      # TWEAK 2: Removed face = "bold" from the legend heading
      legend.title = element_text(size = 14)
    ) +
    # Fixed the lower ylim to 0
    coord_cartesian(ylim = c(0, NA), clip = "off")
  
  # --- 7. Save and Return ---
  if (!is.null(save_path)) {
    ggsave(save_path, plot = p, width = 10, height = 8, dpi = 300)
  }
  
  return(p)
}

plot_responder_proportions <- function(df, save_path = NULL) {
  
  # --- 1. Identify Target Columns ---
  target_cols <- c("stimulus_capsaicin_100nM_has_peak", 
                   "stimulus_KCl_50mM_has_peak", 
                   "stimulus_KCl_75mM_has_peak")
  
  if(!all(target_cols %in% names(df))) {
    stop("One or more target 'has_peak' columns were not found in the dataframe.")
  }
  
  # --- 2. Melt to Long Format and Clean Labels ---
  df_long <- df %>%
    select(sample_id, group, all_of(target_cols)) %>%
    mutate(group = str_replace(as.character(group), "Cytokine", "Treatment")) %>%
    pivot_longer(
      cols = all_of(target_cols),
      names_to = "stimulus", 
      values_to = "responded"
    ) %>%
    mutate(
      responded = case_when(
        toupper(as.character(responded)) %in% c("TRUE", "T", "1", "1.0") ~ 1,
        toupper(as.character(responded)) %in% c("FALSE", "F", "0", "0.0") ~ 0,
        TRUE ~ NA_real_
      ),
      Status = ifelse(responded == 1, "Responder", "NonResponder"),
      stimulus = str_replace(stimulus, "stimulus_", ""),
      stimulus = str_replace(stimulus, "_has_peak", ""),
      stimulus = str_replace_all(stimulus, "_", " "),
      stimulus = str_replace(stimulus, "capsaicin", "Capsaicin"),
      # Added spaces before units
      stimulus = str_replace(stimulus, "100nM", "100 nM"),
      stimulus = str_replace(stimulus, "50mM", "50 mM"),
      stimulus = str_replace(stimulus, "75mM", "75 mM")
    ) %>%
    drop_na(responded) %>%
    # Updated levels to include the space
    mutate(stimulus = factor(stimulus, levels = c("Capsaicin 100 nM", "KCl 50 mM", "KCl 75 mM")))
  
  # --- 3. Calculate Global N for Control and Treatment ---
  legend_n <- df_long %>%
    group_by(stimulus, group) %>%
    summarize(n_count = n(), .groups = "drop") %>%
    group_by(group) %>%
    summarize(final_n = max(n_count), .groups = "drop")
  
  control_n <- legend_n$final_n[legend_n$group == "Control"]
  treatment_n <- legend_n$final_n[legend_n$group == "Treatment"]
  
  # --- 3.5 Calculate Polygons for Vertical Floating Circles ---
  df_pie <- df_long %>%
    group_by(stimulus, group) %>%
    summarize(
      Responder = sum(responded == 1, na.rm = TRUE),
      Total = n(),
      .groups = "drop"
    ) %>%
    mutate(
      Prop_Responder = Responder / Total,
      x_center = as.numeric(stimulus),
      y_center = ifelse(group == "Control", 1.8, 0.2),
      radius = 0.40 
    )
  
  poly_list <- lapply(1:nrow(df_pie), function(i) {
    row <- df_pie[i, ]
    angle_resp <- 2 * pi * row$Prop_Responder
    
    # Responder slice (Left side)
    theta_resp <- seq(pi/2, pi/2 + angle_resp, length.out = max(10, round(100 * row$Prop_Responder)))
    poly_resp <- data.frame(
      x = c(row$x_center, row$x_center + row$radius * cos(theta_resp)),
      y = c(row$y_center, row$y_center + row$radius * sin(theta_resp)),
      Status = paste0(row$group, "_Responder"),
      poly_id = paste0(row$stimulus, "_", row$group, "_Responder")
    )
    
    # Non-Responder slice (Right side)
    theta_nonresp <- seq(pi/2 + angle_resp, pi/2 + 2*pi, length.out = max(10, round(100 * (1 - row$Prop_Responder))))
    poly_nonresp <- data.frame(
      x = c(row$x_center, row$x_center + row$radius * cos(theta_nonresp)),
      y = c(row$y_center, row$y_center + row$radius * sin(theta_nonresp)),
      Status = paste0(row$group, "_NonResponder"),
      poly_id = paste0(row$stimulus, "_", row$group, "_NonResponder")
    )
    
    res <- data.frame()
    if(row$Prop_Responder > 0) res <- rbind(res, poly_resp)
    if(row$Prop_Responder < 1) res <- rbind(res, poly_nonresp)
    return(res)
  })
  df_polygons <- bind_rows(poly_list)
  
  df_text <- bind_rows(
    df_pie %>% filter(Prop_Responder > 0) %>%
      mutate(
        mid_angle = pi/2 + (2 * pi * Prop_Responder) / 2,
        x_text = x_center + (radius * 0.52) * cos(mid_angle),
        y_text = y_center + (radius * 0.52) * sin(mid_angle),
        Label = scales::percent(Prop_Responder, accuracy = 1)
      ),
    df_pie %>% filter(Prop_Responder < 1) %>%
      mutate(
        angle_resp = 2 * pi * Prop_Responder,
        mid_angle = pi/2 + angle_resp + (2 * pi - angle_resp) / 2,
        x_text = x_center + (radius * 0.52) * cos(mid_angle),
        y_text = y_center + (radius * 0.52) * sin(mid_angle),
        Label = scales::percent(1 - Prop_Responder, accuracy = 1)
      )
  )
  
  # --- 3.7 Custom Legend Coordinates ---
  num_stim <- length(levels(df_long$stimulus))
  x_leg_box <- num_stim + 0.70 
  x_leg_txt <- num_stim + 0.775 
  
  df_legend <- data.frame(
    x_box = c(x_leg_box, x_leg_box, x_leg_box, x_leg_box),
    y_box = c(1.85, 1.75, 0.25, 0.15),
    x_text = c(x_leg_txt, x_leg_txt, x_leg_txt, x_leg_txt),
    y_text = c(1.85, 1.75, 0.25, 0.15),
    Status = c("Control_Responder", "Control_NonResponder", "Treatment_Responder", "Treatment_NonResponder"),
    Label = c("Responder", "Non-Responder", "Responder", "Non-Responder")
  )
  
  # --- 4. Build the Plot ---
  custom_colors <- c(
    "Control_Responder" = "#0072B2",      
    "Control_NonResponder" = "#56B4E9",   
    "Treatment_Responder" = "#D55E00",    
    "Treatment_NonResponder" = "#E69F00"  
  )
  
  p <- ggplot() +
    geom_polygon(data = df_polygons, aes(x = x, y = y, fill = Status, group = poly_id), 
                 color = "black", linewidth = 0.5, show.legend = FALSE) +
    
    geom_text(data = df_text, aes(x = x_text, y = y_text, label = Label),
              color = "white", size = 4.5, fontface = "bold", show.legend = FALSE) +
              
    geom_point(data = df_legend, aes(x = x_box, y = y_box, fill = Status), 
               shape = 22, size = 6, color = "black", stroke = 0.5, show.legend = FALSE) +
    
    geom_text(data = df_legend, aes(x = x_text, y = y_text, label = Label), 
              hjust = 0, size = 4.5, show.legend = FALSE) +
              
    coord_fixed(
      ratio = 1, 
      xlim = c(0.5, num_stim + 0.5), 
      # TWEAK: Expanded lower y-limit to -0.8 to push the X-axis line further down
      ylim = c(-0.8, 2.4), 
      clip = "off"
    ) +
    
    scale_fill_manual(values = custom_colors) +
    
    scale_x_continuous(
      breaks = 1:num_stim,
      labels = levels(df_long$stimulus),
      expand = c(0, 0)
    ) +
    
    scale_y_continuous(
      breaks = c(0.2, 1.8),
      labels = c(paste0("Treatment\n(N=", treatment_n, ")"), paste0("Control\n(N=", control_n, ")")),
      expand = c(0, 0)
    ) +
    
    theme_classic(base_size = 14) +
    labs(
      title = "Proportion of Responding Neurons by Stimulus",
      x = "Stimulus Condition",
      fill = NULL 
    ) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 22, margin = margin(t = 15, b = 15)),
      
      # TWEAK: Increased top margin (t = 25) to protect the large title
      plot.margin = margin(t = 5, r = 100, b = 5, l = 5),
      
      axis.title.y = element_blank(),
      axis.text.y = element_text(size = 12, face = "bold", color = "black"),
      axis.ticks.y = element_blank(),
      axis.line.y = element_blank(),
      
      axis.line.x = element_line(linewidth = 0.8, color = "black"), 
      axis.ticks.x = element_line(linewidth = 0.8, color = "black"),
      axis.text.x = element_text(size = rel(1.2), color = "black"),
      axis.title.x = element_text(face = "bold", size = 16, margin = margin(t = 15)),
      
      legend.position = "none"
    )
    
  # --- 5. Save and Return ---
  if (!is.null(save_path)) {
    # TWEAK: Set figure output to 10x8 for better proportions
    ggsave(save_path, plot = p, width = 10, height = 8, dpi = 300)
  }
  
  return(p)
}

export_summary_statistics <- function(df, save_path) {
  
  # Initialize a list to hold dataframes for each feature
  stats_list <- list()
  
  # --- 1. Clean Group Column ---
  df <- df %>%
    mutate(group = str_replace(as.character(group), "Cytokine", "Treatment"))
  
  # --- 2. Process Standard Features ---
  features_to_calc <- c("max_height", "peak_time", "gradient_time", "max_gradient")
  
  for (feat in features_to_calc) {
    col_cap <- paste0("stimulus_capsaicin_100nM_", feat)
    col_kcl_low <- paste0("stimulus_KCl_50mM_", feat)
    col_kcl_high <- paste0("stimulus_KCl_75mM_", feat)
    
    req_cols <- c(col_cap, col_kcl_low, col_kcl_high)
    existing_cols <- req_cols[req_cols %in% names(df)]
    
    if (length(existing_cols) > 0) {
      temp_df <- df
      
      # Mirror the conditional time-shift logic from your plotting function
      if (grepl("time", feat, ignore.case = TRUE)) {
        if (col_cap %in% names(temp_df)) temp_df[[col_cap]] <- temp_df[[col_cap]] - 100
        if (col_kcl_low %in% names(temp_df)) temp_df[[col_kcl_low]] <- temp_df[[col_kcl_low]] - 200
        if (col_kcl_high %in% names(temp_df)) temp_df[[col_kcl_high]] <- temp_df[[col_kcl_high]] - 300
      }
      
      stat_df <- temp_df %>%
        select(group, all_of(existing_cols)) %>%
        pivot_longer(cols = all_of(existing_cols), names_to = "stimulus", values_to = "value") %>%
        mutate(
          # Clean labels to match plots nicely
          stimulus = str_replace(stimulus, paste0("_", feat), ""),
          stimulus = str_replace(stimulus, "stimulus_", ""),
          stimulus = str_replace_all(stimulus, "_", " "),
          stimulus = str_replace(stimulus, "capsaicin", "Capsaicin"),
          stimulus = str_replace(stimulus, "100nM", "100 nM"),
          stimulus = str_replace(stimulus, "50mM", "50 mM"),
          stimulus = str_replace(stimulus, "75mM", "75 mM")
        ) %>%
        drop_na(value) %>%
        group_by(stimulus, group) %>%
        summarize(
          mean_val = mean(value, na.rm = TRUE),
          std_val = sd(value, na.rm = TRUE),
          .groups = "drop"
        )
      
      stats_list[[feat]] <- stat_df
    }
  }
  
  # --- 3. Process Correlation Features ---
  sync_cols <- grep("corr", names(df), value = TRUE)
  if (length(sync_cols) > 0) {
    
    # Replicate the sample-level collapse from plot_sample_correlations
    df_sample <- df %>%
      group_by(sample_id, group) %>%
      summarize(across(all_of(sync_cols), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
    
    corr_stats <- df_sample %>%
      pivot_longer(cols = all_of(sync_cols), names_to = "stimulus", values_to = "value") %>%
      mutate(
        # Mirror the label cleaning exactly
        stimulus = str_replace(stimulus, "sample_corr_", ""),
        stimulus = str_replace(stimulus, "stimulus_corr_", ""),
        stimulus = str_replace(stimulus, "stimulus_", ""),
        stimulus = str_replace_all(stimulus, "_", " "),
        stimulus = paste0(toupper(substr(stimulus, 1, 1)), substr(stimulus, 2, nchar(stimulus))),
        stimulus = str_replace(stimulus, "100nM", "100 nM"),
        stimulus = str_replace(stimulus, "50mM", "50 mM"),
        stimulus = str_replace(stimulus, "75mM", "75 mM")
      ) %>%
      drop_na(value) %>%
      group_by(stimulus, group) %>%
      summarize(
        mean_val = mean(value, na.rm = TRUE),
        std_val = sd(value, na.rm = TRUE),
        .groups = "drop"
      )
    
    stats_list[["correlations"]] <- corr_stats
  }
  
  # --- 4. Write to a YAML text file without extra packages ---
  file_conn <- file(save_path, open = "w")
  
  for (feat_name in names(stats_list)) {
    # Top level: Feature name
    cat(paste0(feat_name, ":\n"), file = file_conn)
    
    feat_data <- stats_list[[feat_name]]
    unique_stimuli <- unique(feat_data$stimulus)
    
    for (stim in unique_stimuli) {
      # Second level: Stimulus condition
      cat(paste0("  \"", stim, "\":\n"), file = file_conn)
      
      stim_data <- feat_data %>% filter(stimulus == stim)
      unique_groups <- unique(stim_data$group)
      
      for (grp in unique_groups) {
        # Third level: Group (Control/Treatment)
        cat(paste0("    \"", grp, "\":\n"), file = file_conn)
        
        row_data <- stim_data %>% filter(group == grp)
        
        # Fourth level: Mean and Std
        cat(paste0("      mean: ", row_data$mean_val, "\n"), file = file_conn)
        cat(paste0("      std: ", row_data$std_val, "\n"), file = file_conn)
      }
    }
  }
  
  close(file_conn)
  print(paste0("Successfully exported summary statistics to: ", save_path))
}

df <- read.csv("/home/jaschneider/projects/DRG-CaSeg/DRG-CaSeg/DRG-CaSeg/bio_analysis/trace_features.csv")
df$group <- as.factor(df$group)
df$sample_id <- as.factor(df$sample_id)


feature_name <- "peak_time"
default_path <- "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/r_plots/xxfeaturexx_plot.png"

responder_proportion_plot <- plot_responder_proportions(
  df = df, 
  save_path = "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/r_plots/responder_proportion_plot.png"
)

corr_plot <- plot_sample_correlations(
  df = df, 
  save_path = "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/r_plots/sample_correlation_plot.png"
)

plot_gradient <- generate_feature_plot(
  df = df, 
  feature_name = feature_name,
  plot_title = "Latency of Peak Amplitude in Stimulus-Responsive Neurons after Stimulation",
  y_axis_label = "Latency [s]",
  p_within_group = FALSE,
  save_path = str_replace(default_path, "xxfeaturexx", feature_name)
)

export_summary_statistics(
  df = df, 
  save_path = "/home/jaschneider/projects/DRG-CaSeg/bio_analysis_plots/r_plots/overall_summary_statistics.yaml"
)