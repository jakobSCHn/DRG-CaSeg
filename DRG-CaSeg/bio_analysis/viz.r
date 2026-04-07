library(ggplot2)
library(tidyr)
library(dplyr)
library(lmerTest)

# --- 1. Prepare the Data ---
df <- read.csv("/home/jaschneider/projects/DRG-CaSeg/DRG-CaSeg/DRG-CaSeg/bio_analysis/trace_features.csv")

df$group <- as.factor(df$group)
df$sample_id <- as.factor(df$sample_id)

model_cap <- lmer(stimulus_capsaicin_100nM_max_height ~ group + (1 | sample_id), data=df)
model_kcl_low <- lmer(stimulus_KCl_50mM_max_height ~ group + (1 | sample_id), data=df)
model_kcl_high <- lmer(stimulus_KCl_75mM_max_height ~ group + (1 | sample_id), data=df)


df_long <- df %>%
  pivot_longer(
    cols = c(stimulus_capsaicin_100nM_max_height, 
             stimulus_KCl_50mM_max_height, 
             stimulus_KCl_75mM_max_height),
    names_to = "stimulus", values_to = "value"
  ) %>%
  mutate(stimulus = factor(stimulus, 
    levels = c("stimulus_capsaicin_100nM_max_height", 
               "stimulus_KCl_50mM_max_height", 
               "stimulus_KCl_75mM_max_height"),
    labels = c("Capsaicin", "KCl Low", "KCl High")))

# --- 2. Extract P-Values from your LMMs ---
# We pull the p-value from the second row of the fixed effects table
get_p <- function(model) {
  summary(model)$coefficients[2, "Pr(>|t|)"]
}

p_values <- data.frame(
  stimulus = c("Capsaicin", "KCl Low", "KCl High"),
  p_label = c(
    paste0("p = ", round(get_p(model_cap), 3)),
    paste0("p = ", round(get_p(model_kcl_low), 3)),
    paste0("p = ", round(get_p(model_kcl_high), 3))
  ),
  # Position the label slightly above the max value of each stimulus
  y_pos = max(df_long$value, na.rm = TRUE) * 1.05 
)

# --- 3. Build the Plot ---
ggplot(df_long, aes(x = stimulus, y = value, fill = group)) +
  # Background Violin Outline
  geom_violin(aes(color = group), fill = NA, position = position_dodge(0.8), linewidth = 1) +
  # Jittered points (colored by group)
  geom_point(aes(color = group), 
             position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8), 
             alpha = 0.4, size = 1) +
  # Crossbar for the Mean (Calculated from raw data for visual reference)
  stat_summary(fun = mean, geom = "crossbar", width = 0.2, 
               position = position_dodge(0.8), color = "black") +
  # Add the LMM P-Values manually
  geom_text(data = p_values, aes(x = stimulus, y = y_pos, label = p_label), 
            inherit.aes = FALSE, size = 4, fontface = "bold") +
  # Aesthetics
  theme_classic() +
  scale_fill_brewer(palette = "Set1") +
  scale_color_brewer(palette = "Set1") +
  labs(title = "LMM Results: Maximum Gradient by Stimulus",
       subtitle = "P-values calculated via Satterthwaite's method (lmerTest)",
       y = "Max Gradient", x = "")

ggsave("/home/jaschneider/projects/DRG-CaSeg/DRG-CaSeg/DRG-CaSeg/bio_analysis/r_test.png", width = 8, height = 6, dpi = 300)