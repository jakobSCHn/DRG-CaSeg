library(lmerTest)
library(dplyr)

df <- read.csv("/home/jaschneider/projects/DRG-CaSeg/DRG-CaSeg/DRG-CaSeg/bio_analysis/trace_features.csv")

df$group <- as.factor(df$group)
df$sample_id <- as.factor(df$sample_id)

model_cap <- lmer(stimulus_capsaicin_100nM_max_gradient ~ group + (1 | sample_id), data=df)
model_kcl_low <- lmer(stimulus_KCl_50mM_max_gradient ~ group + (1 | sample_id), data=df)
model_kcl_high <- lmer(stimulus_KCl_75mM_max_gradient ~ group + (1 | sample_id), data=df)

model_gradient_latency_cap <- lmer(stimulus_capsaicin_100nM_gradient_time ~ group + (1 | sample_id), data=df)
model_gradient_latency_kcllow <- lmer(stimulus_KCl_50mM_gradient_time ~ group + (1 | sample_id), data=df)
model_gradient_latency_kclhigh <- lmer(stimulus_KCl_75mM_gradient_time ~ group + (1 | sample_id), data=df)

model_peak_latency_cap <- lmer(stimulus_capsaicin_100nM_peak_time ~ group + (1 | sample_id), data=df)
model_peak_latency_kcllow <- lmer(stimulus_KCl_50mM_peak_time ~ group + (1 | sample_id), data=df)
model_peak_latency_kclhigh <- lmer(stimulus_KCl_75mM_peak_time ~ group + (1 | sample_id), data=df)

model_peak_height_cap <- lmer(stimulus_capsaicin_100nM_max_height ~ group + (1 | sample_id), data=df)
model_peak_height_kcllow <- lmer(stimulus_KCl_50mM_max_height ~ group + (1 | sample_id), data=df)
model_peak_height_kclhigh <- lmer(stimulus_KCl_75mM_max_height ~ group + (1 | sample_id), data=df)

print("--- CAP RESULTS ---")
print(summary(model_cap))
print("--- KCL LOW RESULTS ---")
print(summary(model_kcl_low))
print("--- KCL HIGH RESULTS ---")
print(summary(model_kcl_high))


print(" --- CAP GRADIENT LATENCY RESULTS ---")
print(summary(model_gradient_latency_cap))
print(" --- KCL LOW GRADIENT LATENCY RESULTS ---")
print(summary(model_gradient_latency_kcllow))
print(" --- KCL HIGH GRADIENT LATENCY RESULTS ---")
print(summary(model_gradient_latency_kclhigh))


print(" --- CAP PEAK LATENCY RESULTS ---")
print(summary(model_peak_latency_cap))
print(" --- KCL LOW PEAK LATENCY RESULTS ---")
print(summary(model_peak_latency_kcllow))
print(" --- KCL HIGH PEAK LATENCY RESULTS ---")
print(summary(model_peak_latency_kclhigh))


print(" --- CAP PEAK HEIGHT RESULTS ---")
print(summary(model_peak_height_cap))
print(" --- KCL LOW PEAK HEIGHT RESULTS ---")
print(summary(model_peak_height_kcllow))
print(" --- KCL HIGH PEAK HEIGHT RESULTS ---")
print(summary(model_peak_height_kclhigh))


df_agg <- df %>%
  group_by(sample_id, group) %>%
  summarize(
    mean_cap = mean(stimulus_capsaicin_100nM_max_gradient, na.rm = TRUE),
    mean_kcl_low = mean(stimulus_KCl_50mM_max_gradient, na.rm = TRUE),
    mean_kcl_high = mean(stimulus_KCl_75mM_max_gradient, na.rm = TRUE),
    .groups = "drop"
  )

ttest_cap <- t.test(mean_cap ~ group, data = df_agg)
ttest_kcl_low <- t.test(mean_kcl_low ~ group, data = df_agg)
ttest_kcl_high <- t.test(mean_kcl_high ~ group, data = df_agg)

print("--- T-TEST: CAP RESULTS ---")
print(ttest_cap)
print("--- T-TEST: KCL LOW RESULTS ---")
print(ttest_kcl_low)
print("--- T-TEST: KCL HIGH RESULTS ---")
print(ttest_kcl_high)

wilcox_cap <- wilcox.test(mean_cap ~ group, data = df_agg)
wilcox_kcl_low <- wilcox.test(mean_kcl_low ~ group, data = df_agg)
wilcox_kcl_high <- wilcox.test(mean_kcl_high ~ group, data = df_agg)

print("--- WILCOXON TEST: CAP RESULTS ---")
print(wilcox_cap)
print("--- WILCOXON TEST: KCL LOW RESULTS ---")
print(wilcox_kcl_low)
print("--- WILCOXON TEST: KCL HIGH RESULTS ---")
print(wilcox_kcl_high)

browser()
# ==========================================
# --- CHI-SQUARED TEST (BINARY CATEGORIES) ---
# ==========================================

threshold_cap      <- 7.0 
threshold_kcl_low  <- 6.75
threshold_kcl_high <- 5.0

# Create the binary categories in the dataframe
df_binary <- df %>%
  mutate(
    cat_cap      = ifelse(stimulus_capsaicin_100nM_max_height > threshold_cap, "High", "Low"),
    cat_kcl_low  = ifelse(stimulus_KCl_50mM_max_height > threshold_kcl_low, "High", "Low"),
    cat_kcl_high = ifelse(stimulus_KCl_75mM_max_height > threshold_kcl_high, "High", "Low")
  )

# Function to run Chi-Squared and print formatted results
run_chi_test <- label <- function(df_func, category_col, stimulus_name) {
  # Create a contingency table (Counts of High/Low vs Control/Treatment)
  tbl <- table(df_func$group, df_func[[category_col]])
  
  print(paste("--- CHI-SQUARED:", stimulus_name, "---"))
  print("Contingency Table:")
  print(tbl)
  
  # Run the test
  # Note: Correct=TRUE applies Yates' continuity correction, which is safer for small counts
  chi_result <- chisq.test(tbl)
  print(chi_result)
  
  # If you get a warning about "Chi-squared approximation may be incorrect", 
  # it's usually because some counts are < 5. In that case, use Fisher's Exact Test:
  if (any(tbl < 5)) {
    print(paste("Note: Small counts detected. Running Fisher's Exact Test for", stimulus_name))
    print(fisher.test(tbl))
  }
}

# Execute for each stimulus
run_chi_test(df_binary, "cat_cap", "CAPSAICIN")
run_chi_test(df_binary, "cat_kcl_low", "KCL LOW")
run_chi_test(df_binary, "cat_kcl_high", "KCL HIGH")

browser()

# ==========================================
# --- THRESHOLD PERCENTILE ANALYSIS ---
# ==========================================

# Function to calculate the percentile of a threshold within the total population
calculate_percentile <- function(data_vector, threshold, stimulus_name) {
  # Remove NAs to ensure accurate distribution
  clean_data <- na.omit(data_vector)
  
  # Create the Empirical Cumulative Distribution Function
  data_ecdf <- ecdf(clean_data)
  
  # Calculate the percentile (result is 0 to 1)
  percentile_val <- data_ecdf(threshold)
  
  # Format as percentage for readability
  percentage <- percentile_val * 100
  
  print(paste("--- PERCENTILE ANALYSIS:", stimulus_name, "---"))
  print(paste("Threshold value:", threshold))
  print(paste("This threshold represents the", round(percentage, 2), "percentile of the total population."))
  print(paste(round(100 - percentage, 2), "% of all traces are above this threshold."))
  print("------------------------------------------")
}

# Execute for each stimulus using the thresholds defined in the previous block
calculate_percentile(df$stimulus_capsaicin_100nM_max_height, threshold_cap, "CAPSAICIN")
calculate_percentile(df$stimulus_KCl_50mM_max_height, threshold_kcl_low, "KCL LOW")
calculate_percentile(df$stimulus_KCl_75mM_max_height, threshold_kcl_high, "KCL HIGH")

browser()

# ==========================================
# --- DYNAMIC 80th PERCENTILE ANALYSIS ---
# ==========================================

# Calculate the 80th percentile threshold for each stimulus (all groups combined)
# na.rm = TRUE is important to ignore missing traces
qt = 0.90
dyn_threshold_cap      <- quantile(df$stimulus_capsaicin_100nM_max_height, qt, na.rm = TRUE)
dyn_threshold_kcl_low  <- quantile(df$stimulus_KCl_50mM_max_height, qt, na.rm = TRUE)
dyn_threshold_kcl_high <- quantile(df$stimulus_KCl_75mM_max_height, qt, na.rm = TRUE)

print("--- CALCULATED 80th PERCENTILE THRESHOLDS OF PEAKS ---")
print(paste("Capsaicin 80% Threshold:", round(dyn_threshold_cap, 4)))
print(paste("KCl Low 80% Threshold:",   round(dyn_threshold_kcl_low, 4)))
print(paste("KCl High 80% Threshold:",  round(dyn_threshold_kcl_high, 4)))

# Re-create binary categories based on these dynamic thresholds
df_dyn_binary_peak <- df %>%
  mutate(
    cat_cap      = ifelse(stimulus_capsaicin_100nM_max_height > dyn_threshold_cap, "High", "Low"),
    cat_kcl_low  = ifelse(stimulus_KCl_50mM_max_height > dyn_threshold_kcl_low, "High", "Low"),
    cat_kcl_high = ifelse(stimulus_KCl_75mM_max_height > dyn_threshold_kcl_high, "High", "Low")
  )


# Execute the tests
run_chi_test(df_dyn_binary_peak, "cat_cap", "CAPSAICIN")
run_chi_test(df_dyn_binary_peak, "cat_kcl_low", "KCL LOW")
run_chi_test(df_dyn_binary_peak, "cat_kcl_high", "KCL HIGH")


dyn_threshold_cap      <- quantile(df$stimulus_capsaicin_100nM_max_gradient, qt, na.rm = TRUE)
dyn_threshold_kcl_low  <- quantile(df$stimulus_KCl_50mM_max_gradient, qt, na.rm = TRUE)
dyn_threshold_kcl_high <- quantile(df$stimulus_KCl_75mM_max_gradient, qt, na.rm = TRUE)

print("--- CALCULATED 80th PERCENTILE THRESHOLDS OF GRADIENTS---")
print(paste("Capsaicin 80% Threshold:", round(dyn_threshold_cap, 4)))
print(paste("KCl Low 80% Threshold:",   round(dyn_threshold_kcl_low, 4)))
print(paste("KCl High 80% Threshold:",  round(dyn_threshold_kcl_high, 4)))

# Re-create binary categories based on these dynamic thresholds
df_dyn_binary_gradient <- df %>%
  mutate(
    cat_cap      = ifelse(stimulus_capsaicin_100nM_max_gradient > dyn_threshold_cap, "High", "Low"),
    cat_kcl_low  = ifelse(stimulus_KCl_50mM_max_gradient > dyn_threshold_kcl_low, "High", "Low"),
    cat_kcl_high = ifelse(stimulus_KCl_75mM_max_gradient > dyn_threshold_kcl_high, "High", "Low")
  )


# Execute the tests
run_chi_test(df_dyn_binary_gradient, "cat_cap", "CAPSAICIN")
run_chi_test(df_dyn_binary_gradient, "cat_kcl_low", "KCL LOW")
run_chi_test(df_dyn_binary_gradient, "cat_kcl_high", "KCL HIGH")