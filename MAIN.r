############################################################
# PHASE 1 — DATA LOADING, CLEANING & PREPROCESSING
# INFO 6105 FINAL PROJECT — GitHub Repository Analysis
############################################################

# ----------------------------------------------------------
# 1. Load Required Libraries
# ----------------------------------------------------------
library(dplyr)

# ----------------------------------------------------------
# 2. Load Dataset
# ----------------------------------------------------------
data <- read.csv("repositories.csv", stringsAsFactors = FALSE)

# Inspect structure
str(data)

# ----------------------------------------------------------
# 3. Downsample to 10,000 Rows (Professor Recommended)
# ----------------------------------------------------------
set.seed(123)
data_sample <- data[sample(nrow(data), 10000), ]

# ----------------------------------------------------------
# 4. Create Log-Transformed Response Variable
# ----------------------------------------------------------
data_sample$log_stars <- log(data_sample$Stars + 1)

# ----------------------------------------------------------
# 5. Extract Year from Created.At
# ----------------------------------------------------------
data_sample$created_year <- as.numeric(substr(data_sample$Created.At, 1, 4))

# ----------------------------------------------------------
# 6. Create Age Group Factor (Old vs New)
# ----------------------------------------------------------
data_sample$age_group <- ifelse(data_sample$created_year <= 2018, "Old", "New")
data_sample$age_group <- factor(data_sample$age_group)

# ----------------------------------------------------------
# 7. Clean Language Column (Factor + Drop NAs)
# ----------------------------------------------------------
data_sample$language <- as.factor(data_sample$Language)
data_sample <- data_sample[!is.na(data_sample$language), ]
data_sample$language <- droplevels(data_sample$language)

# ----------------------------------------------------------
# 8. Keep Only Relevant Variables for Analysis
# ----------------------------------------------------------
data_sample <- data_sample %>% select(
  log_stars, Stars, Forks, Issues, Watchers, Size,
  language, age_group, created_year
)

# ----------------------------------------------------------
# 9. NA Check (Data Quality Check)
# ----------------------------------------------------------
colSums(is.na(data_sample))

# ----------------------------------------------------------
# 10. Convert Numeric Columns Explicitly
# ----------------------------------------------------------
data_sample$Stars    <- as.numeric(data_sample$Stars)
data_sample$Forks    <- as.numeric(data_sample$Forks)
data_sample$Issues   <- as.numeric(data_sample$Issues)
data_sample$Watchers <- as.numeric(data_sample$Watchers)
data_sample$Size     <- as.numeric(data_sample$Size)

# ----------------------------------------------------------
# 11. Remove Redundant Column (Watchers == Stars)
# ----------------------------------------------------------
data_sample <- data_sample %>% select(-Watchers)

# ----------------------------------------------------------
# 12. Trim Extreme Outliers (Top 0.1%) for Regression Stability
# ----------------------------------------------------------
data_sample <- data_sample %>% 
  filter(
    Stars < quantile(Stars, 0.999),
    Forks < quantile(Forks, 0.999),
    Issues < quantile(Issues, 0.999),
    Size  < quantile(Size,  0.999)
  )

# ----------------------------------------------------------
# 13. Log-Transform Size to Reduce Skew
# ----------------------------------------------------------
data_sample$log_size <- log(data_sample$Size + 1)

# Inspect cleaned numeric summaries
summary(data_sample[, c("Stars", "Forks", "Issues", "Size", "log_size")])

# ----------------------------------------------------------
# 14. Filter Languages for ANOVA (Keep Only Groups >= 50 Repos)
# ----------------------------------------------------------
lang_counts <- table(data_sample$language)
valid_langs <- names(lang_counts[lang_counts >= 50])

data_sample <- subset(data_sample, language %in% valid_langs)
data_sample$language <- droplevels(data_sample$language)

# Check balanced language groups
table(data_sample$language)

# ----------------------------------------------------------
# 15. (Optional) Save Cleaned Dataset for Reproducibility
# ----------------------------------------------------------
write.csv(data_sample, "cleaned_data.csv", row.names = FALSE)

############################################################
# PHASE 2 — EXPLORATORY DATA ANALYSIS (EDA)
############################################################
install.packages("GGally")

# ----------------------------------------------------------
# 1. Load Required Libraries
# ----------------------------------------------------------
library(dplyr)
library(ggplot2)
library(GGally)      # Scatterplot Matrix (SPLoM)
library(corrplot)    # Correlation heatmap visualization


# ----------------------------------------------------------
# 2. Summary Statistics for Key Variables
# ----------------------------------------------------------
summary_stats <- summary(data_sample[, c(
  "Stars", "Forks", "Issues", "Size", "log_size", "log_stars"
)])
summary_stats   # Print summary table


# ----------------------------------------------------------
# 3. Scatterplot Matrix (SPLoM)
# ----------------------------------------------------------

GGally::ggpairs(
  data_sample[, c("Stars", "Forks", "Issues", "log_size", "log_stars")],
  title = "Scatterplot Matrix of Key Quantitative Variables"
)


# ----------------------------------------------------------
# 4. Histograms for Distribution Check
# ----------------------------------------------------------
numeric_vars <- c("Stars", "Forks", "Issues", "Size", "log_stars", "log_size")

for (v in numeric_vars) {
  print(
    ggplot(data_sample, aes_string(v)) +
      geom_histogram(bins = 40, fill = "steelblue", color = "black") +
      theme_minimal() +
      ggtitle(paste("Histogram of", v))
  )
}


# ----------------------------------------------------------
# 5. Boxplot of Popularity by Language (ANOVA justification)
# ----------------------------------------------------------
ggplot(data_sample, aes(x = language, y = log_stars)) +
  geom_boxplot(fill = "orange", alpha = 0.75) +
  theme_minimal() +
  labs(
    title = "Distribution of log_stars by Programming Language",
    x = "Programming Language",
    y = "log(stars + 1)"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# ----------------------------------------------------------
# 6. Correlation Heatmap
# ----------------------------------------------------------
corr_data <- data_sample[, c("Stars", "Forks", "Issues", "log_size", "log_stars")]
correlation_matrix <- cor(corr_data)

corrplot(
  correlation_matrix,
  method = "color",
  type = "upper",
  addCoef.col = "black",
  title = "Correlation Heatmap of Key Variables",
  mar = c(0,0,2,0)
)


# ----------------------------------------------------------
# 7. Boxplot of Popularity by Age Group (For Two-Way ANOVA)
# ----------------------------------------------------------
ggplot(data_sample, aes(x = age_group, y = log_stars, fill = age_group)) +
  geom_boxplot(alpha = 0.8) +
  theme_minimal() +
  labs(
    title = "Popularity Comparison (log_stars): Old vs New Repositories",
    x = "Age Group",
    y = "log(stars + 1)"
  )


# ----------------------------------------------------------
# 8. Interaction Plot (Language × Age Group)
# ----------------------------------------------------------
interaction.plot(
  data_sample$language,
  data_sample$age_group,
  data_sample$log_stars,
  xlab = "Programming Language",
  ylab = "Mean log_stars",
  trace.label = "Age Group",
  col = c("red", "blue"),
  lwd = 2
)


############################################################
# PHASE 3 — REGRESSION MODELING
############################################################

# ----------------------------------------------------------
# 0. Load Required Libraries
# ----------------------------------------------------------
library(dplyr)
library(ggplot2)
library(car)          # VIF Check
library(lmtest)       # BP Test
library(sandwich)     # Robust SE

# ----------------------------------------------------------
# 1. Base Linear Regression Model
# ----------------------------------------------------------
model1 <- lm(log_stars ~ Forks + Issues + log_size, data = data_sample)
summary(model1)

# ----------------------------------------------------------
# 2. Add Language + Age Group (Categorical Predictors)
# ----------------------------------------------------------
model2 <- lm(log_stars ~ Forks + Issues + log_size +
               language + age_group,
             data = data_sample)
summary(model2)

# ----------------------------------------------------------
# 3. Interaction Model (Language × Age Group)
# ----------------------------------------------------------
model3 <- lm(log_stars ~ Forks + Issues + log_size +
               language * age_group,
             data = data_sample)
summary(model3)

# ----------------------------------------------------------
# 4. Model Comparison (Which Model Is Best?)
# ----------------------------------------------------------
anova(model1, model2)
anova(model2, model3)

# ----------------------------------------------------------
# 5. Regression Diagnostics (Assumption Checks)
# ----------------------------------------------------------

## 5.1 Residual Plots (Linearity + Homoscedasticity)
par(mfrow = c(2,2))
plot(model2)
par(mfrow = c(1,1))

## 5.2 Normality of Residuals (Q-Q Plot)
qqnorm(residuals(model2))
qqline(residuals(model2))

## 5.3 Residual Histogram
hist(residuals(model2), breaks = 50,
     main = "Residual Histogram",
     col = "skyblue")

## 5.4 Homoscedasticity Test (Breusch-Pagan)
bptest(model2)

## 5.5 Multicollinearity Check (VIF)
vif(model2)

# ----------------------------------------------------------
# 6. Robust Standard Errors (If BP test indicates heteroskedasticity)
# ----------------------------------------------------------
coeftest(model2, vcov = vcovHC(model2, type = "HC1"))

############################################################
# PHASE 4 —  ANOVA - ANALYSES 
 # 4.1 ONE WAY ANOVA
############################################################

library(dplyr)
library(ggplot2)
library(car)   # Only dependency you already have

# 1. Fit ANOVA Model
anova1 <- aov(log_stars ~ language, data = data_sample)
summary(anova1)

# 2. Summary stats by language
aggregate(log_stars ~ language, data_sample, summary)

# 3. Tukey HSD Post-hoc Test
tukey_lang <- TukeyHSD(anova1)
tukey_lang

# 4. Effect Size (MANUAL ETA-SQUARED)
anova_summary <- summary(anova1)[[1]]
SS_language <- anova_summary["language", "Sum Sq"]
SS_residual <- anova_summary["Residuals", "Sum Sq"]
eta_sq <- SS_language / (SS_language + SS_residual)
eta_sq

# 5. Assumption checks
qqnorm(residuals(anova1)); qqline(residuals(anova1))
leveneTest(log_stars ~ language, data = data_sample)

# 6. Boxplot
ggplot(data_sample, aes(language, log_stars)) +
  geom_boxplot(fill="orange") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1)) +
  labs(
    title="log_stars by Programming Language",
    x="Language", y="log(stars + 1)"
  )
############################################################
# 4.2 TWO-WAY ANOVA — Language × Age Group Interaction
############################################################

# Model
anova2 <- aov(log_stars ~ language * age_group, data = data_sample)
summary(anova2)

# Post-hoc Tests
## 1. Pairwise comparisons for Language (main effect)
pairwise.t.test(data_sample$log_stars, data_sample$language,
                p.adjust.method = "BH")

## 2. Pairwise comparisons for Interaction (Language × Age Group)
pairwise.t.test(data_sample$log_stars,
                interaction(data_sample$language, data_sample$age_group),
                p.adjust.method = "BH")

# Interaction Plot
interaction.plot(
  data_sample$language,
  data_sample$age_group,
  data_sample$log_stars,
  xlab = "Programming Language",
  ylab = "Mean log_stars",
  trace.label = "Age Group",
  col = c("red", "blue"),
  lwd = 2
)

# Assumption Checks
## Normality
qqnorm(residuals(anova2)); qqline(residuals(anova2))

## Equal Variances
leveneTest(log_stars ~ language * age_group, data = data_sample)


############################################################
# 4.3 INTERPRETATION PRINTING (SUMMARY)


cat("\n========== ONE-WAY ANOVA SUMMARY ==========\n")
print(summary(anova1))
cat("\nTukey Post-hoc:\n")
print(tukey_lang)
cat("\nEta-Squared:\n")
print(eta_sq)

cat("\n========== TWO-WAY ANOVA SUMMARY ==========\n")
print(summary(anova2))

############################################################


