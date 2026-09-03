suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(ggplot2)
  library(estimatr)
  library(ggsci)
})

.args <- commandArgs(trailingOnly = FALSE)
.self <- sub("^--file=", "", .args[grep("^--file=", .args)])
BASE    <- normalizePath(file.path(dirname(.self), ".."))
IN_DIR  <- file.path(BASE, "data", "fig_table_gen")
OUT_DIR <- file.path(BASE, "output")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

dt <- as.data.table(read_parquet(
  file.path(IN_DIR, "opinionauc_per_varyear.parquet")))

# Standardise `t` once globally (across all tasks). The input has `t`
# equal to raw year, so we z-score it here.
dt[, t := scale(t)[, 1]]

FEATURES <- c("t", "year_2021", "response_rate", "condition_ballot",
              "n_year", "n_response", "two_category",
              "p_dk", "p_refuse", "p_skip", "p_inapplicable",
              "var_response", "cos_sim", "abs_corr_with_ideo")

FEATURE_LABELS <- c(
  "t"                  = "Survey years",
  "year_2021"          = "Year = 2021",
  "response_rate"      = "Yearly Response Rate",
  "condition_ballot"   = "Under Split Ballot Design",
  "n_year"             = "N of collected survey periods",
  "n_response"         = "N individual response for opinion",
  "two_category"       = "binary response (vs three or more)",
  "p_dk"               = "P(Do not know response)",
  "p_skip"             = "P(Skip response)",
  "p_refuse"           = "P(Refuse response)",
  "p_inapplicable"     = "P(Inapplicable response)",
  "abs_corr_with_ideo" = "Absolute Corr. with political ideology",
  "var_response"       = "Variance of response",
  "cos_sim"            = "Average Cosine Similarity from Other Opinions"
)

TASK_ORDER <- c("Missing Data Imputation", "Retrodiction",
                "Unasked Opinion Prediction")
TASK_LABELS <- c("Missing Data Imputation"    = "A. Missing data imputation",
                 "Retrodiction"               = "B. Retrodiction",
                 "Unasked Opinion Prediction" = "C. Unasked Opinion Prediction")

FORMULA <- as.formula(paste("auc ~", paste(FEATURES, collapse = " + ")))

fit_task <- function(task_label) {
  sub <- dt[task == task_label]
  sub <- sub[complete.cases(sub[, c("auc", FEATURES), with = FALSE])]
  if (nrow(sub) < 30) return(NULL)
  fit <- lm_robust(FORMULA, data = sub, se_type = "stata")
  tidied <- as.data.table(broom::tidy(fit, conf.int = TRUE))
  tidied <- tidied[term %in% FEATURES]
  tidied[, task_name := task_label]
  tidied
}

model_out <- rbindlist(lapply(TASK_ORDER, fit_task), use.names = TRUE)

# Pretty labels + row order (top to bottom).
LABEL_ORDER <- rev(unname(FEATURE_LABELS))
model_out[, label := factor(FEATURE_LABELS[term], levels = LABEL_ORDER)]
model_out[, task_label := factor(task_name, levels = TASK_ORDER,
                                 labels = TASK_LABELS[TASK_ORDER])]

model_out[, sig := "insig"]
model_out[conf.high < 0 & estimate < 0, sig := "sig"]
model_out[conf.low  > 0 & estimate > 0, sig := "sig"]
model_out[, direction := factor(ifelse(estimate < 0, "negative", "positive"),
                                levels = c("positive", "negative"))]

p <- ggplot(model_out,
            aes(x = estimate * 100, y = label,
                color = direction, shape = sig)) +
  geom_segment(aes(x = 0, xend = estimate * 100, y = label, yend = label),
               linewidth = 0.4) +
  geom_point(size = 1.6) +
  facet_grid(. ~ task_label) +
  geom_vline(xintercept = 0, color = "grey") +
  scale_color_aaas(name = NULL) +
  scale_shape_manual(name = NULL, values = c("sig" = 19, "insig" = 4)) +
  theme_bw(base_size = 20) +
  theme(legend.position = "none",
        strip.background = element_blank(),
        strip.text = element_text(size = 15),
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 18),
        panel.grid.minor = element_blank()) +
  labs(y = NULL, x = "Marginal Effects on Opinion-level AUC (%)")

out_pdf <- file.path(OUT_DIR, "figure_opinionauc.pdf")
out_png <- file.path(OUT_DIR, "figure_opinionauc.png")
ggsave(out_pdf, p, width = 14, height = 4.8)
ggsave(out_png, p, width = 14, height = 4.8, dpi = 300)
cat("Saved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
