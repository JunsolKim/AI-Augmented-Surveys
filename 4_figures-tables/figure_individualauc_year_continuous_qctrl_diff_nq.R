suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(ggplot2)
  library(estimatr)
  library(modelsummary)
  library(ggsci)
})

.args <- commandArgs(trailingOnly = FALSE)
.self <- sub("^--file=", "", .args[grep("^--file=", .args)])
BASE    <- normalizePath(file.path(dirname(.self), ".."))
IN_DIR  <- file.path(BASE, "data", "fig_table_gen")
OUT_DIR <- file.path(BASE, "output")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

dt   <- as.data.table(read_parquet(file.path(IN_DIR, "individualauc_per_respondent.parquet")))
ctrl <- as.data.table(read_parquet(file.path(IN_DIR, "_period_controls.parquet")))

dt <- merge(dt, ctrl[, .(yearid_id, n_q, mean_diff)], by = "yearid_id", all.x = TRUE)

dt[, sex       := factor(sex,       levels = c("male", "female"))]
dt[, race      := factor(race,      levels = c("White", "Black", "Other"))]
dt[, degree    := factor(degree,    levels = c("below_hs", "hs",
                                               "some_college", "bs", "bs_above"))]
dt[, region4   := factor(region4,   levels = c("NE", "MW", "S", "W"))]
dt[, income_q5 := factor(income_q5, levels = c("Income Q1", "Income Q2",
                                               "Income Q3", "Income Q4",
                                               "Income Q5", "Income NA"))]
dt[, age_group := factor(age_group, levels = c("18-29", "30-44", "45-59",
                                               "60-74", "75-89"))]
dt[, partyid7 := factor(partyid7, levels = c("independent",
                                             "strong democrat",
                                             "weak democrat",
                                             "lean toward democrat",
                                             "lean toward republican",
                                             "weak republican",
                                             "strong republican",
                                             "other party"))]
dt[, year_dec := (as.numeric(year) - 1972) / 10]

FORMULA <- auc ~ sex + race + degree + region4 + income_q5 +
                 age_group + year_dec + partyid7 +
                 n_q + mean_diff

TASK_ORDER <- c("Missing Data Imputation",
                "Retrodiction",
                "Unasked Opinion Prediction")
TASK_LABELS <- c("Missing Data Imputation"    = "A. Missing data imputation",
                 "Retrodiction"               = "B. Retrodiction",
                 "Unasked Opinion Prediction" = "C. Unasked Opinion Prediction")

cm <- c(
  "sexfemale"                                   = "Female (vs Male)",
  "raceBlack"                                   = "Black (vs White)",
  "raceOther"                                   = "Other (vs White)",
  "degreehs"                                    = "HS graduate (vs lt HS)",
  "degreesome_college"                          = "Some college (vs lt HS)",
  "degreebs"                                    = "BS (vs lt HS)",
  "degreebs_above"                              = "MS+ (vs lt HS)",
  "region4MW"                                   = "Midwest (vs East)",
  "region4S"                                    = "South (vs East)",
  "region4W"                                    = "West (vs East)",
  "income_q5Income Q2"                          = "Income Q2 (vs Q1)",
  "income_q5Income Q3"                          = "Income Q3 (vs Q1)",
  "income_q5Income Q4"                          = "Income Q4 (vs Q1)",
  "income_q5Income Q5"                          = "Income Q5 (vs Q1)",
  "income_q5Income NA"                          = "Income missing (vs Q1)",
  "age_group30-44"                              = "Age 30-44 (vs 18-29)",
  "age_group45-59"                              = "Age 45-59 (vs 18-29)",
  "age_group60-74"                              = "Age 60-74 (vs 18-29)",
  "age_group75-89"                              = "Age 75-89 (vs 18-29)",
  "year_dec"                                    = "Year (per +10 yrs)",
  "partyid7strong democrat"                     = "Strong Democrat (vs Independent)",
  "partyid7weak democrat"                       = "Weak Democrat (vs Independent)",
  "partyid7lean toward democrat"                = "Lean toward Democrat (vs Independent)",
  "partyid7lean toward republican"              = "Lean toward Republican (vs Independent)",
  "partyid7weak republican"                     = "Weak Republican (vs Independent)",
  "partyid7strong republican"                   = "Strong Republican (vs Independent)",
  "partyid7other party"                         = "Something else (vs Independent)"
)

fit_task <- function(task_label) {
  sub <- dt[task == task_label]
  needed <- c("auc","year_dec","sex","race","degree","region4","income_q5",
              "age_group","partyid7","n_q","mean_diff")
  sub <- sub[complete.cases(sub[, ..needed])]
  if (nrow(sub) < 50) return(NULL)
  fit <- lm_robust(FORMULA, data = sub, se_type = "stata")
  tidied <- as.data.table(broom::tidy(fit, conf.int = TRUE))
  tidied[, task_name := task_label]
  tidied
}

model_out <- rbindlist(lapply(TASK_ORDER, fit_task), use.names = TRUE)

model_out <- model_out[term %in% names(cm)]
model_out[, term := factor(cm[term], levels = rev(unname(cm)))]

model_out[, covariate_group := fcase(
  grepl("Male", term),         "Sex",
  grepl("White", term),        "Race",
  grepl("lt HS", term),        "Education",
  grepl("East", term),         "Region",
  grepl("Q1", term),           "Income",
  grepl("18-29", term),        "Age",
  grepl("per \\+10 yrs", term),"Period",
  grepl("Independent", term),  "Party ID"
)]
model_out[, covariate_group := factor(covariate_group,
  levels = c("Sex", "Age", "Period", "Race", "Region",
             "Education", "Income", "Party ID"))]

model_out[, task_label := factor(task_name, levels = TASK_ORDER,
                                 labels = TASK_LABELS[TASK_ORDER])]

model_out[, sig := "insig"]
model_out[conf.high < 0 & estimate < 0, sig := "sig"]
model_out[conf.low  > 0 & estimate > 0, sig := "sig"]
model_out[, direction := factor(ifelse(estimate < 0, "negative", "positive"),
                                levels = c("positive", "negative"))]

p <- ggplot(model_out,
            aes(x = estimate * 100, y = term,
                color = direction, shape = sig)) +
  geom_segment(aes(x = 0, xend = estimate * 100, y = term, yend = term),
               linewidth = 0.4) +
  geom_point(size = 1.2) +
  facet_grid(covariate_group ~ task_label, scales = "free_y", space = "free") +
  geom_vline(xintercept = 0, color = "grey") +
  scale_color_aaas(name = NULL) +
  scale_shape_manual(name = NULL, values = c("sig" = 19, "insig" = 4)) +
  theme_bw(base_size = 16) +
  theme(legend.position = "none",
        strip.background = element_blank(),
        panel.grid.minor = element_blank()) +
  labs(y = NULL,
       x = "Marginal Effects on individual-level AUC (%)")

out_pdf <- file.path(OUT_DIR, "figure_individualauc_year_continuous_qctrl_diff_nq.pdf")
out_png <- file.path(OUT_DIR, "figure_individualauc_year_continuous_qctrl_diff_nq.png")
ggsave(out_pdf, p, width = 9, height = 6)
ggsave(out_png, p, width = 9, height = 6, dpi = 300)
cat("Saved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
