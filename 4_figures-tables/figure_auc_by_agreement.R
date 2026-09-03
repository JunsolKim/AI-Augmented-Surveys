suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(ggplot2)
})

.args <- commandArgs(trailingOnly = FALSE)
.self <- sub("^--file=", "", .args[grep("^--file=", .args)])
BASE    <- normalizePath(file.path(dirname(.self), ".."))
IN_DIR  <- file.path(BASE, "data", "fig_table_gen")
OUT_DIR <- file.path(BASE, "output")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

dt_agg <- as.data.table(read_parquet(
  file.path(IN_DIR, "auc_by_agreement.parquet")))
cat("Loaded per-variable AUC table: ", nrow(dt_agg), " variables\n", sep = "")

# --- Bin into (a,b] deciles ----------------------------------------------
setnames(dt_agg, "pct_positive", "mean_response")
setorder(dt_agg, mean_response)
dt_agg[, mean_response_group := cut(mean_response,
                                    breaks = seq(0, 1, by = 0.1))]
dt_agg[mean_response == 0, mean_response_group := "(0,0.1]"]

# --- Per-bin mean + 95% CI via t.test ------------------------------------
tab <- dt_agg[!is.na(mean_response_group), .(
    n         = .N,
    mean_auc  = mean(auc, na.rm = TRUE),
    conf.low  = t.test(auc, conf.level = 0.95)$conf.int[1],
    conf.high = t.test(auc, conf.level = 0.95)$conf.int[2]
  ), by = "mean_response_group"]
setorder(tab, mean_response_group)
print(tab)

# --- Plot ----------------------------------------------------------------
g <- ggplot(tab, aes(x = mean_response_group,
                     y = mean_auc,
                     ymin = conf.low, ymax = conf.high)) +
  geom_pointrange() +
  theme_bw(base_size = 16) +
  theme(legend.position = "top",
        axis.text.x = element_text(size = 8, angle = 30, hjust = 1)) +
  labs(y = "Opinion levels AUC", x = "% Positive Response")

out_pdf <- file.path(OUT_DIR, "figure_auc_by_agreement.pdf")
out_png <- file.path(OUT_DIR, "figure_auc_by_agreement.png")
ggsave(out_pdf, g, width = 5, height = 4)
ggsave(out_png, g, width = 5, height = 4, dpi = 300)
cat("Saved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
