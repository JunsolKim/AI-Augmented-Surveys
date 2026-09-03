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

dt <- as.data.table(read_parquet(
  file.path(IN_DIR, "missing_prop_auc.parquet")))

dt[, split := factor(
  split,
  levels = c("impute", "partial", "total"),
  labels = c("Missing Data Imputation", "Retrodiction",
             "Unasked Opinion Prediction"))]
dt[, model_label := factor(model_label, levels = c("Alpaca-7b", "MF"),
                           labels = c("Alpaca-7b", "Matrix Factorization"))]

model_colors <- c("Alpaca-7b" = "#8B1A1A",
                  "Matrix Factorization" = "grey60")

p <- ggplot(dt, aes(x = missing_pct, y = auc,
                    color = model_label, group = model_label)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.6) +
  facet_wrap(~ split, nrow = 1) +
  scale_color_manual(name = NULL, values = model_colors) +
  scale_x_continuous(breaks = seq(10, 90, 10)) +
  labs(x = "Proportion of Missing Data (%)", y = "AUC") +
  theme_bw(base_size = 22) +
  theme(panel.grid.minor = element_blank(),
        strip.background = element_rect(fill = "grey90", color = NA),
        strip.text = element_text(face = "plain", size = 18),
        axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        legend.text = element_text(size = 18),
        legend.position = "bottom")

out_pdf <- file.path(OUT_DIR, "figure_missing_prop.pdf")
out_png <- file.path(OUT_DIR, "figure_missing_prop.png")
ggsave(out_pdf, p, width = 11, height = 4.5)
ggsave(out_png, p, width = 11, height = 4.5, dpi = 300)
cat("Saved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
