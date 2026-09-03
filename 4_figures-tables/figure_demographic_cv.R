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

TASK_ORDER <- c("Missing Data Imputation", "Retrodiction",
                "Unasked Opinion Prediction")
TASK_COLORS <- c("Missing Data Imputation"    = "#B22222",
                 "Retrodiction"               = "#FF8C00",
                 "Unasked Opinion Prediction" = "#006400")
CONDITION_ORDER <- c("D0P0", "D0P1", "D1P0", "D1P1")

dt <- as.data.table(read_parquet(file.path(IN_DIR, "demographic_cv.parquet")))
dt[, task      := factor(task,      levels = TASK_ORDER)]
dt[, condition := factor(condition, levels = CONDITION_ORDER)]

p <- ggplot(dt, aes(x = condition, y = auc, colour = task)) +
  geom_segment(aes(xend = condition, y = 0.5, yend = auc),
               colour = "black", linewidth = 0.4) +
  geom_point(size = 3) +
  facet_wrap(~ task, nrow = 1) +
  scale_colour_manual(values = TASK_COLORS, guide = "none") +
  coord_cartesian(ylim = c(0.5, 0.9)) +
  labs(x = NULL, y = "AUC") +
  theme_bw(base_size = 22) +
  theme(strip.background = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position  = "none",
        strip.text       = element_text(size = 17),
        axis.text        = element_text(size = 18),
        axis.title       = element_text(size = 22))

out_pdf <- file.path(OUT_DIR, "figure_demographic_cv.pdf")
out_png <- file.path(OUT_DIR, "figure_demographic_cv.png")
ggsave(out_pdf, p, width = 11, height = 4.5)
ggsave(out_png, p, width = 11, height = 4.5, dpi = 300)
cat("saved: ", out_pdf, "\n", "saved: ", out_png, "\n", sep = "")
