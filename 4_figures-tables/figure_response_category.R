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
  file.path(IN_DIR, "response_category_auc.parquet")))
dt <- dt[n_category >= 2 & n_category <= 11]
dt[, parity := factor(parity, levels = c("even", "odd"))]

g <- ggplot(dt, aes(x = n_category, y = auc, colour = parity)) +
  geom_segment(aes(xend = n_category, y = 0.5, yend = auc),
               linewidth = 0.6) +
  geom_point(aes(size = n_vars)) +
  scale_colour_manual(name = "N response option",
    values = c(even = "#0072B2", odd = "#D55E00")) +
  scale_size_continuous(name = "N questions",
    range = c(2, 8), breaks = c(100, 200, 300)) +
  scale_x_continuous(breaks = seq(2, 11, 1), limits = c(1.5, 11.5)) +
  scale_y_continuous(limits = c(0.5, NA), breaks = seq(0.5, 1, 0.1)) +
  labs(x = "N Response Categories", y = "AUC") +
  theme_bw(base_size = 16) +
  theme(legend.position = "top",
        panel.grid.minor = element_blank())

out_pdf <- file.path(OUT_DIR, "figure_response_category.pdf")
out_png <- file.path(OUT_DIR, "figure_response_category.png")
ggsave(out_pdf, g, width = 8, height = 5)
ggsave(out_png, g, width = 8, height = 5, dpi = 300)
cat("Saved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
