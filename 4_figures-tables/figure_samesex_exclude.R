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
  file.path(IN_DIR, "samesex_exclude.parquet")))
setorder(dt, n_exclude)
dt[, n_exclude := factor(n_exclude, levels = sort(unique(n_exclude)))]

p <- ggplot(dt, aes(x = n_exclude, y = auc)) +
  geom_bar(stat = "identity", width = 0.6,
           colour = "black", fill = "black", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.3f", auc)),
            colour = "white", vjust = 2, size = 3.2) +
  labs(x = "N correlated opinions dropped from the training data",
       y = "model AUC for predicting same-sex marriage") +
  theme_minimal(base_size = 16) +
  theme(axis.title.y = element_text(size = 12))

out_pdf <- file.path(OUT_DIR, "figure_samesex_exclude.pdf")
out_png <- file.path(OUT_DIR, "figure_samesex_exclude.png")
ggsave(out_pdf, p, width = 6, height = 4)
ggsave(out_png, p, width = 6, height = 4, dpi = 300)
cat("Saved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
