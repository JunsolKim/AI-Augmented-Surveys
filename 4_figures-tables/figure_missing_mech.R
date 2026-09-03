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
  file.path(IN_DIR, "missing_mech_auc.parquet")))

dt[, mechanism   := factor(mechanism,   levels = c("MCAR", "MAR", "MNAR"))]
dt[, model_label := factor(model_label, levels = c("Alpaca-7b", "MF", "MICE"))]

mech_colors <- c("MCAR" = "#F8766D", "MAR" = "#00BA38", "MNAR" = "#619CFF")

p <- ggplot(dt, aes(x = model_label, y = auc, color = mechanism)) +
  geom_segment(aes(xend = model_label, y = 0.6, yend = auc),
               color = "black", linewidth = 0.5) +
  geom_point(size = 4) +
  geom_text(aes(label = sprintf("%.3f", auc)),
            vjust = -1.0, size = 3.5, color = "black") +
  facet_wrap(~ mechanism, nrow = 1) +
  scale_color_manual(values = mech_colors, guide = "none") +
  scale_y_continuous(limits = c(0.6, 0.95),
                     breaks = seq(0.6, 0.9, 0.1)) +
  labs(x = NULL, y = "AUC") +
  theme_bw(base_size = 17) +
  theme(panel.grid.minor = element_blank(),
        strip.background = element_rect(fill = "grey90", color = NA),
        strip.text = element_text(face = "plain"))

out_pdf <- file.path(OUT_DIR, "figure_missing_mech.pdf")
out_png <- file.path(OUT_DIR, "figure_missing_mech.png")
ggsave(out_pdf, p, width = 7, height = 4.5)
ggsave(out_png, p, width = 7, height = 4.5, dpi = 300)
cat("Saved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
