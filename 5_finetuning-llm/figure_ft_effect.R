suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

.args <- commandArgs(trailingOnly = FALSE)
.self <- sub("^--file=", "", .args[grep("^--file=", .args)])
HERE    <- normalizePath(dirname(.self))
IN_CSV  <- file.path(HERE, "results_3x3.csv")
OUT_DIR <- normalizePath(file.path(HERE, "..", "output"), mustWork = FALSE)
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

MODEL_LEVELS <- c("Baseline\n(frozen)",
                  "Fine-tuned\nLoRA q,v",
                  "Fine-tuned\nLoRA all-linear")
MODEL_COLORS <- c("Baseline\n(frozen)"          = "#444444",
                  "Fine-tuned\nLoRA q,v"        = "#B22222",
                  "Fine-tuned\nLoRA all-linear" = "#FF8C00")

TASK_LEVELS <- c("Missing data imputation",
                 "Retrodiction",
                 "Unasked opinion prediction")
METRIC_LEVELS <- c("AUC", "Accuracy", "F1")

MODEL_KEYS <- c(baseline = MODEL_LEVELS[1],
                lora_qv = MODEL_LEVELS[2],
                lora_all_linear = MODEL_LEVELS[3])

df <- fread(IN_CSV)
df[, task   := factor(task, levels = TASK_LEVELS)]
df[, model  := factor(MODEL_KEYS[model], levels = MODEL_LEVELS)]
df[, metric := factor(metric, levels = METRIC_LEVELS)]
stopifnot(nrow(df) == 27L, !anyNA(df$task), !anyNA(df$model), !anyNA(df$metric))

Y_LO <- 0.5
Y_HI <- 0.93

p <- ggplot(df, aes(x = model, y = value, color = model)) +
  geom_segment(aes(xend = model, y = Y_LO, yend = value), linewidth = 1.0) +
  geom_point(size = 3.5) +
  geom_text(aes(label = sprintf("%.4f", value)),
            vjust = -1.0, fontface = "bold", size = 3.4, show.legend = FALSE) +
  facet_grid(task ~ metric, switch = "y") +
  scale_color_manual(values = MODEL_COLORS, guide = "none") +
  scale_y_continuous(limits = c(Y_LO, Y_HI), expand = c(0, 0)) +
  labs(x = NULL, y = NULL) +
  theme_bw(base_size = 13) +
  theme(
    panel.grid.major   = element_blank(),
    panel.grid.minor   = element_blank(),
    strip.background   = element_blank(),
    strip.text         = element_text(face = "bold", size = 13),
    strip.placement    = "outside",
    axis.text.x        = element_text(size = 12),
    axis.text.y        = element_text(size = 11),
    panel.border       = element_rect(color = "grey30", fill = NA, linewidth = 0.4),
    panel.spacing      = unit(0.8, "lines")
  )

out_pdf <- file.path(OUT_DIR, "figure_ft_effect.pdf")
out_png <- file.path(OUT_DIR, "figure_ft_effect.png")
ggsave(out_pdf, p, width = 10, height = 8.5)
ggsave(out_png, p, width = 10, height = 8.5, dpi = 300)
cat("Saved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
