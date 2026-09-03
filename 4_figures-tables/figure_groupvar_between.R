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

POINT_COLOR <- "#8A3A3A"   # muted brick

theme_paper <- function(base_size = 16) {
  theme_bw(base_size = base_size, base_family = "sans") +
    theme(
      plot.title        = element_text(size = base_size + 1, face = "bold"),
      strip.background  = element_rect(fill = "gray96", color = NA),
      strip.text        = element_text(size = base_size, face = "bold",
                                       margin = margin(4, 0, 4, 0)),
      panel.grid.minor  = element_blank(),
      panel.grid.major  = element_line(color = "gray93", linewidth = 0.25),
      panel.border      = element_rect(color = "gray40", linewidth = 0.4),
      panel.spacing     = unit(0.9, "lines"),
      axis.ticks        = element_line(color = "gray40", linewidth = 0.3),
      axis.title        = element_text(size = base_size),
      axis.text         = element_text(size = base_size - 2, color = "gray25"),
      legend.position   = "none",
      plot.margin       = margin(10, 12, 10, 10)
    )
}

dt <- as.data.table(read_parquet(file.path(IN_DIR, "fig_groupvar_between.parquet")))
sm <- fread(file.path(IN_DIR, "fig_groupvar_between_summary.csv"))

lvls <- c("Age Group", "Gender Group", "Race Group",
          "Political Group", "Education Group")
dt[, demo_group := factor(demo_group, levels = lvls)]
sm[, demo_group := factor(demo_group, levels = lvls)]

annot <- sm[, .(demo_group,
                label = sprintf("ρ = %.3f\nMAE = %.4f", r, mae))]

g <- ggplot(dt, aes(x = pred_std, y = obs_std)) +
  geom_point(alpha = 0.12, size = 0.5, shape = 16, stroke = 0,
             color = POINT_COLOR) +
  facet_wrap(~ demo_group, nrow = 2, scales = "free") +
  geom_text(data = annot,
            aes(label = label, x = -Inf, y = Inf),
            hjust = -0.05, vjust = 1.1,
            size = 4.5, lineheight = 0.95,
            color = "gray15",
            inherit.aes = FALSE) +
  scale_x_continuous(expand = expansion(mult = c(0.02, 0.06))) +
  scale_y_continuous(expand = expansion(mult = c(0.02, 0.06))) +
  labs(x = "Predicted between-group SD",
       y = "Observed between-group SD") +
  theme_paper()

out_pdf <- file.path(OUT_DIR, "figure_groupvar_between.pdf")
out_png <- file.path(OUT_DIR, "figure_groupvar_between.png")
ggsave(out_pdf, g, width = 10, height = 6, device = cairo_pdf)
ggsave(out_png, g, width = 10, height = 6, dpi = 300, type = "cairo")
cat("Saved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
