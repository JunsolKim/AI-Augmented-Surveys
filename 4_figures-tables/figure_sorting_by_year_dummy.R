suppressPackageStartupMessages({
  library(data.table); library(arrow); library(ggplot2); library(ggsci)
})

.args <- commandArgs(trailingOnly = FALSE)
.self <- sub("^--file=", "", .args[grep("^--file=", .args)])
BASE    <- normalizePath(file.path(dirname(.self), ".."))
IN_PATH <- file.path(BASE, "data", "fig_table_gen", "sorting_by_year.parquet")
OUT_DIR <- file.path(BASE, "output")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

dt <- as.data.table(read_parquet(IN_PATH))
cat(sprintf("Loaded %d (variable, year) rows; year range %d-%d\n",
            nrow(dt), min(dt$year), max(dt$year)))

year_levels <- sort(unique(dt$year))
dt[, year_f := factor(year, levels = year_levels)]

cat("\nRows per year:\n"); print(dt[, .N, by = year][order(year)])

fit <- lm(auc ~ abs_corr_with_ideo * year_f, data = dt)
cat("\nFit dim: ", length(coef(fit)), " coefficients\n", sep = "")

levels_corr <- c(low = -1.0, mid = 0.0, high = 1.0)
cat(sprintf("\nCutpoints (z-scored |Corr with Ideology|): low=%.2f mid=%.2f high=%.2f\n",
            levels_corr["low"], levels_corr["mid"], levels_corr["high"]))

newdata <- CJ(abs_corr_with_ideo = unname(levels_corr),
              year_f             = factor(year_levels, levels = year_levels))
pred_mat <- predict(fit, newdata = newdata, interval = "confidence", level = 0.95)
newdata[, `:=`(estimate  = pred_mat[, "fit"],
               conf.low  = pred_mat[, "lwr"],
               conf.high = pred_mat[, "upr"])]
newdata[, level := factor(abs_corr_with_ideo,
                          levels = unname(levels_corr),
                          labels = c("low", "mid", "high"))]
newdata[, year := as.integer(as.character(year_f))]

cat("\nPredicted AUC by year and level:\n")
print(newdata[, .(year, level,
                  estimate = round(estimate, 4),
                  ci = sprintf("[%.3f, %.3f]", conf.low, conf.high))])
cat("\nFirst-year vs last-year by level:\n")
print(newdata[, .(first_year = round(estimate[year == min(year)], 4),
                  last_year  = round(estimate[year == max(year)], 4),
                  delta      = round(estimate[year == max(year)] -
                                     estimate[year == min(year)], 4)),
              by = level])

g <- ggplot(newdata, aes(x = year, y = estimate,
                         ymin = conf.low, ymax = conf.high,
                         colour = level, fill = level, group = level)) +
  geom_ribbon(alpha = 0.25, colour = NA) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.3) +
  scale_colour_d3(name = "|Corr with Ideology|") +
  scale_fill_d3(name   = "|Corr with Ideology|") +
  labs(x = "Year", y = "Predicted AUC at opinion levels") +
  theme_bw(base_size = 16) +
  theme(legend.position = "top",
        panel.grid.minor = element_blank())

out_pdf <- file.path(OUT_DIR, "figure_sorting_by_year_dummy.pdf")
out_png <- file.path(OUT_DIR, "figure_sorting_by_year_dummy.png")
ggsave(out_pdf, g, width = 6, height = 3.8)
ggsave(out_png, g, width = 6, height = 3.8, dpi = 300)
cat("\nSaved:\n  ", out_pdf, "\n  ", out_png, "\n", sep = "")
