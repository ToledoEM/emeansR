# Time e-means against Seurat (FindNeighbors + Louvain) on the GSE233163 PCs.
# Usage: Rscript single-cell-timing.R > timing.log 2>&1   (from any directory, after single-cell-run.R)
# Both methods get the same workers (EMEANSR_WORKERS, default physical cores - 1).
# Flat sweeps run one partition per forked worker, one core each; the bisecting tree
# is sequential across levels, so its emeans() calls get all workers as threads.
# EMEANSR_REPS sets repetitions (default 3); medians are reported.
# Reads and writes data/results/ next to this script: timing.rds, timing.csv,
# timing_summary.csv, and the timing figure (single-cell-time.png) with its values.

suppressPackageStartupMessages({
  library(emeansR)
  library(Seurat)
  library(dplyr)
  library(purrr)
  library(tibble)
  library(ggplot2)
  library(patchwork)
})

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- if (length(script)) dirname(normalizePath(script)) else getwd()
results_dir <- file.path(here, "data", "results")
fig_png <- file.path(results_dir, "single-cell-time.png")
fig_csv <- file.path(results_dir, "single-cell-time.csv")

cores <- parallel::detectCores(logical = FALSE)
workers <- as.integer(Sys.getenv("EMEANSR_WORKERS", max(1L, cores - 1L)))
if (.Platform$OS.type == "windows") workers <- 1L
n_reps <- as.integer(Sys.getenv("EMEANSR_REPS", 3L))
max_depth <- 5
min_cells <- 200
nepochs <- 3

log_msg <- function(...) message(format(Sys.time(), "%H:%M:%S "), sprintf(...))

pcs <- readRDS(file.path(results_dir, "embedding.rds"))$pca[, 1:20]
resolutions <- readRDS(file.path(results_dir, "run_info.rds"))$resolutions

# Run f over x in forked workers, one core each. Each job times itself; the sweep
# wall-clock includes forking and collecting results.
par_jobs <- function(x, f) {
  wall <- system.time(res <- parallel::mclapply(x, function(i) {
    options(emeansR.threads = 1L)
    t <- system.time(v <- f(i))[["elapsed"]]
    list(seconds = t, value = v)
  }, mc.cores = min(workers, length(x)), mc.preschedule = FALSE))[["elapsed"]]
  failed <- vapply(res, inherits, logical(1), "try-error")
  if (any(failed)) stop("worker failed: ", conditionMessage(attr(res[[which(failed)[1]]], "condition")))
  list(wall = wall, seconds = map_dbl(res, "seconds"), value = map_int(res, "value"))
}

# Bisecting tree as in single-cell-run.R, timing each level.
bisect_tree_timed <- function(x, max_depth, min_cells, seed) {
  labels <- rep("1", nrow(x))
  out <- vector("list", max_depth)
  for (depth in seq_len(max_depth)) {
    t <- system.time({
      child <- labels
      nodes <- unique(labels)
      for (i in seq_along(nodes)) {
        idx <- which(labels == nodes[i])
        if (length(idx) < min_cells) next
        fit <- emeans(x[idx, , drop = FALSE], 2, nepochs = nepochs, seed = seed * 1000 + depth * 100 + i)
        child[idx] <- paste0(nodes[i], ".", fit$cluster)
      }
      labels <- child
    })[["elapsed"]]
    out[[depth]] <- tibble(level = depth, clusters = n_distinct(labels), seconds = t)
  }
  bind_rows(out) |> mutate(cumulative = cumsum(seconds))
}

# Warm-up, untimed: load code paths for both methods.
local({
  small <- pcs[seq_len(2000), ]
  g <- FindNeighbors(small, k.param = 20, verbose = FALSE)
  invisible(FindClusters(g$snn, resolution = 0.5, verbose = FALSE))
  invisible(emeans(small, 5, nepochs = nepochs, seed = 1))
})
log_msg("warm-up done; %d cells, %d workers, %d reps", nrow(pcs), workers, n_reps)

graph <- louvain <- flat <- sweep <- tree <- list()
k_values <- NULL
for (rep in seq_len(n_reps)) {
  # Seurat: graph once, then one resolution per worker.
  t_graph <- system.time(snn <- FindNeighbors(pcs, k.param = 20, verbose = FALSE)$snn)[["elapsed"]]
  lv <- par_jobs(resolutions, function(r) n_distinct(FindClusters(snn, resolution = r, verbose = FALSE)[[1]]))
  rm(snn)
  invisible(gc())
  if (is.null(k_values)) k_values <- lv$value
  if (!identical(lv$value, k_values)) stop("Louvain cluster counts changed between reps")
  log_msg("rep %d Seurat: graph %.1f s, Louvain sweep %.1f s", rep, t_graph, lv$wall)

  # e-means: same cluster counts, one k per worker.
  em <- par_jobs(k_values, function(k) n_distinct(emeans(pcs, k, nepochs = nepochs, seed = k)$cluster))
  log_msg("rep %d e-means: sweep %.1f s", rep, em$wall)

  # e-means tree: levels in sequence, all workers as threads.
  options(emeansR.threads = workers)
  tr <- bisect_tree_timed(pcs, max_depth, min_cells, seed = 1)
  log_msg("rep %d e-means tree: %.1f s", rep, sum(tr$seconds))

  graph[[rep]] <- tibble(rep = rep, seconds = t_graph)
  louvain[[rep]] <- tibble(method = "Louvain", resolution = resolutions, k = lv$value, rep = rep,
                           seconds = lv$seconds)
  flat[[rep]] <- tibble(method = "e-means", resolution = resolutions, k = k_values, rep = rep,
                        seconds = em$seconds, k_found = em$value)
  sweep[[rep]] <- tibble(rep = rep,
                         method = c("Seurat: FindNeighbors + Louvain", "e-means", "e-means tree"),
                         seconds = c(t_graph + lv$wall, em$wall, sum(tr$seconds)))
  tree[[rep]] <- mutate(tr, rep = rep)
}

# system.time() reports milliseconds; round away floating-point noise.
r3 <- function(d) mutate(d, across(any_of(c("seconds", "cumulative", "graph_seconds")), ~ round(.x, 3)))
graph <- r3(bind_rows(graph))
per_job <- r3(bind_rows(bind_rows(louvain), bind_rows(flat)))
sweep <- r3(bind_rows(sweep))
tree <- r3(bind_rows(tree))

saved_k <- read.csv(file.path(results_dir, "louvain_sweep.csv"))$clusters
if (!identical(as.integer(saved_k), k_values)) warning("Louvain cluster counts differ from louvain_sweep.csv")

# Medians across reps: the values behind the figure.
partition <- bind_rows(
  per_job |> group_by(method, resolution, k) |> summarise(seconds = median(seconds), .groups = "drop"),
  tree |> group_by(level, k = clusters) |> summarise(seconds = median(cumulative), .groups = "drop") |>
    mutate(method = "e-means tree (cumulative)")
)
totals <- sweep |> group_by(method) |> summarise(seconds = median(seconds), .groups = "drop")
summary_tbl <- bind_rows(
  partition |> mutate(panel = "partition", graph_seconds = NA_real_),
  totals |> mutate(panel = "sweep")
) |>
  mutate(graph_seconds = if_else(method == "Louvain", median(graph$seconds), graph_seconds)) |>
  select(panel, method, resolution, level, k, seconds, graph_seconds) |>
  r3()

info <- list(date = Sys.time(), workers = workers, reps = n_reps, cells = nrow(pcs), dims = ncol(pcs),
             nepochs = nepochs, max_depth = max_depth, min_cells = min_cells, resolutions = resolutions,
             cpu = tryCatch(system("sysctl -n machdep.cpu.brand_string", intern = TRUE), error = function(e) NA),
             session = sessionInfo())
timing <- list(per_job = per_job, graph = graph, sweep = sweep, tree = tree, summary = summary_tbl, info = info)

saveRDS(timing, file.path(results_dir, "timing.rds"))
write.csv(bind_rows(per_job, graph |> mutate(method = "FindNeighbors")) |> select(method, resolution, k, rep, seconds),
          file.path(results_dir, "timing.csv"), row.names = FALSE)
write.csv(summary_tbl, file.path(results_dir, "timing_summary.csv"), row.names = FALSE)
write.csv(summary_tbl, fig_csv, row.names = FALSE)

# Figure: the same code is in single-cell-explore.Rmd and the article.
timing_plot <- function(timing) {
  s <- timing$summary
  colours <- c("Louvain" = "#a04000", "e-means" = "#1f4e79", "e-means tree (cumulative)" = "#7fb3d5",
               "Seurat: FindNeighbors + Louvain" = "#a04000", "e-means tree" = "#7fb3d5")
  part <- filter(s, panel == "partition")
  graph_s <- median(timing$graph$seconds)
  left <- ggplot(part, aes(k, seconds, colour = method)) +
    geom_hline(yintercept = graph_s, linetype = "dashed", colour = "#e59866") +
    annotate("text", x = min(part$k), y = graph_s, label = sprintf("FindNeighbors, once: %.1f s", graph_s),
             hjust = 0, vjust = -0.5, size = 3, colour = "#a04000") +
    geom_line() +
    geom_point(size = 2) +
    scale_x_continuous(trans = "log2", breaks = c(2, 4, 8, 16, 32, 64)) +
    scale_y_log10(labels = scales::label_number(accuracy = 0.01, drop0trailing = TRUE)) +
    scale_colour_manual(values = colours) +
    labs(x = "clusters (log scale)", y = "median seconds (log scale)", colour = NULL,
         title = "One partition") +
    theme_minimal() +
    theme(legend.position = "bottom")
  tot <- filter(s, panel == "sweep") |>
    mutate(method = factor(method, levels = c("Seurat: FindNeighbors + Louvain", "e-means", "e-means tree")))
  right <- ggplot(tot, aes(method, seconds, fill = method)) +
    geom_col(width = 0.6) +
    geom_text(aes(label = sprintf("%.1f s", seconds)), vjust = -0.4, size = 3.2) +
    scale_fill_manual(values = colours, guide = "none") +
    scale_x_discrete(labels = c("Seurat\n14 resolutions", "e-means\n14 k", "e-means tree\n5 levels")) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(x = NULL, y = "wall-clock seconds",
         title = sprintf("Whole sweep, %d workers", timing$info$workers)) +
    theme_minimal()
  left + right + plot_layout(widths = c(3, 2)) +
    plot_annotation(title = sprintf("Clustering %s cells (GSE233163, 20 PCs)",
                                    format(timing$info$cells, big.mark = ",")))
}

ggsave(fig_png, timing_plot(timing), width = 9, height = 4.4, dpi = 110, bg = "white")
log_msg("saved %s, %s, timing.rds/csv in %s", fig_png, fig_csv, results_dir)
print(totals)
