# Recursive e-means tree on GSE233163: compute everything and save every result.
# Usage: Rscript single-cell-run.R [input_rds] [metadata_csv] [out_dir]   (from any directory)
# Defaults read the files download.R saved in data/ next to this script and write to data/results/.
# Independent jobs (seed reruns, per-level stability, Louvain resolutions) run in
# forked workers via parallel::mclapply; set EMEANSR_WORKERS to change their number.
# Each step saves its output as soon as it finishes; on a re-run, a step whose
# file already exists is loaded instead of recomputed. Delete a file to redo it.

suppressPackageStartupMessages({
  library(emeansR)
  library(Seurat)
  library(mclust)
  library(dplyr)
  library(purrr)
  library(tibble)
})

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
data_dir <- file.path(if (length(script)) dirname(normalizePath(script)) else getwd(), "data")
args <- commandArgs(TRUE)
rds_path <- if (length(args) >= 1) args[1] else file.path(data_dir, "GSE233163_seurat_object.rds.gz")
meta_path <- if (length(args) >= 2) args[2] else file.path(data_dir, "GSE233163_cells_metadata.csv.gz")
out_dir <- if (length(args) >= 3) args[3] else file.path(data_dir, "results")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cores <- parallel::detectCores(logical = FALSE)
threads <- max(1L, cores - 1L)
options(emeansR.threads = threads)
workers <- as.integer(Sys.getenv("EMEANSR_WORKERS", max(1L, cores - 1L)))
if (.Platform$OS.type == "windows") workers <- 1L  # mclapply cannot fork on Windows

# Parallel map over independent jobs; each worker gets `inner` emeans threads.
par_map <- function(x, f, inner = 1L) {
  parallel::mclapply(x, function(i) {
    options(emeansR.threads = inner)
    f(i)
  }, mc.cores = min(workers, length(x)), mc.preschedule = FALSE)
}
max_depth <- 5
min_cells <- 200
n_reruns <- 10
resolutions <- c(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2)

log_msg <- function(...) message(format(Sys.time(), "%H:%M:%S "), sprintf(...))

# Load the saved result if present, else compute, save and time it.
step <- function(name, expr) {
  path <- file.path(out_dir, paste0(name, ".rds"))
  if (file.exists(path)) {
    log_msg("%s: loaded %s", name, path)
    return(readRDS(path))
  }
  log_msg("%s: computing", name)
  t <- system.time(value <- force(expr))[["elapsed"]]
  saveRDS(value, path)
  log_msg("%s: %.1f s, saved %s", name, t, path)
  timings[[name]] <<- t
  value
}
timings_path <- file.path(out_dir, "timings.rds")
timings <- if (file.exists(timings_path)) readRDS(timings_path) else list()

# 1. Embedding: PCA and UMAP with the authors' logged settings (integrated assay).
embedding <- step("embedding", local({
  full <- readRDS(gzcon(gzfile(rds_path, "rb")))
  integrated <- full@assays$integrated
  rm(full)
  invisible(gc())
  obj <- suppressWarnings(CreateSeuratObject(CreateAssay5Object(data = integrated@data), assay = "integrated"))
  VariableFeatures(obj) <- integrated@var.features
  rm(integrated)
  invisible(gc())
  obj <- ScaleData(obj, verbose = FALSE)
  obj <- RunPCA(obj, npcs = 30, seed.use = 12345, verbose = FALSE)
  obj <- RunUMAP(obj, dims = 1:20, n.neighbors = 30, metric = "cosine", min.dist = 0.3,
                 seed.use = 12345, verbose = FALSE)
  list(pca = Embeddings(obj, "pca"), umap = Embeddings(obj, "umap"))
}))
pcs <- embedding$pca[, 1:20]
umap <- embedding$umap

meta <- read.csv(meta_path)
cell_type <- meta$cell_type[match(rownames(pcs), meta$id)]

# 2. Recursive bisection.
bisect_tree <- function(x, max_depth, min_cells, seed) {
  labels <- matrix("1", nrow(x), max_depth + 1, dimnames = list(rownames(x), paste0("level_", 0:max_depth)))
  for (depth in seq_len(max_depth)) {
    parent <- labels[, depth]
    child <- parent
    nodes <- unique(parent)
    for (i in seq_along(nodes)) {
      idx <- which(parent == nodes[i])
      if (length(idx) < min_cells) next
      fit <- emeans(x[idx, , drop = FALSE], 2, nepochs = 3, seed = seed * 1000 + depth * 100 + i) # recursive e-means clustering
      child[idx] <- paste0(nodes[i], ".", fit$cluster)
    }
    labels[, depth + 1] <- child
  }
  labels
}

tree <- step("tree", bisect_tree(pcs, max_depth, min_cells, seed = 1))
levels <- seq_len(max_depth)
n_clusters <- map_int(levels, ~ n_distinct(tree[, .x + 1]))

# 3. Seed reruns and stability.
reruns <- step("reruns", par_map(seq_len(n_reruns) + 1, function(s) bisect_tree(pcs, max_depth, min_cells, seed = s)))

cell_jaccard <- function(a, b) {
  tab <- table(a, b)
  jac <- tab / (outer(rowSums(tab), colSums(tab), "+") - tab)
  jac[cbind(match(a, rownames(tab)), match(b, colnames(tab)))]
}

seed_ari <- step("seed_ari", map_dfr(seq_along(reruns), function(r) {
  tibble(run = r, seed = r + 1, level = levels,
         ari = map_dbl(levels, ~ adjustedRandIndex(tree[, .x + 1], reruns[[r]][, .x + 1])))
}))

stability <- step("stability", {
  m <- do.call(cbind, par_map(levels, function(l) {
    rowMeans(sapply(reruns, function(run) cell_jaccard(tree[, l + 1], run[, l + 1])))
  }))
  dimnames(m) <- list(rownames(pcs), paste0("stability_", levels))
  m
})

# 4. Louvain sweep on the SNN graph of the same 20 PCs.
louvain <- step("louvain", {
  graphs <- FindNeighbors(pcs, k.param = 20, verbose = FALSE)
  # Louvain is single-threaded; one resolution per worker.
  cols <- par_map(resolutions, function(r) FindClusters(graphs$snn, resolution = r, verbose = FALSE))
  do.call(cbind, cols)
})
louvain_k <- map_int(louvain, n_distinct)
match_res <- map_int(n_clusters, ~ which.min(abs(louvain_k - .x)))

# 5. Agreement tables.
agreement <- step("agreement", tibble(
  level = levels,
  clusters = n_clusters,
  resolution = resolutions[match_res],
  louvain_clusters = louvain_k[match_res],
  ari_louvain = map2_dbl(levels, match_res, ~ adjustedRandIndex(tree[, .x + 1], louvain[[.y]])),
  ari_cell_type = map_dbl(levels, ~ adjustedRandIndex(tree[, .x + 1], cell_type))
))
louvain_sweep <- tibble(resolution = resolutions, clusters = louvain_k,
                        ari_cell_type = map_dbl(louvain, ~ adjustedRandIndex(.x, cell_type)))

# 6. Tree nodes and edges for the dendrogram.
node_stability <- map_dfr(levels, function(l) {
  tibble(name = tree[, l + 1], stability = stability[, l]) |>
    group_by(name) |>
    summarise(stability = mean(stability), .groups = "drop") |>
    mutate(depth = l)
})
nodes <- map_dfr(0:max_depth, ~ count(tibble(name = tree[, .x + 1]), name, name = "cells") |> mutate(depth = .x)) |>
  group_by(name) |>
  slice_min(depth) |>
  ungroup() |>
  left_join(node_stability, by = c("name", "depth"))
edges <- nodes |>
  filter(depth > 0) |>
  transmute(from = sub("\\.[^.]+$", "", name), to = name)

# 7. Per-cell table and plain-text copies of every table.
cells <- tibble(id = rownames(pcs), cell_type = cell_type, umap_1 = umap[, 1], umap_2 = umap[, 2])
for (l in levels) {
  cells[[paste0("level_", l)]] <- tree[, l + 1]
  cells[[paste0("stability_", l)]] <- stability[, l]
  cells[[paste0("louvain_", l)]] <- louvain[[match_res[l]]]
}

# 8. saving data
write.csv(cells, gzfile(file.path(out_dir, "cells.csv.gz")), row.names = FALSE)
write.csv(cbind(id = rownames(pcs), louvain), gzfile(file.path(out_dir, "louvain_all_resolutions.csv.gz")),
          row.names = FALSE)
write.csv(seed_ari, file.path(out_dir, "seed_ari.csv"), row.names = FALSE)
write.csv(agreement, file.path(out_dir, "agreement.csv"), row.names = FALSE)
write.csv(louvain_sweep, file.path(out_dir, "louvain_sweep.csv"), row.names = FALSE)
write.csv(nodes, file.path(out_dir, "tree_nodes.csv"), row.names = FALSE)
write.csv(edges, file.path(out_dir, "tree_edges.csv"), row.names = FALSE)
saveRDS(timings, timings_path)
saveRDS(list(date = Sys.time(), threads = threads, max_depth = max_depth, min_cells = min_cells,
             n_reruns = n_reruns, resolutions = resolutions, rds_path = rds_path, meta_path = meta_path,
             session = sessionInfo()),
        file.path(out_dir, "run_info.rds"))

log_msg("workers: %d, emeans threads: %d", workers, threads)
log_msg("done: %d cells, %d levels, results in %s", nrow(cells), max_depth, out_dir)
print(agreement)
