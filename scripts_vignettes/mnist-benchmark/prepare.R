# MNIST benchmark: download, decode, PCA-50, nested subsets, UMAP.
# Usage: Rscript prepare.R   (from any directory)
# Writes everything under out/ next to this script. Every output is skipped if it already exists.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- if (length(script)) dirname(normalizePath(script)) else getwd()
out <- file.path(here, "out")
dir.create(file.path(out, "raw"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out, "data"), recursive = TRUE, showWarnings = FALSE)
log <- function(...) message(format(Sys.time(), "%H:%M:%S "), sprintf(...))

base <- "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/"
files <- c(train = "train-00000-of-00001.parquet", test = "test-00000-of-00001.parquet")
for (f in files) {
  dest <- file.path(out, "raw", f)
  if (!file.exists(dest)) {
    log("download %s", f)
    utils::download.file(paste0(base, f), dest, mode = "wb", quiet = TRUE)
  }
}

# Pixels 0-1, label, split
mnist_file <- file.path(out, "data", "mnist.rds")
if (!file.exists(mnist_file)) {
  log("decode png")
  parts <- lapply(names(files), function(s) {
    tb <- as.data.frame(arrow::read_parquet(file.path(out, "raw", files[[s]])))
    bytes <- tb$image$bytes
    px <- t(vapply(bytes, function(b) as.vector(t(png::readPNG(b))), numeric(784)))
    list(x = px, label = as.integer(tb$label), split = rep(s, nrow(tb)))
  })
  mnist <- list(
    x = do.call(rbind, lapply(parts, `[[`, "x")),
    label = unlist(lapply(parts, `[[`, "label")),
    split = unlist(lapply(parts, `[[`, "split"))
  )
  dimnames(mnist$x) <- NULL
  saveRDS(mnist, mnist_file)
  log("saved %s (%d x %d)", mnist_file, nrow(mnist$x), ncol(mnist$x))
}
mnist <- readRDS(mnist_file)
labels_file <- file.path(out, "data", "labels.rds")
if (!file.exists(labels_file)) saveRDS(mnist$label, labels_file)

# PCA-50 on all rows
pca_file <- file.path(out, "data", "pca50.rds")
if (!file.exists(pca_file)) {
  log("pca")
  p <- stats::prcomp(mnist$x, center = TRUE, scale. = FALSE, rank. = 50)
  pca <- list(scores = unname(p$x), rotation = unname(p$rotation), center = p$center,
              sdev = p$sdev, var_explained = p$sdev^2 / sum(p$sdev^2))
  saveRDS(pca, pca_file)
  log("saved %s, var explained by 50 PCs %.3f", pca_file, sum(pca$var_explained[1:50]))
}
pca <- readRDS(pca_file)

# Nested stratified subsets: each size contains the smaller ones
sub_file <- file.path(out, "data", "subsets.rds")
if (!file.exists(sub_file)) {
  set.seed(20261114)
  n_all <- length(mnist$label)
  # Random order within each label, then interleave by rank so any prefix is stratified
  ord <- unlist(lapply(split(seq_len(n_all), mnist$label), function(i) i[sample.int(length(i))]))
  rank_in_label <- ave(seq_along(ord), mnist$label[ord], FUN = seq_along)
  frac <- rank_in_label / tabulate(mnist$label[ord] + 1L, 10)[mnist$label[ord] + 1L]
  ord <- ord[order(frac, stats::runif(length(ord)))]
  sizes <- c(1000, 5000, 10000, 20000, 30000, 70000)
  subsets <- stats::setNames(lapply(sizes, function(s) sort(ord[seq_len(s)])), sizes)
  saveRDS(subsets, sub_file)
  log("saved %s", sub_file)
}

# UMAP 2-D for plotting
umap_file <- file.path(out, "data", "umap.rds")
if (!file.exists(umap_file)) {
  log("umap")
  set.seed(1)
  um <- uwot::umap(pca$scores, n_neighbors = 15, min_dist = 0.1, n_threads = 7, seed = 1)
  saveRDS(unname(um), umap_file)
  log("saved %s", umap_file)
}
log("done")

# Raw-pixel subsets: each run loads only its rows
subsets <- readRDS(sub_file)
for (s in names(subsets)) {
  f <- file.path(out, "data", sprintf("raw_n%s.rds", s))
  if (!file.exists(f)) {
    saveRDS(mnist$x[subsets[[s]], , drop = FALSE], f)
    log("saved %s", f)
  }
}
log("raw subsets done")
