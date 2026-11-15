# Run ONE benchmark configuration in a fresh R process (raw pixels).
# Usage: Rscript run-one.R <out_dir> <run_id>
# Reads <out_dir>/grid.rds; writes <out_dir>/results/<run_id>.rds plus labels/centers files.
# Hierarchical runs build the tree once and give one result row per k.
args <- commandArgs(TRUE)
out <- args[1]
run_id <- args[2]
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "methods.R"))

cfg <- as.list(readRDS(file.path(out, "grid.rds"))[run_id, ])
# Input data written by prepare.R, next to this script
data_dir <- file.path(here, "out", "data")
x <- readRDS(file.path(data_dir, sprintf("raw_n%d.rds", cfg$n)))
truth <- readRDS(file.path(data_dir, "labels.rds"))[readRDS(file.path(data_dir, "subsets.rds"))[[as.character(cfg$n)]]]
m <- methods[[cfg$method]]
suppressPackageStartupMessages(library(m$package, character.only = TRUE))
for (d in c("results", "labels", "centers")) dir.create(file.path(out, d), showWarnings = FALSE)
invisible(gc())
rss_before_mb <- ps::ps_memory_info()[["rss"]] / 2^20

warn <- character()
t0 <- proc.time()
res <- withCallingHandlers(
  m$f(x, cfg),
  warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") }
)
pt <- proc.time() - t0

base_row <- function(k, fit_id) {
  data.frame(
    run_id = run_id, fit_id = fit_id, method = cfg$method, family = m$family, package = m$package,
    package_version = as.character(utils::packageVersion(m$package)),
    n = cfg$n, d = ncol(x), k = k, threads = 1L, precision = cfg$precision, nepochs = cfg$nepochs,
    seed = cfg$seed, rep = cfg$rep, timeout_s = cfg$timeout_s,
    user_s = pt[["user.self"]], sys_s = pt[["sys.self"]],
    cpu_util = (pt[["user.self"]] + pt[["sys.self"]]) / pt[["elapsed"]],
    rss_before_mb = rss_before_mb,
    dist_mb = if (m$family == "hierarchical" && cfg$method != "fastcluster_vector")
      8 * cfg$n * (cfg$n - 1) / 2 / 2^20 else 0,
    warning = gsub("\\s+", " ", paste(unique(warn), collapse = " | "))
  )
}

rows <- list()
extra <- list()
if (m$family == "hierarchical") {
  ks <- as.integer(strsplit(cfg$ks, ",")[[1]])
  for (k in ks) {
    fit_id <- sub("_kall_", sprintf("_k%d_", k), run_id)
    ct <- phase(stats::cutree(res$tree, k))
    cl <- as.integer(ct$val)
    row <- cbind(base_row(k, fit_id), quality(x, cl, truth))
    row$dist_s <- res$times[["dist_s"]]
    row$tree_s <- res$times[["tree_s"]]
    row$cut_s <- ct$t
    row$elapsed_s <- sum(res$times, na.rm = TRUE) + ct$t
    saveRDS(cl, file.path(out, "labels", paste0(fit_id, ".rds")))
    rows[[length(rows) + 1]] <- row
  }
  if (cfg$method == "stats_hclust" && cfg$n == 5000) {
    saveRDS(list(tree = res$tree, n = cfg$n), file.path(out, "hclust_5k.rds"))
  }
} else {
  cl <- as.integer(res$cluster)
  row <- cbind(base_row(cfg$k, run_id), quality(x, cl, truth))
  row$elapsed_s <- pt[["elapsed"]]
  for (nm in names(res$times)) row[[nm]] <- res$times[[nm]]
  row$iter <- if (is.null(res$extra$iter)) NA_integer_ else as.integer(res$extra$iter)
  saveRDS(cl, file.path(out, "labels", paste0(run_id, ".rds")))
  if (isTRUE(cfg$save_centers)) {
    centers <- list(label_means = label_means(x, cl))
    if (!is.null(res$centers)) centers$own <- unname(as.matrix(res$centers))
    saveRDS(centers, file.path(out, "centers", paste0(run_id, ".rds")))
  }
  rows[[1]] <- row
  extra <- res$extra
}
saveRDS(list(rows = rows, extra = extra), file.path(out, "results", paste0(run_id, ".rds")))
