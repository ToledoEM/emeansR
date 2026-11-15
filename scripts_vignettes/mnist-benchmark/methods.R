# Method registry and metrics for the MNIST benchmark (raw pixels, single thread).
# Partitioning/model-based: function(x, cfg) -> list(cluster, centers, times, extra).
# Hierarchical: function(x, cfg) -> list(tree, times); run-one.R cuts it at each k.

phase <- function(expr) {
  t0 <- proc.time()
  val <- force(expr)
  list(val = val, t = (proc.time() - t0)[["elapsed"]])
}

hier <- function(dist_fun, tree_fun) {
  function(x, cfg) {
    d <- phase(dist_fun(x))
    dist_s <- if (is.null(d$val)) NA_real_ else d$t
    tr <- phase(tree_fun(d$val, x))
    list(tree = tr$val, times = c(dist_s = dist_s, tree_s = tr$t))
  }
}

methods <- list(
  emeans = list(
    family = "partitioning", package = "emeansR",
    f = function(x, cfg) {
      fit <- emeansR::emeans(x, cfg$k, nepochs = cfg$nepochs, precision = cfg$precision,
                             seed = as.integer(cfg$seed), n_threads = 1)
      r <- fit$recipe
      list(cluster = fit$cluster, centers = fit$centers, times = c(),
           extra = list(trace = fit$iteration_stats,
                        recipe = data.frame(batch_size = r$batch_size, alpha0 = r$alpha0,
                                            epoch_rows = r$epoch_rows[1], total_steps = r$total_steps,
                                            batch_law = r$batch_law, unused_threshold = r$unused_threshold,
                                            nepochs = cfg$nepochs, n_splits = fit$n_splits)))
    }
  ),
  kmeans_hw = list(
    family = "partitioning", package = "stats",
    f = function(x, cfg) {
      set.seed(cfg$seed)
      fit <- stats::kmeans(x, cfg$k, iter.max = 100, nstart = 1, algorithm = "Hartigan-Wong")
      list(cluster = fit$cluster, centers = fit$centers, times = c(), extra = list(iter = fit$iter))
    }
  ),
  kmeans_lloyd = list(
    family = "partitioning", package = "stats",
    f = function(x, cfg) {
      set.seed(cfg$seed)
      fit <- stats::kmeans(x, cfg$k, iter.max = 100, nstart = 1, algorithm = "Lloyd")
      list(cluster = fit$cluster, centers = fit$centers, times = c(), extra = list(iter = fit$iter))
    }
  ),
  clusterr_kmeans = list(
    family = "partitioning", package = "ClusterR",
    f = function(x, cfg) {
      fit <- ClusterR::KMeans_rcpp(x, cfg$k, num_init = 1, max_iters = 100,
                                   initializer = "kmeans++", seed = cfg$seed)
      list(cluster = as.integer(fit$clusters), centers = fit$centroids, times = c(),
           extra = list(iter = length(fit$total_SSE)))
    }
  ),
  clusterr_minibatch = list(
    family = "partitioning", package = "ClusterR",
    f = function(x, cfg) {
      fit <- phase(ClusterR::MiniBatchKmeans(x, cfg$k, batch_size = min(1024L, nrow(x)), num_init = 1,
                                             max_iters = 100, initializer = "kmeans++", seed = cfg$seed))
      pr <- phase(ClusterR::predict_MBatchKMeans(x, fit$val$centroids))
      list(cluster = as.integer(pr$val), centers = fit$val$centroids,
           times = c(fit_s = fit$t, assign_s = pr$t), extra = list())
    }
  ),
  clara = list(
    family = "partitioning", package = "cluster",
    f = function(x, cfg) {
      set.seed(cfg$seed)
      fit <- cluster::clara(x, cfg$k, samples = 50, rngR = TRUE, pamLike = TRUE)
      list(cluster = as.integer(fit$clustering), centers = fit$medoids, times = c(), extra = list())
    }
  ),
  mclust_eii = list(
    family = "model-based", package = "mclust",
    f = function(x, cfg) {
      fit <- mclust::Mclust(x, G = cfg$k, modelNames = "EII", verbose = FALSE)
      list(cluster = as.integer(fit$classification), centers = t(fit$parameters$mean),
           times = c(), extra = list(loglik = fit$loglik, bic = fit$bic))
    }
  ),
  stats_hclust = list(
    family = "hierarchical", package = "stats",
    f = hier(function(x) stats::dist(x), function(d, x) stats::hclust(d, "ward.D2"))
  ),
  flashclust = list(
    family = "hierarchical", package = "flashClust",
    # Ward on squared distances: same merges as ward.D2
    f = hier(function(x) stats::dist(x)^2, function(d, x) flashClust::flashClust(d, "ward"))
  ),
  fastcluster_hclust = list(
    family = "hierarchical", package = "fastcluster",
    f = hier(function(x) stats::dist(x), function(d, x) fastcluster::hclust(d, "ward.D2"))
  ),
  fastcluster_vector = list(
    family = "hierarchical", package = "fastcluster",
    # No dist matrix: O(n d) memory
    f = hier(function(x) NULL, function(d, x) fastcluster::hclust.vector(x, method = "ward"))
  ),
  # Smoke only: checks the timeout kill path
  timeout_probe = list(
    family = "probe", package = "base",
    f = function(x, cfg) { Sys.sleep(3600); NULL }
  )
)

# Cluster means from labels (same for every method)
label_means <- function(x, cl) {
  s <- rowsum(x, cl)
  s / tabulate(cl)[as.integer(rownames(s))]
}

quality <- function(x, cl, truth) {
  m <- label_means(x, cl)
  resid <- x - m[match(cl, as.integer(rownames(m))), , drop = FALSE]
  tab <- table(cl, truth)
  sizes <- tabulate(cl)
  data.frame(
    ari = aricode::ARI(cl, truth), nmi = aricode::NMI(cl, truth), ami = aricode::AMI(cl, truth),
    purity = sum(apply(tab, 1, max)) / length(cl),
    inertia = sum(resid^2) / nrow(x),
    n_clusters_found = length(unique(cl)), min_cluster_size = min(sizes[sizes > 0])
  )
}

# Fixed column order of runs.csv
run_cols <- c(
  "run_id", "fit_id", "method", "family", "package", "package_version", "n", "d", "k", "threads",
  "precision", "nepochs", "seed", "rep", "timeout_s", "elapsed_s", "user_s", "sys_s", "cpu_util",
  "dist_s", "tree_s", "cut_s", "fit_s", "assign_s", "rss_before_mb", "max_rss_mb", "dist_mb",
  "process_wall_s", "ari", "nmi", "ami", "purity", "inertia", "n_clusters_found", "min_cluster_size",
  "iter", "warning", "status", "error_msg"
)
