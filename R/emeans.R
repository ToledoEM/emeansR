#' @useDynLib emeansR, .registration = TRUE
#' @importFrom stats predict fitted
#' @keywords internal
"_PACKAGE"

# Check and coerce to double
as_data_matrix <- function(x, what = "x") {
  if (is.data.frame(x)) x <- as.matrix(x)
  if (!is.matrix(x) || !is.numeric(x)) stop("`", what, "` must be a numeric matrix or data frame", call. = FALSE)
  if (anyNA(x) || any(!is.finite(x))) stop("`", what, "` contains NA or non-finite values", call. = FALSE)
  storage.mode(x) <- "double"
  x
}

# Threads: argument, option, cores
default_threads <- function(n_threads) {
  if (!is.null(n_threads)) return(max(1L, as.integer(n_threads)))
  opt <- getOption("emeansR.threads")
  if (!is.null(opt)) return(max(1L, as.integer(opt)))
  cores <- max(1L, parallel::detectCores(logical = FALSE) - 1L, na.rm = TRUE)
  # R CMD check --as-cran sets this; CRAN allows at most two cores.
  limit <- tolower(Sys.getenv("_R_CHECK_LIMIT_CORES_", ""))
  if (nzchar(limit) && limit != "false") cores <- min(cores, 2L)
  cores
}

# Warn on non-unit rows
check_unit_rows <- function(x) {
  rows <- if (nrow(x) > 1000) x[seq(1, nrow(x), length.out = 1000), , drop = FALSE] else x
  if (any(abs(rowSums(rows^2) - 1) > 1e-3)) {
    warning("spherical = TRUE expects unit-norm rows; normalize x first", call. = FALSE)
  }
}

#' e-means clustering
#'
#' Mini-batch k-means with exponential-moving-average centroid updates
#' (Bessa et al., 2026, arXiv:2611.09501). Batch size and step size come from
#' fitted scaling laws, so the only inputs are the data, `k`, and the epoch
#' budget. In a few epochs it typically reaches a lower within-cluster sum of
#' squares than 25 Lloyd iterations.
#'
#' Each epoch visits the rows in a fresh random order in batches of
#' `batch_size`. Every batch is assigned to the nearest centroids, and each
#' touched centroid moves towards its batch mean with weight
#' `alpha * n_j / N_j`, where the mass `N_j` is itself an exponential moving
#' average and `alpha` follows a cosine schedule from `alpha0` to
#' `alpha0 / 10`. Every five updates, clusters with mass below
#' `unused_threshold * mean(mass)` are revived by splitting large clusters
#' (picked with probability proportional to mass). The returned centers are
#' the exact means of the points assigned during the last epoch.
#'
#' @param x Numeric matrix or data frame, one row per point.
#' @param k Number of clusters, or a `k x d` matrix of initial centers.
#' @param nepochs Epoch budget; fractions allowed (e.g. `1.5`).
#' @inheritParams emeans_recipe
#' @param spherical Cluster unit vectors by cosine similarity. Rows of `x`
#'   must already be unit-norm; centers are returned as unit vectors.
#' @param max_points_per_centroid If set and `nrow(x) > max_points_per_centroid * k`,
#'   train on a random subset of that many rows (as `faiss.Kmeans`).
#' @param precision `"double"`, or `"single"` to store the training data and
#'   compute distances in float32 (about 2x faster, half the memory). Batch
#'   statistics, centroids and masses always accumulate in double precision,
#'   and `withinss` is computed from the original data. As in the faiss FP32
#'   path, single precision assigns poorly when the data sit far from the
#'   origin (the error grows with the squared norm of the rows); centre the
#'   data first, e.g. with `scale(x, scale = FALSE)`. Double precision hits
#'   the same limit at offsets around 1e8.
#' @param final_assign If `TRUE` (default), assign every row of `x` to the
#'   final centers, so `cluster`, `withinss` and `size` describe the returned
#'   centers (as [stats::kmeans()]). If `FALSE`, `cluster` holds labels from
#'   the last training pass, and rows left out by `max_points_per_centroid`
#'   are `NA`.
#' @param seed Integer seed for initialization, shuffling and revival.
#'   Does not touch R's random number generator.
#' @param n_threads Threads for assignment. Defaults to
#'   `getOption("emeansR.threads")`, else physical cores minus one (at most
#'   two when the `_R_CHECK_LIMIT_CORES_` environment variable is set).
#' @param verbose Print the recipe and per-epoch objective.
#' @return An object of class `c("emeans", "kmeans")` with components
#'   `cluster`, `centers`, `totss`, `withinss`, `tot.withinss`, `betweenss`,
#'   `size`, `iter` (epochs), `ifault` (0), plus `obj` (moving-pass objective
#'   per epoch), `iteration_stats`, `recipe`, `n_splits`, `spherical` and
#'   `precision`.
#' @examples
#' set.seed(1)
#' x <- rbind(matrix(rnorm(2000, 0), ncol = 2), matrix(rnorm(2000, 5), ncol = 2))
#' fit <- emeans(x, k = 2, nepochs = 3, n_threads = 1)
#' fit$centers
#' table(fit$cluster)
#' @export
emeans <- function(x, k, nepochs = 5,
                   batch_law = c("auto", "throughput", "quality"),
                   batch_size = NULL, alpha0 = NULL, spherical = FALSE,
                   unused_threshold = NULL, max_points_per_centroid = NULL,
                   precision = c("double", "single"), final_assign = TRUE,
                   seed = 1234L, n_threads = NULL, verbose = FALSE) {
  # Validate inputs
  batch_law <- match.arg(batch_law)
  precision <- match.arg(precision)
  single <- precision == "single"
  x <- as_data_matrix(x)
  n <- nrow(x)
  d <- ncol(x)
  init <- NULL
  # Matrix k gives centers
  if (is.matrix(k)) {
    init <- as_data_matrix(k, "k")
    if (ncol(init) != d) stop("initial centers must have ", d, " columns", call. = FALSE)
    k <- nrow(init)
  }
  k <- as.integer(k)
  if (length(k) != 1 || is.na(k) || k < 1) stop("`k` must be a positive integer", call. = FALSE)
  if (k > n) stop("more clusters than rows", call. = FALSE)
  if (spherical) check_unit_rows(x)
  threads <- default_threads(n_threads)
  seed <- as.numeric(seed)

  # Optional training subset
  keep <- NULL
  if (!is.null(max_points_per_centroid)) {
    if (max_points_per_centroid < 1) stop("max_points_per_centroid must be >= 1", call. = FALSE)
    limit <- floor(max_points_per_centroid) * k  # whole points, as faiss
    if (n > limit) {
      keep <- sort(em_sample(n, limit, seed + 1))
      if (verbose) message(sprintf("Sampling %d of %d rows (max_points_per_centroid = %d)",
                                   length(keep), n, as.integer(max_points_per_centroid)))
    }
  }
  xt <- if (is.null(keep)) x else x[keep, , drop = FALSE]
  nt <- nrow(xt)
  if (k > nt) stop("more clusters than training rows", call. = FALSE)

  # Recipe and initial centers
  recipe <- emeans_recipe(nt, k, d, nepochs, batch_law, batch_size, alpha0, unused_threshold)
  if (is.null(init)) {
    init <- if (nt == k) xt else xt[em_sample(nt, k, seed), , drop = FALSE]
  }
  if (verbose) {
    message(sprintf("Clustering %d points in %dD to %d clusters, %g epochs (batch %d, alpha0 %.3g, %s law%s)",
                    nt, d, k, sum(recipe$epoch_rows) / nt, as.integer(recipe$batch_size), recipe$alpha0,
                    if (is.na(recipe$batch_law)) "given" else recipe$batch_law,
                    if (spherical) ", spherical" else ""))
  }

  # Rust data and state
  data <- em_data(xt, single)
  st <- em_new(init, nt, spherical, single, recipe$batch_size, recipe$alpha0,
               recipe$epoch_rows, recipe$unused_threshold, seed, threads)
  # Train in chunks; allow interrupts
  chunk_steps <- max(1, floor(2e9 / (recipe$batch_size * k * d)))
  last_epoch <- 0
  start <- proc.time()[["elapsed"]]
  repeat {
    done <- em_train_steps(st, data, chunk_steps, !final_assign)
    if (verbose) {
      # Report finished epochs
      info <- em_info(st)
      while (last_epoch < length(info$obj)) {
        last_epoch <- last_epoch + 1
        message(sprintf("  Epoch %d / %d (%.2f s): objective=%g, splits=%d",
                        last_epoch, info$n_epochs, proc.time()[["elapsed"]] - start,
                        info$obj[last_epoch], as.integer(info$nsplit[last_epoch])))
      }
    }
    if (done) break
  }
  rm(data)
  # Exact closing centers
  centers <- em_finalize(st)
  info <- em_info(st)
  colnames(centers) <- colnames(x)

  # Labels: reassign or last pass
  dist2 <- NULL
  if (final_assign) {
    a <- em_assign(centers, x, spherical, single, threads)
    cluster <- a$cluster
    # Spherical gives similarities
    if (!spherical) dist2 <- a$distance
  } else {
    labels <- em_labels(st)
    cluster <- rep(NA_integer_, n)
    cluster[if (is.null(keep)) seq_len(n) else keep] <- labels
  }
  new_emeans(x, centers, cluster, k, recipe, info, spherical, precision, dist2)
}

# Build kmeans-compatible result
new_emeans <- function(x, centers, cluster, k, recipe, info, spherical, precision,
                       dist2 = NULL) {
  ok <- !is.na(cluster)
  all_rows <- all(ok)
  xs <- if (all_rows) x else x[ok, , drop = FALSE]
  cl <- cluster[ok]
  if (is.null(dist2)) {
    dist2 <- if (length(cl)) rowSums((xs - centers[cl, , drop = FALSE])^2) else numeric()
  }
  # Withinss from squared distances
  withinss <- numeric(k)
  if (length(cl)) {
    s <- rowsum(dist2, cl, reorder = TRUE)
    withinss[as.integer(rownames(s))] <- s[, 1]
  }
  totss <- em_totss(xs)
  names(cluster) <- rownames(x)
  structure(
    list(
      cluster = cluster,
      centers = centers,
      totss = totss,
      withinss = withinss,
      tot.withinss = sum(withinss),
      betweenss = totss - sum(withinss),
      size = tabulate(cl, k),
      iter = length(info$obj),
      ifault = 0L,
      obj = info$obj,
      iteration_stats = data.frame(epoch = seq_along(info$obj), obj = info$obj,
                                   imbalance = info$imbalance, nsplit = info$nsplit),
      recipe = recipe,
      n_splits = info$n_splits,
      spherical = spherical,
      precision = precision
    ),
    class = c("emeans", "kmeans")
  )
}

# Print fit summary
#' @export
print.emeans <- function(x, ...) {
  cat(sprintf("e-means clustering with %d clusters of sizes %s\n", nrow(x$centers),
              paste(utils::head(x$size, 20), collapse = ", ")),
      if (length(x$size) > 20) "...\n" else "", sep = "")
  cat(sprintf("Epochs: %g (batch %d, alpha0 %.3g), splits: %d%s\n",
              x$recipe$nepochs,
              as.integer(x$recipe$batch_size), x$recipe$alpha0, as.integer(x$n_splits),
              if (x$spherical) ", spherical" else ""))
  cat(sprintf("Within cluster sum of squares: %g (between_SS / total_SS = %5.1f %%)\n",
              x$tot.withinss, 100 * x$betweenss / x$totss))
  cat("Available components:\n")
  print(names(x))
  invisible(x)
}

#' Assign new data to e-means centers
#'
#' @param object An `emeans` fit.
#' @param newdata Numeric matrix or data frame with the same columns.
#' @param distance If `TRUE`, return a data frame with `cluster` and
#'   `distance`: the squared Euclidean distance, or, when spherical, the
#'   inner product with the unit centers (the cosine similarity if `newdata`
#'   rows are unit-norm).
#' @param n_threads Threads; see [emeans()].
#' @param ... Unused.
#' @return Integer cluster labels, or a data frame when `distance = TRUE`.
#' @export
predict.emeans <- function(object, newdata, distance = FALSE, n_threads = NULL, ...) {
  # Nearest center per row
  newdata <- as_data_matrix(newdata, "newdata")
  res <- em_assign(object$centers, newdata, object$spherical,
                   object$precision == "single", default_threads(n_threads))
  if (distance) data.frame(cluster = res$cluster, distance = res$distance) else res$cluster
}

# Centers or labels per row
#' @export
fitted.emeans <- function(object, method = c("centers", "classes"), ...) {
  method <- match.arg(method)
  if (is.null(object$cluster)) stop("no cluster labels; use predict()", call. = FALSE)
  if (method == "centers") object$centers[object$cluster, , drop = FALSE] else object$cluster
}
