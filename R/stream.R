#' Streaming e-means for data that does not fit in memory
#'
#' Caller-driven training: the caller supplies rows in chunks, in any order
#' it chooses (shuffle each epoch for best results). Each epoch must supply
#' exactly `recipe$epoch_rows[epoch]` rows, grouped into batches of
#' `emeans_stream_status(s)$expected_batch` rows; one batch may be split
#' across several `emeans_accumulate()` calls.
#'
#' The loop is:
#' ```
#' s <- emeans_stream(n, init, nepochs = 3)
#' for (epoch in seq_along(s$recipe$epoch_rows)) {
#'   left <- s$recipe$epoch_rows[epoch]
#'   while (left > 0) {
#'     b <- emeans_stream_status(s)$expected_batch
#'     emeans_accumulate(s, next_rows(b))  # next_rows(): your data reader
#'     emeans_update(s)
#'     left <- left - b
#'   }
#'   emeans_end_epoch(s)
#' }
#' fit <- emeans_finalize(s)
#' ```
#' See the "Streaming" section of the vignette for a complete example.
#'
#' @param n Total number of training rows (one epoch).
#' @param init `k x d` matrix of initial centers (e.g. random training rows).
#' @inheritParams emeans
#' @return `emeans_stream()` returns an `emeans_stream` object.
#' @export
emeans_stream <- function(n, init, nepochs = 5,
                          batch_law = c("auto", "throughput", "quality"),
                          batch_size = NULL, alpha0 = NULL, spherical = FALSE,
                          unused_threshold = NULL,
                          precision = c("double", "single"),
                          seed = 1234L, n_threads = NULL) {
  # Create streaming training state
  batch_law <- match.arg(batch_law)
  precision <- match.arg(precision)
  if (!is.numeric(n) || length(n) != 1 || !is.finite(n) || n < 1 || n != floor(n)) {
    stop("`n` must be a single whole number >= 1", call. = FALSE)
  }
  init <- as_data_matrix(init, "init")
  k <- nrow(init)
  d <- ncol(init)
  if (k > n) stop("more clusters than rows", call. = FALSE)
  recipe <- emeans_recipe(n, k, d, nepochs, batch_law, batch_size, alpha0, unused_threshold)
  ptr <- em_new(init, n, spherical, precision == "single", recipe$batch_size,
                recipe$alpha0, recipe$epoch_rows, recipe$unused_threshold,
                as.numeric(seed), default_threads(n_threads))
  # Wrap pointer in environment
  s <- new.env(parent = emptyenv())
  s$ptr <- ptr
  s$recipe <- recipe
  s$k <- k
  s$d <- d
  s$spherical <- spherical
  s$precision <- precision
  s$colnames <- colnames(init)
  s$unit_checked <- FALSE
  class(s) <- "emeans_stream"
  s
}

#' @rdname emeans_stream
#' @param s An `emeans_stream` object.
#' @return `emeans_stream_status()` returns a list with `step`, `epoch`
#'   (completed epochs), `n_epochs`, `expected_batch` (rows the next update
#'   needs), `n_splits`, and the per-epoch objective so far.
#' @export
emeans_stream_status <- function(s) {
  # Progress from Rust state
  stopifnot(inherits(s, "emeans_stream"))
  em_info(s$ptr)
}

#' @rdname emeans_stream
#' @param chunk Numeric matrix of rows to add to the current batch.
#' @return `emeans_accumulate()` returns the chunk's cluster labels against
#'   the centroids of the current batch (invisibly).
#' @export
emeans_accumulate <- function(s, chunk) {
  # Add rows to batch
  stopifnot(inherits(s, "emeans_stream"))
  chunk <- as_data_matrix(chunk, "chunk")
  info <- stream_running(s)
  # Check width and overflow
  if (ncol(chunk) != s$d) stop(sprintf("chunk has %d columns, model has %d", ncol(chunk), s$d), call. = FALSE)
  if (info$pending + nrow(chunk) > info$expected_batch) {
    stop(sprintf("batch overflow: update expects %d rows, got %d",
                 as.integer(info$expected_batch), as.integer(info$pending + nrow(chunk))), call. = FALSE)
  }
  # Warn once on non-unit rows
  if (s$spherical && !s$unit_checked) {
    s$unit_checked <- TRUE
    check_unit_rows(chunk)
  }
  invisible(em_accumulate(s$ptr, chunk))
}

# Error unless training active
stream_running <- function(s) {
  info <- em_info(s$ptr)
  if (info$finalized) stop("stream already finalized", call. = FALSE)
  if (info$epoch >= info$n_epochs) stop("all epochs are complete; call emeans_finalize()", call. = FALSE)
  info
}

#' @rdname emeans_stream
#' @export
emeans_update <- function(s) {
  # Apply EMA update
  stopifnot(inherits(s, "emeans_stream"))
  info <- stream_running(s)
  if (info$pending != info$expected_batch) {
    stop(sprintf("update expects a batch of %d rows, accumulated %d",
                 as.integer(info$expected_batch), as.integer(info$pending)), call. = FALSE)
  }
  em_update(s$ptr)
  invisible(s)
}

#' @rdname emeans_stream
#' @export
emeans_end_epoch <- function(s) {
  # Close completed epoch
  stopifnot(inherits(s, "emeans_stream"))
  info <- stream_running(s)
  rows <- s$recipe$epoch_rows[info$epoch + 1]
  if (info$epoch_done_rows != rows || info$pending != 0) {
    stop(sprintf("epoch %d expects %d rows, updated %d",
                 as.integer(info$epoch + 1), as.integer(rows), as.integer(info$epoch_done_rows)), call. = FALSE)
  }
  em_end_epoch(s$ptr)
  invisible(s)
}

#' @rdname emeans_stream
#' @return `emeans_finalize()` returns a list of class `emeans_streamed` with
#'   `centers`, `obj`, `iteration_stats`, `recipe`, `n_splits`; use
#'   [predict.emeans()] on it to label data.
#' @export
emeans_finalize <- function(s) {
  # Exact last-epoch centers
  stopifnot(inherits(s, "emeans_stream"))
  info <- em_info(s$ptr)
  if (!info$finalized && info$epoch < info$n_epochs) {
    stop("finalize needs all ", as.integer(info$n_epochs), " epochs completed", call. = FALSE)
  }
  centers <- em_finalize(s$ptr)
  colnames(centers) <- s$colnames
  info <- em_info(s$ptr)
  structure(
    list(
      centers = centers,
      obj = info$obj,
      iteration_stats = data.frame(epoch = seq_along(info$obj), obj = info$obj,
                                   imbalance = info$imbalance, nsplit = info$nsplit),
      recipe = s$recipe,
      n_splits = info$n_splits,
      spherical = s$spherical,
      precision = s$precision
    ),
    class = c("emeans_streamed", "emeans")
  )
}

# Print stream progress
#' @export
print.emeans_stream <- function(x, ...) {
  info <- em_info(x$ptr)
  cat(sprintf("e-means stream: k = %d, d = %d, epoch %d / %d, step %d / %d%s\n",
              x$k, x$d, as.integer(info$epoch), as.integer(info$n_epochs),
              as.integer(info$step), as.integer(info$total_steps),
              if (info$finalized) " (finalized)" else ""))
  invisible(x)
}

# Print streamed centers
#' @export
print.emeans_streamed <- function(x, ...) {
  cat(sprintf("e-means centers from a stream: %d clusters in %dD, %d epochs, %d splits\n",
              nrow(x$centers), ncol(x$centers), length(x$obj), as.integer(x$n_splits)))
  invisible(x)
}
