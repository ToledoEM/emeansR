#' Total sum of squares about the column means (two passes per column).
#' @noRd
em_totss <- function(x) .Call(wrap__em_totss, x)

#' Copy a numeric matrix into a row-major training buffer.
#' @noRd
em_data <- function(x, single) .Call(wrap__em_data, x, single)

#' Create a training state.
#' @noRd
em_new <- function(init, n, spherical, single, batch_size, alpha0, epoch_rows, unused_threshold, seed, n_threads) .Call(wrap__em_new, init, n, spherical, single, batch_size, alpha0, epoch_rows, unused_threshold, seed, n_threads)

#' Run up to `max_steps` updates on in-memory data. TRUE when done.
#' @noRd
em_train_steps <- function(st, data, max_steps, capture) .Call(wrap__em_train_steps, st, data, max_steps, capture)

#' Accumulate one chunk of rows into the pending batch; returns 1-based labels.
#' @noRd
em_accumulate <- function(st, x) .Call(wrap__em_accumulate, st, x)

#' Apply the EMA update for the pending batch.
#' @noRd
em_update <- function(st) .Call(wrap__em_update, st)

#' Close the current epoch.
#' @noRd
em_end_epoch <- function(st) .Call(wrap__em_end_epoch, st)

#' Closing centroids (k x d).
#' @noRd
em_finalize <- function(st) .Call(wrap__em_finalize, st)

#' Current centroid readout (k x d).
#' @noRd
em_centroids <- function(st) .Call(wrap__em_centroids, st)

#' Last-epoch labels captured by em_train_steps (1-based, NA if unseen).
#' @noRd
em_labels <- function(st) .Call(wrap__em_labels, st)

#' Progress and per-epoch statistics.
#' @noRd
em_info <- function(st) .Call(wrap__em_info, st)

#' Nearest-centroid assignment for new data.
#' @noRd
em_assign <- function(centers, x, spherical, single, n_threads) .Call(wrap__em_assign, centers, x, spherical, single, n_threads)

#' First `take` entries of a seeded random permutation of 1..n.
#' @noRd
em_sample <- function(n, take, seed) .Call(wrap__em_sample, n, take, seed)

