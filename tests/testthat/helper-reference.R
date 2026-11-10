# Slow pure-R e-means (no revival), driven by explicit row orders.
# Oracle for the Rust update math.
reference_emeans <- function(x, init, recipe, orders, spherical = FALSE) {
  unit <- function(m) {
    nr <- sqrt(rowSums(m^2))
    nr[nr == 0] <- 1
    m / nr
  }
  k <- nrow(init)
  cent <- if (spherical) unit(init) else init
  usage <- rep(2^-16, k)
  step <- 0
  B <- recipe$batch_size
  n_ep <- length(recipe$epoch_rows)
  for (e in seq_len(n_ep)) {
    rows <- recipe$epoch_rows[e]
    ord <- orders[[e]][seq_len(rows)]
    pass_sums <- matrix(0, k, ncol(x))
    pass_counts <- numeric(k)
    for (start in seq(1, rows, by = B)) {
      idx <- ord[start:min(start + B - 1, rows)]
      xb <- x[idx, , drop = FALSE]
      ro <- if (spherical) unit(cent) else cent
      score <- if (spherical) -xb %*% t(ro) else
        matrix(rowSums(ro^2), nrow(xb), k, byrow = TRUE) - 2 * xb %*% t(ro)
      lab <- max.col(-score, ties.method = "first")
      cnt <- tabulate(lab, k)
      sums <- rowsum(xb, factor(lab, levels = seq_len(k)), reorder = TRUE)
      sums[is.na(sums)] <- 0
      pass_counts <- pass_counts + cnt
      pass_sums <- pass_sums + sums
      frac <- step / max(1, recipe$total_steps)
      alpha <- recipe$alpha0 * (0.1 + 0.9 * 0.5 * (1 + cos(pi * min(frac, 1))))
      usage <- usage * (1 - alpha)
      t <- cnt > 0
      mass <- usage[t] + alpha * cnt[t]
      w <- alpha * cnt[t] / mass
      mean <- sums[t, , drop = FALSE] / cnt[t]
      cent[t, ] <- cent[t, , drop = FALSE] + w * (mean - cent[t, , drop = FALSE])
      usage[t] <- mass
      step <- step + 1
    }
  }
  out <- cent
  ok <- pass_counts > 0
  out[ok, ] <- pass_sums[ok, , drop = FALSE] / pass_counts[ok]
  if (spherical) out <- unit(out)
  out
}

blobs <- function(n = 2000, d = 4, centers = 5, sd = 0.5, seed = 1) {
  set.seed(seed)
  cent <- matrix(rnorm(centers * d, sd = 5), centers)
  cent[sample(centers, n, TRUE), , drop = FALSE] + matrix(rnorm(n * d, sd = sd), n)
}

run_stream <- function(x, init, orders, ...) {
  s <- emeans_stream(nrow(x), init, n_threads = 1, ...)
  for (e in seq_along(s$recipe$epoch_rows)) {
    ord <- orders[[e]]
    pos <- 0
    while (pos < s$recipe$epoch_rows[e]) {
      b <- emeans_stream_status(s)$expected_batch
      # Split each batch in two chunks to exercise partial accumulation.
      h <- max(1, b %/% 2)
      emeans_accumulate(s, x[ord[(pos + 1):(pos + h)], , drop = FALSE])
      if (b > h) emeans_accumulate(s, x[ord[(pos + h + 1):(pos + b)], , drop = FALSE])
      emeans_update(s)
      pos <- pos + b
    }
    emeans_end_epoch(s)
  }
  emeans_finalize(s)
}
