# Streaming e-means for data that does not fit in memory

Caller-driven training: the caller supplies rows in chunks, in any order
it chooses (shuffle each epoch for best results). Each epoch must supply
exactly `recipe$epoch_rows[epoch]` rows, grouped into batches of
`emeans_stream_status(s)$expected_batch` rows; one batch may be split
across several `emeans_accumulate()` calls.

## Usage

``` r
emeans_stream(
  n,
  init,
  nepochs = 5,
  batch_law = c("auto", "throughput", "quality"),
  batch_size = NULL,
  alpha0 = NULL,
  spherical = FALSE,
  unused_threshold = NULL,
  precision = c("double", "single"),
  seed = 1234L,
  n_threads = NULL
)

emeans_stream_status(s)

emeans_accumulate(s, chunk)

emeans_update(s)

emeans_end_epoch(s)

emeans_finalize(s)
```

## Arguments

- n:

  Total number of training rows (one epoch).

- init:

  `k x d` matrix of initial centers (e.g. random training rows).

- nepochs:

  Epoch budget; fractions allowed (e.g. `1.5`).

- batch_law:

  `"auto"` (quality law when `d <= 16`, else throughput),
  `"throughput"`, or `"quality"` (batch about 8 times smaller).

- batch_size, alpha0:

  Optional overrides. Giving `batch_size` alone keeps the step law for
  `alpha0`.

- spherical:

  Cluster unit vectors by cosine similarity. Rows of `x` must already be
  unit-norm; centers are returned as unit vectors.

- unused_threshold:

  Revival threshold tau: clusters with mass below `tau * mean(mass)` are
  revived. `NULL` gives 0.1, or 0 (off) when `d <= 2`.

- precision:

  `"double"`, or `"single"` to store the training data and compute
  distances in float32 (about 2x faster, half the memory). Batch
  statistics, centroids and masses always accumulate in double
  precision, and `withinss` is computed from the original data. As in
  the faiss FP32 path, single precision assigns poorly when the data sit
  far from the origin (the error grows with the squared norm of the
  rows); centre the data first, e.g. with `scale(x, scale = FALSE)`.
  Double precision hits the same limit at offsets around 1e8.

- seed:

  Integer seed for initialization, shuffling and revival. Does not touch
  R's random number generator.

- n_threads:

  Threads for assignment. Defaults to `getOption("emeansR.threads")`,
  else physical cores minus one (at most two when the
  `_R_CHECK_LIMIT_CORES_` environment variable is set).

- s:

  An `emeans_stream` object.

- chunk:

  Numeric matrix of rows to add to the current batch.

## Value

`emeans_stream()` returns an `emeans_stream` object.

`emeans_stream_status()` returns a list with `step`, `epoch` (completed
epochs), `n_epochs`, `expected_batch` (rows the next update needs),
`n_splits`, and the per-epoch objective so far.

`emeans_accumulate()` returns the chunk's cluster labels against the
centroids of the current batch (invisibly).

`emeans_finalize()` returns a list of class `emeans_streamed` with
`centers`, `obj`, `iteration_stats`, `recipe`, `n_splits`; use
[`predict.emeans()`](https://toledoem.github.io/emeansR/reference/predict.emeans.md)
on it to label data.

## Details

The loop is:

    s <- emeans_stream(n, init, nepochs = 3)
    for (epoch in seq_along(s$recipe$epoch_rows)) {
      left <- s$recipe$epoch_rows[epoch]
      while (left > 0) {
        b <- emeans_stream_status(s)$expected_batch
        emeans_accumulate(s, next_rows(b))  # next_rows(): your data reader
        emeans_update(s)
        left <- left - b
      }
      emeans_end_epoch(s)
    }
    fit <- emeans_finalize(s)

See the "Streaming" section of the vignette for a complete example.
