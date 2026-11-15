# e-means clustering

Mini-batch k-means with exponential-moving-average centroid updates
(Bessa et al., 2026, arXiv:2611.09501). Batch size and step size come
from fitted scaling laws, so the only inputs are the data, `k`, and the
epoch budget. In a few epochs it typically reaches a lower
within-cluster sum of squares than 25 Lloyd iterations.

## Usage

``` r
emeans(
  x,
  k,
  nepochs = 5,
  batch_law = c("auto", "throughput", "quality"),
  batch_size = NULL,
  alpha0 = NULL,
  spherical = FALSE,
  unused_threshold = NULL,
  max_points_per_centroid = NULL,
  precision = c("double", "single"),
  final_assign = TRUE,
  seed = 1234L,
  n_threads = NULL,
  verbose = FALSE
)
```

## Arguments

- x:

  Numeric matrix or data frame, one row per point.

- k:

  Number of clusters, or a `k x d` matrix of initial centers.

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

- max_points_per_centroid:

  If set and `nrow(x) > max_points_per_centroid * k`, train on a random
  subset of that many rows (as `faiss.Kmeans`).

- precision:

  `"double"`, or `"single"` to store the training data and compute
  distances in float32 (about 2x faster, half the memory). Batch
  statistics, centroids and masses always accumulate in double
  precision, and `withinss` is computed from the original data. As in
  the faiss FP32 path, single precision assigns poorly when the data sit
  far from the origin (the error grows with the squared norm of the
  rows); centre the data first, e.g. with `scale(x, scale = FALSE)`.
  Double precision hits the same limit at offsets around 1e8.

- final_assign:

  If `TRUE` (default), assign every row of `x` to the final centers, so
  `cluster`, `withinss` and `size` describe the returned centers (as
  [`stats::kmeans()`](https://rdrr.io/r/stats/kmeans.html)). If `FALSE`,
  `cluster` holds labels from the last training pass, and rows left out
  by `max_points_per_centroid` are `NA`.

- seed:

  Integer seed for initialization, shuffling and revival. Does not touch
  R's random number generator.

- n_threads:

  Threads for assignment. Defaults to `getOption("emeansR.threads")`,
  else physical cores minus one (at most two when the
  `_R_CHECK_LIMIT_CORES_` environment variable is set).

- verbose:

  Print the recipe and per-epoch objective.

## Value

An object of class `c("emeans", "kmeans")` with components `cluster`,
`centers`, `totss`, `withinss`, `tot.withinss`, `betweenss`, `size`,
`iter` (epochs), `ifault` (0), plus `obj` (moving-pass objective per
epoch), `iteration_stats`, `recipe`, `n_splits`, `spherical` and
`precision`.

## Details

Each epoch visits the rows in a fresh random order in batches of
`batch_size`. Every batch is assigned to the nearest centroids, and each
touched centroid moves towards its batch mean with weight
`alpha * n_j / N_j`, where the mass `N_j` is itself an exponential
moving average and `alpha` follows a cosine schedule from `alpha0` to
`alpha0 / 10`. Every five updates, clusters with mass below
`unused_threshold * mean(mass)` are revived by splitting large clusters
(picked with probability proportional to mass). The returned centers are
the exact means of the points assigned during the last epoch.

## Examples

``` r
set.seed(1)
x <- rbind(matrix(rnorm(2000, 0), ncol = 2), matrix(rnorm(2000, 5), ncol = 2))
fit <- emeans(x, k = 2, nepochs = 3, n_threads = 1)
fit$centers
#>            [,1]        [,2]
#> [1,] -0.0150379 -0.01895623
#> [2,]  5.0136701  5.01438322
table(fit$cluster)
#> 
#>    1    2 
#>  999 1001 
```
