# Changelog

## emeansR 0.1.0

First release.

- [`emeans()`](https://toledoem.github.io/emeansR/reference/emeans.md)
  fits e-means (Bessa et al., 2026, arXiv:2611.09501): mini-batch
  k-means with exponential moving average centroid updates. The batch
  size and initial step come from the scaling laws of the reference
  library, `faiss/contrib/e_means`, and match it exactly for the same
  data size, number of clusters, dimension and epoch budget.
- The result also inherits from `kmeans`, so `cluster`, `centers`,
  `size`, `withinss` and `tot.withinss` work as usual.
  [`predict()`](https://rdrr.io/r/stats/predict.html) labels new rows.
- [`emeans_stream()`](https://toledoem.github.io/emeansR/reference/emeans_stream.md)
  and its companions train on data supplied in chunks, for data that do
  not fit in memory.
- `spherical = TRUE` clusters unit vectors by cosine similarity;
  `precision = "single"` computes distances in single precision;
  `max_points_per_centroid` trains on a random subset.
- [`emeans_recipe()`](https://toledoem.github.io/emeansR/reference/emeans_recipe.md)
  shows the batch size and step the scaling laws choose.
- The compute core is written in Rust and runs on several threads. All
  Rust dependencies are bundled, so installation needs no network
  access. Needs rustc 1.71 or later.
- The pkgdown site has two articles: a port of the authors’ CPU demo,
  and a benchmark against
  [`kmeans()`](https://rdrr.io/r/stats/kmeans.html),
  [`hclust()`](https://rdrr.io/r/stats/hclust.html) and `Mclust()`.
