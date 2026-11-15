# Assign new data to e-means centers

Assign new data to e-means centers

## Usage

``` r
# S3 method for class 'emeans'
predict(object, newdata, distance = FALSE, n_threads = NULL, ...)
```

## Arguments

- object:

  An `emeans` fit.

- newdata:

  Numeric matrix or data frame with the same columns.

- distance:

  If `TRUE`, return a data frame with `cluster` and `distance`: the
  squared Euclidean distance, or, when spherical, the inner product with
  the unit centers (the cosine similarity if `newdata` rows are
  unit-norm).

- n_threads:

  Threads; see
  [`emeans()`](https://toledoem.github.io/emeansR/reference/emeans.md).

- ...:

  Unused.

## Value

Integer cluster labels, or a data frame when `distance = TRUE`.
