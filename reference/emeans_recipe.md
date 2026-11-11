# Inspect the e-means training recipe

Resolves batch size and initial step size from the fitted scaling laws
of the e-means paper, exactly as
[`emeans()`](https://toledoem.github.io/emeansR/reference/emeans.md)
does.

## Usage

``` r
emeans_recipe(
  n,
  k,
  d,
  nepochs = 5,
  batch_law = c("auto", "throughput", "quality"),
  batch_size = NULL,
  alpha0 = NULL,
  unused_threshold = NULL
)
```

## Arguments

- n:

  Number of training rows.

- k:

  Number of clusters.

- d:

  Dimension (selects the `"auto"` batch law and revival default).

- nepochs:

  Epoch budget; fractions allowed.

- batch_law:

  `"auto"` (quality law when `d <= 16`, else throughput),
  `"throughput"`, or `"quality"` (batch about 8 times smaller).

- batch_size, alpha0:

  Optional overrides. Giving `batch_size` alone keeps the step law for
  `alpha0`.

- unused_threshold:

  Revival threshold tau: clusters with mass below `tau * mean(mass)` are
  revived. `NULL` gives 0.1, or 0 (off) when `d <= 2`.

## Value

A list with `batch_size`, `alpha0`, `epoch_rows`, `epoch_steps`,
`total_steps`, `batch_law`, and `unused_threshold`.

## Examples

``` r
emeans_recipe(n = 1e6, k = 1024, d = 128, nepochs = 3)
#> $batch_size
#> [1] 32768
#> 
#> $alpha0
#> [1] 0.6955212
#> 
#> $epoch_rows
#> [1] 1e+06 1e+06 1e+06
#> 
#> $epoch_steps
#> [1] 31 31 31
#> 
#> $total_steps
#> [1] 93
#> 
#> $batch_law
#> [1] "throughput"
#> 
#> $unused_threshold
#> [1] 0.1
#> 
#> $nepochs
#> [1] 3
#> 
```
