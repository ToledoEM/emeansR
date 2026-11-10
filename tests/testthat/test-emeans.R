test_that("streaming matches the pure-R reference", {
  x <- blobs(1500, 4, 6)
  init <- x[1:6, ]
  set.seed(2)
  for (spherical in c(FALSE, TRUE)) {
    xx <- if (spherical) x / sqrt(rowSums(x^2)) else x
    recipe <- emeans_recipe(nrow(xx), 6, 4, nepochs = 2.5, batch_size = 128, unused_threshold = 0)
    orders <- lapply(seq_along(recipe$epoch_rows), function(e) sample(nrow(xx)))
    want <- reference_emeans(xx, init, recipe, orders, spherical = spherical)
    got <- run_stream(xx, init, orders, nepochs = 2.5, batch_size = 128,
                      unused_threshold = 0, spherical = spherical)
    expect_equal(unname(got$centers), unname(want), tolerance = 1e-10)
  }
})

test_that("emeans beats Lloyd at 25 iterations on blobs", {
  x <- blobs(20000, 8, 50, sd = 1)
  set.seed(3)
  km <- suppressWarnings(kmeans(x, x[sample(nrow(x), 50), ], iter.max = 25, algorithm = "Lloyd"))
  fit <- emeans(x, 50, nepochs = 3, n_threads = 1)
  expect_lte(fit$tot.withinss, km$tot.withinss * 1.01)
  expect_s3_class(fit, "kmeans")
  expect_equal(sum(fit$size), nrow(x))
  expect_equal(fit$tot.withinss + fit$betweenss, fit$totss)
})

test_that("results are reproducible and leave R's RNG alone", {
  x <- blobs()
  set.seed(10)
  before <- .Random.seed
  a <- emeans(x, 5, n_threads = 1)
  expect_identical(.Random.seed, before)
  b <- emeans(x, 5, n_threads = 2)
  expect_equal(a$centers, b$centers)
  c <- emeans(x, 5, seed = 99, n_threads = 1)
  expect_false(isTRUE(all.equal(a$centers, c$centers)))
})

test_that("predict equals brute-force nearest centers", {
  x <- blobs(3000, 6, 8)
  fit <- emeans(x, 8, n_threads = 1)
  newx <- blobs(500, 6, 8, seed = 4)
  d2 <- as.matrix(dist(rbind(fit$centers, newx)))[-(1:8), 1:8]^2
  p <- predict(fit, newx, distance = TRUE)
  expect_equal(p$cluster, unname(max.col(-d2, ties.method = "first")))
  expect_equal(p$distance, unname(apply(d2, 1, min)), tolerance = 1e-8)
  expect_equal(predict(fit, x), fit$cluster, ignore_attr = TRUE)
})

test_that("single precision is close to double", {
  x <- blobs(5000, 16, 20)
  a <- emeans(x, 20, n_threads = 1)
  b <- emeans(x, 20, precision = "single", n_threads = 1)
  expect_equal(b$tot.withinss, a$tot.withinss, tolerance = 1e-3)
})

test_that("spherical centers are unit vectors", {
  x <- blobs(2000, 5, 4)
  x <- x / sqrt(rowSums(x^2))
  fit <- emeans(x, 4, spherical = TRUE, n_threads = 1)
  expect_equal(rowSums(fit$centers^2), rep(1, 4), tolerance = 1e-12)
  expect_warning(emeans(x * 2, 4, spherical = TRUE, n_threads = 1), "unit-norm")
})

test_that("revival runs above d = 2 and is off at d <= 2", {
  # Many clusters started in one blob force dead clusters.
  x <- blobs(5000, 4, 3, sd = 0.3)
  init <- x[rep(1, 30), ] + matrix(rnorm(120, sd = 1e-3), 30)
  fit <- emeans(x, init, nepochs = 3, n_threads = 1)
  expect_gt(fit$n_splits, 0)
  x2 <- blobs(5000, 2, 3)
  expect_equal(emeans(x2, 10, n_threads = 1)$n_splits, 0)
})

test_that("max_points_per_centroid subsamples", {
  x <- blobs(5000, 4, 5)
  fit <- emeans(x, 5, max_points_per_centroid = 100, final_assign = FALSE, n_threads = 1)
  expect_equal(sum(!is.na(fit$cluster)), 500)
  expect_equal(fit$recipe$epoch_rows[1], 500)
  full <- emeans(x, 5, max_points_per_centroid = 100, n_threads = 1)
  expect_false(anyNA(full$cluster))
})

test_that("input validation", {
  expect_error(emeans(matrix(c(1, NA), 2), 1), "NA")
  expect_error(emeans(matrix(1:4, 2), 3), "more clusters")
  expect_error(emeans(letters, 2), "numeric")
  s <- emeans_stream(100, matrix(rnorm(6), 3), batch_size = 10)
  expect_error(emeans_accumulate(s, matrix(rnorm(40), 20)), "batch overflow")
  expect_error(emeans_update(s), "expects a batch")
})

test_that("print shows fractional epochs", {
  fit <- emeans(blobs(2000, 4, 5), 5, nepochs = 0.4, n_threads = 1)
  expect_output(print(fit), "Epochs: 0.4 ")
})

test_that("fitted needs cluster labels", {
  x <- blobs(200, 2, 3)
  s <- emeans_stream(nrow(x), x[1:3, ], nepochs = 1, batch_size = 200, n_threads = 1)
  emeans_accumulate(s, x)
  emeans_update(s)
  emeans_end_epoch(s)
  expect_error(fitted(emeans_finalize(s)), "use predict")
})

test_that("emeans_stream validates n", {
  init <- matrix(rnorm(6), 3)
  expect_error(emeans_stream(10.5, init), "whole number")
  expect_error(emeans_stream(-1, init), "whole number")
  expect_error(emeans_stream(c(10, 20), init), "whole number")
})

test_that("fractional max_points_per_centroid truncates", {
  x <- blobs(5000, 4, 5)
  fit <- emeans(x, 5, max_points_per_centroid = 100.7, final_assign = FALSE, n_threads = 1)
  expect_equal(sum(!is.na(fit$cluster)), 500)
})
