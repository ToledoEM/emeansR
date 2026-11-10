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

test_that("withinss and predict distances are exact far from the origin", {
  set.seed(1)
  x <- matrix(rnorm(20000 * 4), ncol = 4) + 1e6
  for (prec in c("double", "single")) {
    fit <- emeans(x, 8, precision = prec, n_threads = 1)
    exact <- rowSums((x - fit$centers[fit$cluster, ])^2)
    want <- as.vector(tapply(exact, factor(fit$cluster, 1:8), sum))
    want[is.na(want)] <- 0
    expect_equal(fit$withinss, want, tolerance = 1e-8)
    p <- predict(fit, x[1:200, ], distance = TRUE)
    expect_equal(p$distance, unname(exact[1:200]), tolerance = 1e-8)
  }
})

test_that("alpha0 = 1 (replace path) matches the reference", {
  x <- blobs(1200, 3, 4)
  init <- x[1:4, ]
  recipe <- emeans_recipe(nrow(x), 4, 3, nepochs = 2, batch_size = 100, alpha0 = 1, unused_threshold = 0)
  set.seed(5)
  orders <- lapply(seq_along(recipe$epoch_rows), function(e) sample(nrow(x)))
  want <- reference_emeans(x, init, recipe, orders)
  got <- run_stream(x, init, orders, nepochs = 2, batch_size = 100, alpha0 = 1, unused_threshold = 0)
  expect_equal(unname(got$centers), unname(want), tolerance = 1e-10)
})

test_that("nepochs too small to train errors", {
  expect_error(emeans_recipe(1000, 10, 5, nepochs = 1e-4), "nepochs too small")
})

test_that("spherical stream warns once on non-unit chunks", {
  x <- blobs(400, 3, 2)
  xs <- x / sqrt(rowSums(x^2))
  s <- emeans_stream(400, xs[1:2, ], nepochs = 1, batch_size = 100, spherical = TRUE, n_threads = 1)
  expect_warning(emeans_accumulate(s, x[1:50, ]), "unit-norm")
  expect_no_warning(emeans_accumulate(s, x[51:100, ]))
})

test_that("emeans validates its inputs", {
  x <- blobs(300, 3, 3)
  expect_error(emeans(x, x[1:3, 1:2], n_threads = 1), "must have 3 columns")
  expect_error(emeans(x, 0, n_threads = 1), "positive integer")
  expect_error(emeans(x, NA, n_threads = 1), "positive integer")
  expect_error(emeans(x[1:2, ], 3, n_threads = 1), "more clusters than rows")
  expect_error(emeans(x, 3, max_points_per_centroid = 0.5, n_threads = 1), ">= 1")
})

test_that("emeans takes data frames and returns fitted centers", {
  x <- blobs(600, 3, 3)
  fit <- emeans(as.data.frame(x), 3, nepochs = 2, n_threads = 1)
  expect_equal(fit$cluster, emeans(x, 3, nepochs = 2, n_threads = 1)$cluster)
  expect_equal(unname(fitted(fit)), unname(fit$centers[fit$cluster, ]))
  expect_equal(fitted(fit, "classes"), fit$cluster)
})

test_that("verbose reports sampling, recipe and epochs", {
  x <- blobs(2000, 3, 4)
  msgs <- character()
  withCallingHandlers(
    emeans(x, 4, nepochs = 2, max_points_per_centroid = 100, verbose = TRUE, n_threads = 1),
    message = function(m) {
      msgs <<- c(msgs, conditionMessage(m))
      invokeRestart("muffleMessage")
    }
  )
  expect_match(msgs, "Sampling 400 of 2000", all = FALSE)
  expect_match(msgs, "Clustering 400 points", all = FALSE)
  expect_match(msgs, "Epoch 2 / 2", all = FALSE)
})

test_that("thread defaults follow the option and the CRAN check limit", {
  old <- options(emeansR.threads = 3)
  on.exit(options(old), add = TRUE)
  expect_equal(default_threads(NULL), 3L)
  expect_equal(default_threads(5), 5L)
  options(emeansR.threads = NULL)
  prev <- Sys.getenv("_R_CHECK_LIMIT_CORES_", NA)
  on.exit(if (is.na(prev)) Sys.unsetenv("_R_CHECK_LIMIT_CORES_") else Sys.setenv(`_R_CHECK_LIMIT_CORES_` = prev),
          add = TRUE)
  Sys.setenv(`_R_CHECK_LIMIT_CORES_` = "TRUE")
  expect_lte(default_threads(NULL), 2L)
})

test_that("explicit batch laws resolve", {
  q <- emeans_recipe(1e5, 64, 32, nepochs = 3, batch_law = "quality")
  t <- emeans_recipe(1e5, 64, 32, nepochs = 3, batch_law = "throughput")
  expect_equal(q$batch_law, "quality")
  expect_equal(t$batch_law, "throughput")
})

test_that("stream calls out of order fail with R errors", {
  x <- blobs(200, 2, 3)
  expect_error(emeans_stream(2, x[1:3, ], n_threads = 1), "more clusters than rows")
  s <- emeans_stream(nrow(x), x[1:3, ], nepochs = 1, batch_size = 200, n_threads = 1)
  expect_output(print(s), "epoch 0 / 1")
  expect_error(emeans_accumulate(s, x[, 1, drop = FALSE]), "chunk has 1 columns, model has 2")
  expect_error(emeans_end_epoch(s), "epoch 1 expects 200 rows, updated 0")
  expect_error(emeans_finalize(s), "finalize needs all 1 epochs")
  emeans_accumulate(s, x)
  emeans_update(s)
  emeans_end_epoch(s)
  expect_error(emeans_update(s), "all epochs are complete")
  res <- emeans_finalize(s)
  expect_output(print(res), "3 clusters in 2D, 1 epochs")
  expect_output(print(s), "\\(finalized\\)")
  expect_error(emeans_update(s), "already finalized")
})
