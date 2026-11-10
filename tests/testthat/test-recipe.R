# Values produced by faiss/contrib/e_means/base.py (_resolve_recipe).
test_that("recipe matches the faiss reference", {
  ref <- list(
    list(c(1e5, 256, 32, 3), 4096, 0.5715175033166802),
    list(c(1e6, 1024, 128, 3), 32768, 0.6955211692520258),
    list(c(5e4, 10, 2, 5), 256, 0.2723319441596358),
    list(c(2e6, 65536, 768, 2.5), 131072, 0.2650927404763027),
    list(c(1000, 7, 3, 0.4), 1, 0.01654316026018749)
  )
  for (r in ref) {
    p <- r[[1]]
    got <- emeans_recipe(p[1], p[2], p[3], p[4])
    expect_equal(got$batch_size, r[[2]])
    expect_equal(got$alpha0, r[[3]], tolerance = 1e-12)
  }
})

test_that("auto law and revival default follow the dimension", {
  expect_equal(emeans_recipe(1e5, 100, 16)$batch_law, "quality")
  expect_equal(emeans_recipe(1e5, 100, 17)$batch_law, "throughput")
  expect_equal(emeans_recipe(1e5, 100, 2)$unused_threshold, 0)
  expect_equal(emeans_recipe(1e5, 100, 3)$unused_threshold, 0.1)
})

test_that("fractional epochs keep a closing full pass", {
  r <- emeans_recipe(1000, 10, 5, nepochs = 2.5)
  expect_equal(r$epoch_rows, c(1000, 500, 1000))
  expect_equal(emeans_recipe(1000, 10, 5, nepochs = 0.4)$epoch_rows, 400)
  expect_equal(r$total_steps, sum(ceiling(r$epoch_rows / r$batch_size)))
})

test_that("overrides are respected", {
  r <- emeans_recipe(1e4, 10, 5, batch_size = 100, alpha0 = 0.3)
  expect_equal(r$batch_size, 100)
  expect_equal(r$alpha0, 0.3)
  expect_true(is.na(r$batch_law))
})
