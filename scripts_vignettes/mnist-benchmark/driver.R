# Build the benchmark grid (raw pixels, single thread, k up to 1000) and run each
# configuration in its own R process under /usr/bin/time -l (peak RSS) and gtimeout.
# Safeguard: a run longer than 15 min is killed and saved with status "timeout".
# Skip rule: after a timeout/oom, larger n and remaining seeds of that (method, k) are "skipped".
# Resumes: run_ids already in runs.csv (any status but error) are not rerun.
# Usage: caffeinate -ims Rscript driver.R [--smoke]   (from any directory, after prepare.R)
#   Reads and writes out/ (out-smoke/ with --smoke) next to this script.
#   Several workers: run one driver per worker with --worker <i>; each claims the next run
#   (atomic mkdir in tmp/claims) and appends results under a lock. Trash tmp/claims before a launch.

# Arguments and paths
argv <- commandArgs(TRUE)
smoke <- "--smoke" %in% argv
worker <- if ("--worker" %in% argv) argv[which(argv == "--worker") + 1] else "1"
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- if (length(script)) dirname(normalizePath(script)) else getwd()
source(file.path(here, "methods.R"))
out <- file.path(here, if (smoke) "out-smoke" else "out")
dir.create(file.path(out, "tmp", "claims"), recursive = TRUE, showWarnings = FALSE)

# Time limit, lock, claim and log helpers
timeout_s <- 900
log_file <- file.path(out, "driver.log")
lock_dir <- file.path(out, "tmp", "lock")
with_lock <- function(expr) {
  while (!dir.create(lock_dir, showWarnings = FALSE)) Sys.sleep(0.05)
  on.exit(unlink(lock_dir, recursive = TRUE))
  force(expr)
}
claim <- function(id) dir.create(file.path(out, "tmp", "claims", id), showWarnings = FALSE)
log <- function(...) {
  line <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S "), "w", worker, " ", sprintf(...))
  message(line)
  cat(line, "\n", file = log_file, append = TRUE, sep = "")
}
# Single thread for every library in the children
Sys.setenv(OMP_NUM_THREADS = "1", VECLIB_MAXIMUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
           RCPP_PARALLEL_NUM_THREADS = "1")

# Grid
g <- function(method, n, k = NA, seed = 1, precision = "double", nepochs = NA, ks = NA) {
  expand.grid(method = method, n = n, k = k, seed = seed, rep = 1L, precision = precision,
              nepochs = nepochs, ks = ks, stringsAsFactors = FALSE)
}
sizes <- if (smoke) 1000 else c(5000, 10000, 20000, 30000, 70000)
ks <- if (smoke) c(10, 100) else c(10, 100, 1000)
seeds <- if (smoke) 1 else 1:3
ks_str <- paste(ks, collapse = ",")
grid <- rbind(
  g("emeans", sizes, ks, seeds, precision = c("double", "single"), nepochs = 5),
  g("emeans", max(sizes), if (smoke) 10 else 100, seeds, nepochs = c(1, 2, 3, 5, 10)),
  g(c("kmeans_hw", "kmeans_lloyd", "clusterr_kmeans", "clusterr_minibatch", "clara"), sizes, ks, seeds),
  g("mclust_eii", sizes[sizes <= 10000], 10),
  g(c("stats_hclust", "flashclust", "fastcluster_hclust"), sizes[sizes <= 30000], ks = ks_str),
  g("fastcluster_vector", sizes, ks = ks_str)
)
grid$timeout_s <- timeout_s
if (smoke) grid <- rbind(grid, transform(g("timeout_probe", 1000, 10), timeout_s = 2))
grid$run_id <- with(grid, sprintf("%s_%s_e%s_n%d_k%s_s%d", method, precision,
                                  ifelse(is.na(nepochs), "NA", nepochs), n,
                                  ifelse(is.na(k), "all", k), seed))
grid <- grid[!duplicated(grid$run_id), ]
grid$save_centers <- grid$n == max(sizes) & grid$seed == 1
grid <- grid[order(grid$n, grid$method != "emeans", grid$method, grid$k, grid$seed), ]
rownames(grid) <- grid$run_id
with_lock({
  tmp_grid <- file.path(out, "tmp", paste0("grid-", worker, ".rds"))
  saveRDS(grid, tmp_grid)
  file.rename(tmp_grid, file.path(out, "grid.rds"))
  write.csv(grid, file.path(out, "grid.csv"), row.names = FALSE)
})

# Skip key: a timeout/oom skips later runs that share it
fail_key <- function(cfg) {
  if (methods[[cfg$method]]$family == "hierarchical") cfg$method
  else paste(cfg$method, cfg$k, cfg$precision, cfg$nepochs)
}

# Environment
env_file <- file.path(out, "env.rds")
if (!file.exists(env_file) && worker == "1") {
  pkgs <- c("emeansR", "stats", "ClusterR", "cluster", "mclust", "flashClust", "fastcluster", "aricode")
  env <- list(
    cpu = system("sysctl -n machdep.cpu.brand_string", intern = TRUE),
    cores_physical = as.integer(system("sysctl -n hw.physicalcpu", intern = TRUE)),
    ram_gb = as.numeric(system("sysctl -n hw.memsize", intern = TRUE)) / 2^30,
    os = paste(system("sw_vers -productName", intern = TRUE), system("sw_vers -productVersion", intern = TRUE)),
    r = R.version.string, blas = extSoftVersion()[["BLAS"]], lapack = La_library(),
    emeansR_git = system("git rev-parse --short HEAD", intern = TRUE),
    thread_env = Sys.getenv(c("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                              "RCPP_PARALLEL_NUM_THREADS")),
    input = "raw 784 pixels scaled 0-1", threads = 1L,
    packages = sessioninfo::package_info(pkgs, dependencies = FALSE),
    timeout_s = timeout_s, started = Sys.time()
  )
  saveRDS(env, env_file)
  capture.output(str(env[setdiff(names(env), "packages")]), print(env$packages), file = file.path(out, "env.txt"))
}

# Output helpers
append_csv <- function(df, file) {
  write.table(df, file, sep = ",", row.names = FALSE, qmethod = "double",
              col.names = !file.exists(file), append = file.exists(file))
}
fix_cols <- function(row) {
  for (cc in setdiff(run_cols, names(row))) row[[cc]] <- NA
  row[, run_cols]
}
# Rows for a run that produced no result (timeout, oom, error, skipped)
fail_rows <- function(cfg, status, wall, max_rss_mb, msg) {
  k <- if (is.na(cfg$k)) as.integer(strsplit(cfg$ks, ",")[[1]]) else cfg$k
  m <- methods[[cfg$method]]
  do.call(rbind, lapply(k, function(kk) fix_cols(data.frame(
    run_id = cfg$run_id, fit_id = sub("_kall_", sprintf("_k%d_", kk), cfg$run_id),
    method = cfg$method, family = m$family, package = m$package, n = cfg$n, d = 784L, k = kk,
    threads = 1L, precision = cfg$precision, nepochs = cfg$nepochs, seed = cfg$seed, rep = cfg$rep,
    timeout_s = cfg$timeout_s, elapsed_s = if (status == "timeout") cfg$timeout_s else NA,
    max_rss_mb = max_rss_mb, process_wall_s = wall, status = status, error_msg = msg))))
}

# Resume state (re-read before each run: other workers append too)
runs_file <- file.path(out, "runs.csv")
read_state <- function() {
  prev <- if (file.exists(runs_file)) with_lock(read.csv(runs_file)) else NULL
  bad <- prev[prev$status %in% c("timeout", "oom"), ]
  list(done = unique(prev$run_id[prev$status != "error"]),
       failed = unique(vapply(unique(bad$run_id), function(r) fail_key(grid[r, ]), "")))
}
st0 <- read_state()
todo <- setdiff(grid$run_id, st0$done)
log("grid %d runs, %d done, %d to run", nrow(grid), length(st0$done), length(todo))

# Main loop: one configuration per child process
for (id in todo) {
  state <- read_state()
  if (id %in% state$done || !claim(id)) next
  cfg <- grid[id, ]
  key <- fail_key(cfg)
  if (key %in% state$failed) {
    with_lock(append_csv(fail_rows(cfg, "skipped", NA, NA, "earlier timeout/oom for this method and k"), runs_file))
    log("%-55s skipped", id)
    next
  }
  # Run under /usr/bin/time (peak RSS) and gtimeout (time limit)
  time_file <- file.path(out, "tmp", paste0(id, ".time"))
  t0 <- Sys.time()
  status <- system2("/usr/bin/time",
                    c("-l", "gtimeout", "-k", "30", cfg$timeout_s, "Rscript", file.path(here, "run-one.R"), out, id),
                    stdout = time_file, stderr = time_file)
  wall <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  txt <- readLines(time_file, warn = FALSE)
  rss_line <- grep("maximum resident set size", txt, value = TRUE)
  max_rss_mb <- if (length(rss_line)) as.numeric(sub("^\\s*(\\d+).*", "\\1", rss_line[1])) / 2^20 else NA
  # Collect the result rows, or classify the failure as timeout, oom or error
  res_file <- file.path(out, "results", paste0(id, ".rds"))
  if (status == 0 && file.exists(res_file)) {
    res <- readRDS(res_file)
    rows <- do.call(rbind, lapply(res$rows, function(r) {
      r$max_rss_mb <- max_rss_mb
      r$process_wall_s <- wall
      r$status <- "ok"
      r$error_msg <- ""
      fix_cols(r)
    }))
    with_lock({
      if (!is.null(res$extra$trace)) append_csv(cbind(run_id = id, res$extra$trace), file.path(out, "emeans_trace.csv"))
      if (!is.null(res$extra$recipe)) append_csv(cbind(run_id = id, res$extra$recipe), file.path(out, "emeans_recipe.csv"))
      if (!is.null(res$extra$bic)) append_csv(data.frame(run_id = id, loglik = res$extra$loglik, bic = res$extra$bic),
                                              file.path(out, "mclust_fit.csv"))
    })
  } else {
    st <- if (status %in% c(124, 137) && wall >= cfg$timeout_s) "timeout"
          else if (status == 137 || any(grepl("cannot allocate|vector memory", txt))) "oom"
          else "error"
    err <- gsub("[\r\n,]", " ", paste(head(grep("^Error", txt, value = TRUE), 2), collapse = " "))
    rows <- fail_rows(cfg, st, wall, max_rss_mb, err)
  }
  with_lock(append_csv(rows, runs_file))
  log("%-55s %-7s %8.1f s  rss %6.0f MB  inertia %s", id, rows$status[1], wall, max_rss_mb,
      paste(sprintf("%.2f", rows$inertia), collapse = "/"))
}

# Finish: record end time and worker count
with_lock({
  env <- readRDS(env_file)
  env$finished <- Sys.time()
  env$workers <- max(as.integer(env$workers), as.integer(worker), na.rm = TRUE)
  saveRDS(env, env_file)
})
log("finished")
