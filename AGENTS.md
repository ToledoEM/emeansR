# AGENTS.md

## Project overview

emeansR is an R package implementing e-means (Bessa et al., 2026, arXiv:2611.09501): mini-batch k-means with exponential-moving-average centroid updates, and batch size and step size set by fitted scaling laws. CPU only. The compute core is Rust (extendr), and the R layer handles the recipe, input validation and S3 methods.

The reference implementation is `faiss/contrib/e_means` (Python). For the same `n`, `k`, `d` and `nepochs`, `emeans_recipe()` must give exactly the batch size and `alpha0` that faiss gives.

## Layout

- `R/recipe.R`: scaling laws (`emeans_recipe()`), constants copied from faiss `base.py`.
- `R/emeans.R`: `emeans()`, `print`/`predict`/`fitted` methods.
- `R/stream.R`: caller-driven streaming API (`emeans_stream()`, `emeans_accumulate()`, `emeans_update()`, `emeans_end_epoch()`, `emeans_finalize()`). Preconditions are checked in R so users get R errors, not Rust panics.
- `R/extendr-wrappers.R`: generated, do not edit by hand (see below).
- `src/rust/src/`: `kernel.rs` (blocked GEMM + fused argmin, f32/f64), `state.rs` (EMA update, cosine schedule, revival, finalize), `rng.rs` (xoshiro256**), `lib.rs` (extendr glue, `em_*` functions).
- `src/rust/vendor.tar.xz`: vendored crates for offline builds. `inst/AUTHORS` lists their authors and licenses.
- `configure`, `configure.win`, `tools/config.sh`: find cargo/rustc, check the minimum version, write `src/Makevars[.win]` from `src/Makevars[.win].in`. `cleanup` removes the generated Makevars.
- `vignettes/`: articles built by pkgdown. `.github/workflows/`: CI.
- `.dev/`: development notes and scripts. Ignored by git and by `R CMD build`. Never move anything from there into the package.

## Build and test commands

```sh
# Install into a library and run tests
R CMD INSTALL .
Rscript -e 'testthat::test_local(".")'

# Full check, as CRAN would run it
R CMD build . && R CMD check --as-cran --no-manual emeansR_*.tar.gz

# Docs and NAMESPACE (roxygen2 owns both)
Rscript -e 'roxygen2::roxygenise()'

# Site
Rscript -e 'pkgdown::build_site()'

# Rust only
cd src/rust && cargo clippy --release
```

After changing an `#[extendr]` signature, regenerate the wrappers from the installed library (rextendr is not required):

```sh
Rscript -e 'dyn.load(system.file("libs", "emeansR.so", package = "emeansR")); cat(.Call("wrap__make_emeansR_wrappers", use_symbols = TRUE, package_name = "emeansR", PACKAGE = "emeansR"))' > R/extendr-wrappers.R
```

After changing `src/rust/Cargo.toml`, re-vendor with `sh .dev/vendor-update.sh` (refreshes `vendor.tar.xz` and `inst/AUTHORS`).

## Rust and CRAN rules

- The build must work offline: `cargo build --offline --locked` against the vendored crates. Never add a dependency without re-vendoring.
- Minimum Rust is 1.71 (`rust-version` in `Cargo.toml`, `SystemRequirements` in DESCRIPTION, `tools/config.sh`). Do not use std APIs newer than that (e.g. `f64::next_up`, `div_ceil`). `cargo clippy` warns about them through `incompatible_msrv`. Resolve new dependencies with `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback`, and keep `Cargo.lock` at lockfile version 3.
- Builds use `-j 2` and a `CARGO_HOME` inside `src/`, never `~/.cargo`.
- `rust_clean` in the Makevars deletes the static library after linking. That keeps R's compiled-code check from flagging `abort` in Rust std.
- No printing from Rust, and no `R_CheckUserInterrupt` (longjmp across Rust frames). `emeans()` returns to R between chunks of steps so interrupts work.

## Code style

- R: lintr defaults with 120-character lines (`.github/.lintr`). CI fails on any lint, so `lintr::lint_package()` must report none; run it with the current build installed, or it flags new Rust entry points as undefined. Paths in the `.lintr` exclusions are relative to `.github/`. Base R only at runtime; Imports are `parallel`, `stats`, `utils`.
- Rust: `cargo clippy` clean except style lints. Centroids, masses and statistics accumulate in f64; only distance computation uses f32 when `precision = "single"`.
- The package never touches R's RNG. All randomness comes from `seed` through `rng.rs`.
- Labels are 0-based in Rust and 1-based in R.

## Testing instructions

- `tests/testthat/helper-reference.R` holds a slow pure-R e-means (no revival) used as an oracle. The streaming test compares it to the Rust path with tolerance 1e-10. Keep that test passing after any change to `state.rs`.
- `test-recipe.R` pins values produced by faiss `_resolve_recipe`. Do not edit those numbers to make a test pass. If the recipe changes, re-derive them from faiss.
- `.dev/xcheck/` holds the cross-check against faiss `NumpyEmeans` (run with `uv run --with faiss-cpu --with numpy python -I`).
- Tests and examples must use `n_threads = 1` or `2`.

## Commits and pull requests

- `git config core.hooksPath .githooks` enables the commit-msg hook (`.githooks/commit-lint.sh`): Conventional Commits `type(scope): imperative summary`, at most 72 characters, no trailing period, body at most 5 lines and 150 characters.

## Security considerations

- Installation must not download anything. Do not add `build.rs` scripts or crates that fetch at build time.
- `unsafe` is limited to the GEMM calls in `kernel.rs`. Slice bounds are fixed by the caller, so any change there needs matching shape checks.
- Do not commit `.dev/`, `docs/`, `src/rust/target/` or `src/rust/vendor/`.
