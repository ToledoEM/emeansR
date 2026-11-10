## Submission

First submission of emeansR 0.1.0.

## Test environments

* local macOS 27.2 (aarch64), R 4.6.1, rustc 1.99.1
* Docker rocker/r-ver 4.5.1 (Ubuntu), rustc 1.71.0 (minimum version): installs and runs

## R CMD check results

0 errors | 0 warnings | 1 note

* New submission.

## Rust

The compute core is written in Rust and linked through extendr.

* All Rust dependencies are vendored in `src/rust/vendor.tar.xz` and built with
  `cargo build --offline --locked -j 2`, so installation needs no network
  access. `CARGO_HOME` points to a temporary folder inside `src/`, never `~/.cargo`.
* `configure` reports the versions of `rustc` and `cargo` used, and stops with
  a message if `rustc` is older than 1.71 (`SystemRequirements`).
* Authors and licenses of the vendored crates are listed in `inst/AUTHORS`
  and acknowledged in `Authors@R` and the `Copyright` field.
* Examples, tests and the vignette use at most two threads.
