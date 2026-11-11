## Resubmission

This is a resubmission of emeansR 0.1.0, a new package. The incoming pretest
of 2026-11-11 failed to install on r-devel-windows-x86_64. I fixed two problems
in `src/Makevars.win.in`:

* The Rust linker was set from `$(CC)`, which on Windows ends in a space
  (`gcc `), so cargo could not find it. It now uses `$(firstword $(CC))`.
* rustc links `x86_64-pc-windows-gnu` with `-lgcc_eh`, which Rtools45 does not
  provide. The build now adds an empty `libgcc_eh.a` to `LIBRARY_PATH`, inside
  a temporary folder in `src/` that is removed after linking.

## Test environments

* local macOS 27.2 (aarch64), R 4.6.1, rustc 1.99.1
* Windows 11 x64, R 4.6.2, Rtools45 (gcc 14.3.0), rustc 1.92.0
* Docker rocker/r-ver 4.5.1 (Ubuntu), rustc 1.71.0 (minimum version): installs and runs

## R CMD check results

0 errors | 0 warnings | 1 note

* New submission.
* Possibly misspelled words in DESCRIPTION: "Bessa", "et", "al". These are the
  author citation of the method (Bessa et al. 2026), not misspellings.

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
