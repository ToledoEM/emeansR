#!/bin/sh
# Locate cargo and rustc, report their versions, check the minimum Rust
# version, then write src/Makevars (or src/Makevars.win) from its template.
# Usage: sh tools/config.sh [win]

MSRV_MAJOR=1
MSRV_MINOR=71

# PATH first, then personal installs (rustup), then Homebrew prefixes: GUI
# sessions such as RStudio on macOS start with a minimal PATH that omits
# /opt/homebrew/bin and /usr/local/bin.
find_tool() {
  if command -v "$1" >/dev/null 2>&1; then
    command -v "$1"
    return
  fi
  for dir in "${CARGO_HOME:-$HOME/.cargo}/bin" "$HOME/.cargo/bin" /opt/homebrew/bin /usr/local/bin; do
    for exe in "$1" "$1.exe"; do
      if [ -x "$dir/$exe" ]; then
        echo "$dir/$exe"
        return
      fi
    done
  done
}

CARGO=$(find_tool cargo)
RUSTC=$(find_tool rustc)

if [ -z "$CARGO" ] || [ -z "$RUSTC" ]; then
  cat <<MSG
------------------------- RUST TOOLCHAIN NOT FOUND -------------------------
emeansR needs 'cargo' and 'rustc' (Rust >= $MSRV_MAJOR.$MSRV_MINOR) to compile.
Install them with your system package manager, for example:
  Debian/Ubuntu:  sudo apt-get install cargo rustc
  Fedora/RHEL:    sudo dnf install cargo
  macOS:          brew install rust
or from https://rustup.rs/ , then reinstall emeansR.
On Windows also run: rustup target add x86_64-pc-windows-gnu
(aarch64-pc-windows-gnullvm for R on Windows ARM)
-----------------------------------------------------------------------------
MSG
  exit 1
fi

echo "using rustc: $("$RUSTC" --version)"
echo "using cargo: $("$CARGO" --version)"

VERSION=$("$RUSTC" --version | sed -n 's/^rustc \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p')
MAJOR=${VERSION% *}
MINOR=${VERSION#* }
if [ -z "$VERSION" ] || [ "$MAJOR" -lt "$MSRV_MAJOR" ] || \
   { [ "$MAJOR" -eq "$MSRV_MAJOR" ] && [ "$MINOR" -lt "$MSRV_MINOR" ]; }; then
  echo "ERROR: emeansR needs rustc >= $MSRV_MAJOR.$MSRV_MINOR; found $("$RUSTC" --version)."
  echo "Update Rust (e.g. 'rustup update') or install a newer system rustc."
  exit 1
fi

RUST_BIN_DIR=$(dirname "$RUSTC")
if [ "$1" = "win" ]; then
  IN=src/Makevars.win.in; OUT=src/Makevars.win
else
  IN=src/Makevars.in; OUT=src/Makevars
fi
sed -e "s|@CARGO@|$CARGO|" -e "s|@RUST_BIN_DIR@|$RUST_BIN_DIR|" "$IN" > "$OUT"
echo "wrote $OUT"
