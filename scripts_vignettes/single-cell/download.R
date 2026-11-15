# Download the GEO GSE233163 files used by the single-cell article (about 3 GB in total).
# Usage: Rscript download.R [out_dir]   (from any directory, before single-cell-run.R)
# Default out_dir is data/ next to this script. A file that already exists is not downloaded again.
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
here <- if (length(script)) dirname(normalizePath(script)) else getwd()
args <- commandArgs(TRUE)
out_dir <- if (length(args) >= 1) args[1] else file.path(here, "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

base <- "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE233nnn/GSE233163/suppl/"
files <- c("GSE233163_seurat_object.rds.gz", "GSE233163_cells_metadata.csv.gz")
options(timeout = max(3600, getOption("timeout")))
for (f in files) {
  dest <- file.path(out_dir, f)
  if (file.exists(dest)) {
    message("exists: ", dest)
    next
  }
  message("download: ", f)
  utils::download.file(paste0(base, f), dest, mode = "wb", quiet = TRUE)
}
