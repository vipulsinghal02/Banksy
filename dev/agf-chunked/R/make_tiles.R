# Build a large synthetic dataset for TIME and MEMORY tests by tiling copies
# of a real one side by side in space. Each copy redraws every count as
# Poisson(original count), so copies are realistic noisy replicates rather
# than exact duplicates, then is normalised with the same genes.
#
# Do NOT use tiled data for accuracy or clustering comparisons: the copies
# share one covariance (the PCs are the original ones), and near-duplicate
# cells become each other's nearest neighbours in PC space. See PLAN.md.
#
#   Rscript dev/agf-chunked/R/make_tiles.R data=rds:/data/xenium.rds \
#       copies=20 out=/data/tiles20 backend=bpcells
#
# backend=bpcells writes a BPCells matrix directory (no 2^31 nonzero limit;
# needs the BPCells package) plus <out>.rds, a SpatialExperiment whose "data"
# assay points at it. backend=dgc writes a single in-memory dgCMatrix RDS.

source("dev/agf-chunked/R/common.R")
source("dev/agf-chunked/R/load_data.R")

a <- parse_args(list(data = "dlpfc:151673", n_hvg = 2000, copies = 4,
                     out = "dev/agf-chunked/results/tiles", seed = 1,
                     backend = "dgc"))

se <- load_dataset(a$data, a$n_hvg)
stopifnot("source needs a counts assay" = "counts" %in% assayNames(se))
cnt <- as(assay(se, "counts"), "dgCMatrix")
if (is.null(colnames(cnt))) colnames(cnt) <- paste0("cell", seq_len(ncol(cnt)))
xy <- spatialCoords(se)[, 1:2]
w <- diff(range(xy[, 1])) * 1.05
h <- diff(range(xy[, 2])) * 1.05
n_col <- ceiling(sqrt(a$copies))
set.seed(a$seed)

normalise <- function(m) {
    lib <- Matrix::colSums(m)
    lib[lib == 0] <- 1
    x <- as(m %*% Matrix::Diagonal(x = 1e4 / lib), "dgCMatrix")
    x@x <- log1p(x@x)
    dimnames(x) <- dimnames(m)
    x
}

one_copy <- function(i) {
    m <- cnt
    m@x <- as.numeric(rpois(length(m@x), m@x))
    m <- Matrix::drop0(m)
    colnames(m) <- paste0(colnames(cnt), "_t", i)
    shift <- c(((i - 1) %% n_col) * w, ((i - 1) %/% n_col) * h)
    list(x = normalise(m), xy = sweep(xy, 2, shift, `+`))
}

if (a$backend == "bpcells") {
    stopifnot(requireNamespace("BPCells", quietly = TRUE))
    dir.create(a$out, recursive = TRUE, showWarnings = FALSE)
    parts <- vector("list", a$copies)
    coords <- vector("list", a$copies)
    for (i in seq_len(a$copies)) {
        cp <- one_copy(i)
        parts[[i]] <- BPCells::write_matrix_dir(
            cp$x, file.path(a$out, sprintf("copy%05d", i)), overwrite = TRUE)
        coords[[i]] <- cp$xy
        if (i %% 10 == 0) log_msg(i, " / ", a$copies, " copies")
    }
    mat <- do.call(cbind, parts)
    mat <- BPCells::write_matrix_dir(mat, file.path(a$out, "all"),
                                     overwrite = TRUE)
    unlink(file.path(a$out, sprintf("copy%05d", seq_len(a$copies))),
           recursive = TRUE)
} else {
    parts <- lapply(seq_len(a$copies), one_copy)
    nnz <- sum(vapply(parts, function(p) length(p$x@x), numeric(1)))
    if (nnz > .Machine$integer.max)
        stop(nnz, " nonzeros exceed the dgCMatrix limit; use backend=bpcells")
    mat <- do.call(cbind, lapply(parts, `[[`, "x"))
    coords <- lapply(parts, `[[`, "xy")
}
xy_all <- do.call(rbind, coords)
rownames(xy_all) <- colnames(mat)
out_se <- SpatialExperiment(
    assays = list(data = mat), spatialCoords = xy_all,
    colData = S4Vectors::DataFrame(
        sample = rep(seq_len(a$copies), each = ncol(cnt)),
        row.names = colnames(mat)))
saveRDS(out_se, paste0(a$out, ".rds"))
log_msg("wrote ", ncol(out_se), " cells x ", nrow(out_se), " genes to ",
        a$out, ".rds")
