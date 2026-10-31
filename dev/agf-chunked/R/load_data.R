# Dataset loaders. Each returns a SpatialExperiment with
#   assay "data": normalised log expression (genes x cells), HVGs only
#   assay "counts": raw counts of the same genes (for make_tiles.R)
#   colData$truth: ground-truth labels when the dataset has them (else NULL)
#   colData$sample: sample of origin
# Source after common.R.

# Library-size normalisation to 1e4 counts, log1p, then the n_hvg genes with
# the highest variance of the log values. Deliberately simple, and identical
# for every method compared, which is all the comparisons need.
normalise_hvg <- function(counts, n_hvg = 2000) {
    counts <- as(counts, "dgCMatrix")
    lib <- Matrix::colSums(counts)
    keep_cells <- lib > 0
    counts <- counts[, keep_cells, drop = FALSE]
    x <- counts %*% Matrix::Diagonal(x = 1e4 / lib[keep_cells])
    x@x <- log1p(x@x)
    x <- as(x, "dgCMatrix")
    dimnames(x) <- dimnames(counts)  # %*% Diagonal() drops colnames
    m <- Matrix::rowMeans(x)
    v <- Matrix::rowMeans(x^2) - m^2
    hv <- order(v, decreasing = TRUE)[seq_len(min(n_hvg, nrow(x)))]
    list(x = x[sort(hv), , drop = FALSE],
         counts = counts[sort(hv), , drop = FALSE], keep_cells = keep_cells)
}

# 10x Visium human DLPFC (Maynard et al. 2021) through spatialLIBD; manual
# layer annotations as truth. 12 samples, ~3.5-4.8k spots each. The first
# call downloads ~1-2 GB into the ExperimentHub cache.
load_dlpfc <- function(samples = "151673", n_hvg = 2000) {
    suppressPackageStartupMessages({
        library(spatialLIBD)
        library(ExperimentHub)
    })
    spe <- spatialLIBD::fetch_data(type = "spe",
                                   eh = ExperimentHub::ExperimentHub())
    spe <- spe[, spe$sample_id %in% samples & spe$in_tissue]
    nh <- normalise_hvg(counts(spe), n_hvg)
    spe <- spe[, nh$keep_cells]
    xy <- spatialCoords(spe)
    # Visium coordinates per sample overlap: shift samples apart so global
    # neighbours never cross samples (the lazy path also uses per-group kNN
    # when group = "sample")
    s <- match(spe$sample_id, samples)
    xy[, 1] <- xy[, 1] + (s - 1) * (diff(range(xy[, 1])) * 1.1)
    out <- SpatialExperiment(
        assays = list(data = nh$x, counts = nh$counts),
        spatialCoords = xy,
        colData = S4Vectors::DataFrame(
            truth = as.character(spe$layer_guess_reordered),
            sample = spe$sample_id))
    rownames(out) <- rownames(spe)[match(rownames(nh$x), rownames(spe))]
    out
}

# Any SpatialExperiment saved as RDS with a "data" assay (and optionally
# colData truth / sample), e.g. produced by a conversion script on EC2
load_rds <- function(path) readRDS(path)

# Dispatch on a dataset spec string:
#   dlpfc:151673            one DLPFC sample
#   dlpfc:151673+151674     several (shifted apart in space)
#   rds:/path/to/file.rds   a prepared SpatialExperiment
load_dataset <- function(spec, n_hvg = 2000) {
    kind <- sub(":.*", "", spec)
    arg <- sub("^[^:]*:", "", spec)
    switch(kind,
           dlpfc = load_dlpfc(strsplit(arg, "+", fixed = TRUE)[[1]], n_hvg),
           rds = load_rds(arg),
           stop("unknown dataset spec: ", spec))
}
