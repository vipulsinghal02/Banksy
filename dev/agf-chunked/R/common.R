# Shared helpers for the chunked-AGF experiments. Source from the other
# scripts: source("dev/agf-chunked/R/common.R")

suppressPackageStartupMessages({
    library(Banksy)
    library(SpatialExperiment)
    library(SingleCellExperiment)
    library(SummarizedExperiment)
    library(Matrix)
})

# Command-line arguments as key=value, merged into defaults and converted to
# the default's type. Vectors: comma-separated (n_iter=0,1,2,4).
parse_args <- function(defaults) {
    for (a in commandArgs(trailingOnly = TRUE)) {
        kv <- strsplit(a, "=", fixed = TRUE)[[1]]
        if (length(kv) != 2L || !kv[1] %in% names(defaults))
            stop("bad argument '", a, "'; known: ",
                 paste(names(defaults), collapse = ", "))
        v <- strsplit(kv[2], ",", fixed = TRUE)[[1]]
        d <- defaults[[kv[1]]]
        defaults[[kv[1]]] <- if (is.null(d) || is.character(d)) v
                             else if (is.logical(d)) as.logical(v)
                             else as.numeric(v)
    }
    defaults
}

# --- accuracy metrics --------------------------------------------------------

# Sine of the largest principal angle between the leading-k column spaces:
# 0 = same subspace, 1 = some direction entirely missed
sin_theta <- function(e1, e2, k) {
    q1 <- qr.Q(qr(e1[, seq_len(k), drop = FALSE]))
    q2 <- qr.Q(qr(e2[, seq_len(k), drop = FALSE]))
    sqrt(max(0, 1 - min(svd(crossprod(q1, q2))$d)^2))
}

knn_index <- function(emb, k) dbscan::kNN(emb, k = k)$id

# Mean over cells of |N_a(i) & N_b(i)| / |N_a(i) | N_b(i)|
knn_jaccard <- function(ia, ib) {
    k <- ncol(ia)
    inter <- vapply(seq_len(nrow(ia)), function(i)
        length(intersect(ia[i, ], ib[i, ])), numeric(1))
    mean(inter / (2 * k - inter))
}

# Leiden on an unweighted, symmetrised kNN graph. Deliberately independent of
# clusterBanksy (whose igraph::as_undirected import breaks on igraph < 2.1),
# and the same function is used for every embedding, so comparisons are fair.
leiden_labels <- function(emb, k = 50, resolution = 0.5, seed = 1,
                          knn = NULL) {
    if (is.null(knn)) knn <- knn_index(emb, k)
    # Sort each neighbour list: Leiden depends on edge order, so the same
    # kNN sets must give the same edge list for "same graph, same result"
    knn <- t(apply(knn, 1, sort))
    k <- ncol(knn)
    edges <- cbind(rep(seq_len(nrow(knn)), k), as.vector(knn))
    g <- igraph::graph_from_edgelist(edges, directed = FALSE)
    g <- igraph::simplify(g)
    set.seed(seed)
    igraph::membership(igraph::cluster_leiden(
        g, objective_function = "modularity",
        resolution_parameter = resolution, n_iterations = -1))
}

ari <- function(a, b) {
    ok <- !is.na(a) & !is.na(b)
    aricode::ARI(a[ok], b[ok])
}

# --- running the methods -------------------------------------------------------

# Chunked AGF PCA through the public API. Returns the embedding, the wall time
# and the solver diagnostics.
run_agf <- function(se, lambda, npcs, k_geom, assay_name, group = NULL,
                    seed = NULL, num_cores = 1L, ...) {
    t0 <- proc.time()[["elapsed"]]
    out <- runBanksyPCA(se, assay_name = assay_name, use_agf = TRUE,
                        lambda = lambda, npcs = npcs, k_geom = k_geom,
                        lazy = TRUE, group = group, seed = seed,
                        verbose = FALSE, agf_control = list(
                            num_cores = num_cores, ...))
    nm <- sprintf("PCA_M1_lam%s", lambda)
    list(emb = reducedDim(out, nm),
         time = proc.time()[["elapsed"]] - t0,
         diag = metadata(out)$BANKSY_agf[[sprintf("lam%s", lambda)]])
}

# Standard path (materialised BANKSY matrix + irlba). Only for data whose
# dense matrix fits: about 5 copies of 3 * n_genes * n_cells doubles.
run_standard <- function(se, lambda, npcs, k_geom, assay_name, M = 1,
                         group = NULL, seed = 1) {
    t0 <- proc.time()[["elapsed"]]
    ref <- computeBanksy(se, assay_name = assay_name, compute_agf = M == 1,
                         k_geom = if (M == 1) k_geom else k_geom[1],
                         verbose = FALSE)
    out <- suppressMessages(runBanksyPCA(
        ref, use_agf = M == 1, lambda = lambda, npcs = npcs, lazy = FALSE,
        group = group, seed = seed))
    list(emb = reducedDim(out, sprintf("PCA_M%d_lam%s", M, lambda)),
         time = proc.time()[["elapsed"]] - t0)
}

standard_fits <- function(se, mem_cap_gb) {
    5 * 3 * as.double(nrow(se)) * ncol(se) * 8 / 1e9 < mem_cap_gb
}

# --- memory ---------------------------------------------------------------------

# Peak resident memory of this R process in MB (Linux: VmHWM; includes C/C++
# allocations). Forked workers are separate processes: measure the whole tree
# with dev/agf-chunked/memwatch.sh instead.
peak_rss_mb <- function() {
    f <- "/proc/self/status"
    if (!file.exists(f)) return(NA_real_)
    line <- grep("^VmHWM", readLines(f), value = TRUE)
    as.numeric(gsub("\\D", "", line)) / 1024
}

# Append a one-row data.frame to a CSV, writing the header on first use
append_csv <- function(row, path) {
    dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
    write.table(row, path, sep = ",", row.names = FALSE,
                col.names = !file.exists(path), append = file.exists(path))
}

log_msg <- function(...) message(format(Sys.time(), "%H:%M:%S "), ...)
