# Test funcs. in agf_chunked.R

library(SummarizedExperiment)
library(SingleCellExperiment)
library(SpatialExperiment)

set.seed(2024)
spe <- simulateDataset(n_cells = 1500, n_genes = 60, n_rings = 6)

npcs <- 10L
lambda <- 0.3
k_geom <- c(15L, 30L)
pca_name <- sprintf("PCA_M1_lam%s", lambda)

# Sine of the largest principal angle between the leading-k embeddings: 0
# means the same subspace. Compare subspaces, since PC signs are arbitrary.
sin_theta <- function(e1, e2, k) {
    q1 <- qr.Q(qr(e1[, seq_len(k), drop = FALSE]))
    q2 <- qr.Q(qr(e2[, seq_len(k), drop = FALSE]))
    sqrt(max(0, 1 - min(svd(crossprod(q1, q2))$d)^2))
}

dense_pca <- function(se, ...) {
    ref <- computeBanksy(se, assay_name = "counts", compute_agf = TRUE,
                         k_geom = k_geom, verbose = FALSE)
    reducedDim(suppressMessages(runBanksyPCA(
        ref, use_agf = TRUE, lambda = lambda, npcs = npcs, lazy = FALSE,
        seed = 1000, ...)), pca_name)
}

agf_pca <- function(se, ...) {
    reducedDim(suppressMessages(runBanksyPCA(
        se, assay_name = "counts", use_agf = TRUE, lambda = lambda,
        npcs = npcs, k_geom = k_geom, lazy = TRUE, verbose = FALSE, ...)),
        pca_name)
}

# Two copies of spe far apart in space, the second with doubled counts so
# per-group scaling matters. Built directly: cbind() on SpatialExperiment
# does not dispatch from the package test environment.
two_samples <- function() {
    xy <- spatialCoords(spe)
    xy_b <- xy
    xy_b[, 1] <- xy_b[, 1] + 100
    cnt <- counts(spe)
    cnt_b <- cnt * 2
    colnames(cnt_b) <- paste0(colnames(cnt), "_b")
    rownames(xy_b) <- colnames(cnt_b)
    SpatialExperiment(
        assays = list(counts = cbind(cnt, cnt_b)),
        spatialCoords = rbind(xy, xy_b),
        colData = S4Vectors::DataFrame(
            sample = rep(c("a", "b"), each = ncol(spe))))
}
both <- two_samples()

test_that("chunks rebuild the BANKSY matrix exactly", {
    ref <- computeBanksy(spe, assay_name = "counts", compute_agf = TRUE,
                         k_geom = k_geom, verbose = FALSE)
    z_ref <- unname(as.matrix(getBanksyMatrix(
        ref, M = 1, lambda = lambda, scale = TRUE, verbose = FALSE)))

    locs <- spatialCoords(spe)
    knn <- lapply(k_geom, function(k) computeNeighbors(locs, k_geom = k))
    ctrl <- .agfControl(list(chunk_size = 137L), npcs, NULL)
    prep <- .agfPrepare(counts(spe), knn[[1]], knn[[2]], locs, NULL,
                        scale = TRUE, split_scale = FALSE, ctrl = ctrl,
                        npcs = npcs, seed = NULL, verbose = FALSE)
    expect_gt(length(prep$chunks), 1L)
    expect_equal(.agfFullMatrix(prep, lambda), z_ref, tolerance = 1e-10)
})

test_that("gram matches the dense path", {
    e_dense <- dense_pca(spe)
    e_gram <- agf_pca(spe, agf_control = list(method = "gram",
                                              chunk_size = 400L))
    expect_equal(dim(e_gram), dim(e_dense))
    expect_lt(sin_theta(e_dense, e_gram, npcs - 2L), 1e-4)
    # Singular values match: columns of embeddings are V * d
    expect_equal(sqrt(colSums(e_gram^2)), sqrt(colSums(e_dense^2)),
                 tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("gram matches the dense path without scaling", {
    e_dense <- dense_pca(spe, scale = FALSE)
    e_gram <- agf_pca(spe, scale = FALSE, agf_control = list(method = "gram"))
    expect_lt(sin_theta(e_dense, e_gram, npcs - 2L), 1e-4)
})

test_that("gram matches the dense path with per-group scaling", {
    e_dense <- dense_pca(both, group = "sample")
    e_gram <- agf_pca(both, group = "sample",
                      agf_control = list(method = "gram"))
    expect_lt(sin_theta(e_dense, e_gram, npcs - 2L), 1e-4)
})

test_that("rsvd recovers the leading PCs", {
    e_gram <- agf_pca(spe, agf_control = list(method = "gram"))
    for (grouped in c(FALSE, TRUE)) {
        se <- if (grouped) both else spe
        grp <- if (grouped) "sample" else NULL
        e_ref <- if (grouped) agf_pca(se, group = grp,
                                      agf_control = list(method = "gram"))
                 else e_gram
        e_rsvd <- agf_pca(se, group = grp, agf_control = list(n_iter = 8L))
        expect_lt(sin_theta(e_ref, e_rsvd, 4L), 1e-3)
    }
})

test_that("results do not depend on chunking or worker count", {
    ctrl <- list(n_iter = 2L, chunk_size = 200L)
    e1 <- agf_pca(spe, agf_control = ctrl)
    e2 <- agf_pca(spe, agf_control = modifyList(ctrl,
                                                list(chunk_size = 1500L)))
    expect_equal(e1, e2, tolerance = 1e-8)
    skip_on_os("windows")
    e3 <- agf_pca(spe, agf_control = modifyList(ctrl, list(num_cores = 2L)))
    expect_equal(e1, e3, tolerance = 1e-8)
})

test_that("rsvd stops early with tol and reports diagnostics", {
    out <- suppressMessages(runBanksyPCA(
        spe, assay_name = "counts", use_agf = TRUE, lambda = lambda,
        npcs = npcs, k_geom = k_geom, lazy = TRUE, verbose = FALSE,
        agf_control = list(n_iter = 20L, tol = 0.5)))
    d <- metadata(out)$BANKSY_agf[[sprintf("lam%s", lambda)]]
    expect_equal(d$n_iter, 2L)
    expect_equal(d$passes, d$n_iter + 2L)
    expect_true(d$captured > 0 && d$captured <= 1)
})

test_that("lazy path rejects unsupported M and settings", {
    expect_error(agf_pca(spe, M = 2), "single M of 0 or 1")
    expect_error(runBanksyPCA(spe, assay_name = "counts",
                              use_agf = c(FALSE, TRUE), lazy = TRUE,
                              verbose = FALSE), "single M")
    expect_error(runBanksyPCA(spe, assay_name = "counts", use_agf = TRUE,
                              lazy = TRUE, k_geom = c(10, 15, 20),
                              verbose = FALSE), "k_geom")
    expect_error(agf_pca(spe, agf_control = list(nope = 1)), "unknown")
})

test_that("planBanksyAGF recommends a configuration that fits", {
    p <- planBanksyAGF(1e6, 2000, ram_gb = 64, cores = 8, verbose = FALSE)
    expect_true(p$num_cores >= 1L && p$num_cores <= 8L)
    expect_lt(p$peak_gb, 64)
    expect_warning(planBanksyAGF(1e7, 5000, ram_gb = 512, cores = 8,
                                 verbose = FALSE), "2\\^31")
})
