
# Chunked PCA of the BANKSY matrix with the AGF (M = 1)
#
# The lazy operator in lazy.R relies on H0 = X W being linear in X, so its
# products factor as X (W v). The AGF H1 = |X C| is not linear, so no such
# factoring exists. Instead, every pass over the data rebuilds the BANKSY
# matrix Z one chunk of cells at a time: a chunk's columns depend only on the
# expression of those cells and of their spatial neighbours. Z is never
# stored. Two solvers run on top of the chunks:
#
#   rsvd  randomized subspace iteration (Halko, Martinsson & Tropp 2011).
#         n_iter + 3 passes (n_iter + 2 without groups), F x ell working
#         memory. Approximate: leading PCs converge fast, PCs in the flat
#         tail of the spectrum do not.
#   gram  feature covariance Z Z^T accumulated over chunks, then eigen().
#         3 passes, F x F working memory per worker. Exact.
#
# F = 3 * n_genes is the number of BANKSY features. Peak memory is the sparse
# inputs (X and three n x n neighbour matrices) plus, per worker, a few dense
# F x chunk_size buffers.
#
# Matches the standard path computeBanksy(M = 1) + getBanksyMatrix(scale =
# TRUE) + runBanksyPCA(lazy = FALSE): H0 = |X W0|, H1 = |X W1 - (X U) s|
# (centred), each row z-scored (sample sd, constant rows set to 0), blocks
# weighted by sqrt(c(1 - lambda, 2 lambda / 3, lambda / 3)).

# Defaults for agf_control in runBanksyPCA
.agfControlDefaults <- function(npcs) {
    list(method = "rsvd",       # "rsvd" or "gram"
         n_iter = 4L,           # power passes (max passes if tol is set)
         oversample = 10L,      # ell = npcs + oversample
         tol = NULL,            # stop early when captured-variance gain < tol
         chunk_size = NULL,     # cells per chunk; NULL derives from memory
         mem_per_worker_gb = 2, # target memory per worker for chunk buffers
         num_cores = 1L,        # parallel workers over chunks (fork, unix)
         center = TRUE)         # centre H1 in local neighbourhoods
}

.agfControl <- function(agf_control, npcs, num_cores) {
    ctrl <- .agfControlDefaults(npcs)
    if (!is.null(num_cores)) ctrl$num_cores <- as.integer(num_cores)
    unknown <- setdiff(names(agf_control), names(ctrl))
    if (length(unknown) > 0L)
        stop('unknown agf_control entries: ', paste(unknown, collapse = ', '))
    ctrl[names(agf_control)] <- agf_control
    ctrl$method <- match.arg(ctrl$method, c("rsvd", "gram"))
    if (.Platform$OS.type != "unix" && ctrl$num_cores > 1L) {
        warning('agf_control$num_cores > 1 needs fork (unix); using 1')
        ctrl$num_cores <- 1L
    }
    ctrl
}

# lazy = TRUE, M = 1 branch of runBanksyPCA
.runBanksyPCA_agf <- function(se, lambda, npcs, assay_name, scale, group,
                              seed, coord_names, k_geom, spatial_mode,
                              split_scale, num_cores, agf_control, verbose,
                              ...) {
    if (is.null(assay_name)) {
        assay_name <- SummarizedExperiment::assayNames(se)[1]
        if (verbose) message('Using assay: ', assay_name)
    }
    # Not coerced: chunks read X[, cells] from any backend that supports
    # column subsetting (dgCMatrix, base matrix, BPCells, DelayedArray)
    data_own <- assay(se, assay_name)
    locs <- getLocs(se, coord_names)
    if (length(k_geom) == 1L) k_geom <- rep(k_geom, 2L)
    if (length(k_geom) != 2L)
        stop('k_geom must have length 1 or 2 when use_agf = TRUE')
    ctrl <- .agfControl(agf_control, npcs, num_cores)

    if (!is.null(group)) {
        groups_vec <- colData(se)[, group]
        ugroups <- unique(groups_vec)
        group_idx <- lapply(ugroups, function(g) which(g == groups_vec))
        .checkGroupIdx(group_idx)
    } else {
        group_idx <- NULL
    }

    if (verbose) message('Computing neighbors (k_geom = ',
                         paste(k_geom, collapse = ', '), ')')
    knn <- lapply(k_geom, function(kg)
        .agfNeighbors(locs, group_idx, kg, spatial_mode, ...))

    prep <- .agfPrepare(
        data_own, knn[[1]], knn[[2]], locs, group_idx,
        scale = scale, split_scale = !is.null(group) && split_scale,
        ctrl = ctrl, npcs = npcs, seed = seed, verbose = verbose)
    rm(knn)

    for (lam in lambda) {
        if (verbose && length(lambda) > 1L)
            message('--- lambda = ', lam, ' ---')
        res <- .agfSolve(prep, lam, npcs, ctrl, verbose)
        pca_x <- res$embeddings
        colnames(pca_x) <- paste0("PC", seq_len(npcs))
        attr(pca_x, "percentVar") <- 100 * res$d^2 / sum(res$d^2)
        reducedDim(se, sprintf("PCA_M1_lam%s", lam)) <- pca_x
        metadata(se)$BANKSY_agf[[sprintf("lam%s", lam)]] <- res$diagnostics
    }

    metadata(se)$BANKSY_params$lambda <- lambda
    metadata(se)$BANKSY_params$npcs <- npcs
    metadata(se)$BANKSY_params$k_geom <- k_geom
    metadata(se)$BANKSY_params$agf_control <- ctrl
    if (verbose) message('Done.')
    se
}

# kNN over all cells, or per group with indices mapped back to the full
# object. Per-group neighbours equal global ones when samples do not overlap
# in space, and avoid pairing cells across samples when they do.
.agfNeighbors <- function(locs, group_idx, k_geom, spatial_mode, ...) {
    if (is.null(group_idx))
        return(computeNeighbors(locs, spatial_mode = spatial_mode,
                                k_geom = k_geom, verbose = FALSE, ...))
    parts <- lapply(group_idx, function(cid) {
        d <- computeNeighbors(locs[cid, , drop = FALSE],
                              spatial_mode = spatial_mode, k_geom = k_geom,
                              verbose = FALSE, ...)
        # rNN_gauss writes 0 as a sentinel index for isolated cells; keep it
        from <- d[["from"]]; to <- d[["to"]]
        from[from > 0] <- cid[from[from > 0]]
        to[to > 0] <- cid[to[to > 0]]
        data.table(from = from, to = to, weight = d[["weight"]],
                   phi = as.numeric(d[["phi"]]))
    })
    rbindlist(parts)
}

# Neighbour matrices, columns indexed by the receiving cell (as in
# computeHarmonics): H0[, j] = |X W0[, j]|, H1[, j] = |X Cre[, j] + i X Cim[, j]|.
# Centring folds into the coefficients: c_ij = w_ij e^{i phi_ij} - s_j / k_j,
# with s_j the sum of cell j's complex weights and k_j its neighbour count.
.agfWeightMats <- function(knn0, knn1, n_cells, center = TRUE) {
    filt <- function(d) {
        from <- d[["from"]]; to <- d[["to"]]
        ok <- to > 0L & from > 0L
        list(from = from[ok], to = to[ok], w = d[["weight"]][ok],
             phi = as.numeric(d[["phi"]])[ok])
    }
    k0 <- filt(knn0)
    W0 <- sparseMatrix(i = k0$to, j = k0$from, x = k0$w,
                       dims = c(n_cells, n_cells))
    k1 <- filt(knn1)
    re <- k1$w * cos(k1$phi)
    im <- k1$w * sin(k1$phi)
    if (center) {
        k_per <- tabulate(k1$from, nbins = n_cells)
        k_per[k_per == 0] <- 1L
        s_re <- as.numeric(rowsum(re, k1$from, reorder = TRUE))
        s_im <- as.numeric(rowsum(im, k1$from, reorder = TRUE))
        # rowsum drops cells with no neighbours; expand to all cells
        has <- sort(unique(k1$from))
        s_re_all <- numeric(n_cells); s_re_all[has] <- s_re
        s_im_all <- numeric(n_cells); s_im_all[has] <- s_im
        re <- re - s_re_all[k1$from] / k_per[k1$from]
        im <- im - s_im_all[k1$from] / k_per[k1$from]
    }
    list(W0 = W0,
         Cre = sparseMatrix(i = k1$to, j = k1$from, x = re,
                            dims = c(n_cells, n_cells)),
         Cim = sparseMatrix(i = k1$to, j = k1$from, x = im,
                            dims = c(n_cells, n_cells)))
}

# Z-order (Morton) key on the first two coordinates. Sorting by it makes each
# chunk a compact patch of tissue, so its neighbours mostly lie inside it and
# each pass reads close to n cells of X rather than n * k.
.mortonKey <- function(locs, bits = 15L) {
    xy <- as.matrix(locs[, seq_len(min(2L, ncol(locs))), drop = FALSE])
    key <- numeric(nrow(xy))
    for (d in seq_len(ncol(xy))) {
        v <- xy[, d]
        r <- range(v)
        s <- if (r[2] > r[1]) (2^bits - 1) / (r[2] - r[1]) else 0
        qd <- as.integer(floor((v - r[1]) * s))
        for (b in 0:(bits - 1L))
            key <- key + bitwAnd(bitwShiftR(qd, b), 1L) *
                2^(ncol(xy) * b + d - 1L)
    }
    key
}

.agfChunks <- function(locs, chunk_size, cell_group = NULL) {
    n <- nrow(locs)
    key <- .mortonKey(locs)
    ord <- if (is.null(cell_group)) order(key) else order(cell_group, key)
    unname(split(ord, ceiling(seq_len(n) / chunk_size)))
}

# Rows the chunk solvers can multiply: dgCMatrix and base matrix pass through,
# other backends (BPCells, DelayedArray) are realized per chunk.
.agfAsChunk <- function(x) {
    if (is.matrix(x) || inherits(x, 'dgCMatrix')) x else as(x, 'dgCMatrix')
}

# Raw (unscaled) BANKSY columns for cells J: rbind(own, H0, H1), F x |J|.
.agfChunkRaw <- function(X, mats, J) {
    nj <- length(J)
    W <- cbind(mats$W0[, J, drop = FALSE], mats$Cre[, J, drop = FALSE],
               mats$Cim[, J, drop = FALSE])
    nb <- sort.int(unique.default(c(J, W@i + 1L)))
    W <- W[nb, , drop = FALSE]
    # Densify the neighbour slice: dense %*% sparse is several times faster
    # than sparse %*% sparse here, and the product is nearly dense anyway
    Xn <- as.matrix(.agfAsChunk(X[, nb, drop = FALSE]))
    P <- as.matrix(Xn %*% W)
    own <- Xn[, match(J, nb), drop = FALSE]
    re <- P[, nj + seq_len(nj), drop = FALSE]
    im <- P[, 2L * nj + seq_len(nj), drop = FALSE]
    out <- rbind(own, abs(P[, seq_len(nj), drop = FALSE]),
                 sqrt(re * re + im * im))
    dimnames(out) <- NULL
    out
}

# Z for cells J: d * (raw - mu), with per-group mu and d under split scaling.
# d already carries the lambda block weight, and is 0 for constant rows.
.agfChunkScaled <- function(R, J, sc) {
    if (ncol(sc$mu) == 1L) return(sc$d[, 1] * (R - sc$mu[, 1]))
    gj <- sc$cell_group[J]
    for (g in unique(gj)) {
        cols <- which(gj == g)
        R[, cols] <- sc$d[, g] * (R[, cols, drop = FALSE] - sc$mu[, g])
    }
    R
}

# Run f(J, t) over all chunks and fold the results with combine(acc, x)
# (combine(NULL, x) must return x). With num_cores > 1, chunks are dealt
# round-robin to forked workers, each folding its own share.
.agfMapChunks <- function(chunks, f, combine, num_cores = 1L) {
    run <- function(ts) {
        acc <- NULL
        for (t in ts) acc <- combine(acc, f(chunks[[t]], t))
        acc
    }
    if (num_cores <= 1L || length(chunks) == 1L)
        return(run(seq_along(chunks)))
    shares <- split(seq_along(chunks),
                    rep_len(seq_len(num_cores), length(chunks)))
    parts <- parallel::mclapply(shares, run, mc.cores = num_cores,
                                mc.preschedule = TRUE)
    # mclapply returns a try-error for a failed worker, and NULL for one the
    # OS killed (usually out of memory)
    for (p in parts) {
        if (is.null(p))
            stop('a chunk worker was killed, most likely out of memory: ',
                 'lower agf_control$chunk_size or num_cores')
        if (inherits(p, 'try-error'))
            stop('a chunk worker failed: ', as.character(p))
    }
    Reduce(combine, parts, NULL)
}

# Chan et al. pairwise merge of (n, mean, m2) plus running min/max per row
.agfStatsMerge <- function(a, b) {
    if (is.null(a)) return(b)
    if (is.null(b)) return(a)
    n <- a$n + b$n
    delta <- b$mean - a$mean
    list(n = n,
         mean = a$mean + delta * (b$n / n),
         m2 = a$m2 + b$m2 + delta * delta * (a$n * b$n / n),
         min = pmin(a$min, b$min), max = pmax(a$max, b$max))
}

.agfStatsChunk <- function(R) {
    mu <- rowMeans(R)
    list(n = ncol(R), mean = mu, m2 = rowSums((R - mu)^2),
         min = MatrixGenerics::rowMins(R), max = MatrixGenerics::rowMaxs(R))
}

# Starting block for the subspace iteration: Gaussian. With a seed, drawn
# from the caller's RNG after set.seed(seed). Without one, drawn from a fixed
# seed and the caller's RNG state is restored, so the result is reproducible
# without side effects (as in the lazy path).
.agfStartBlock <- function(F, ell, seed) {
    if (!is.null(seed)) {
        verbose.seed(seed)
        return(matrix(stats::rnorm(F * ell), F, ell))
    }
    genv <- globalenv()
    old <- if (exists('.Random.seed', envir = genv, inherits = FALSE))
        get('.Random.seed', envir = genv) else NULL
    on.exit(if (is.null(old)) rm('.Random.seed', envir = genv)
            else assign('.Random.seed', old, envir = genv))
    set.seed(1L)
    matrix(stats::rnorm(F * ell), F, ell)
}

# Everything that does not depend on lambda: neighbour matrices, chunks,
# feature statistics and (without groups) the raw start product.
.agfPrepare <- function(X, knn0, knn1, locs, group_idx, scale, split_scale,
                        ctrl, npcs, seed, verbose) {
    n_genes <- nrow(X)
    n_cells <- ncol(X)
    F <- 3L * n_genes
    ell <- as.integer(min(npcs + ctrl$oversample, F, n_cells))
    if (npcs >= min(F, n_cells))
        stop('npcs must be < min(3 * n_genes, n_cells)')

    cell_group <- NULL
    if (!is.null(group_idx)) {
        cell_group <- integer(n_cells)
        for (g in seq_along(group_idx)) cell_group[group_idx[[g]]] <- g
    }
    # Per-group statistics only when scaling per group; otherwise the whole
    # dataset is one group (also for scale = FALSE, where the standard path
    # centres globally inside irlba)
    stat_group <- if (scale && split_scale) cell_group else NULL
    n_stat_groups <- if (is.null(stat_group)) 1L else length(group_idx)

    chunk_size <- ctrl$chunk_size
    if (is.null(chunk_size)) {
        # About five dense F x chunk buffers live at once in a worker
        chunk_size <- floor(ctrl$mem_per_worker_gb * 1e9 / (5 * 8 * F))
        chunk_size <- max(500L, min(as.integer(chunk_size), 50000L))
    }
    chunk_size <- as.integer(min(chunk_size, n_cells))
    .agfCheckMemory(X, n_cells, n_genes, npcs, ctrl, chunk_size, knn0, knn1)
    chunks <- .agfChunks(locs, chunk_size, cell_group)
    if (verbose) message('BANKSY AGF: ', n_cells, ' cells, ', F,
                         ' features, ', length(chunks), ' chunks of <= ',
                         chunk_size, ' cells, ', ctrl$num_cores, ' worker(s)')

    mats <- .agfWeightMats(knn0, knn1, n_cells, center = ctrl$center)

    # Pass 1: statistics. Without per-group scaling, the same pass also
    # accumulates T = sum_J R_J (R_J^T P) for a fixed block P, from which the
    # start product Z Z^T (.) follows for every lambda without another pass
    merge_start <- ctrl$method == "rsvd" && n_stat_groups == 1L
    P <- if (merge_start) .agfStartBlock(F, ell, seed) else NULL
    t0 <- proc.time()[["elapsed"]]
    acc <- .agfMapChunks(chunks, function(J, t) {
        R <- .agfChunkRaw(X, mats, J)
        gj <- if (is.null(stat_group)) rep(1L, length(J)) else stat_group[J]
        st <- lapply(seq_len(n_stat_groups), function(g) {
            cols <- which(gj == g)
            if (length(cols) == 0L) NULL
            else .agfStatsChunk(R[, cols, drop = FALSE])
        })
        list(st = st, T = if (merge_start) R %*% crossprod(R, P))
    }, function(a, b) {
        if (is.null(a)) return(b)
        list(st = Map(.agfStatsMerge, a$st, b$st),
             T = if (merge_start) a$T + b$T)
    }, ctrl$num_cores)
    t_stats <- proc.time()[["elapsed"]] - t0
    if (verbose) message(sprintf('  pass 1 (statistics%s): %.1fs',
                                 if (merge_start) ' + start' else '', t_stats))

    st <- acc$st
    mu <- vapply(st, `[[`, numeric(F), 'mean')
    if (!is.matrix(mu)) mu <- matrix(mu, ncol = 1L)
    if (scale) {
        sdv <- vapply(st, function(s) sqrt(s$m2 / (s$n - 1)), numeric(F))
        if (!is.matrix(sdv)) sdv <- matrix(sdv, ncol = 1L)
        const <- vapply(st, function(s) s$max == s$min, logical(F))
        if (!is.matrix(const)) const <- matrix(const, ncol = 1L)
        inv_sd <- ifelse(const, 0, 1 / sdv)
    } else {
        inv_sd <- matrix(1, F, ncol(mu))
    }

    list(X = X, mats = mats, chunks = chunks, chunk_size = chunk_size,
         mu = mu, inv_sd = inv_sd, m2 = vapply(st, `[[`, numeric(F), 'm2'),
         cell_group = stat_group, n_genes = n_genes, n_cells = n_cells,
         F = F, ell = ell, seed = seed, scale = scale,
         T = acc$T, P = P, n_total = n_cells, t_stats = t_stats,
         gene_names = rownames(X), cell_names = colnames(X))
}

.agfScaling <- function(prep, lambda) {
    a <- rep(getLambdas(lambda, n_harmonics = 2), each = prep$n_genes)
    list(mu = prep$mu, d = a * prep$inv_sd, cell_group = prep$cell_group)
}

# Sum of the top-p eigenvalues of Q^T Z Z^T Q = Q^T Y: the variance the
# current subspace captures in its best p directions
.agfCaptured <- function(Q, Y, p) {
    M <- crossprod(Q, Y)
    ev <- eigen((M + t(M)) / 2, symmetric = TRUE, only.values = TRUE)$values
    sum(ev[seq_len(p)])
}

.agfOrth <- function(Y) qr.Q(qr(Y))

.agfSolve <- function(prep, lambda, npcs, ctrl, verbose) {
    sc <- .agfScaling(prep, lambda)
    X <- prep$X; mats <- prep$mats; chunks <- prep$chunks
    zchunk <- function(J) .agfChunkScaled(.agfChunkRaw(X, mats, J), J, sc)
    nc <- ctrl$num_cores
    # Total variance of Z, from the statistics (for diagnostics)
    total_ss <- sum(sc$d^2 * prep$m2)
    timings <- c(stats = prep$t_stats)

    if (ctrl$method == "gram") {
        t0 <- proc.time()[["elapsed"]]
        G <- .agfMapChunks(chunks, function(J, t) tcrossprod(zchunk(J)),
                           function(a, b) if (is.null(a)) b else a + b, nc)
        eg <- .agfTopEigen(G, npcs)
        rm(G)
        V <- eg$vectors
        d <- sqrt(pmax(eg$values, 0))
        timings["gram"] <- proc.time()[["elapsed"]] - t0
        if (verbose) message(sprintf('  pass 2 (covariance): %.1fs',
                                     timings[["gram"]]))
        t0 <- proc.time()[["elapsed"]]
        emb <- .agfEmbed(chunks, function(J) crossprod(zchunk(J), V),
                         prep$n_cells, npcs, nc)
        timings["embed"] <- proc.time()[["elapsed"]] - t0
        if (verbose) message(sprintf('  pass 3 (embeddings): %.1fs',
                                     timings[["embed"]]))
        diagnostics <- list(method = "gram", passes = 3L,
                            captured = sum(d^2) / total_ss,
                            timings = timings)
        return(.agfResult(prep, emb, V, d, diagnostics))
    }

    # rsvd. Start: Y = Z Z^T Q0. Merged into pass 1 when possible:
    # Z Z^T (D^-1 P) = D (T - n mu mu^T P) with D = diag(d), i.e. a start
    # block D^-1 P that is as random as P itself.
    t0 <- proc.time()[["elapsed"]]
    if (!is.null(prep$T)) {
        Y <- sc$d[, 1] * (prep$T - prep$n_total *
                              tcrossprod(prep$mu[, 1], crossprod(prep$P, prep$mu[, 1])))
        passes <- 1L
    } else {
        Q <- .agfOrth(.agfStartBlock(prep$F, prep$ell, prep$seed))
        Y <- .agfMapChunks(chunks, function(J, t) {
            Z <- zchunk(J)
            Z %*% crossprod(Z, Q)
        }, function(a, b) if (is.null(a)) b else a + b, nc)
        passes <- 2L
        timings["start"] <- proc.time()[["elapsed"]] - t0
        if (verbose) message(sprintf('  pass 2 (start): %.1fs',
                                     timings[["start"]]))
    }

    # Power passes, with the free captured-variance monitor
    captured <- numeric(0)
    gain <- numeric(0)
    it <- 0L
    while (it < ctrl$n_iter) {
        it <- it + 1L
        Q <- .agfOrth(Y)
        t0 <- proc.time()[["elapsed"]]
        Y <- .agfMapChunks(chunks, function(J, t) {
            Z <- zchunk(J)
            Z %*% crossprod(Z, Q)
        }, function(a, b) if (is.null(a)) b else a + b, nc)
        passes <- passes + 1L
        timings[paste0("power", it)] <- proc.time()[["elapsed"]] - t0
        captured[it] <- .agfCaptured(Q, Y, npcs) / total_ss
        gain[it] <- if (it > 1L)
            (captured[it] - captured[it - 1L]) / captured[it] else NA_real_
        if (verbose) message(sprintf(
            '  pass %d (power %d): %.1fs, captured %.4f%s', passes, it,
            timings[[paste0("power", it)]], captured[it],
            if (it > 1L) sprintf(', gain %.2e', gain[it]) else ''))
        if (!is.null(ctrl$tol) && it > 1L && gain[it] < ctrl$tol) break
    }

    # Final pass: Rayleigh-Ritz on span(Q). C_J = Z_J^T Q is the chunk's
    # rows of B^T; G = B B^T = sum_J C_J^T C_J.
    Q <- .agfOrth(Y)
    t0 <- proc.time()[["elapsed"]]
    fin <- .agfMapChunks(chunks, function(J, t) {
        C <- crossprod(zchunk(J), Q)
        list(G = crossprod(C), C = list(C), J = list(J))
    }, function(a, b) {
        if (is.null(a)) return(b)
        list(G = a$G + b$G, C = c(a$C, b$C), J = c(a$J, b$J))
    }, nc)
    passes <- passes + 1L
    timings["final"] <- proc.time()[["elapsed"]] - t0
    eg <- eigen(fin$G, symmetric = TRUE)
    Ut <- eg$vectors[, seq_len(npcs), drop = FALSE]
    d <- sqrt(pmax(eg$values[seq_len(npcs)], 0))
    emb <- matrix(0, prep$n_cells, npcs)
    for (i in seq_along(fin$J)) emb[fin$J[[i]], ] <- fin$C[[i]] %*% Ut
    if (verbose) message(sprintf('  pass %d (final): %.1fs, captured %.4f',
                                 passes, timings[["final"]],
                                 sum(d^2) / total_ss))
    diagnostics <- list(method = "rsvd", passes = passes, n_iter = it,
                        ell = prep$ell, captured_trace = captured,
                        gain_trace = gain, captured = sum(d^2) / total_ss,
                        sv_ritz = sqrt(pmax(eg$values, 0)),
                        timings = timings)
    .agfResult(prep, emb, Q %*% Ut, d, diagnostics)
}

# Top-k eigenpairs of a symmetric PSD matrix. A full eigen() is O(F^3) and
# dominates for thousands of features; a converged partial solver gives the
# same leading pairs.
.agfTopEigen <- function(G, k) {
    if (nrow(G) <= 3000L) {
        eg <- eigen(G, symmetric = TRUE)
        return(list(values = eg$values[seq_len(k)],
                    vectors = eg$vectors[, seq_len(k), drop = FALSE]))
    }
    # For PSD G the singular vectors are the eigenvectors
    sv <- irlba::irlba(G, nv = k, nu = k, tol = 1e-10, maxit = 5000L,
                       work = k + 20L)
    list(values = sv$d, vectors = sv$v)
}

.agfEmbed <- function(chunks, f, n_cells, npcs, nc) {
    parts <- .agfMapChunks(chunks, function(J, t)
        list(E = list(f(J)), J = list(J)),
        function(a, b) if (is.null(a)) b else
            list(E = c(a$E, b$E), J = c(a$J, b$J)), nc)
    emb <- matrix(0, n_cells, npcs)
    for (i in seq_along(parts$J)) emb[parts$J[[i]], ] <- parts$E[[i]]
    emb
}

.agfResult <- function(prep, emb, loadings, d, diagnostics) {
    # Fix each PC's sign (largest-magnitude loading positive), so results do
    # not depend on summation order, chunking or worker count
    flip <- sign(loadings[cbind(apply(abs(loadings), 2, which.max),
                                seq_len(ncol(loadings)))])
    flip[flip == 0] <- 1
    loadings <- sweep(loadings, 2, flip, `*`)
    emb <- sweep(emb, 2, flip, `*`)
    diagnostics$chunk_size <- prep$chunk_size
    diagnostics$n_chunks <- length(prep$chunks)
    gn <- prep$gene_names
    if (is.null(gn)) gn <- paste0("gene", seq_len(prep$n_genes))
    rownames(emb) <- prep$cell_names
    rownames(loadings) <- c(gn, paste0(gn, "_H0"), paste0(gn, "_H1"))
    colnames(emb) <- colnames(loadings) <- paste0("PC", seq_len(ncol(emb)))
    list(embeddings = emb, loadings = loadings, d = d,
         diagnostics = diagnostics)
}

# Full scaled BANKSY matrix assembled from chunks. Small data only: used by
# the tests to check chunk recomputation against getBanksyMatrix.
.agfFullMatrix <- function(prep, lambda) {
    sc <- .agfScaling(prep, lambda)
    Z <- matrix(0, prep$F, prep$n_cells)
    for (J in prep$chunks)
        Z[, J] <- .agfChunkScaled(.agfChunkRaw(prep$X, prep$mats, J), J, sc)
    Z
}

# --- Memory model and resource planning -----------------------------------

# Bytes of the pieces of a lazy AGF run (see planBanksyAGF for the model)
.agfMemoryModel <- function(n_cells, n_genes, density, k_geom, npcs,
                            oversample, x_on_disk, method) {
    F <- 3 * n_genes
    ell <- npcs + oversample
    if (length(k_geom) == 1L) k_geom <- rep(k_geom, 2L)
    # dgCMatrix: 8 (value) + 4 (row index) bytes per nonzero
    x <- if (x_on_disk) 0 else 12 * density * n_genes * n_cells
    nbr <- 12 * n_cells * (k_geom[1] + 2 * k_geom[2])
    # kNN tables (from, to, weight, phi, distance) and sparseMatrix
    # construction copies; freed before the passes start
    build <- 48 * n_cells * sum(k_geom)
    out <- 8 * n_cells * (ell + npcs) * 2
    list(F = F, ell = ell, x = x, nbr = nbr, build = build, out = out,
         per_cell = 5 * 8 * F,
         gram = if (method == "gram") 2 * 8 * F^2 else 0,
         base = 1.5e9)
}

# Warn when the estimated peak of this run exceeds the machine's RAM
.agfCheckMemory <- function(X, n_cells, n_genes, npcs, ctrl, chunk_size,
                            knn0, knn1) {
    ram <- .availableRamGb()
    if (is.na(ram)) return(invisible(NULL))
    on_disk <- !(is.matrix(X) || inherits(X, 'dgCMatrix'))
    density <- if (inherits(X, 'dgCMatrix'))
        length(X@x) / (as.double(n_genes) * n_cells) else 1
    k <- c(nrow(knn0), nrow(knn1)) / n_cells
    m <- .agfMemoryModel(n_cells, n_genes, density, k, npcs,
                         ctrl$oversample, on_disk, ctrl$method)
    # X is already in memory and counted by R; count only what the run adds
    peak <- m$base + m$nbr + m$out +
        ctrl$num_cores * (m$per_cell * chunk_size + m$gram)
    if (peak > 0.9 * ram * 1e9)
        warning(sprintf(paste0(
            'estimated peak %.0f GB is close to or above RAM (%.0f GB): ',
            'lower agf_control$chunk_size or num_cores; see planBanksyAGF()'),
            peak / 1e9, ram))
    invisible(peak)
}

# Physical RAM in GB, or NA when it cannot be determined
.availableRamGb <- function() {
    gb <- NA_real_
    if (file.exists('/proc/meminfo')) {
        mi <- readLines('/proc/meminfo', n = 3L)
        line <- grep('^MemTotal', mi, value = TRUE)
        if (length(line)) gb <- as.numeric(gsub('\\D', '', line)) / 1024^2
    } else if (Sys.info()[["sysname"]] == "Darwin") {
        b <- suppressWarnings(tryCatch(
            as.numeric(system('sysctl -n hw.memsize', intern = TRUE)),
            error = function(e) NA_real_))
        gb <- b / 1024^3
    }
    gb
}

#' Plan memory and cores for lazy PCA with the AGF.
#'
#' @details
#' Estimates the memory of \code{runBanksyPCA(lazy = TRUE, use_agf = TRUE)}
#' and recommends \code{chunk_size} and \code{num_cores} for a machine.
#'
#' Memory has three parts. Fixed: the expression matrix (unless on disk, e.g.
#' BPCells), three sparse neighbour matrices, about 1.5 GB of R overhead, and
#' the output. Per worker: about five dense buffers of
#' \code{3 * n_genes x chunk_size} doubles (plus a \code{3 * n_genes} square
#' matrix for \code{method = "gram"}). Workers share the fixed part, so
#' memory grows with \code{num_cores} only through the per-worker part.
#'
#' The recommendation keeps a headroom of 15\% of RAM (at least 4 GB) unused,
#' uses as many of the cores as fit, and then the largest chunk (at most
#' 20,000 cells) that still fits. Sparse products are limited by memory
#' bandwidth, so speed-up from extra cores is sublinear; measure before
#' assuming more cores help.
#'
#' @param n_cells Number of cells.
#' @param n_genes Number of genes (features of the expression assay).
#' @param ram_gb RAM of the machine in GB. Detected when NULL (Linux, macOS).
#' @param cores Cores you are willing to use. Defaults to all but one.
#' @param density Fraction of nonzero entries in the expression matrix.
#' @param k_geom Neighbours for the mean neighbourhood and the AGF.
#' @param npcs Number of PCs.
#' @param oversample Extra subspace dimensions (\code{agf_control}).
#' @param x_on_disk Whether the expression matrix is on disk (e.g. BPCells).
#' @param method "rsvd" or "gram".
#' @param headroom Fraction of RAM to leave free.
#' @param verbose Print a summary.
#'
#' @return Invisibly, a list with the recommended \code{chunk_size} and
#'   \code{num_cores} and the memory breakdown in GB.
#'
#' @export
#'
#' @examples
#' planBanksyAGF(n_cells = 1e7, n_genes = 2000, ram_gb = 128, cores = 16)
planBanksyAGF <- function(n_cells, n_genes, ram_gb = NULL, cores = NULL,
                          density = 0.1, k_geom = c(15, 30), npcs = 50,
                          oversample = 10, x_on_disk = FALSE,
                          method = c("rsvd", "gram"), headroom = 0.15,
                          verbose = TRUE) {
    method <- match.arg(method)
    if (is.null(ram_gb)) ram_gb <- .availableRamGb()
    if (is.na(ram_gb)) stop('could not detect RAM; supply ram_gb')
    if (is.null(cores))
        cores <- max(1L, parallel::detectCores(logical = FALSE) - 1L)
    m <- .agfMemoryModel(n_cells, n_genes, density, k_geom, npcs,
                         oversample, x_on_disk, method)
    if (!x_on_disk && density * n_genes * n_cells > .Machine$integer.max)
        warning('about ', signif(density * n_genes * n_cells, 2),
                ' nonzeros: over the 2^31 - 1 limit of an in-memory ',
                'dgCMatrix. Keep the expression matrix on disk (BPCells) and ',
                'set x_on_disk = TRUE')
    gb <- 1e9
    reserve <- max(headroom * ram_gb, 4) * gb
    fixed <- m$base + m$x + m$nbr + m$out
    budget <- ram_gb * gb - reserve - fixed
    # Construction peak (before the passes) must also fit
    if (ram_gb * gb - reserve < m$base + m$x + m$nbr + m$build)
        warning('building the neighbour matrices alone may not fit')
    if (budget <= 0) {
        if (verbose) message('Fixed memory (', round(fixed / gb, 1),
                             ' GB) exceeds the budget; this will not fit.')
        return(invisible(list(chunk_size = NA, num_cores = 0L)))
    }
    pick <- NULL
    for (c in c(20000, 10000, 5000, 2000, 1000, 500)) {
        per_worker <- m$per_cell * min(c, n_cells) + m$gram
        w <- min(cores, floor(budget / per_worker))
        if (is.null(pick) || w > pick$w) pick <- list(c = min(c, n_cells),
                                                     w = w, pw = per_worker)
        if (w >= cores) break
    }
    if (pick$w < 1L) {
        if (verbose) message('Even one worker with 500-cell chunks does not ',
                             'fit; reduce n_genes or use method = "rsvd".')
        return(invisible(list(chunk_size = NA, num_cores = 0L)))
    }
    peak <- fixed + pick$w * pick$pw
    res <- list(chunk_size = as.integer(pick$c), num_cores = as.integer(pick$w),
                per_worker_gb = pick$pw / gb, fixed_gb = fixed / gb,
                peak_gb = peak / gb, ram_gb = ram_gb,
                headroom_gb = (ram_gb * gb - peak) / gb,
                breakdown_gb = c(expression = m$x, neighbours = m$nbr,
                                 output = m$out, r_base = m$base) / gb)
    if (verbose) {
        message(sprintf(paste0(
            'Fixed memory %.1f GB (expression %.1f, neighbour matrices %.1f, ',
            'output %.1f, R %.1f).\n',
            'Each worker needs %.1f GB with chunk_size = %d.\n',
            'With %.0f GB RAM and %d cores: use num_cores = %d, ',
            'chunk_size = %d. Peak about %.0f GB, leaving %.0f GB free.'),
            res$fixed_gb, res$breakdown_gb[["expression"]],
            res$breakdown_gb[["neighbours"]], res$breakdown_gb[["output"]],
            res$breakdown_gb[["r_base"]], res$per_worker_gb, res$chunk_size,
            ram_gb, as.integer(cores), res$num_cores, res$chunk_size,
            res$peak_gb, res$headroom_gb))
        if (res$num_cores < cores)
            message('Memory, not cores, is the limit: more workers would ',
                    'need smaller chunks than 500 cells or more RAM.')
    }
    invisible(res)
}
