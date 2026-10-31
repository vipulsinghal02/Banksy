# Accuracy sweep for the chunked AGF PCA on a dataset small enough for exact
# references. For each (oversample, n_iter or tol, start seed) runs rsvd and
# compares it with the exact answer (method = "gram") and with two "nobody
# would care" yardsticks:
#   - the standard path (dense matrix + irlba) at different seeds
#   - the exact PCA with fewer PCs (npcs - 5, npcs - 10)
#   - Leiden run on the exact embedding with different seeds
#
# Usage (from the repo root, after installing the branch):
#   Rscript dev/agf-chunked/R/sweep_params.R data=dlpfc:151673 \
#       lambda=0.2 k_geom=15,30 npcs=20 out=dev/agf-chunked/results/sweep.csv
#
# One row per run is appended to `out` as it finishes, so a partial run still
# leaves usable results.

source("dev/agf-chunked/R/common.R")
source("dev/agf-chunked/R/load_data.R")

a <- parse_args(list(
    data = "dlpfc:151673", n_hvg = 2000, lambda = 0.2, k_geom = c(15, 30),
    npcs = 20, oversample = c(5, 10, 20, 40), n_iter = c(0, 1, 2, 3, 4, 6, 8),
    tol = c(3e-2, 1e-2, 3e-3, 1e-3), tol_max_iter = 15, seeds = 3,
    knn_k = 20, cluster_k = 50, resolution = 0.5, chunk_size = 5000,
    num_cores = 1, mem_cap_gb = 6, out = "dev/agf-chunked/results/sweep.csv"))

log_msg("loading ", a$data)
se <- load_dataset(a$data, a$n_hvg)
group <- if (length(unique(se$sample)) > 1L) "sample" else NULL
truth <- if ("truth" %in% colnames(colData(se))) se$truth else NULL
log_msg(ncol(se), " cells, ", nrow(se), " genes, group = ",
        if (is.null(group)) "none" else group)

base <- data.frame(data = a$data, n_cells = ncol(se), n_genes = nrow(se),
                   lambda = a$lambda, npcs = a$npcs)
emit <- function(...) append_csv(cbind(base, data.frame(...)), a$out)

agf <- function(...) run_agf(se, a$lambda, a$npcs, a$k_geom, "data",
                             group = group, num_cores = a$num_cores,
                             chunk_size = a$chunk_size, ...)

# --- exact reference -----------------------------------------------------------
log_msg("exact reference (gram)")
ref <- agf(method = "gram")
ref_knn <- knn_index(ref$emb, a$knn_k)
ref_lab <- leiden_labels(ref$emb, a$cluster_k, a$resolution, seed = 1)
ref_cap <- ref$diag$captured
half <- max(1L, a$npcs %/% 2L)

# Metrics of an embedding against the exact one. The Leiden seed matches the
# reference's, so any ARI loss comes from the embedding, not the seed.
score <- function(emb) {
    lab <- leiden_labels(emb, a$cluster_k, a$resolution, seed = 1)
    data.frame(
        sin5 = sin_theta(ref$emb, emb, min(5L, a$npcs)),
        sin_half = sin_theta(ref$emb, emb, half),
        sin_all = sin_theta(ref$emb, emb, a$npcs),
        jaccard = knn_jaccard(ref_knn, knn_index(emb, a$knn_k)),
        ari_vs_exact = ari(ref_lab, lab),
        ari_truth = if (is.null(truth)) NA else ari(truth, lab),
        n_clusters = length(unique(lab)))
}

emit(method = "gram", oversample = NA, n_iter = NA, tol = NA, seed = NA,
     passes = 3, n_iter_used = NA, time = ref$time,
     captured = ref_cap, captured_rel = 1, score(ref$emb))

# --- yardsticks ------------------------------------------------------------------
for (drop in c(5, 10)) {
    k <- a$npcs - drop
    if (k < 2) next
    e <- ref$emb[, seq_len(k)]
    emit(method = sprintf("exact_%dpcs", k), oversample = NA, n_iter = NA,
         tol = NA, seed = NA, passes = NA, n_iter_used = NA, time = NA,
         captured = NA, captured_rel = NA,
         sin5 = NA, sin_half = NA, sin_all = NA,
         jaccard = knn_jaccard(ref_knn, knn_index(e, a$knn_k)),
         ari_vs_exact = ari(ref_lab, leiden_labels(e, a$cluster_k,
                                                   a$resolution, 1)),
         ari_truth = NA, n_clusters = NA)
}
for (s in 2:(a$seeds + 1)) {
    lab <- leiden_labels(ref$emb, a$cluster_k, a$resolution, seed = s)
    emit(method = "leiden_seed", oversample = NA, n_iter = NA, tol = NA,
         seed = s, passes = NA, n_iter_used = NA, time = NA, captured = NA,
         captured_rel = NA, sin5 = NA, sin_half = NA, sin_all = NA,
         jaccard = 1, ari_vs_exact = ari(ref_lab, lab),
         ari_truth = if (is.null(truth)) NA else ari(truth, lab),
         n_clusters = length(unique(lab)))
}
if (standard_fits(se, a$mem_cap_gb)) {
    for (s in seq_len(a$seeds)) {
        log_msg("standard path, seed ", s)
        st <- run_standard(se, a$lambda, a$npcs, a$k_geom, "data",
                           group = group, seed = s)
        emit(method = "standard_irlba", oversample = NA, n_iter = NA,
             tol = NA, seed = s, passes = NA, n_iter_used = NA,
             time = st$time, captured = NA, captured_rel = NA,
             score(st$emb))
    }
} else {
    log_msg("skipping the standard path: dense matrix exceeds mem_cap_gb")
}

# --- rsvd grid ------------------------------------------------------------------
for (os in a$oversample) for (s in seq_len(a$seeds)) {
    for (q in a$n_iter) {
        log_msg(sprintf("rsvd oversample=%d n_iter=%d seed=%d", os, q, s))
        r <- agf(oversample = os, n_iter = q, seed = s)
        emit(method = "rsvd", oversample = os, n_iter = q, tol = NA,
             seed = s, passes = r$diag$passes, n_iter_used = r$diag$n_iter,
             time = r$time, captured = r$diag$captured,
             captured_rel = r$diag$captured / ref_cap, score(r$emb))
    }
    for (tl in a$tol) {
        log_msg(sprintf("rsvd oversample=%d tol=%g seed=%d", os, tl, s))
        r <- agf(oversample = os, n_iter = a$tol_max_iter, tol = tl, seed = s)
        emit(method = "rsvd_tol", oversample = os, n_iter = a$tol_max_iter,
             tol = tl, seed = s, passes = r$diag$passes,
             n_iter_used = r$diag$n_iter, time = r$time,
             captured = r$diag$captured,
             captured_rel = r$diag$captured / ref_cap, score(r$emb))
    }
}
log_msg("done: ", a$out)
