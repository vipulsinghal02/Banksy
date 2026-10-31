# Step 0: is the AGF worth it on this dataset? Compares M = 0 (lazy, exact)
# with M = 1 (chunked gram, exact) against ground-truth labels, over Leiden
# seeds and resolutions. If M = 1 does not beat M = 0 by more than the spread
# across seeds, the dataset cannot show a benefit from a scalable AGF.
#
#   Rscript dev/agf-chunked/R/step0_m0_vs_m1.R data=dlpfc:151673 \
#       lambda=0.2 k_geom=15,30 out=dev/agf-chunked/results/step0.csv

source("dev/agf-chunked/R/common.R")
source("dev/agf-chunked/R/load_data.R")

a <- parse_args(list(
    data = "dlpfc:151673", n_hvg = 2000, lambda = c(0.2, 0.8),
    k_geom = c(15, 30), npcs = 20, cluster_k = 50,
    resolution = c(0.2, 0.4, 0.6, 0.8, 1.0), seeds = 5,
    out = "dev/agf-chunked/results/step0.csv"))

se <- load_dataset(a$data, a$n_hvg)
stopifnot("dataset has no truth labels" = "truth" %in% colnames(colData(se)))
group <- if (length(unique(se$sample)) > 1L) "sample" else NULL

for (lam in a$lambda) {
    m0 <- runBanksyPCA(se, assay_name = "data", lambda = lam, npcs = a$npcs,
                       k_geom = a$k_geom[1], lazy = TRUE, group = group,
                       verbose = FALSE)
    emb <- list(M0 = reducedDim(m0, sprintf("PCA_M0_lam%s", lam)),
                M1 = run_agf(se, lam, a$npcs, a$k_geom, "data", group = group,
                             method = "gram")$emb)
    for (M in names(emb)) {
        knn <- knn_index(emb[[M]], a$cluster_k)
        for (res in a$resolution) for (s in seq_len(a$seeds)) {
            lab <- leiden_labels(emb[[M]], resolution = res, seed = s,
                                 knn = knn)
            append_csv(data.frame(
                data = a$data, lambda = lam, M = M, resolution = res,
                seed = s, n_clusters = length(unique(lab)),
                n_truth = length(unique(na.omit(se$truth))),
                ari_truth = ari(se$truth, lab)), a$out)
        }
        log_msg("lambda=", lam, " ", M, " done")
    }
}
