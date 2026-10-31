# One timed run of the chunked AGF PCA, for scaling and core-count tests.
# Run one configuration per process, under memwatch.sh so the memory of all
# forked workers is measured:
#
#   dev/agf-chunked/memwatch.sh dev/agf-chunked/results/mem.csv -- \
#     Rscript dev/agf-chunked/R/bench_scale.R data=rds:/data/tiles20.rds \
#       method=rsvd n_iter=4 chunk_size=10000 num_cores=8 \
#       out=dev/agf-chunked/results/bench.csv
#
# Appends one row to `out`: sizes, settings, wall time per pass, total time,
# and the master process's peak RSS (memwatch.sh records the whole tree).

source("dev/agf-chunked/R/common.R")
source("dev/agf-chunked/R/load_data.R")

a <- parse_args(list(
    data = "dlpfc:151673", n_hvg = 2000, lambda = 0.2, k_geom = c(15, 30),
    npcs = 20, method = "rsvd", n_iter = 4, oversample = 10, chunk_size = 0,
    num_cores = 1, group = FALSE, tag = "",
    out = "dev/agf-chunked/results/bench.csv"))

t_load <- proc.time()[["elapsed"]]
se <- load_dataset(a$data, a$n_hvg)
t_load <- proc.time()[["elapsed"]] - t_load
log_msg(ncol(se), " cells, ", nrow(se), " genes, assay class ",
        class(assay(se, "data"))[1])
plan <- planBanksyAGF(ncol(se), nrow(se), cores = a$num_cores,
                      k_geom = a$k_geom, npcs = a$npcs,
                      oversample = a$oversample, method = a$method,
                      x_on_disk = !inherits(assay(se, "data"), "dgCMatrix"))

r <- run_agf(se, a$lambda, a$npcs, a$k_geom, "data",
             group = if (a$group) "sample" else NULL,
             num_cores = a$num_cores, method = a$method,
             n_iter = a$n_iter, oversample = a$oversample,
             chunk_size = if (a$chunk_size > 0) a$chunk_size else NULL)
tm <- r$diag$timings
append_csv(data.frame(
    tag = a$tag, data = a$data, n_cells = ncol(se), n_genes = nrow(se),
    assay_class = class(assay(se, "data"))[1], method = a$method,
    n_iter = a$n_iter, oversample = a$oversample,
    chunk_size = r$diag$chunk_size, n_chunks = r$diag$n_chunks,
    num_cores = a$num_cores, passes = r$diag$passes,
    t_load = t_load, t_total = r$time, t_stats = tm[["stats"]],
    t_pass_median = median(tm[names(tm) != "stats"]),
    captured = r$diag$captured, peak_rss_master_mb = peak_rss_mb(),
    plan_peak_gb = plan$peak_gb, plan_per_worker_gb = plan$per_worker_gb,
    host = Sys.info()[["nodename"]]), a$out)
log_msg(sprintf("total %.1fs, %d passes, master peak %.0f MB", r$time,
                r$diag$passes, peak_rss_mb()))
