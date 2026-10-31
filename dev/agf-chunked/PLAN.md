# Chunked AGF PCA: handoff and work plan

This file is for a Claude Code session (or a person) picking up the work on a
large machine, e.g. an EC2 instance with 128-256 GB RAM and 16-64 cores. It is
self-contained: the laptop-side notes and memory are not in git.

## 1. Context

- **Package:** Banksy (`prabhakarlab/Banksy`), an R/Bioconductor package for
  spatial clustering. Maintainer: Joseph Lee (`jleechung`). The person driving
  this work is Vipul Singhal, a co-author.
- **Branch:** `feat-agf-rsvd`, on Vipul's fork `vipulsinghal02/Banksy`
  (remote `fork`), made from upstream `devel` at 1.9.6.
  - Push only to `fork`.
  - Don't push to `prabhakarlab` or open a PR unless Vipul asks.
- **Working style:**
  - Vipul prefers short, direct answers.
  - He runs workflows he needs to understand himself.
  - Claude may run tests and experiments itself.
  - Ask before anything outward-facing (pushing to upstream, posting on
    GitHub).
- **Background reading:**
  - `dev/agf-chunked/notes/agf_randomized_svd.pdf`, a 24-page explanation of
    why the AGF needs this and how it works. Sections 4–8 matter most.
  - `notes/agf_rsvd_demo.py`, the Python simulation behind the PDF's numbers.

### The problem in one paragraph

BANKSY's PCA runs on Z = [√(1−λ)·zscore(X); √(2λ/3)·zscore(H0); √(λ/3)·zscore(H1)],
a 3g × n matrix (g genes, n cells).
- Without the AGF (M = 0), the lazy path in `R/lazy.R` never builds Z. It
  multiplies through X and the neighbour matrix W, using (XW)v = X(Wv).
- The AGF H1 = |X C| has a modulus, so it isn't linear and that trick fails.
- Dense Z at 10M cells × 5k genes is 1.2 TB.
- The fix is to rebuild Z one chunk of cells at a time on every pass. A chunk's
  columns depend only on those cells and their spatial neighbours. A solver
  that needs only a few passes then runs on top of the chunks.

## 2. What is implemented (all on this branch)

**`R/agf_chunked.R`** (new)
- **Entry point:** `runBanksyPCA(se, use_agf = TRUE, lazy = TRUE, ...)`, which
  dispatches to `.runBanksyPCA_agf`.
- **Settings:** `agf_control = list(...)`. Defaults are in `.agfControlDefaults`:
  - `method = "rsvd"`
  - `n_iter = 4`
  - `oversample = 10`
  - `tol = NULL`
  - `chunk_size = NULL` (derived from `mem_per_worker_gb = 2`)
  - `num_cores = 1`
  - `center = TRUE`
- **Chunk builder** (`.agfChunkRaw`):
  - For cells J, it reads X at J and their neighbours (`X[, nb]`), so any
    backend with column subsetting works: dgCMatrix, base matrix, BPCells,
    DelayedArray.
  - It does one sparse product with `cbind(W0, Cre, Cim)[nb, J]` and returns
    `rbind(own, |H0|, |H1|)`.
  - Centring is folded into the coefficients: c_ij = w_ij·e^{iφ} − s_j/k_j.
- **Chunk order:** chunks are spatially compact (Morton/Z-order key) and
  sorted by group first.
- **Pass 1, statistics:**
  - Per-row mean, M2, min and max, merged with Chan's formula.
  - Optionally per group (`split_scale`).
  - Constant rows are scaled to 0, matching `scaler()`'s NaN → 0.
  - Without per-group scaling, the same pass also accumulates
    T = Σ R_J R_Jᵀ P. That gives the start product D(T − nμμᵀP) for every λ,
    saving one pass.
- **`method = "rsvd"`:**
  - Subspace iteration in feature space: Y ← Σ_J Z_J (Z_Jᵀ Q).
  - Then a final Rayleigh–Ritz pass.
  - Passes: `n_iter + 2` without groups, `n_iter + 3` with per-group scaling.
  - **Stopping rule (`tol`):** after each power pass it records the captured
    variance (sum of the top-npcs eigenvalues of QᵀY, divided by total
    variance), at no extra cost. It stops early when the relative gain is
    below `tol`.
- **`method = "gram"`:** exact. Accumulates the F × F covariance G = Σ Z_J Z_Jᵀ,
  then `eigen()`, then a pass to compute the embeddings. 3 passes, F² memory
  per worker. This is the reference for all accuracy work.
- **Parallelism:** `num_cores` forked workers (`parallel::mclapply`, unix
  only). Chunks are dealt round-robin. Each worker returns an F × ℓ partial
  sum (or F × F for gram).
- **Determinism:** PC signs are fixed (largest loading positive). Results are
  identical across chunk sizes and worker counts (tested to 1e-8).
- **Diagnostics:** in `metadata(se)$BANKSY_agf$lam<λ>`: passes, n_iter used,
  captured-variance trace, gains, Ritz values and per-pass timings.
- **Memory:**
  - `planBanksyAGF(n_cells, n_genes, ram_gb, cores, ...)` (exported)
    recommends `chunk_size` and `num_cores` and prints a breakdown.
  - `.agfCheckMemory` warns during a run if the estimated peak exceeds RAM.
  - **The model's constants are guesses until Phase 2 measures them:** five
    F × chunk dense buffers per worker, 1.5 GB R base, and a 15% / 4 GB
    headroom.

**`R/reduction.R`**
- The lazy branch now accepts M ∈ {0, 1} and rejects anything else.
- New `agf_control` argument.
- Docs updated.

**Tests**
- `tests/testthat/test-agf-chunked.R`: 21 expectations. Chunks rebuild
  `getBanksyMatrix` exactly; gram matches dense (scaled, unscaled, grouped);
  rsvd recovers leading PCs; chunk and worker invariance; `tol`; argument
  errors; the planner.
- `test-lazy.R`: "rejects the AGF" became "takes M = 1, rejects M = 2".
- All pass, together with test-lazy, test-reduction, test-computation and
  test-sparse-matmul.

**Results so far** (laptop, simulated rings, 4k cells, 120 genes)
- **Gram vs. standard path:** sin θ(top 10) = 6e-8.
- **rsvd (as in the Python demo):** the top 5 PCs converge fast
  (sin θ 1e-2 → 3e-5 → 4e-8 at n_iter 1/4/8). Directions in the flat noise
  bulk never converge.
- **Expected consequence:** the 50-PC kNN graph changes about as much as using
  35–40 PCs instead of 50. Whether that matters is what Phase 1 decides.

**First real-data sweep** (laptop, DLPFC 151673: 3,639 spots, 2,000 HVGs,
λ = 0.2, npcs = 20, k_geom = 15/30; `sweep_params.R` with 2 start seeds)

| run | passes | time (s) | sin θ top 5 | sin θ top 10 | kNN Jaccard vs exact | ARI vs truth |
|---|---|---|---|---|---|---|
| exact (gram) | 3 | 32 | 0 | 0 | 1 | 0.483 |
| standard irlba (dense) | – | 7 | 8e-8 | 7e-8 | ≈1.000 | 0.492 |
| exact with 15 PCs | – | – | – | – | 0.634 | – |
| exact with 10 PCs | – | – | – | – | 0.491 | – |
| rsvd n_iter=2 | 4 | 7 | 0.04 | 0.6 | 0.42 | 0.36–0.51 |
| rsvd n_iter=4 | 6 | 11 | 3e-3 | 0.4 | 0.50 | 0.45–0.48 |
| rsvd n_iter=8 | 10 | 19 | 2e-5 | 0.05–0.08 | 0.61 | 0.47 |
| rsvd n_iter=8, oversample 20 | 10 | 22 | 2e-5 | 0.04 | 0.67–0.68 | 0.43–0.44 |
| rsvd tol=1e-2 / 3e-3 | 6 / 8 | 11 / 15 | (stopped at n_iter 4 / 6 on both seeds) | | | |

What this run shows:
- **The rsvd tail is worse on real data than on the simulation.** The
  spectrum is flat after the first few PCs, so with npcs = 20 most components
  sit in the noise bulk. Even n_iter = 8 only matches the kNN graph of "exact
  with 15 PCs".
- **Clustering ARI against truth** is within about ±0.05 of exact. With only 2
  seeds that's not conclusive; use at least 5.
- **Leiden is very seed-sensitive here.** At resolution 0.5, ARI between seeds
  of the same exact embedding was 0.78–0.98. Even standard irlba (kNN Jaccard
  ≈ 1) gave ARI 0.87 against exact. So `ari_vs_exact` is mostly Leiden noise
  on this dataset; trust `jaccard` and `sin_*` more. (`leiden_labels` now
  sorts neighbour lists, so identical kNN sets give identical graphs.)
- **New question for Phase 1/2: is exact `gram` the better default?** It has
  no accuracy question at all. Its cost is F²·n flops in one BLAS-3 pass plus
  a partial eigensolve. rsvd costs ~4·F·n·ℓ per pass but needs about 10
  passes, and each pass also rebuilds the chunks (sparse work). Here gram
  (32 s) vs rsvd n_iter=8 (19 s) were within 2×, on R's slow reference BLAS.
  For g ≤ ~2–3k genes gram may win outright. Measure both with an optimised
  BLAS.
- **BLAS matters a lot.** The laptop R uses reference BLAS (`libRblas`), and
  the dense `Z Zᵀ Q` step ran at about 2.6 GFLOP/s. On EC2, use OpenBLAS or
  MKL (check `sessionInfo()`, which shows BLAS/LAPACK). Decide BLAS threads
  vs. forked workers by measurement.

### Known environment issues (laptop; may recur)

- **igraph:** igraph < 2.1 doesn't export `as_undirected`, which
  `R/cluster.R` imports, so the package fails to load. Fix by updating
  igraph. Don't change the import.
- **C++ standard:** `src/banksy_lazy.cpp` needs C++11 or newer. A
  `~/.R/Makevars` with `CXX=g++` and no `-std` breaks it.
- **roxygen2 7.3.3 vs 7.3.2:** 7.3.3 reorders NAMESPACE and deletes
  `man/dim.BanksyLazy.Rd`. Don't commit that churn. Copy only the Rd files
  you changed, and keep `RoxygenNote: 7.3.2`.

## 3. Setup on the big machine

```bash
git clone -b feat-agf-rsvd https://github.com/vipulsinghal02/Banksy.git
cd Banksy
git remote add upstream https://github.com/prabhakarlab/Banksy.git
# R >= 4.4, Bioconductor; then in R:
#   BiocManager::install(c("SpatialExperiment", "spatialLIBD", "ExperimentHub",
#     "scuttle"), update = FALSE)
#   install.packages(c("igraph", "leidenAlg", "dbscan", "irlba", "aricode",
#     "uwot", "RcppHungarian", "data.table", "testthat", "devtools"))
#   remotes::install_github("bnprks/BPCells/r")   # needs libhdf5-dev
R CMD INSTALL .
Rscript -e 'devtools::test(filter = "agf-chunked|lazy|reduction")'
```

- Run long jobs as scripts inside `tmux`, not in RStudio Server.
- Measure memory with `dev/agf-chunked/memwatch.sh` (it sums PSS over the
  master and its forked workers). `/usr/bin/time` only sees the largest
  single process.
- **BLAS threads:** with `num_cores > 1`, set `OPENBLAS_NUM_THREADS=1` (or
  `OMP_NUM_THREADS=1`) so workers don't oversubscribe. With `num_cores = 1`,
  also try letting BLAS use threads. That's an alternative to forked workers
  worth comparing.

## 4. Scripts (`dev/agf-chunked/R/`)

All take `key=value` arguments (vectors comma-separated) and append rows to a
CSV as they go.

| Script | Purpose |
|---|---|
| `common.R` | metrics (sin θ, kNN Jaccard, Leiden, ARI), `run_agf`, `run_standard`, peak RSS |
| `load_data.R` | `load_dataset("dlpfc:151673+151674")` or `"rds:/path.rds"`. Returns an SPE with assays `data` (log-normalised HVGs) and `counts`, plus colData `truth` and `sample` |
| `step0_m0_vs_m1.R` | Is the AGF worth it? M=0 vs M=1 against truth over Leiden seeds and resolutions |
| `sweep_params.R` | Accuracy grid: oversample × n_iter × start seed, plus `tol` runs, against exact gram, with yardsticks (standard irlba at other seeds, exact with npcs−5 / −10 PCs, Leiden seed noise) |
| `make_tiles.R` | Poisson-resampled tiled copies for **time and memory only** (BPCells or dgC) |
| `bench_scale.R` | One timed run (`method`, `n_iter`, `chunk_size`, `num_cores`) and its planner prediction; run under `memwatch.sh` |

## 5. Experiments

### Phase 0: is the AGF worth it? (cheap; do first)
- **What to run:** `step0_m0_vs_m1.R` on each DLPFC sample (12) and on
  STARmap PLUS, at λ = 0.2 and 0.8.
- **Note:** the package vignettes no longer use the AGF for these datasets.
- **What to look for:** if M = 1's ARI against truth doesn't beat M = 0's by
  more than the seed-to-seed spread, report it. The scalable AGF then needs a
  different motivating dataset (gradient-rich tissue).

### Phase 1: accuracy and default settings
Run `sweep_params.R`, including `npcs=20` (package default) and `npcs=50`, on:
- DLPFC samples one at a time and in groups of four (exercises
  `group = "sample"` and per-group scaling);
- STARmap PLUS well11 (43k cells, 1,022 genes);
- one or two unlabelled mid-size datasets where the standard path still fits:
  dense Z costs about 5 × 3g × n × 8 bytes, so g = 2000 and n = 300k is
  about 72 GB. Example: a 100–300k-cell Xenium 5K subset.

Questions to answer:
1. **How many power passes (`n_iter`) are needed** before the rsvd embedding
   is as close to exact as the yardsticks are? Concretely, all three of:
   - median `ari_vs_exact` ≥ median ARI of `leiden_seed` (Leiden's own seed
     noise);
   - `jaccard` ≥ `exact_<npcs−5>pcs`;
   - `jaccard` ≥ `standard_irlba` (what the current dense path itself gives).
2. **Does `oversample` (5, 10, 20, 40) buy more** than an extra pass of the
   same cost?
3. **Is a `tol` rule reliable?** Does it stop at about the n_iter from
   question 1 on every dataset? If yes, make `tol` the default (with an
   `n_iter` cap); if not, recommend a fixed `n_iter`.
4. **Is `ari_truth` the same** for rsvd and exact? (Any difference should be
   within seed noise.)

5. **Gram vs. rsvd, end to end:** at what (g, n) does exact gram stop being
   competitive in time (F²·n flops; F² × 8 bytes per worker)? If gram is
   within about 2× of rsvd at the accuracy rsvd needs, recommend gram up to
   that g. In that case rsvd is only needed for very wide panels (Xenium 5K:
   F = 15,000, so G is 1.8 GB per worker and eigensolving is still fine).

Pick the default as the cheapest setting (passes × cost) that meets
question 1's three criteria on every dataset. Update `.agfControlDefaults` and
the `runBanksyPCA` docs.

### Phase 2: performance model (mid-size, exact comparisons possible)
- **What to run:** `bench_scale.R` under `memwatch.sh`, over:
  - `chunk_size` ∈ {1000, 2000, 5000, 10000, 20000};
  - `num_cores` ∈ {1, 2, 4, 8, 12, 16, …up to the machine};
  - X as dgCMatrix vs. BPCells.
- **Outputs:**
  - Time per pass vs. chunk size: where does it flatten? (The guess: per-chunk
    overhead and border rereads stop mattering above about 5–20k cells.)
  - Speed-up vs. cores: sparse products are bandwidth-bound, so expect
    sublinear. Where does it stop paying?
  - Measured peak PSS vs. `planBanksyAGF()`'s `plan_peak_gb`. Recalibrate
    `.agfMemoryModel`: the factor-5 buffers, the base, and mclapply's return
    copies.
  - Update the planner's chunk-vs-cores policy from the data. It currently
    maximises workers first and then chunk size, which may be wrong if
    workers stop helping.
- **Goal:** `planBanksyAGF()` should output a statement like "on 16 cores /
  128 GB: num_cores = N, chunk_size = C, peak ≈ X GB, leaving Y GB",
  validated to within about 20%.

### Phase 3: scale
- **Real atlases** (also used for accuracy wherever exact gram fits:
  F² × 8 bytes, e.g. g = 5000 needs 1.8 GB per worker).
  - **Allen Brain Cell Atlas MERFISH** (public S3, no credentials):
    `aws s3 ls --no-sign-request s3://allen-brain-cell-atlas/`.
    Zhuang-ABCA-1…4 (about 1,100 genes, millions of cells) and
    MERFISH-C57BL6J-638850 (about 500 genes, about 4M cells). They have cell
    metadata with coordinates per section and mapped cell-type labels.
    **Verify the exact paths and sizes.** Convert h5ad to BPCells with
    `BPCells::open_matrix_anndata_hdf5()` then `write_matrix_dir()`, and build
    an SPE with section-offset coordinates.
  - **10x Xenium Prime 5K public datasets** (about 5,000 genes, 100k–700k
    cells per dataset; download links on 10x's site). These are relevant to
    issue #68 (a 5K panel, 12 samples, 512 GB / 53 h in `clusterBanksy`).
  - **STARmap PLUS:** Shi et al. 2023, Broad Single Cell Portal SCP1830. It
    may need a Google login, in which case ask Vipul to download
    `well11processed_expression_pd.csv.gz` and `well11_spatial.csv.gz`.
    Conversion recipe: `vignettes/domain-segment.Rmd`, the "Data
    preprocessing" chunk. **The labels bundled in `inst/extdata/STARmap.rds`
    are BANKSY M=0 output, not ground truth.**
- **Tiles:** `make_tiles.R` (Poisson resampling) from the largest real
  dataset, up to 10M and 20M cells, with BPCells.
- **What to report:**
  - time and peak memory vs. n for rsvd (chosen defaults) and gram;
  - the M = 0 lazy path's time on the same data, for comparison;
  - where it breaks, and why.

### Phase 4: optional optimisation (only after profiling)
Candidates, in order of expected value:
- **C++ chunk builder:** gather neighbours and do the complex sums directly
  from CSC slots, OpenMP inside a worker. Currently R and Matrix do the sparse
  product with a sparse intermediate.
- **Multiple λ in one sweep:** chunks are λ-independent; only the block weights
  change.
- **Block Krylov** (Musco & Musco 2015) instead of plain subspace iteration,
  if Phase 1 shows the tail converging too slowly.
- **Hybrid:** lazy products for the own and H0 blocks, rebuilding only H1.

## 6. Reporting back

- Write `dev/agf-chunked/results/summary.md`:
  - what was run (machine, data, versions);
  - one table per phase;
  - the recommended defaults;
  - the planner's validated statement;
  - open problems.
- Commit the CSVs (small) and the summary, not data, to `feat-agf-rsvd`, and
  push to `fork`.
- Tell Vipul plainly what was found, including anything that didn't work.

## 7. Before any PR to upstream

- Decide whether `dev/` stays (it's in `.Rbuildignore`) or is dropped from the
  PR.
- Run `R CMD check`.
- Rebuild the docs only for files touched (see the roxygen note above).
- Add a NEWS entry.
- Joseph should review the API: `agf_control` and `planBanksyAGF` naming.
