#!/usr/bin/env python3
"""
Numerical demonstration for the notes in ../agf_randomized_svd.tex
("PCA of the BANKSY matrix with the AGF, one chunk at a time").

What this script does
---------------------
1. Simulates a spatial dataset: cells in a square tissue with five spatial
   domains, ten cell types, domain genes, smooth spatial programs and
   gradient genes.
2. Builds the BANKSY features the way Banksy::computeBanksy and
   Banksy::getBanksyMatrix do for M = 1:
       own  = X
       H0   = |X W0|                       (kNN_median weights, k = 15)
       H1   = |sum_i w_ij e^{i phi_ij} (x_i - xbar_j)|   (centred AGF, k = 30)
   then z-scores every row and multiplies the three blocks by
   sqrt(1 - lambda), sqrt(2 lambda / 3), sqrt(lambda / 3).
3. Computes the exact top-p PCA with a dense SVD (possible only because the
   example is small).
4. Runs randomized SVD
     (a) chunk by chunk: every chunk of Z is rebuilt from X and the neighbour
         lists, the stored Z is never touched;
     (b) on the stored Z, for the parameter sweeps;
   and checks that (a) and (b) agree to rounding error. Also runs the exact
   alternative (feature covariance, one chunked pass) for comparison.
5. Sweeps the number of power iterations q and the block size ell, and
   records subspace angles, singular-value errors, captured variance,
   kNN-graph overlap and clustering agreement.
6. Counts how many matrix-vector products a Lanczos-type solver (ARPACK,
   through scipy) needs on the same matrix.

Outputs: figures in ../figures, numbers in ../generated (results.json,
numbers.tex with LaTeX macros, and table snippets).

Run with:  ~/miniconda3/bin/python3 agf_rsvd_demo.py
"""

import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, svds
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
GEN = os.path.join(HERE, "..", "generated")
os.makedirs(FIG, exist_ok=True)
os.makedirs(GEN, exist_ok=True)

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
N_CELLS = 10000
N_GENES = 400
K0 = 15            # neighbours for H0 (mean neighbourhood)
K1 = 30            # neighbours for H1 (AGF)
LAMBDA = 0.8       # domain segmentation setting
P = 50             # number of PCs wanted
ELLS = [50, 55, 60, 70]
QS = [0, 1, 2, 3, 4, 6, 8]
SEEDS = [11, 12, 13, 14, 15]
ELL_MAIN = 60
CHUNK = 500        # cells per chunk in the chunked runs
KNN_EMB = 20       # neighbours in PC space (Seurat FindNeighbors default)

# ----------------------------------------------------------------------------
# Chart styling (reference palette from the dataviz guidance, light mode)
# ----------------------------------------------------------------------------
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
       "#0d366b"]
ORDINAL = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # for ell, light -> dark

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.facecolor": SURFACE,
    "figure.facecolor": "white",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "pdf.fonttype": 42,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
})
SEQ_CMAP = ListedColormap(SEQ)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(GEN, name + "_preview.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# 1. Synthetic tissue
# ----------------------------------------------------------------------------
def smooth_field(rng, x, y, n_bumps=6):
    f = np.zeros_like(x)
    for _ in range(n_bumps):
        cx, cy = rng.uniform(0, 100, 2)
        s = rng.uniform(8, 25)
        f += rng.uniform(-1, 1) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) /
                                          (2 * s ** 2))
    return (f - f.mean()) / f.std()


def simulate(rng):
    locs = rng.uniform(0, 100, size=(N_CELLS, 2))
    x, y = locs[:, 0], locs[:, 1]

    # Domains: four wavy layers plus a round inclusion
    yw = y + 6 * np.sin(2 * np.pi * 1.5 * x / 100)
    domain = np.digitize(yw, [25, 50, 75])                    # 0..3
    domain[(x - 68) ** 2 + (y - 40) ** 2 < 13 ** 2] = 4       # 4 = inclusion

    # Cell types: domain-specific mixtures, types shared across domains
    comp = np.zeros((5, 10))
    comp[0, [0, 1, 2]] = [0.5, 0.3, 0.2]
    comp[1, [1, 2, 3, 4]] = [0.2, 0.4, 0.2, 0.2]
    comp[2, [3, 4, 5, 6]] = [0.3, 0.3, 0.2, 0.2]
    comp[3, [5, 6, 7]] = [0.3, 0.4, 0.3]
    comp[4, [0, 7, 8, 9]] = [0.2, 0.2, 0.3, 0.3]
    ctype = np.array([rng.choice(10, p=comp[d]) for d in domain])

    lograte = np.tile(rng.normal(np.log(0.08), 1.0, size=(N_GENES, 1)),
                      (1, N_CELLS))
    g = 0
    for t in range(10):                       # 120 type markers, folds 3..12
        lograte[g:g + 12, ctype == t] += np.log(3 + t)
        g += 12
    for d, fold in enumerate([3, 4, 5, 6, 8]):  # 40 domain genes
        lograte[g:g + 8, domain == d] += np.log(fold)
        g += 8
    strengths = [1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    for s in strengths:                       # 80 genes in 8 spatial programs
        lograte[g:g + 10, :] += s * smooth_field(rng, x, y)
        g += 10
    for j in range(10):                       # 10 gradient genes
        coord = x if j % 2 == 0 else y
        sign = 1 if j < 5 else -1
        lograte[g, :] += 2.0 * sign * (coord / 100 - 0.5)
        g += 1
    # remaining genes: noise only

    size = rng.lognormal(0, 0.3, size=N_CELLS)
    counts = rng.poisson(size * np.exp(lograte))
    lib = counts.sum(axis=0).astype(float)
    lib[lib == 0] = 1.0
    X = np.log1p(counts / lib * np.median(lib))               # g x n
    return locs, domain, ctype, X, counts


# ----------------------------------------------------------------------------
# 2. Neighbours and BANKSY features (mirrors R/computation.R)
# ----------------------------------------------------------------------------
def knn_median(locs, k):
    """Neighbour indices, kNN_median weights and angles, one row per cell."""
    tree = cKDTree(locs)
    dist, idx = tree.query(locs, k=k + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]                       # drop self
    med = np.median(dist, axis=1, keepdims=True)
    w = np.exp(-dist ** 2 / med ** 2)
    w = w / w.sum(axis=1, keepdims=True)
    d = locs[idx] - locs[:, None, :]                          # neighbour - cell
    phi = np.mod(np.arctan2(d[..., 1], d[..., 0]), 2 * np.pi)
    return idx, w, phi


def agf_coefficients(w, phi):
    """c_ij = w_ij e^{i phi_ij} - s_j / k_j  (centred AGF as one complex sum)."""
    e = w * np.exp(1j * phi)
    s = e.sum(axis=1, keepdims=True)
    return e - s / w.shape[1]


class ChunkBuilder:
    """Builds columns of the BANKSY matrix for any set of cells, using only X
    for those cells and their neighbours. This is the 'chunk recomputation'."""

    def __init__(self, X, nbr0, w0, nbr1, c1, lam):
        self.X, self.nbr0, self.w0, self.nbr1, self.c1 = X, nbr0, w0, nbr1, c1
        g = X.shape[0]
        a = np.sqrt([1 - lam, 2 * lam / 3, lam / 3])
        self.a = np.repeat(a, g)                              # block weights
        self.mu = None
        self.sd = None

    def raw(self, idx):
        X = self.X
        own = X[:, idx]
        h0 = np.abs(np.einsum("gbk,bk->gb", X[:, self.nbr0[idx]],
                              self.w0[idx]))
        h1 = np.abs(np.einsum("gbk,bk->gb", X[:, self.nbr1[idx]],
                              self.c1[idx]))
        return np.vstack([own, h0, h1])

    def set_stats(self, s1, s2, n):
        mu = s1 / n
        var = (s2 - n * mu ** 2) / (n - 1)
        sd = np.sqrt(np.maximum(var, 0))
        sd[sd == 0] = 1.0
        self.mu, self.sd = mu, sd

    def scaled(self, idx):
        R = self.raw(idx)
        return self.a[:, None] * (R - self.mu[:, None]) / self.sd[:, None]


def stats_pass(builder, chunks):
    F = builder.a.size
    n = sum(len(c) for c in chunks)
    s1, s2 = np.zeros(F), np.zeros(F)
    for idx in chunks:
        R = builder.raw(idx)
        s1 += R.sum(axis=1)
        s2 += (R * R).sum(axis=1)
    builder.set_stats(s1, s2, n)


def full_banksy(X, nbr0, w0, nbr1, c1, lam):
    """The stored BANKSY matrix (only possible because the example is small)."""
    n = X.shape[1]
    W0 = csc_matrix((w0.ravel(), (nbr0.ravel(), np.repeat(np.arange(n), K0))),
                    shape=(n, n))
    C1 = csc_matrix((c1.ravel(), (nbr1.ravel(), np.repeat(np.arange(n), K1))),
                    shape=(n, n))
    own = X
    h0 = np.abs((W0.T @ X.T).T)
    h1 = np.abs((C1.T @ X.T.astype(complex)).T)

    def zscore(A):
        mu = A.mean(axis=1, keepdims=True)
        sd = A.std(axis=1, ddof=1, keepdims=True)
        return (A - mu) / np.where(sd == 0, 1.0, sd)

    a = np.sqrt([1 - lam, 2 * lam / 3, lam / 3])
    Z = np.vstack([a[0] * zscore(own), a[1] * zscore(h0), a[2] * zscore(h1)])
    return Z, own, h0, h1


# ----------------------------------------------------------------------------
# 3. Randomized SVD
# ----------------------------------------------------------------------------
def omega_chunk(seed, t, b, ell):
    return np.random.default_rng([seed, t]).standard_normal((b, ell))


def rayleigh_ritz(Q, B, p):
    """Given Q and B = Q^T Z, return loadings, singular values, embeddings."""
    G = B @ B.T
    evals, Ut = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals, Ut = evals[order], Ut[:, order]
    svals = np.sqrt(np.maximum(evals, 0))
    return Q @ Ut[:, :p], svals[:p], B.T @ Ut[:, :p]


def rsvd_inmemory(Z, p, ell, q, seed, chunks, start="cell"):
    F, n = Z.shape
    if start == "cell":
        Omega = np.vstack([omega_chunk(seed, t, len(c), ell)
                           for t, c in enumerate(chunks)])
        Y = Z @ Omega
    else:
        G0 = np.random.default_rng(seed).standard_normal((F, ell))
        Q0, _ = np.linalg.qr(G0)
        Y = Z @ (Z.T @ Q0)
    Q, _ = np.linalg.qr(Y)
    for _ in range(q):
        Y = Z @ (Z.T @ Q)
        Q, _ = np.linalg.qr(Y)
    load, sv, emb = rayleigh_ritz(Q, Q.T @ Z, p)
    return load, sv, emb, Q


def rsvd_chunked(builder, chunks, p, ell, q, seed):
    """The streaming algorithm: every pass rebuilds each chunk from X."""
    F = builder.a.size
    passes = 0

    stats_pass(builder, chunks)                               # pass 1
    passes += 1

    Y = np.zeros((F, ell))                                    # pass 2
    for t, idx in enumerate(chunks):
        Zt = builder.scaled(idx)
        Y += Zt @ omega_chunk(seed, t, len(idx), ell)
    Q, _ = np.linalg.qr(Y)
    passes += 1

    for _ in range(q):                                        # power passes
        Y = np.zeros((F, ell))
        for idx in chunks:
            Zt = builder.scaled(idx)
            Y += Zt @ (Zt.T @ Q)
        Q, _ = np.linalg.qr(Y)
        passes += 1

    Bt = []                                                   # final pass
    G = np.zeros((ell, ell))
    for idx in chunks:
        Zt = builder.scaled(idx)
        Bc = Q.T @ Zt
        Bt.append(Bc)
        G += Bc @ Bc.T
    passes += 1
    evals, Ut = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals, Ut = evals[order], Ut[:, order]
    sv = np.sqrt(np.maximum(evals[:p], 0))
    emb = np.vstack([Bc.T @ Ut[:, :p] for Bc in Bt])
    return Q @ Ut[:, :p], sv, emb, Q, passes


def gram_chunked(builder, chunks, p):
    """The exact alternative: feature covariance in one pass, then project."""
    F = builder.a.size
    G = np.zeros((F, F))
    for idx in chunks:                                        # pass 2
        Zt = builder.scaled(idx)
        G += Zt @ Zt.T
    evals, V = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1][:p]
    load = V[:, order]
    sv = np.sqrt(np.maximum(evals[order], 0))
    emb = np.vstack([builder.scaled(idx).T @ load for idx in chunks])  # pass 3
    return load, sv, emb


def monitor_run(Z, p, ell, n_power, seed, chunks, S):
    """Free stopping monitor. In power pass i we have Q (from before the pass)
    and Y = Z Z^T Q. The small matrix Q^T Y gives the top-p Ritz values of
    span(Q) at no extra cost; their sum is the variance the current estimate
    would capture. Monitor = relative increase since the previous pass."""
    Omega = np.vstack([omega_chunk(seed, t, len(c), ell)
                       for t, c in enumerate(chunks)])
    Q, _ = np.linalg.qr(Z @ Omega)
    rows, prev = [], None
    best = float(np.sum(S[:p] ** 2))
    for i in range(1, n_power + 1):
        Y = Z @ (Z.T @ Q)
        M = Q.T @ Y
        evals = np.sort(np.linalg.eigvalsh((M + M.T) / 2))[::-1]
        cap = float(np.sum(evals[:p]))
        if prev is not None:
            rows.append({"pass": i, "monitor": (cap - prev) / cap,
                         "true_shortfall": 1 - cap / best})
        prev = cap
        Q, _ = np.linalg.qr(Y)
    return rows


# ----------------------------------------------------------------------------
# 4. Metrics
# ----------------------------------------------------------------------------
def sin_theta(U_true, U_est):
    """Sine of the largest principal angle between span(U_true), span(U_est)."""
    R = U_true - U_est @ (U_est.T @ U_true)
    return float(np.linalg.norm(R, 2))


def knn_sets(emb, k):
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(emb)
    _, idx = nn.kneighbors(emb)
    return idx[:, 1:]


def jaccard(idx_a, idx_b):
    tot = 0.0
    for a, b in zip(idx_a, idx_b):
        sa, sb = set(a), set(b)
        tot += len(sa & sb) / len(sa | sb)
    return tot / len(idx_a)


def louvain_labels(knn_idx, resolution, seed, prune=1 / 15):
    """Seurat-style clustering: shared-nearest-neighbour (Jaccard) weights on
    the kNN edges, prune weak edges, then Louvain."""
    n = knn_idx.shape[0]
    sets = [set(row.tolist()) | {i} for i, row in enumerate(knn_idx)]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in knn_idx[i]:
            j = int(j)
            w = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
            if w >= prune:
                G.add_edge(i, j, weight=w)
    comms = nx.community.louvain_communities(G, weight="weight",
                                             resolution=resolution, seed=seed)
    lab = np.empty(n, dtype=int)
    for c, members in enumerate(comms):
        lab[list(members)] = c
    return lab


def fmt_sci(v):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    if -2 <= e <= 0:
        return f"{v:.3f}" if e >= -1 else f"{v:.4f}"
    mant = v / 10 ** e
    return f"{mant:.1f}\\times 10^{{{e}}}"


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    results = {}
    rng = np.random.default_rng(2026)
    locs, domain, ctype, X, counts = simulate(rng)
    sparsity = float((counts == 0).mean())

    nbr0, w0, _ = knn_median(locs, K0)
    nbr1, w1, phi1 = knn_median(locs, K1)
    c1 = agf_coefficients(w1, phi1)

    # Check: the one-complex-sum form equals the textbook centred AGF
    j = np.arange(0, N_CELLS, 997)
    direct = np.empty((N_GENES, len(j)))
    for col, cell in enumerate(j):
        xi = X[:, nbr1[cell]]                       # g x k
        xbar = xi.mean(axis=1, keepdims=True)       # unweighted neighbour mean
        direct[:, col] = np.abs(((xi - xbar) * (w1[cell] *
                                 np.exp(1j * phi1[cell]))).sum(axis=1))
    compact = np.abs(np.einsum("gbk,bk->gb", X[:, nbr1[j]], c1[j]))
    results["agf_form_maxdiff"] = float(np.abs(direct - compact).max())
    results["agf_coef_sum_max"] = float(np.abs(c1.sum(axis=1)).max())

    Z, own, h0, h1 = full_banksy(X, nbr0, w0, nbr1, c1, LAMBDA)
    F = Z.shape[0]

    # Exact PCA (dense SVD of the stored matrix)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    emb_exact = Vt[:P].T * S[:P]
    best_var = float(np.sum(S[:P] ** 2))
    results["svals"] = S[:150].tolist()
    for a, b in [(61, 50), (51, 50), (61, 8), (9, 8), (61, 20), (21, 20), (51, 8)]:
        results[f"ratio_{a}_{b}"] = float(S[a - 1] / S[b - 1])

    chunks = [np.arange(s, min(s + CHUNK, N_CELLS))
              for s in range(0, N_CELLS, CHUNK)]

    # --- (a) chunked run versus (b) in-memory run, same random numbers -------
    builder = ChunkBuilder(X, nbr0, w0, nbr1, c1, LAMBDA)
    q_demo = 2
    load_c, sv_c, emb_c, Q_c, passes = rsvd_chunked(builder, chunks, P,
                                                    ELL_MAIN, q_demo,
                                                    SEEDS[0])
    results["chunk_vs_stored_Z_maxdiff"] = float(
        np.abs(builder.scaled(chunks[3]) - Z[:, chunks[3]]).max())
    load_m, sv_m, emb_m, Q_m = rsvd_inmemory(Z, P, ELL_MAIN, q_demo,
                                             SEEDS[0], chunks)
    results["chunked_passes"] = passes
    results["chunked_vs_inmem_sv_reldiff"] = float(
        np.max(np.abs(sv_c - sv_m) / sv_m))
    results["chunked_vs_inmem_subspace"] = sin_theta(load_m, load_c)

    # --- The exact alternative: feature covariance ---------------------------
    load_g, sv_g, emb_g = gram_chunked(builder, chunks, P)
    results["gram_sv_relerr"] = float(np.max(np.abs(sv_g - S[:P]) / S[:P]))
    results["gram_sin50"] = sin_theta(U[:, :P], load_g)

    # --- Reference: how much does the kNN graph depend on the number of PCs?
    knn_exact = knn_sets(emb_exact, KNN_EMB)
    ref_jac = {}
    for k in [20, 30, 40, 45]:
        ref_jac[k] = jaccard(knn_exact, knn_sets(Vt[:k].T * S[:k], KNN_EMB))
    results["ref_jaccard_fewer_pcs"] = ref_jac
    results["gram_jaccard"] = jaccard(knn_exact, knn_sets(emb_g, KNN_EMB))

    # --- Parameter sweep ----------------------------------------------------
    sweep = []
    for ell in ELLS:
        for q in QS:
            for seed in SEEDS:
                load, sv, emb, Q = rsvd_inmemory(Z, P, ell, q, seed, chunks)
                sweep.append({
                    "ell": ell, "q": q, "seed": seed, "passes": q + 3,
                    "sin8": sin_theta(U[:, :8], load[:, :8]),
                    "sin20": sin_theta(U[:, :20], load[:, :20]),
                    "sin50": sin_theta(U[:, :50], load[:, :50]),
                    "cap8": sin_theta(U[:, :8], Q),
                    "cap50": sin_theta(U[:, :50], Q),
                    "sv8": float(np.max(np.abs(sv[:8] - S[:8]) / S[:8])),
                    "sv50": float(np.max(np.abs(sv[:50] - S[:50]) / S[:50])),
                    "short50": 1 - float(np.sum(sv ** 2)) / best_var,
                    "jaccard": jaccard(knn_exact, knn_sets(emb, KNN_EMB))})
    results["sweep"] = sweep

    def med(ell, q, key):
        vals = [r[key] for r in sweep if r["ell"] == ell and r["q"] == q]
        return float(np.median(vals)), float(np.min(vals)), float(np.max(vals))

    # Start in cell space (HMT) versus start in feature space, same passes
    start_cmp = []
    for npass in [3, 4, 5, 6, 7]:   # sampling passes after the stats pass
        q_cell, q_feat = npass - 2, npass - 2
        rc = [rsvd_inmemory(Z, P, ELL_MAIN, q_cell, s, chunks, "cell")
              for s in SEEDS]
        rf = [rsvd_inmemory(Z, P, ELL_MAIN, q_feat, s, chunks, "feature")
              for s in SEEDS]
        start_cmp.append({
            "passes_total": npass + 1,
            "cell_sin8": float(np.median([sin_theta(U[:, :8], r[0][:, :8])
                                           for r in rc])),
            "feat_sin8": float(np.median([sin_theta(U[:, :8], r[0][:, :8])
                                           for r in rf])),
            "cell_short": float(np.median([1 - np.sum(r[1] ** 2) / best_var
                                           for r in rc])),
            "feat_short": float(np.median([1 - np.sum(r[1] ** 2) / best_var
                                           for r in rf]))})
    results["start_compare"] = start_cmp

    mon_rows = monitor_run(Z, P, ELL_MAIN, 9, SEEDS[0], chunks, S)
    results["monitor"] = mon_rows

    # --- Clustering agreement (Louvain on the kNN graph in PC space) --------
    # Fixed resolution: stable on the exact embedding (seed-to-seed ARI
    # 0.82-0.98). The simulated tissue has spatial structure beyond the five
    # labelled domains (the smooth programs), so ARI against the labels is not
    # a meaningful yardstick here; we compare approximate against exact.
    res_used = 0.1
    lab_exact = louvain_labels(knn_exact, res_used, 0)
    ari_truth_exact = adjusted_rand_score(domain, lab_exact)
    nclust = len(np.unique(lab_exact))
    seed_floor = [adjusted_rand_score(lab_exact,
                                      louvain_labels(knn_exact, res_used, s))
                  for s in [1, 2, 3, 4]]
    lab30 = louvain_labels(knn_sets(Vt[:30].T * S[:30], KNN_EMB), res_used, 0)
    results.update({"louvain_resolution": res_used,
                    "ari_exact_vs_domains": ari_truth_exact,
                    "n_clusters_exact": nclust,
                    "ari_louvain_seed_floor": seed_floor,
                    "ari_exact30_vs_exact50": adjusted_rand_score(lab_exact,
                                                                  lab30)})
    clus = []
    for q in [0, 1, 2, 4]:
        aris, aris_truth = [], []
        for seed in SEEDS:
            emb = rsvd_inmemory(Z, P, ELL_MAIN, q, seed, chunks)[2]
            lab = louvain_labels(knn_sets(emb, KNN_EMB), res_used, 0)
            aris.append(adjusted_rand_score(lab_exact, lab))
            aris_truth.append(adjusted_rand_score(domain, lab))
        clus.append({"q": q, "ari_vs_exact": float(np.median(aris)),
                     "ari_vs_exact_min": float(np.min(aris)),
                     "ari_vs_domains": float(np.median(aris_truth))})
    results["clustering"] = clus

    # --- Lanczos-type solver: how many products? ----------------------------
    counter = {"mv": 0, "rmv": 0}

    def mv(v):
        counter["mv"] += 1
        return Z @ v

    def rmv(u):
        counter["rmv"] += 1
        return Z.T @ u

    op = LinearOperator(Z.shape, matvec=mv, rmatvec=rmv, dtype=float)
    _, s_arp, _ = svds(op, k=P, tol=1e-5, solver="arpack",
                       random_state=np.random.default_rng(0))
    s_arp = np.sort(s_arp)[::-1]
    results["arpack_matvec"] = counter["mv"]
    results["arpack_rmatvec"] = counter["rmv"]
    results["arpack_sv50_relerr"] = float(np.max(np.abs(s_arp - S[:P]) / S[:P]))
    results["runtime_s"] = time.time() - t0

    # ------------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------------
    genes = slice(120 + 8 * 4, 120 + 8 * 5)   # the 8 domain genes of D5
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05))
    names = ["D1", "D2", "D3", "D4", "D5"]
    ax = axes[0]
    for d in range(5):
        m = domain == d
        ax.scatter(locs[m, 0], locs[m, 1], s=0.5, c=CAT[d], lw=0,
                   rasterized=True)
    label_pos = {0: (20, 10), 1: (20, 37), 2: (35, 62), 3: (40, 90),
                 4: (68, 40)}
    for d in range(5):
        ax.text(*label_pos[d], names[d], ha="center", va="center",
                fontsize=7.5, color=INK, weight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                          alpha=0.85))
    ax.set_title("(a) domains")
    panels = [(own[genes].sum(axis=0), "(b) own: D5 genes"),
              (h0[genes].sum(axis=0), "(c) H0: neighbour mean"),
              (h1[genes].sum(axis=0), "(d) H1: AGF")]
    for ax, (vals, title) in zip(axes[1:], panels):
        order = np.argsort(vals)
        ax.scatter(locs[order, 0], locs[order, 1], s=0.8, c=vals[order],
                   cmap=SEQ_CMAP, lw=0, rasterized=True)
        ax.set_title(title)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "fig_tissue")

    # Spectrum
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    idx = np.arange(1, 121)
    ax.semilogy(idx, S[:120], color=CAT[0], marker="o", ms=2.4, lw=1.2)
    for xline, lab, side in [(P, r"$p=50$", "right"),
                             (ELL_MAIN + 1, r"$\ell+1=61$", "left")]:
        ax.axvline(xline, color=MUTED, lw=0.9, ls="--")
        ax.text(xline + (-1.5 if side == "right" else 1.5), S[0], lab,
                color=INK2, fontsize=7.5, va="top", ha=side)
    ax.text(14, S[3], "signal PCs", color=INK2, fontsize=7.5, va="center")
    ax.text(118, S[40] * 1.12, "noise bulk: nearly flat", color=INK2,
            fontsize=7.5, ha="right", va="bottom")
    ax.set_xlabel(r"index $i$")
    ax.set_ylabel(r"singular value $\sigma_i$ (log scale)")
    ax.set_xlim(0, 121)
    save(fig, "fig_spectrum")

    # Convergence
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    ax = axes[0]
    for c, (key, lab) in zip(CAT[:3], [("sin8", "top 8 PCs"),
                                       ("sin20", "top 20 PCs"),
                                       ("sin50", "top 50 PCs")]):
        m = np.array([med(ELL_MAIN, q, key) for q in QS])
        ax.fill_between(QS, m[:, 1], m[:, 2], color=c, alpha=0.15, lw=0)
        ax.semilogy(QS, m[:, 0], color=c, marker="o", label=lab)
        ax.text(QS[-1] + 0.3, m[-1, 0], lab, color=INK2, fontsize=7.5,
                va="center")
    ax.set_title(r"$\ell=60$: leading PCs converge first")
    ax.set_xlabel(r"power iterations $q$")
    ax.set_ylabel("sin of largest angle to exact subspace")
    ax.set_xlim(-0.3, QS[-1] + 3.2)
    ax.legend(loc="lower left")
    ax = axes[1]
    for c, ell in zip(ORDINAL, ELLS):
        m = np.array([med(ell, q, "sin8") for q in QS])
        ax.semilogy(QS, m[:, 0], color=c, marker="o", label=rf"$\ell={ell}$")
    ax.set_title("top 8 PCs: oversampling speeds it up")
    ax.set_xlabel(r"power iterations $q$")
    ax.set_xlim(-0.3, QS[-1] + 0.5)
    ax.legend(loc="lower left")
    save(fig, "fig_convergence")

    # Captured variance
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for c, ell in zip(ORDINAL, ELLS):
        m = np.array([med(ell, q, "short50") for q in QS])
        ax.semilogy(QS, m[:, 0], color=c, marker="o", label=rf"$\ell={ell}$")
    ax.set_xlabel(r"power iterations $q$")
    ax.set_ylabel("variance shortfall, top 50")
    ax.legend(loc="upper right")
    save(fig, "fig_capvar")

    # Theory: capture angle versus the predicted rate
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for c, (key, k) in zip(CAT[:2], [("cap8", 8), ("cap50", 50)]):
        m = np.array([med(ELL_MAIN, q, key) for q in QS])
        rate = S[ELL_MAIN] / S[k - 1]
        pred = m[0, 0] * rate ** (2 * np.array(QS, dtype=float))
        ax.semilogy(QS, m[:, 0], color=c, marker="o",
                    label=f"measured, top {k}")
        ax.semilogy(QS, pred, color=c, ls="--", lw=1.1,
                    label=rf"predicted slope $(\sigma_{{61}}/\sigma_{{{k}}})^{{2q+1}}$")
    ax.set_xlabel(r"power iterations $q$")
    ax.set_ylabel(r"$\|(I-QQ^{\top})\,U_k\|$")
    ax.set_ylim(1e-5, 2)
    ax.legend(loc="lower left")
    save(fig, "fig_theory")

    # kNN graph overlap, with the fewer-exact-PCs reference
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    for c, ell in zip(ORDINAL, ELLS):
        m = np.array([med(ell, q, "jaccard") for q in QS])
        ax.plot(QS, m[:, 0], color=c, marker="o", label=rf"$\ell={ell}$")
    for k, x0, ha, dy, va in [(40, 0, "left", 0.012, "bottom"),
                              (30, QS[-1], "right", -0.012, "top")]:
        ax.axhline(ref_jac[k], color=MUTED, lw=0.9, ls="--")
        ax.text(x0, ref_jac[k] + dy, f"exact, {k} PCs instead of 50",
                color=INK2, fontsize=7, ha=ha, va=va)
    ax.set_xlabel(r"power iterations $q$")
    ax.set_ylabel("mean kNN Jaccard vs exact")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    save(fig, "fig_knn")

    # ------------------------------------------------------------------------
    # Numbers for the LaTeX document
    # ------------------------------------------------------------------------
    words = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 6: "Six",
             8: "Eight"}
    macros = {
        "demoN": f"{N_CELLS:,}".replace(",", "{,}"),
        "demoG": str(N_GENES),
        "demoF": f"{F:,}".replace(",", "{,}"),
        "demoSparsity": f"{100 * sparsity:.0f}",
        "demoChunk": str(CHUNK),
        "demoNChunks": str(len(chunks)),
        "demoPasses": str(passes),
        "demoChunkZdiff": fmt_sci(results["chunk_vs_stored_Z_maxdiff"]),
        "demoChunkSVdiff": fmt_sci(results["chunked_vs_inmem_sv_reldiff"]),
        "demoChunkSubdiff": fmt_sci(results["chunked_vs_inmem_subspace"]),
        "demoAgfFormDiff": fmt_sci(results["agf_form_maxdiff"]),
        "demoAgfCoefSum": fmt_sci(results["agf_coef_sum_max"]),
        "demoGramSV": fmt_sci(results["gram_sv_relerr"]),
        "demoGramSin": fmt_sci(results["gram_sin50"]),
        "demoGramJac": f"{results['gram_jaccard']:.4f}",
        "demoRatioSixtyOneFifty": f"{results['ratio_61_50']:.3f}",
        "demoRatioFiftyOneFifty": f"{results['ratio_51_50']:.3f}",
        "demoRatioSixtyOneEight": f"{results['ratio_61_8']:.3f}",
        "demoRatioFiftyOneEight": f"{results['ratio_51_8']:.3f}",
        "demoRatioNineEight": f"{results['ratio_9_8']:.3f}",
        "demoRatioSixtyOneTwenty": f"{results['ratio_61_20']:.3f}",
        "demoRatioTwentyOneTwenty": f"{results['ratio_21_20']:.3f}",
        "demoArpackMV": str(results["arpack_matvec"]),
        "demoArpackRMV": str(results["arpack_rmatvec"]),
        "demoArpackTotal": str(results["arpack_matvec"] +
                               results["arpack_rmatvec"]),
        "demoLouvainRes": f"{res_used}",
        "demoNClust": str(nclust),
        "demoAriTruth": f"{ari_truth_exact:.2f}",
        "demoAriSeedFloorMin": f"{min(seed_floor):.2f}",
        "demoAriSeedFloorMax": f"{max(seed_floor):.2f}",
        "demoAriThirtyFifty": f"{results['ari_exact30_vs_exact50']:.2f}",
        "demoRefJacThirty": f"{ref_jac[30]:.2f}",
        "demoRefJacForty": f"{ref_jac[40]:.2f}",
        "demoRefJacFortyFive": f"{ref_jac[45]:.2f}",
        "demoRuntime": f"{results['runtime_s']:.0f}",
        "demoSigmaOne": f"{S[0]:.0f}",
        "demoSigmaEight": f"{S[7]:.1f}",
        "demoSigmaNine": f"{S[8]:.1f}",
        "demoSigmaFifty": f"{S[49]:.1f}",
        "demoSigmaSixtyOne": f"{S[60]:.1f}",
    }
    for q in QS:
        w = words[q]
        macros[f"demoJacQ{w}"] = f"{med(ELL_MAIN, q, 'jaccard')[0]:.2f}"
        macros[f"demoSinEightQ{w}"] = fmt_sci(med(ELL_MAIN, q, "sin8")[0])
        macros[f"demoSinFiftyQ{w}"] = fmt_sci(med(ELL_MAIN, q, "sin50")[0])
        macros[f"demoShortQ{w}"] = fmt_sci(med(ELL_MAIN, q, "short50")[0])
    for r in clus:
        w = words[r["q"]]
        macros[f"demoAriQ{w}"] = f"{r['ari_vs_exact']:.2f}"
        macros[f"demoAriQ{w}Min"] = f"{r['ari_vs_exact_min']:.2f}"
        macros[f"demoAriTruthQ{w}"] = f"{r['ari_vs_domains']:.2f}"
    with open(os.path.join(GEN, "numbers.tex"), "w") as fh:
        fh.write("% generated by demo/agf_rsvd_demo.py -- do not edit\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    lines = []
    for q in QS:
        vals = [med(ELL_MAIN, q, k)[0] for k in
                ("sin8", "sin20", "sin50", "sv8", "sv50", "short50")]
        lines.append(f"{q} & {q + 3} & " +
                     " & ".join(f"${fmt_sci(v)}$" for v in vals) +
                     f" & {med(ELL_MAIN, q, 'jaccard')[0]:.2f} \\\\")
    with open(os.path.join(GEN, "tab_convergence.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    lines = [f"{r['passes_total']} & ${fmt_sci(r['cell_sin8'])}$ & "
             f"${fmt_sci(r['feat_sin8'])}$ & ${fmt_sci(r['cell_short'])}$ & "
             f"${fmt_sci(r['feat_short'])}$ \\\\" for r in start_cmp]
    with open(os.path.join(GEN, "tab_start.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    lines = [f"{r['pass']} & ${fmt_sci(r['monitor'])}$ & "
             f"${fmt_sci(r['true_shortfall'])}$ \\\\" for r in mon_rows]
    with open(os.path.join(GEN, "tab_monitor.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    lines = [f"{r['q']} & {r['q'] + 3} & {r['ari_vs_exact']:.2f} & "
             f"{r['ari_vs_exact_min']:.2f} \\\\" for r in clus]
    with open(os.path.join(GEN, "tab_cluster.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    with open(os.path.join(GEN, "results.json"), "w") as fh:
        json.dump(results, fh, indent=1, default=float)

    # Console summary
    print(f"F={F}, sparsity of counts: {sparsity:.3f}")
    print(f"AGF compact-form check: {results['agf_form_maxdiff']:.2e}, "
          f"coef sums {results['agf_coef_sum_max']:.2e}")
    print(f"chunk Z vs stored Z: {results['chunk_vs_stored_Z_maxdiff']:.2e}")
    print(f"chunked vs in-memory: sv {results['chunked_vs_inmem_sv_reldiff']:.2e}"
          f", subspace {results['chunked_vs_inmem_subspace']:.2e}, "
          f"passes {passes}")
    print(f"gram: sv {results['gram_sv_relerr']:.1e}, sin50 "
          f"{results['gram_sin50']:.1e}, jaccard {results['gram_jaccard']:.4f}")
    print("top singular values:", np.round(S[:30], 1))
    print({k: round(v, 3) for k, v in results.items() if k.startswith("ratio")})
    print("reference jaccard (fewer exact PCs):",
          {k: round(v, 3) for k, v in ref_jac.items()})
    for q in QS:
        print(f"ell=60 q={q}: sin8 {med(60, q, 'sin8')[0]:.2e} "
              f"sin20 {med(60, q, 'sin20')[0]:.2e} "
              f"sin50 {med(60, q, 'sin50')[0]:.2e} "
              f"sv8 {med(60, q, 'sv8')[0]:.1e} "
              f"sv50 {med(60, q, 'sv50')[0]:.1e} "
              f"short {med(60, q, 'short50')[0]:.1e} "
              f"jac {med(60, q, 'jaccard')[0]:.3f}")
    for ell in ELLS:
        print(f"ell={ell}: sin8 by q", [f"{med(ell, q, 'sin8')[0]:.1e}"
                                         for q in QS])
    print("start compare:", start_cmp)
    print("monitor:", mon_rows)
    print(f"louvain res {res_used}, clusters {nclust}, ARI vs domains "
          f"{ari_truth_exact:.2f}, seed floor {np.round(seed_floor, 3)}, "
          f"30 vs 50 PCs {results['ari_exact30_vs_exact50']:.3f}")
    print("clustering:", clus)
    print(f"ARPACK products: {counter['mv']} + {counter['rmv']}, "
          f"sv err {results['arpack_sv50_relerr']:.1e}")
    print(f"runtime {results['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
