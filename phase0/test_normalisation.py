"""
Test normalization strategies for HVG/PCA embedding.

Context:
  PMC-E (Evren lab, 1500-3000ng loading) has HVG top-100 = 0.008 -- essentially random.
  PMC-A = 0.213, PMC-B = 0.101, PMC-D = 0.040. Cell-line cosine separation = 0.037.
  The paper (Tesorai Search) gets HVG = 0.665.

  The Tahoe-x1 model uses per-sample quantile binning before training.
  The paper likely applies the same normalization to its HVG/PCA baselines.

Ideas tested:
  1. Raw log2 LFQ (current -- baseline to beat)
  2. Per-sample median centering  
  3. Per-sample z-score (detected proteins only)
  4. Per-sample quantile rank (0-1, like Tahoe binning)
  5. ComBat-style mean shift batch correction (PMC-E as one batch)
  6. HVG selected from individual samples, not pair means

Run: pipenv run python test_normalisation.py
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from scipy.stats import rankdata, pearsonr

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from parse_lfq import load_pxd014791
from config import N_HVG, K_NEIGHBORS
from benchmark import _pairwise_complete_knn, _detected_mask, compute_metrics

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
print("Loading data ...")
matrix, meta = load_pxd014791()
drug_meta = meta[~meta["is_control"]]
ctrl_meta  = meta[meta["is_control"]]

# Per-context control means
ctrl_by_ctx = {}
for ctx, grp in ctrl_meta.groupby("donor"):
    samps = grp.index.intersection(matrix.index)
    if len(samps):
        ctrl_by_ctx[ctx] = np.nanmean(matrix.loc[samps].values.astype(float), axis=0)

# Build pair list
from dataclasses import dataclass, field
@dataclass
class Pair:
    drug: str
    context: str
    is_held_out: bool
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    c: np.ndarray = field(default_factory=lambda: np.array([]))
    @property
    def delta(self): return self.r - self.c

from config import PXD014791_HELD_OUT_DRUGS

raw_pairs = []
for (drug, ctx), grp in drug_meta.groupby(["drug", "donor"]):
    if ctx not in ctrl_by_ctx:
        continue
    samps = grp.index.intersection(matrix.index)
    if not len(samps):
        continue
    r = np.nanmean(matrix.loc[samps].values.astype(float), axis=0)
    c = ctrl_by_ctx[ctx]
    raw_pairs.append(Pair(drug, ctx, drug in PXD014791_HELD_OUT_DRUGS, r, c))

held_out = [p for p in raw_pairs if p.is_held_out]
print(f"  {len(held_out)} held-out pairs, {len(raw_pairs)} total")

# ------------------------------------------------------------------
# Normalization functions (applied to r vectors before embedding)
# ------------------------------------------------------------------

def norm_raw(r: np.ndarray) -> np.ndarray:
    """No change -- raw log2 LFQ."""
    return r.copy()

def norm_median(r: np.ndarray) -> np.ndarray:
    """Subtract per-sample median of detected proteins."""
    det = np.isfinite(r)
    out = r.copy()
    if det.sum() > 0:
        out[det] -= np.median(r[det])
        out[~det] = 0.
    return out

def norm_zscore(r: np.ndarray) -> np.ndarray:
    """Z-score across detected proteins; missing -> 0."""
    det = np.isfinite(r)
    out = np.zeros_like(r)
    if det.sum() > 1:
        mu = r[det].mean()
        sd = r[det].std() + 1e-8
        out[det] = (r[det] - mu) / sd
    return out

def norm_quantile(r: np.ndarray) -> np.ndarray:
    """Replace detected values with their quantile rank in [0,1]; missing -> -1."""
    det = np.isfinite(r)
    out = np.full_like(r, -1.0)
    if det.sum() > 0:
        out[det] = rankdata(r[det]) / det.sum()
    return out

def norm_combat(r: np.ndarray, ctx: str, batch_means: dict) -> np.ndarray:
    """Subtract batch mean for each protein, re-add grand mean."""
    out = r.copy()
    if ctx in batch_means:
        bm = batch_means[ctx]
        gm = batch_means.get("__grand__", np.zeros_like(bm))
        det = np.isfinite(r)
        out[det] = r[det] - bm[det] + gm[det]
        out[~det] = 0.
    else:
        out = norm_zscore(r)
    return out

# Compute batch means for ComBat correction
batch_means = {}
for ctx in ["PMC-A", "PMC-B", "PMC-D", "PMC-E"]:
    ctx_pairs = [p for p in raw_pairs if p.context == ctx]
    if ctx_pairs:
        batch_means[ctx] = np.nanmean([p.r for p in ctx_pairs], axis=0)
grand = np.nanmean([p.r for p in raw_pairs], axis=0)
batch_means["__grand__"] = grand

# ------------------------------------------------------------------
# Evaluation function
# ------------------------------------------------------------------

def cosine_separation(pairs_r: np.ndarray, pairs_cl: list[str]) -> float:
    """Cosine similarity: same cell line vs different cell line."""
    m = normalize(np.nan_to_num(pairs_r, nan=0.0), norm="l2")
    sim = m @ m.T
    same, diff = [], []
    for i in range(len(pairs_cl)):
        for j in range(i+1, len(pairs_cl)):
            (same if pairs_cl[i]==pairs_cl[j] else diff).append(sim[i,j])
    return np.mean(same) - np.mean(diff)


def run_hvg_benchmark(pairs: list[Pair], norm_fn, label: str) -> dict:
    """Run HVG kNN benchmark with given normalisation function."""
    held  = [p for p in pairs if p.is_held_out]
    pool  = pairs  # LOOCV: full pool, remove current pair per iteration

    all_r     = np.stack([norm_fn(p.r) for p in pairs])
    all_cl    = [p.context for p in pairs]
    sep       = cosine_separation(all_r, all_cl)

    # HVG selection from all pairs
    variances = np.var(all_r, axis=0)
    hvg_idx   = np.argsort(variances)[-N_HVG:]

    top100_pearson = []
    for pair in held:
        pool_excl = [p for p in pairs if not (p.drug==pair.drug and p.context==pair.context)]
        pool_r = np.stack([norm_fn(p.r) for p in pool_excl])
        query  = norm_fn(pair.r)

        p_i     = _pairwise_complete_knn(query, pool_r, hvg_idx)
        p_raw   = _pairwise_complete_knn(query, np.stack([p.r for p in pool_excl]), hvg_idx)
        # Predict delta using raw pair.c (control is not normalised)
        delta_hat = p_i - norm_fn(pair.c)
        detected  = _detected_mask(pair.r, p_i)
        _, top100 = compute_metrics(delta_hat, pair.delta, detected)
        if np.isfinite(top100):
            top100_pearson.append(top100)

    mean_top100 = np.mean(top100_pearson) if top100_pearson else np.nan
    return {"label": label, "sep": sep, "top100": mean_top100,
            "n": len(top100_pearson)}


# ------------------------------------------------------------------
# Run all normalisations
# ------------------------------------------------------------------
print("\nNormalisation comparison (HVG top-100 DE Pearson + cell-line separation):")
print(f"{'Method':30s}  {'Separation':>12s}  {'top-100 Pearson':>16s}  {'n':>5s}")
print("-" * 70)

results = []
for label, fn in [
    ("raw log2 (current)",          lambda r: norm_raw(r)),
    ("per-sample median centre",    lambda r: norm_median(r)),
    ("per-sample z-score",          lambda r: norm_zscore(r)),
    ("per-sample quantile rank",    lambda r: norm_quantile(r)),
    ("ComBat mean-shift",           None),   # special case
]:
    if label == "ComBat mean-shift":
        def fn_combat(pair):
            return norm_combat(pair.r, pair.context, batch_means)
        # wrap: norm_fn receives pair object
        class _WrapPair:
            """Adapter so norm_fn(p.r) works for most but fn_combat needs p."""
            pass
        # For ComBat we need context; pass pair object differently
        held  = [p for p in raw_pairs if p.is_held_out]
        all_r_cb  = np.stack([norm_combat(p.r, p.context, batch_means) for p in raw_pairs])
        all_cl_cb = [p.context for p in raw_pairs]
        sep_cb    = cosine_separation(all_r_cb, all_cl_cb)
        variances_cb = np.var(all_r_cb, axis=0)
        hvg_cb   = np.argsort(variances_cb)[-N_HVG:]
        top100s  = []
        for pair in held:
            pool_excl = [p for p in raw_pairs if not (p.drug==pair.drug and p.context==pair.context)]
            pool_r_cb = np.stack([norm_combat(p.r, p.context, batch_means) for p in pool_excl])
            query_cb  = norm_combat(pair.r, pair.context, batch_means)
            p_i = _pairwise_complete_knn(query_cb, pool_r_cb, hvg_cb)
            delta_hat = p_i - norm_combat(pair.c, pair.context, batch_means)
            detected  = _detected_mask(pair.r, p_i)
            _, top100 = compute_metrics(delta_hat, pair.delta, detected)
            if np.isfinite(top100): top100s.append(top100)
        mean_t = np.mean(top100s) if top100s else np.nan
        print(f"{'ComBat mean-shift':30s}  {sep_cb:12.4f}  {mean_t:16.4f}  {len(top100s):5d}")
        results.append({"label":"ComBat","sep":sep_cb,"top100":mean_t})
        continue

    res = run_hvg_benchmark(raw_pairs, fn, label)
    print(f"{label:30s}  {res['sep']:12.4f}  {res['top100']:16.4f}  {res['n']:5d}")
    results.append(res)

print("-" * 70)
print(f"{'Paper (Tesorai Search)':30s}  {'(unknown)':>12s}  {0.665:16.4f}")

# ------------------------------------------------------------------
# Additional: use individual samples (not pair means) for HVG selection
# ------------------------------------------------------------------
print("\n--- Bonus: HVG selected from individual samples (not pair means) ---")
print("(Paper says 'embed all available samples then aggregate' -- variance")
print(" from 317 samples may give better cell-line signal than 61 pair means)")

indiv_r  = matrix.loc[drug_meta.index.intersection(matrix.index)].values.astype(float)
indiv_cl = [drug_meta.loc[s, "donor"] if s in drug_meta.index else "?" 
             for s in matrix.index if s in drug_meta.index]

# z-score each sample
indiv_z  = np.stack([norm_zscore(r) for r in indiv_r])
var_indiv = np.var(indiv_z, axis=0)
hvg_indiv = np.argsort(var_indiv)[-N_HVG:]

# Now re-run HVG benchmark using these proteins with z-scored pair means
top100s = []
for pair in [p for p in raw_pairs if p.is_held_out]:
    pool_excl = [p for p in raw_pairs if not (p.drug==pair.drug and p.context==pair.context)]
    pool_r    = np.stack([norm_zscore(p.r) for p in pool_excl])
    query     = norm_zscore(pair.r)
    p_i       = _pairwise_complete_knn(query, pool_r, hvg_indiv)
    delta_hat = p_i - norm_zscore(pair.c)
    detected  = _detected_mask(pair.r, p_i)
    _, top100 = compute_metrics(delta_hat, pair.delta, detected)
    if np.isfinite(top100): top100s.append(top100)

print(f"  HVG from 317 individual z-scored samples, pair-mean z-score embedding:")
print(f"  top-100 DE Pearson = {np.mean(top100s):.4f}  (n={len(top100s)})")
