"""
Diagnose why perturbation-mean dropped from 0.518 -> 0.477 after adding PMC-D,
and check if HVG cosine similarity actually separates cell lines.

Run: pipenv run python diagnose_benchmark.py
"""

import sys, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize

sys.path.insert(0, ".")
from parse_lfq import load_pxd014791
from config import PXD014791_HELD_OUT_DRUGS, N_HVG

warnings.filterwarnings("ignore")

# ---- 1. Per-pair breakdown from results CSV ----
results_path = Path("results/pxd014791_per_pair.csv")
if results_path.exists():
    r = pd.read_csv(results_path)
    print("=" * 60)
    print("1. Per-pair top-100 Pearson by cell line")
    print("=" * 60)
    for bl in ["perturbation-mean", "additive-linear", "HVG", "PCA"]:
        sub = r[r["baseline"] == bl]
        by_ctx = sub.groupby("context")["pearson_top100"].agg(["mean","count"]).round(3)
        print(f"\n  {bl}:")
        print(by_ctx.to_string())
    
    print("\n" + "=" * 60)
    print("2. Worst 10 perturbation-mean pairs")
    print("=" * 60)
    pm = r[r["baseline"] == "perturbation-mean"].nsmallest(10, "pearson_top100")
    print(pm[["drug","context","pearson_top100","pearson_full"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("3. Trametinib pairs specifically")
    print("=" * 60)
    tram = r[r["drug"] == "Trametinib"]
    print(tram[["baseline","context","pearson_top100","pearson_full"]].to_string(index=False))
else:
    print("results/pxd014791_per_pair.csv not found -- run run_phase0.py first")

# ---- 2. Spot-check metadata->column mapping ----
print("\n" + "=" * 60)
print("4. Spot-check: 5 metadata rows vs LFQ column lookup")
print("=" * 60)
xl = pd.read_excel("data/PXD014791/Metadata-Table.xlsx")
pg = pd.read_csv("data/PXD014791/proteinGroups.txt", sep="\t", low_memory=False, nrows=0)
lfq_cols_lower = {c.replace("LFQ intensity ", "").strip().lower(): c
                   for c in pg.columns if c.startswith("LFQ intensity ")}

sample_rows = xl.sample(5, random_state=42)
for _, row in sample_rows.iterrows():
    fname = row["Raw Mass Spectrometry File"].replace(".raw","").strip()
    key = fname.lower()
    found = key in lfq_cols_lower
    print(f"  {'OK' if found else 'MISS'} Drug={row['Drug Treatment']:5s} CL={row['Cell Line']:6s}  {fname[:55]}")

# ---- 3. HVG cosine similarity: same vs different cell line ----
print("\n" + "=" * 60)
print("5. HVG cosine similarity: same vs different cell line")
print("   (if same > different, kNN is correctly capturing cell identity)")
print("=" * 60)

matrix, meta = load_pxd014791()
drug_meta = meta[~meta["is_control"]]

# Build pair-level means
pairs = {}
for (drug, ctx), grp in drug_meta.groupby(["drug", "donor"]):
    samps = grp.index.intersection(matrix.index)
    if len(samps) == 0:
        continue
    r = np.nanmean(matrix.loc[samps].values.astype(float), axis=0)
    pairs[(drug, ctx)] = {"r": r, "cell_line": ctx}

pair_keys = list(pairs.keys())
pair_r = np.stack([pairs[k]["r"] for k in pair_keys])
pair_cl = [pairs[k]["cell_line"] for k in pair_keys]

# Select HVG by nanvar
variances = np.nanvar(pair_r, axis=0)
variances = np.where(np.isfinite(variances), variances, 0.0)
hvg_idx = np.argsort(variances)[-N_HVG:]

# 0-impute and L2-normalize
pair_r_hvg = np.nan_to_num(pair_r[:, hvg_idx], nan=0.0)
pair_r_hvg_normed = normalize(pair_r_hvg, norm="l2")

# Compute cosine similarity matrix
sim_matrix = pair_r_hvg_normed @ pair_r_hvg_normed.T

same_sims, diff_sims = [], []
for i in range(len(pair_keys)):
    for j in range(i+1, len(pair_keys)):
        s = sim_matrix[i, j]
        if pair_cl[i] == pair_cl[j]:
            same_sims.append(s)
        else:
            diff_sims.append(s)

print(f"  Same cell line pairs:  n={len(same_sims):4d}  mean={np.mean(same_sims):.4f}  std={np.std(same_sims):.4f}")
print(f"  Diff cell line pairs:  n={len(diff_sims):4d}  mean={np.mean(diff_sims):.4f}  std={np.std(diff_sims):.4f}")
sep = np.mean(same_sims) - np.mean(diff_sims)
print(f"  Separation (same-diff): {sep:.4f}")
if sep < 0.02:
    print("  *** POOR SEPARATION -- HVG is NOT capturing cell line identity ***")
    print("  This explains the low kNN performance, not data quality.")
else:
    print("  Good separation -- kNN should be correctly finding same-cell-line neighbors.")
    print("  Low HVG/PCA Pearson is then a data quality issue (MaxQuant vs Tesorai Search).")