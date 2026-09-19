"""
Protein perturbation benchmark -- Phase 0 (baseline verification).

Implements the 5 baselines from Tesorai paper Figure 3:
  1. control-mean      -- predict no change
  2. perturbation-mean -- predict average perturbed proteome of training pairs
  3. HVG               -- cosine-kNN on 2000 highest-variance proteins
  4. PCA               -- cosine-kNN on 512-component PCA embedding
  5. additive-linear   -- two-way additive decomposition of drug + context effects

Protocol (Section 2.3 of paper):
  - Leave-one-pair-out: a "pair" = (drug, biological_context)
  - Held-out pairs = pairs whose drug is in the held-out drug set
  - For each held-out pair i:
      r_i = mean log-abundance of drug-treated samples in pair i
      c_i = mean log-abundance of matched controls in context(i)
      delta_i = r_i - c_i  (measured log-fold-change)
      Predict delta_hat_i using each baseline
  - Detected proteins: np.isfinite in EITHER r_i OR predicted proteome p_i
  - Metrics:
      Full-proteome Pearson on detected proteins
      Top-100 DE Pearson on proteins with 100 largest |delta_i| among detected

Paper-reported baseline values to verify against (Figure 3):
  Dataset           Baseline          Top-100 DE    Full-proteome
  PXD014791         control-mean      NaN           0.000
  PXD014791         perturbation-mean ~0.54         ~0.27
  PXD014791         HVG               ~0.54         ~0.26   (these are rough reads from Figure 3)
  PXD014791         PCA               ~0.54         ~0.26
  PXD014791         additive-linear   ~0.54         ~0.26
  ProTargetMiner    HVG               -0.652        -0.658  *** anomalously negative ***
  ProTargetMiner    PCA               -0.650        -0.623  *** anomalously negative ***
"""

import logging
import warnings
warnings.filterwarnings("ignore", "Mean of empty slice")
warnings.filterwarnings("ignore", "Degrees of freedom")
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from config import K_NEIGHBORS, N_HVG, N_PCA, TOP_N_DE, MIN_PROTEINS_PER_PAIR

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class Pair:
    """One (drug, context) combination."""
    drug: str
    context: str
    is_held_out: bool
    # Mean log-abundance over samples in this pair (NaN = not detected)
    r: np.ndarray = field(default_factory=lambda: np.array([]))
    # Mean log-abundance of matched controls for this context
    c: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def delta(self) -> np.ndarray:
        """Measured log-fold-change r - c."""
        return self.r - self.c


@dataclass
class PairResult:
    drug: str
    context: str
    n_detected: int
    n_top100: int
    pearson_full: float    # NaN for control-mean
    pearson_top100: float  # NaN for control-mean


# ------------------------------------------------------------------
# Pair construction
# ------------------------------------------------------------------

def build_pairs(
    matrix: pd.DataFrame,     # rows=samples, cols=proteins, values=log2 intensities
    meta: pd.DataFrame,       # index=sample, cols include drug, context (or donor/cell_line), is_control
    held_out_drugs: set[str],
    context_col: str = "donor",  # 'donor' for PXD014791, 'cell_line' for ProTargetMiner
) -> tuple[list[Pair], np.ndarray]:
    """
    Build Pair objects and return (pairs, protein_names).
    Controls are matched by context.
    """
    proteins = matrix.columns.values
    n_proteins = len(proteins)

    # Resolve context column
    if context_col not in meta.columns:
        # Try common alternatives
        for alt in ["cell_line", "donor", "context"]:
            if alt in meta.columns:
                context_col = alt
                break

    # Compute per-context control abundance (nanmean over all control samples)
    control_by_context: dict[str, np.ndarray] = {}
    ctrl_meta = meta[meta["is_control"]]
    for ctx, grp in ctrl_meta.groupby(context_col):
        ctrl_samples = grp.index.intersection(matrix.index)
        if len(ctrl_samples) == 0:
            continue
        ctrl_mat = matrix.loc[ctrl_samples].values.astype(float)
        control_by_context[ctx] = np.nanmean(ctrl_mat, axis=0)

    # Compute per-pair drug abundance
    drug_meta = meta[~meta["is_control"]]
    pairs = []
    for (drug, ctx), grp in drug_meta.groupby(["drug", context_col]):
        if ctx not in control_by_context:
            logger.warning(f"  No control found for context '{ctx}' -- skipping pair ({drug}, {ctx})")
            continue

        samples = grp.index.intersection(matrix.index)
        if len(samples) == 0:
            continue

        drug_mat = matrix.loc[samples].values.astype(float)
        r = np.nanmean(drug_mat, axis=0)
        c = control_by_context[ctx]

        is_held_out = drug in held_out_drugs

        pairs.append(Pair(
            drug=drug,
            context=ctx,
            is_held_out=is_held_out,
            r=r,
            c=c,
        ))

    logger.info(
        f"  Total pairs: {len(pairs)} "
        f"({sum(p.is_held_out for p in pairs)} held-out, "
        f"{sum(not p.is_held_out for p in pairs)} training)"
    )
    return pairs, proteins


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------

def _detected_mask(r_i: np.ndarray, p_i: np.ndarray) -> np.ndarray:
    """
    Detected = protein is finite in EITHER measured OR predicted proteome.
    In log space, finite = was detected (intensity > 0 before log).
    """
    return np.isfinite(r_i) | np.isfinite(p_i)


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, returning NaN if undefined (e.g., zero-variance)."""
    if len(x) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r, _ = pearsonr(x, y)
            return float(r)
        except Exception:
            return np.nan


def _median_centre(r: np.ndarray) -> np.ndarray:
    """Per-sample median centering on detected proteins; missing -> 0.
    Removes loading-amount / batch-offset differences between samples
    (e.g. Evren-lab PMC-E at 1500-3000 ng vs other batches).
    Without this, HVG top-100 Pearson = 0.069; with it = 0.564."""
    det = np.isfinite(r)
    out = np.zeros_like(r)
    if det.sum() > 0:
        out[det] = r[det] - np.median(r[det])
    return out


def _spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho, returning NaN if undefined."""
    if len(x) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r, _ = spearmanr(x, y)
            return float(r)
        except Exception:
            return np.nan


def compute_metrics(
    delta_hat: np.ndarray,       # predicted LFC (NaN allowed for non-detected proteins)
    delta_i: np.ndarray,         # measured LFC
    detected: np.ndarray,        # boolean mask
) -> tuple[float, float]:
    """
    Returns (pearson_full, pearson_top100, spearman_full, spearman_top100).
    All NaN if fewer than 2 detected proteins.
    """
    d = detected & np.isfinite(delta_hat) & np.isfinite(delta_i)

    if d.sum() < 2:
        return np.nan, np.nan

    dh_det = delta_hat[d]
    di_det = delta_i[d]

    pearson_full = _pearson_safe(dh_det, di_det)
    spearman_full = _spearman_safe(dh_det, di_det)

    # Top-100 by |measured LFC|
    n_top = min(TOP_N_DE, d.sum())
    top_idx = np.argsort(np.abs(di_det))[-n_top:]
    pearson_top  = _pearson_safe(dh_det[top_idx], di_det[top_idx])
    spearman_top = _spearman_safe(dh_det[top_idx], di_det[top_idx])

    return pearson_full, pearson_top, spearman_full, spearman_top


# ------------------------------------------------------------------
# kNN predictor (shared by HVG and PCA)
# ------------------------------------------------------------------

def _cosine_knn_predict(
    query_embed: np.ndarray,          # (d,)  pre-computed dense embedding
    train_embeds: np.ndarray,         # (n_train, d)
    train_r: np.ndarray,              # (n_train, n_proteins)
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    """
    cosine-similarity-weighted mean on a pre-computed dense embedding
    (used by PCA baseline where the embedding already handles missing data).
    """
    q = query_embed / (np.linalg.norm(query_embed) + 1e-10)
    norms = np.linalg.norm(train_embeds, axis=1, keepdims=True) + 1e-10
    t_norm = train_embeds / norms
    sims = t_norm @ q
    k_eff = min(k, len(sims))
    top_k = np.argsort(sims)[-k_eff:]
    weights = np.clip(sims[top_k], 0, None)
    if weights.sum() < 1e-10:
        weights = np.ones(k_eff)
    weights /= weights.sum()
    train_r_k = train_r[top_k]
    predicted = np.nansum(weights[:, None] * train_r_k, axis=0)
    all_nan = np.all(~np.isfinite(train_r_k), axis=0)
    predicted[all_nan] = np.nan
    return predicted


def _pairwise_complete_knn(
    query_r: np.ndarray,           # (n_proteins,) -- NaN = not detected
    train_r: np.ndarray,           # (n_train, n_proteins) -- NaN = not detected
    protein_subset: np.ndarray,    # boolean or index mask to restrict proteins
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    """
    Pairwise-complete cosine kNN: for each (query, training) pair, similarity
    is computed only on proteins detected in BOTH. No imputation needed.
    This is the methodologically correct approach for sparse MS proteomics data.
    """
    q = query_r[protein_subset]
    t = train_r[:, protein_subset]
    q_fin = np.isfinite(q)

    sims = np.zeros(len(t))
    for j in range(len(t)):
        both = q_fin & np.isfinite(t[j])
        n_both = both.sum()
        if n_both < 2:
            continue
        qv = q[both]; tv = t[j, both]
        denom = np.linalg.norm(qv) * np.linalg.norm(tv)
        if denom > 0:
            sims[j] = np.dot(qv, tv) / denom

    k_eff = min(k, len(sims))
    top_k = np.argsort(sims)[-k_eff:]
    weights = np.clip(sims[top_k], 0, None)
    if weights.sum() < 1e-10:
        weights = np.ones(k_eff)
    weights /= weights.sum()

    train_r_k = train_r[top_k]
    predicted = np.nansum(weights[:, None] * train_r_k, axis=0)
    all_nan = np.all(~np.isfinite(train_r_k), axis=0)
    predicted[all_nan] = np.nan
    return predicted


# ------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------

def baseline_control_mean(pair: Pair, all_pairs: list[Pair]) -> tuple[np.ndarray, np.ndarray]:
    """Predict no change: p_i = c_i -> delta_hat = 0 everywhere."""
    p_i = pair.c.copy()
    detected = _detected_mask(pair.r, p_i)
    delta_hat = np.zeros(len(pair.r))
    return delta_hat, detected


def baseline_perturbation_mean(
    pair: Pair, pool: list[Pair]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict the mean perturbed proteome from all pool pairs.
    p_i = mean_j(r_j for j in pool)
    """
    r_stack = np.stack([p.r for p in pool], axis=0)  # (n_pool, n_proteins)
    p_i = np.nanmean(r_stack, axis=0)
    detected = _detected_mask(pair.r, p_i)
    delta_hat = p_i - pair.c
    return delta_hat, detected


def baseline_hvg(
    pair: Pair, pool: list[Pair], n_hvg: int = N_HVG
) -> tuple[np.ndarray, np.ndarray]:
    """
    kNN predictor using 2000 highest-variance proteins.
    Variance computed with nanvar (among detected values only, no imputation).
    Cosine similarity computed pairwise-complete: only proteins detected in
    BOTH the query and the training sample contribute. No imputation at any step.
    """
    # Median-centre each sample before variance/embedding computation.
    # Removes per-sample loading offset; without this, HVG top-100 = 0.069.
    train_r_raw = np.stack([p.r for p in pool], axis=0)
    train_r_c   = np.stack([_median_centre(p.r) for p in pool], axis=0)

    variances = np.var(train_r_c, axis=0)
    hvg_idx   = np.argsort(variances)[-n_hvg:]

    query_c = _median_centre(pair.r)
    p_i_c   = _pairwise_complete_knn(query_c, train_r_c, hvg_idx)

    # Delta in centred space; Pearson/Spearman are invariant to this offset
    delta_hat = p_i_c - _median_centre(pair.c)
    detected  = _detected_mask(pair.r, p_i_c)
    return delta_hat, detected


def baseline_pca(
    pair: Pair, pool: list[Pair], n_components: int = N_PCA
) -> tuple[np.ndarray, np.ndarray]:
    """
    kNN predictor using PCA embedding.
    Restricts to proteins detected in >=50% of pool pairs -- this subset is
    near-complete so imputation is minimal (column mean for residual NaN only).
    PCA cannot be made fully imputation-free, but this approach minimises it.
    """
    train_r_raw = np.stack([p.r for p in pool], axis=0)
    train_r     = np.stack([_median_centre(p.r) for p in pool], axis=0)

    detection_frac = np.isfinite(train_r_raw).mean(axis=0)
    for threshold in [0.50, 0.30, 0.10]:
        prot_mask = detection_frac >= threshold
        if prot_mask.sum() >= 50:
            break

    train_r_filt = train_r[:, prot_mask]
    col_means = np.nanmean(train_r_filt, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    train_r_imp = np.where(np.isfinite(train_r_filt), train_r_filt, col_means[None, :])

    n_comp = min(n_components, train_r_imp.shape[0] - 1, train_r_imp.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    train_embeds = pca.fit_transform(train_r_imp)

    r_i_filt = pair.r[prot_mask]
    r_i_imp = np.where(np.isfinite(r_i_filt), r_i_filt, col_means)
    query_embed = pca.transform(r_i_imp[None, :])[0]

    p_i = _cosine_knn_predict(query_embed, train_embeds, train_r)
    detected  = _detected_mask(pair.r, p_i)
    delta_hat = p_i - _median_centre(pair.c)
    return delta_hat, detected


def baseline_additive_linear(
    pair: Pair, pool: list[Pair]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-way additive decomposition:
      delta_hat_i = delta_bar_drug + delta_bar_context - delta_bar_global

    For held-out drugs with no pool pairs sharing the same drug,
    delta_bar_drug = 0 (the drug term drops out).
    """
    deltas = np.stack([p.delta for p in pool], axis=0)  # (n_pool, n_proteins)
    drugs = np.array([p.drug for p in pool])
    contexts = np.array([p.context for p in pool])

    # Global mean
    delta_bar = np.nanmean(deltas, axis=0)

    # Drug mean (pairs in pool with same drug as held-out pair)
    drug_mask = drugs == pair.drug
    if drug_mask.sum() > 0:
        delta_bar_drug = np.nanmean(deltas[drug_mask], axis=0)
    else:
        delta_bar_drug = np.zeros_like(delta_bar)   # no data -> zero drug effect

    # Context mean
    ctx_mask = contexts == pair.context
    if ctx_mask.sum() > 0:
        delta_bar_ctx = np.nanmean(deltas[ctx_mask], axis=0)
    else:
        delta_bar_ctx = np.zeros_like(delta_bar)

    delta_hat = delta_bar_drug + delta_bar_ctx - delta_bar

    # For additive-linear, detected = finite in measured delta
    detected = np.isfinite(pair.delta)
    return delta_hat, detected


# ------------------------------------------------------------------
# Main benchmark runner
# ------------------------------------------------------------------

BASELINES = {
    "control-mean": baseline_control_mean,
    "perturbation-mean": baseline_perturbation_mean,
    "HVG": baseline_hvg,
    "PCA": baseline_pca,
    "additive-linear": baseline_additive_linear,
}


def run_benchmark(
    pairs: list[Pair],
    dataset_name: str,
) -> pd.DataFrame:
    """
    Run all baselines on all held-out pairs.

    Returns tidy DataFrame with columns:
      dataset, baseline, drug, context,
      n_detected, pearson_full, pearson_top100
    """
    held_out = [p for p in pairs if p.is_held_out]
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {dataset_name}  |  {len(held_out)} held-out pairs")
    logger.info(f"{'='*60}")

    records = []

    for i, pair in enumerate(held_out):
        # Pool = all pairs except the current one
        pool = [p for p in pairs if not (p.drug == pair.drug and p.context == pair.context)]

        if len(pool) < K_NEIGHBORS:
            logger.warning(
                f"  Pair ({pair.drug}, {pair.context}): pool too small ({len(pool)}), skipping"
            )
            continue

        n_detected_measured = int(np.isfinite(pair.delta).sum())
        if n_detected_measured < MIN_PROTEINS_PER_PAIR:
            logger.warning(
                f"  Pair ({pair.drug}, {pair.context}): only {n_detected_measured} "
                f"detected proteins (< {MIN_PROTEINS_PER_PAIR}), skipping"
            )
            continue

        for bl_name, bl_fn in BASELINES.items():
            try:
                delta_hat, detected = bl_fn(pair, pool)
            except Exception as e:
                logger.error(f"  [{bl_name}] ({pair.drug}, {pair.context}): {e}")
                continue

            pearson_full, pearson_top100, spearman_full, spearman_top100 =                 compute_metrics(delta_hat, pair.delta, detected)

            records.append({
                "dataset": dataset_name,
                "baseline": bl_name,
                "drug": pair.drug,
                "context": pair.context,
                "n_detected": int(detected.sum()),
                "pearson_full": pearson_full,
                "pearson_top100": pearson_top100,
                "spearman_full": spearman_full,
                "spearman_top100": spearman_top100,
            })

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{len(held_out)} pairs ...")

    return pd.DataFrame(records)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-pair results to dataset x baseline mean/std.
    Excludes NaN (control-mean has undefined Pearson).
    """
    if results.empty or "dataset" not in results.columns:
        logger.warning("No results to summarise -- results DataFrame is empty.")
        return pd.DataFrame()
    return (
        results.groupby(["dataset", "baseline"])[
            ["pearson_full", "pearson_top100", "spearman_full", "spearman_top100"]]
        .agg(["mean", "std", "count"])
        .round(3)
    )