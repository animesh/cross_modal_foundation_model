"""
Corrected Phase 1 diagnostic based on the supplied run_phase1.py.
Fixes TSV parsing, explicit sample mapping, symbol alignment, five folds,
missing dependency labels, and training/validation gene leakage.
Preserves original standardized-count PCA loadings and logistic classifier.
These are NOT verified equivalents of the paper's upstream RF/MLP benchmarks.
Default: DepMap diagnostic only. --msigdb-proxy explicitly enables the proxy.
Run from phase1: python run_phase1.py --eval
Required preflight mapping: results/phase1_preflight/sample_mapping.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

BENCH   = Path(r"D:\data\phase1\benchmarks")
RESULTS = Path(__file__).resolve().parent / "results/phase1"
N_PCA   = 512


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gct(path: Path) -> pd.DataFrame:
    """Read plain Name/Description TSV or GCT #1.2; preserve every gene row.
    Duplicate gene symbols are summed at the count level, explicitly reported.
    """
    with path.open(encoding="utf-8-sig") as handle:
        first = handle.readline().rstrip("\r\n")
    if first == "#1.2":
        skiprows = 2
    elif first.split("\t")[:2] == ["Name", "Description"]:
        skiprows = 0
    else:
        raise ValueError(f"Unsupported expression header: {first[:200]!r}")
    mapping_path = RESULTS.parent / "phase1_preflight" / "sample_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Run check_phase1_data.py first: missing {mapping_path}")
    mapping = pd.read_csv(mapping_path, dtype=str)
    if not {"expression_sample", "cell-line"}.issubset(mapping.columns):
        raise ValueError("Invalid preflight sample mapping schema")
    if mapping.isna().any().any() or not mapping["expression_sample"].is_unique or not mapping["cell-line"].is_unique:
        raise ValueError("Sample mapping must be nonblank and one-to-one")
    header = pd.read_csv(path, sep="\t", skiprows=skiprows, nrows=0).columns
    missing = set(mapping["expression_sample"]) - set(header)
    if missing:
        raise ValueError(f"Mapping refers to {len(missing)} absent expression columns")
    selected = mapping["expression_sample"].tolist()
    df = pd.read_csv(path, sep="\t", skiprows=skiprows,
                     usecols=["Name", "Description"] + selected,
                     dtype={column: np.float32 for column in selected})
    symbols = df["Description"].fillna("").astype(str).str.strip()
    counts = df[selected].copy()
    if not np.isfinite(counts.to_numpy()).all() or (counts.to_numpy() < 0).any():
        raise ValueError("Expression counts contain nonfinite/negative values before aggregation")
    counts.index = symbols
    counts = counts.loc[counts.index != ""]
    duplicate_rows = int(counts.index.duplicated().sum())
    print(f"  Duplicate symbol rows summed at count level: {duplicate_rows}")
    counts = counts.groupby(level=0, sort=False).sum()
    if not np.isfinite(counts.to_numpy()).all() or (counts.to_numpy() < 0).any():
        raise ValueError("Expression counts contain nonfinite/negative values")
    counts.columns = mapping.set_index("expression_sample").loc[counts.columns, "cell-line"].tolist()
    return counts



def load_ccle_expression() -> pd.DataFrame:
    gct = BENCH / "depmap/raw/ccle-counts.gct"
    if not gct.exists():
        raise FileNotFoundError(gct)
    print(f"  Loading {gct} ...")
    return load_gct(gct).T



def load_depmap() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dep = pd.read_csv(BENCH / "depmap/raw/depmap-gene-dependencies.csv", index_col=0)
    dep.columns = [column.split(" (")[0].strip() for column in dep.columns]
    if not dep.index.is_unique or not dep.columns.is_unique:
        raise ValueError("Dependency sample IDs and stripped gene symbols must be unique")
    split_cls = pd.read_csv(BENCH / "depmap/misc/split-cls.csv", dtype=str)
    split_genes = pd.read_csv(BENCH / "depmap/misc/split-genes-lt5gt70.csv", dtype=str)
    for table, key in [(split_cls, "cell-line"), (split_genes, "gene")]:
        if key not in table or table[key].isna().any() or not table[key].is_unique:
            raise ValueError(f"Invalid split identifiers: {key}")
        for fold in [f"fold-{i}" for i in range(5)]:
            if fold not in table:
                raise ValueError(f"Missing split column: {fold}")
            table[fold] = table[fold].str.strip().str.lower()
            if set(table[fold]) != {"train", "val"}:
                raise ValueError(f"Unexpected or empty split labels: {fold}")
    return dep, split_cls, split_genes



def load_msigdb() -> dict[str, set[str]]:
    gmt = BENCH / "msigdb/sigs/genesets.v2023.2.Hs_hallmark.gmt"
    gene_sets = {}
    for line in gmt.read_text().strip().splitlines():
        parts = line.strip().split("\t")
        gene_sets[parts[0]] = set(parts[2:])
    print(f"  Loaded {len(gene_sets)} Hallmark gene sets")
    return gene_sets


# ---------------------------------------------------------------------------
# PCA gene embeddings
# ---------------------------------------------------------------------------

def compute_pca_gene_embeddings(
    expr: pd.DataFrame, train_cls: list[str] | None = None,
    n_components: int = N_PCA,
) -> pd.DataFrame:
    """Preserve original standardized-count PCA and gene-loading representation.
    This representation is a diagnostic, not verified upstream PCA equivalence.
    """
    if train_cls is None:
        raise ValueError("Explicit training cell lines are required")
    missing = set(train_cls) - set(expr.index)
    if missing:
        raise ValueError(f"{len(missing)} training cell lines lack mapped expression")
    fit_data = expr.loc[train_cls]
    if min(fit_data.shape) <= n_components:
        raise ValueError(f"PCA({n_components}) unsupported for {fit_data.shape}; no silent rank reduction")
    print(f"  PCA({n_components}): {len(fit_data)} training lines x {fit_data.shape[1]} genes")
    X = StandardScaler().fit_transform(fit_data.to_numpy())
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return pd.DataFrame(pca.components_.T, index=fit_data.columns,
                        columns=[f"PC{i+1}" for i in range(n_components)])



# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def eval_depmap(
    gene_emb: pd.DataFrame, dep: pd.DataFrame, split_cls: pd.DataFrame,
    split_genes: pd.DataFrame, threshold: float = 0.5, fold: int = 0,
) -> float:
    """Corrected original logistic diagnostic, NOT the upstream RF benchmark.
    Fit on training genes using mean dependency across training cell lines.
    Score validation genes separately in each validation cell line.
    Nonfinite labels are excluded; no missing label becomes inessential.
    """
    column = f"fold-{fold}"
    train_cls = split_cls.loc[split_cls[column] == "train", "cell-line"].tolist()
    val_cls = split_cls.loc[split_cls[column] == "val", "cell-line"].tolist()
    if not set(train_cls + val_cls).issubset(dep.index):
        raise ValueError("Dependency matrix does not cover all split cell lines")
    train_genes = gene_emb.index.intersection(dep.columns).intersection(
        split_genes.loc[split_genes[column] == "train", "gene"])
    val_genes = gene_emb.index.intersection(dep.columns).intersection(
        split_genes.loc[split_genes[column] == "val", "gene"])
    if set(train_genes) & set(val_genes) or not len(train_genes) or not len(val_genes):
        raise ValueError("Training/validation genes must be nonempty and disjoint")
    values = dep.loc[train_cls, train_genes].to_numpy(dtype=float)
    finite = np.isfinite(values)
    n_observed = finite.sum(axis=0)
    means = np.divide(np.where(finite, values, 0).sum(axis=0), n_observed,
                      out=np.full(len(train_genes), np.nan), where=n_observed > 0)
    keep = np.isfinite(means)
    y_train = (means[keep] > threshold).astype(int)
    if len(np.unique(y_train)) != 2:
        raise ValueError(f"fold-{fold}: training labels contain fewer than two classes")
    clf = LogisticRegression(max_iter=300, C=1.0, random_state=42)
    clf.fit(gene_emb.loc[train_genes[keep]].to_numpy(), y_train)
    scores = clf.predict_proba(gene_emb.loc[val_genes].to_numpy())[:, 1]
    rows = []
    for sample in val_cls:
        labels = dep.loc[sample, val_genes].to_numpy(dtype=float)
        valid = np.isfinite(labels)
        y = (labels[valid] > threshold).astype(int)
        positives = int(y.sum())
        if positives < 5 or len(y) - positives < 5:
            continue
        rows.append({"fold": fold, "cell-line": sample, "n_genes": int(valid.sum()),
                     "positives": positives, "auprc": average_precision_score(y, scores[valid])})
    if not rows:
        raise ValueError(f"fold-{fold}: no evaluable validation cell lines")
    pd.DataFrame(rows).to_csv(RESULTS / f"depmap_fold{fold}_per_cell.csv", index=False)
    result = float(np.mean([row["auprc"] for row in rows]))
    print(f"  DepMap diagnostic fold-{fold}: {result:.4f}, {len(rows)} validation lines")
    return result



def eval_msigdb(
    gene_emb: pd.DataFrame, gene_sets: dict[str, set[str]],
    split_genes: pd.DataFrame, fold: int = 0,
) -> float:
    """CCLE/logistic Hallmark PROXY with held-out genes, not Figure 2 MSigDB.
    Uses the supplied gene split; does not substitute this for upstream MLP/data.
    """
    column = f"fold-{fold}"
    train_genes = gene_emb.index.intersection(split_genes.loc[split_genes[column] == "train", "gene"])
    val_genes = gene_emb.index.intersection(split_genes.loc[split_genes[column] == "val", "gene"])
    if not len(train_genes) or not len(val_genes) or set(train_genes) & set(val_genes):
        raise ValueError("Invalid Hallmark proxy gene split")
    X_train = gene_emb.loc[train_genes].to_numpy()
    X_val = gene_emb.loc[val_genes].to_numpy()
    rows = []
    for name, members in tqdm(gene_sets.items(), desc=f"  Hallmark proxy fold-{fold}", leave=False):
        y_train = np.array([gene in members for gene in train_genes], dtype=int)
        y_val = np.array([gene in members for gene in val_genes], dtype=int)
        if min(y_train.sum(), len(y_train)-y_train.sum(), y_val.sum(), len(y_val)-y_val.sum()) < 5:
            continue
        clf = LogisticRegression(max_iter=300, C=1.0, random_state=42)
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_val)[:, 1]
        rows.append({"fold": fold, "pathway": name, "train_genes": len(train_genes),
                     "val_genes": len(val_genes), "auprc": average_precision_score(y_val, scores)})
    if not rows:
        raise ValueError(f"fold-{fold}: no evaluable Hallmark pathways")
    pd.DataFrame(rows).to_csv(RESULTS / f"hallmark_proxy_fold{fold}.csv", index=False)
    return float(np.mean([row["auprc"] for row in rows]))



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Corrected Phase 1 diagnostic; not exact Figure 2 reproduction")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(5), default=None,
                        help="Run one fold; default runs all five")
    parser.add_argument("--msigdb-proxy", action="store_true",
                        help="Also run explicitly labelled CCLE/logistic Hallmark proxy")
    args = parser.parse_args()
    if not args.eval:
        parser.print_help()
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=== Phase 1 corrected PCA/logistic diagnostic (NOT exact Figure 2) ===")
    expr = load_ccle_expression()
    dep, split_cls, split_genes = load_depmap()
    if not set(split_cls["cell-line"]).issubset(expr.index):
        raise ValueError("Expression mapping does not cover the complete cell-line split")
    gene_sets = load_msigdb() if args.msigdb_proxy else None
    rows = []
    folds = [args.fold] if args.fold is not None else list(range(5))
    for fold in folds:
        train_cls = split_cls.loc[split_cls[f"fold-{fold}"] == "train", "cell-line"].tolist()
        gene_emb = compute_pca_gene_embeddings(expr, train_cls=train_cls, n_components=N_PCA)
        value = eval_depmap(gene_emb, dep, split_cls, split_genes, fold=fold)
        rows.append({"fold": fold, "task": "DepMap PCA/logistic diagnostic", "auprc": value,
                     "status": "not_upstream_RF_protocol"})
        if args.msigdb_proxy:
            value = eval_msigdb(gene_emb, gene_sets, split_genes, fold=fold)
            rows.append({"fold": fold, "task": "Hallmark CCLE/logistic proxy", "auprc": value,
                         "status": "not_Figure2_MSigDB_protocol"})
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "phase1_results.csv", index=False)
    summary = results.groupby(["task", "status"]).agg(mean_auprc=("auprc", "mean"),
                  std_auprc=("auprc", "std"), folds=("fold", "count")).reset_index()
    summary.to_csv(RESULTS / "phase1_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved results under: {RESULTS}")
    print("Exact Figure 2 still requires upstream PCA/RF and MSigDB MLP/data protocols.")



if __name__ == "__main__":
    main()