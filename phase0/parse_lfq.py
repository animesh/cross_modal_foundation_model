"""
Parse PXD014791 MaxQuant LFQ output into a tidy sample x protein matrix.

Expected input: proteinGroups.txt from MaxQuant combined/txt/
Key columns used:
  - Gene names           (preferred protein identifier)
  - Protein IDs          (fallback UniProt)
  - LFQ intensity <sample>  (one column per sample)
  - Only identified by site / Reverse / Potential contaminant  (filter flags)

Sample naming in PXD014791 follows Xiong et al. (Scientific Data 2022):
  <CellLine>_<Drug>_<Replicate>  e.g.  iCell_Afatinib_1
The cell line is one of the 4 iPSC-CM donor lines (iCell, Cor4U, Pluricel, CDI).
Controls are named with DMSO or Vehicle.
"""

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, PXD014791_ACCESSION, PXD014791_LFQ_PREFIX,
    PXD014791_CONTROL_KEYWORDS, PXD014791_HELD_OUT_DRUGS,
    PXD014791_TRAINING_DRUGS, MQ_FILTER_COLS,
)

logger = logging.getLogger(__name__)

# Known donor names in this dataset (from the Scientific Data paper)
KNOWN_DONORS = {"iCell", "Cor4U", "Pluricel", "CDI"}

# Some drugs appear under alternate spellings in file names
DRUG_ALIASES = {
    "gefitinib": "Gefitinib",
    "erlotinib": "Erlotinib",
    "dasatinib": "Dasatinib",
    # extend if needed after inspecting column names
}


def _canonical_drug(raw: str) -> str:
    """Normalise drug name to Title Case, apply known aliases."""
    raw = raw.strip()
    lower = raw.lower()
    if lower in DRUG_ALIASES:
        return DRUG_ALIASES[lower]
    return raw.title()


def _is_control(sample_token: str) -> bool:
    lower = sample_token.lower()
    return any(kw in lower for kw in PXD014791_CONTROL_KEYWORDS)


def _parse_sample_name(col_suffix: str) -> dict | None:
    """
    Parse a column name suffix (after 'LFQ intensity ') into metadata.
    Expected patterns (from the dataset):
      <Donor>_<Drug>_<Rep>
      <Donor>_DMSO_<Rep>
    Returns dict with keys: donor, drug, replicate, is_control
    or None if pattern is unrecognised.
    """
    parts = col_suffix.strip().split("_")
    if len(parts) < 3:
        return None

    donor = parts[0]
    drug_raw = "_".join(parts[1:-1])   # handles multi-word drug names
    rep = parts[-1]

    is_ctrl = _is_control(drug_raw)
    drug = "CONTROL" if is_ctrl else _canonical_drug(drug_raw)

    return {
        "sample": col_suffix.strip(),
        "donor": donor,
        "drug": drug,
        "replicate": rep,
        "is_control": is_ctrl,
    }


def load_pxd014791(protein_groups_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PXD014791 proteinGroups.txt.

    Returns
    -------
    matrix : DataFrame, shape (n_samples, n_proteins)
        log2-transformed LFQ intensities; NaN where protein not detected.
        Index = sample names, columns = gene names.
    meta   : DataFrame, shape (n_samples, 4)
        Columns: donor, drug, replicate, is_control.
    """
    if protein_groups_path is None:
        protein_groups_path = Path(DATA_DIR) / PXD014791_ACCESSION / "proteinGroups.txt"

    if not protein_groups_path.exists():
        raise FileNotFoundError(
            f"File not found: {protein_groups_path}\n"
            f"Please download proteinGroups.txt from PRIDE {PXD014791_ACCESSION} "
            f"(combined/txt/proteinGroups.txt) and place it there."
        )

    logger.info(f"Loading {protein_groups_path} ...")
    pg = pd.read_csv(protein_groups_path, sep="\t", low_memory=False)
    logger.info(f"  Raw: {pg.shape[0]} protein groups, {pg.shape[1]} columns")

    # ---- Filter MaxQuant decoys / contaminants / site-only ----
    mask = pd.Series(True, index=pg.index)
    for col, bad_value in MQ_FILTER_COLS.items():
        if col in pg.columns:
            mask &= pg[col].fillna("").astype(str) != bad_value
    pg = pg[mask].copy()
    logger.info(f"  After MQ filters: {pg.shape[0]} protein groups")

    # ---- Choose protein identifier ----
    if "Gene names" in pg.columns:
        pg["_id"] = pg["Gene names"].fillna("").str.split(";").str[0].str.strip()
    else:
        pg["_id"] = pg["Protein IDs"].str.split(";").str[0].str.strip()

    # Drop rows with empty ID or duplicate IDs (keep first)
    pg = pg[pg["_id"] != ""].copy()
    pg = pg.drop_duplicates(subset="_id", keep="first")
    pg = pg.set_index("_id")

    # ---- Extract LFQ intensity columns ----
    lfq_cols = [c for c in pg.columns if c.startswith(PXD014791_LFQ_PREFIX)]
    if not lfq_cols:
        raise ValueError(
            f"No columns starting with '{PXD014791_LFQ_PREFIX}' found.\n"
            f"First 30 columns: {list(pg.columns[:30])}"
        )
    logger.info(f"  Found {len(lfq_cols)} LFQ intensity columns")

    intensity_matrix = pg[lfq_cols].copy().astype(float)

    # ---- Parse sample metadata from column names ----
    records = []
    for col in lfq_cols:
        suffix = col[len(PXD014791_LFQ_PREFIX):]
        meta = _parse_sample_name(suffix)
        if meta is None:
            logger.warning(f"  Could not parse column name: '{col}' -- skipping")
            continue
        records.append(meta)

    meta_df = pd.DataFrame(records)
    meta_df = meta_df.set_index("sample")

    # Align matrix and meta
    valid_cols = [PXD014791_LFQ_PREFIX + s for s in meta_df.index]
    intensity_matrix = intensity_matrix[valid_cols]
    intensity_matrix.columns = meta_df.index   # strip prefix for cleaner column names

    # ---- log2-transform; zeros -> NaN (not detected) ----
    intensity_matrix = intensity_matrix.replace(0, np.nan)
    log_matrix = np.log2(intensity_matrix)

    # Transpose: rows = samples, columns = proteins
    log_matrix = log_matrix.T

    logger.info(
        f"  Final matrix: {log_matrix.shape[0]} samples x {log_matrix.shape[1]} proteins"
    )
    logger.info(f"  Donors found: {sorted(meta_df['donor'].unique())}")
    logger.info(f"  Drugs found: {sorted(meta_df[~meta_df['is_control']]['drug'].unique())}")

    return log_matrix, meta_df
