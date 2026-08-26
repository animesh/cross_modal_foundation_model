"""
Parse ProTargetMiner TMT datasets (PXD009775, PXD009644, PXD013134).

Tesorai paper uses "absolute isotope-corrected reporter intensities"
from MaxQuant output (NOT the ratio columns), then maps TMT channels to
drug/condition using Supplementary Table 2 from Saei et al. 2019.

Two-stage approach:
  1. Try to auto-detect the channel-to-drug mapping from the PRIDE
     experimental design file (SDRF or experiment_design.txt), if present.
  2. Fall back to HARDCODED_CHANNEL_MAP defined below.

*** IMPORTANT ***
If auto-detection fails, you MUST fill in HARDCODED_CHANNEL_MAP from
Supplementary Table 2 of:
  Saei et al. (2019) Nature Communications 10:5715
  https://doi.org/10.1038/s41467-019-13582-8

The supplementary file is freely available at:
  https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-13582-8/
  MediaObjects/41467_2019_13582_MOESM4_ESM.xlsx   (Supplementary Data 2)

Format expected in HARDCODED_CHANNEL_MAP:
  { (plex_id, channel_index): {"cell_line": "A549", "drug": "Doxorubicin",
                                "is_control": False} }
where plex_id is a string matching the raw file batch name, and channel_index
is 0-based (0 = Reporter intensity corrected 0, ..., 9 = Reporter intensity 9).

This is intentionally left as an empty dict so the pipeline fails loudly
rather than silently producing wrong results. Fill it in before running.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, PTM_ACCESSIONS, PTM_REPORTER_PREFIX, PTM_CHANNELS,
    PTM_CONTEXTS, PTM_CONTROL_KEYWORDS, PTM_HELD_OUT_DRUGS,
    PTM_TRAINING_DRUGS, MQ_FILTER_COLS,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# FILL THIS IN from Saei et al. 2019 Supp. Table 2
# Example structure (replace with real values):
#
# HARDCODED_CHANNEL_MAP = {
#     ("PXD009775_batch1", 0): {"cell_line": "A549", "drug": "Doxorubicin", "is_control": False},
#     ("PXD009775_batch1", 9): {"cell_line": "A549", "drug": "DMSO",        "is_control": True},
#     ...
# }
# -----------------------------------------------------------------------
HARDCODED_CHANNEL_MAP: dict = {}


def _try_load_sdrf(folder: Path) -> pd.DataFrame | None:
    """
    Load PRIDE SDRF / experimental design file if present.
    Returns tidy DataFrame with columns [raw_file, channel, cell_line, drug, is_control]
    or None if no file found.
    """
    candidates = list(folder.glob("*.sdrf.tsv")) + list(folder.glob("*experimental_design*.txt"))
    if not candidates:
        return None
    path = candidates[0]
    logger.info(f"  Found design file: {path.name}")
    df = pd.read_csv(path, sep="\t", low_memory=False)
    logger.info(f"  Design columns: {list(df.columns)}")
    # Attempt to find relevant columns -- PRIDE SDRF uses standard names
    # This is dataset-specific; return None if we can't parse it
    needed = {"characteristics[cell line]", "comment[label]", "factor value[compound]"}
    if not needed.issubset({c.lower() for c in df.columns}):
        logger.warning("  SDRF found but doesn't have expected columns -- falling back to manual map")
        return None
    return df


def _channel_index_from_label(label: str) -> int | None:
    """
    TMT channel labels like 'TMT126', 'TMT127N', '127C', etc.
    Map to 0-9 channel index.
    """
    # Strip 'TMT' prefix
    label = re.sub(r"^TMT", "", label.strip(), flags=re.I)
    order = ["126", "127N", "127C", "128N", "128C", "129N", "129C", "130N", "130C", "131"]
    for i, t in enumerate(order):
        if label.upper() == t:
            return i
    return None


def load_protargetminer(
    data_dirs: list[Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and merge all three ProTargetMiner PRIDE deposits.

    Returns
    -------
    matrix : DataFrame (n_samples, n_proteins)
        log2 absolute reporter intensities; NaN = not detected.
    meta   : DataFrame (n_samples, 3+)
        Columns: cell_line, drug, plex_id, channel, is_control.
    """
    if data_dirs is None:
        data_dirs = [Path(DATA_DIR) / acc for acc in PTM_ACCESSIONS]

    all_matrices = []
    all_metas = []

    for folder in data_dirs:
        pg_path = folder / "proteinGroups.txt"
        if not pg_path.exists():
            logger.warning(
                f"Missing: {pg_path}\n"
                f"Download from https://www.ebi.ac.uk/pride/archive/projects/{folder.name}\n"
                f"(combined/txt/proteinGroups.txt)"
            )
            continue

        logger.info(f"\nLoading {pg_path} ...")
        pg = pd.read_csv(pg_path, sep="\t", low_memory=False)
        logger.info(f"  Raw: {pg.shape[0]} protein groups, {pg.shape[1]} columns")

        # Filter
        mask = pd.Series(True, index=pg.index)
        for col, bad in MQ_FILTER_COLS.items():
            if col in pg.columns:
                mask &= pg[col].fillna("").astype(str) != bad
        pg = pg[mask].copy()

        # Protein ID
        if "Gene names" in pg.columns:
            pg["_id"] = pg["Gene names"].fillna("").str.split(";").str[0].str.strip()
        else:
            pg["_id"] = pg["Protein IDs"].str.split(";").str[0].str.strip()
        pg = pg[pg["_id"] != ""].drop_duplicates("_id").set_index("_id")

        # Find reporter intensity corrected columns
        rep_cols = [c for c in pg.columns if c.startswith(PTM_REPORTER_PREFIX)]
        if not rep_cols:
            raise ValueError(
                f"No '{PTM_REPORTER_PREFIX}*' columns in {pg_path}.\n"
                f"First 30 cols: {list(pg.columns[:30])}"
            )
        logger.info(f"  Found {len(rep_cols)} reporter intensity columns")

        intensity_matrix = pg[rep_cols].astype(float).replace(0, np.nan)
        log_matrix = np.log2(intensity_matrix)

        # ---- Map channels to drug/condition ----
        design = _try_load_sdrf(folder)
        channel_meta = _assign_channels(rep_cols, design, folder.name)

        if channel_meta is None:
            logger.error(
                f"\n*** CHANNEL MAP MISSING for {folder.name} ***\n"
                f"    Auto-detection failed. Fill in HARDCODED_CHANNEL_MAP in parse_tmt.py\n"
                f"    using Saei et al. 2019 Supplementary Table 2.\n"
            )
            continue

        # Build sample-level dataframe
        # Each column = one TMT channel = one pseudo-sample
        sample_records = []
        valid_cols = []
        for col in rep_cols:
            ch_idx = int(re.search(r"\d+$", col).group())  # trailing number
            key = (folder.name, ch_idx)
            if key not in channel_meta:
                logger.warning(f"  No channel map entry for {key} -- skipping column {col}")
                continue
            info = channel_meta[key]
            sample_name = f"{folder.name}_ch{ch_idx}"
            sample_records.append({
                "sample": sample_name,
                "cell_line": info["cell_line"],
                "drug": info["drug"],
                "plex_id": folder.name,
                "channel": ch_idx,
                "is_control": info["is_control"],
            })
            valid_cols.append(col)

        if not sample_records:
            continue

        meta_chunk = pd.DataFrame(sample_records).set_index("sample")
        mat_chunk = log_matrix[valid_cols].T
        mat_chunk.index = meta_chunk.index

        all_matrices.append(mat_chunk)
        all_metas.append(meta_chunk)

    if not all_matrices:
        raise RuntimeError(
            "No data loaded for ProTargetMiner. "
            "Check download and channel map (see HARDCODED_CHANNEL_MAP in parse_tmt.py)."
        )

    # Merge across deposits on shared protein IDs
    matrix = pd.concat(all_matrices, axis=0, join="outer")
    meta = pd.concat(all_metas, axis=0)

    logger.info(
        f"\nProTargetMiner merged: {matrix.shape[0]} samples x {matrix.shape[1]} proteins"
    )
    return matrix, meta


def _assign_channels(
    rep_cols: list[str],
    sdrf: pd.DataFrame | None,
    folder_name: str,
) -> dict | None:
    """
    Return {(folder_name, channel_index): {cell_line, drug, is_control}}
    or None if we can't determine the mapping.
    """
    # Try SDRF
    if sdrf is not None:
        return _channels_from_sdrf(sdrf, folder_name)

    # Try HARDCODED_CHANNEL_MAP
    relevant = {k: v for k, v in HARDCODED_CHANNEL_MAP.items() if k[0] == folder_name}
    if relevant:
        return relevant

    # Last resort: try to infer from column names (e.g., if drug names appear)
    inferred = _channels_from_column_names(rep_cols, folder_name)
    if inferred:
        return inferred

    return None


def _channels_from_sdrf(sdrf: pd.DataFrame, folder_name: str) -> dict | None:
    """Parse PRIDE SDRF format into channel map."""
    result = {}
    # Normalise column names
    sdrf.columns = [c.lower().strip() for c in sdrf.columns]
    for _, row in sdrf.iterrows():
        label = str(row.get("comment[label]", ""))
        ch = _channel_index_from_label(label)
        if ch is None:
            continue
        cell_line = str(row.get("characteristics[cell line]", "unknown")).strip()
        drug = str(row.get("factor value[compound]", "DMSO")).strip()
        is_ctrl = drug.lower() in PTM_CONTROL_KEYWORDS
        result[(folder_name, ch)] = {
            "cell_line": cell_line,
            "drug": "CONTROL" if is_ctrl else drug,
            "is_control": is_ctrl,
        }
    return result if result else None


def _channels_from_column_names(rep_cols: list[str], folder_name: str) -> dict | None:
    """
    If column names encode drug/condition (rare but possible), parse them.
    E.g. 'Reporter intensity corrected 0 A549_Doxorubicin'
    Returns None if not parseable.
    """
    result = {}
    for col in rep_cols:
        m = re.search(r"(\d+)\s+(.+)$", col)
        if not m:
            return None
        ch = int(m.group(1))
        condition = m.group(2).strip()
        is_ctrl = any(kw in condition.lower() for kw in PTM_CONTROL_KEYWORDS)
        # Try to extract cell_line from known list
        cell_line = next((c for c in PTM_CONTEXTS if c.lower() in condition.lower()), "unknown")
        drug_name = condition if not is_ctrl else "CONTROL"
        result[(folder_name, ch)] = {
            "cell_line": cell_line,
            "drug": drug_name,
            "is_control": is_ctrl,
        }
    return result if result else None
