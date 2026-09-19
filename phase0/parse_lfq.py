"""
Parse PXD014791 MaxQuant LFQ output into a tidy sample x protein matrix.

Column naming in this dataset is highly irregular across gel batches:
  031616_Gel1_BOS_Tube_11
  031616_Gel2_Exp20_AFA_Tube_17
  Evren_Gel1_Tube11_TOF_Exp_32_2000ng
  09022016_Gel21_EXP59_PRE_tube6
  Gel12_Tube1_CTRL_Exp_43
  2nd_Experiment_39_SUN_1

Strategy: scan each column for a known drug abbreviation rather than
trying to parse fixed-position structure. Donor proxy comes from
Metadata-Table.xlsx if available; otherwise we use the experiment
number extracted from the column name.

Run `python parse_metadata.py` first to download and display
Metadata-Table.xlsx so you can confirm the donor mapping.
"""

import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DATA_DIR, PXD014791_ACCESSION, PXD014791_LFQ_PREFIX,
    PXD014791_HELD_OUT_DRUGS, PXD014791_TRAINING_DRUGS, MQ_FILTER_COLS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drug abbreviation -> canonical name
# All abbreviations seen in the FTP raw file listing for PXD014791
# ---------------------------------------------------------------------------
DRUG_MAP = {
    "AFA":  "Afatinib",
    "AXI":  "Axitinib",
    "BOS":  "Bosutinib",
    "CAB":  "Cabozantinib",
    "DAB":  "Dabrafenib",
    "DAS":  "Dasatinib",
    "ERL":  "Erlotinib",
    "EST":  "Estaurosporine",   # not in held-out list -- will be training
    "GEF":  "Gefitinib",
    "IMA":  "Imatinib",
    "LAP":  "Lapatinib",
    "NIL":  "Nilotinib",
    "PAZ":  "Pazopanib",
    "PON":  "Ponatinib",
    "PRE":  "Trametinib",       # hypothesis -- confirm via Metadata-Table.xlsx
    "REG":  "Regorafenib",
    "RUX":  "Ruxolitinib",
    "SOR":  "Sorafenib",
    "SUN":  "Sunitinib",
    "TOF":  "Tofacitinib",
    "TRA":  "Trastuzumab",
    "TRS":  "Trametinib",       # confirmed: only unaccounted held-out drug among 22 Drug Treatment codes
    "USN":  "Vandetanib",       # confirmed: metadata maps 031616_Gel1_USN_Tube_12 -> Drug Treatment=VAN
    "VAN":  "Vandetanib",
    "VEM":  "Vemurafenib",
    # Longer forms also seen in some columns
    "TRAM": "Trametinib",
    "SUNIT": "Sunitinib",
}

# Tokens that identify a control sample
CTRL_TOKENS = {"ctrl", "ctrol", "control", "dmso", "vehicle"}

# Tokens that identify HeLa QC samples -- skip entirely
HELA_TOKENS = {"hela"}

# Tokens to ignore when extracting drug (not drug names)
NON_DRUG = {
    "gel", "tube", "exp", "experiment", "nd", "ng", "hela", "ctrl",
    "ctrol", "control", "2000ng", "1500ng", "3000ng", "2nd",
}


def _extract_drug_from_tokens(tokens: list[str]) -> str | None:
    """Return canonical drug name from first matching abbreviation, or None."""
    for t in tokens:
        upper = t.upper()
        if upper in DRUG_MAP:
            return DRUG_MAP[upper]
    return None


def _extract_donor(tokens: list[str], col_suffix: str) -> str:
    """
    Extract a donor/batch proxy from the column name.

    Priority:
      1. Experiment number (Exp32, Exp_32, EXP59, etc.) -- most consistent
      2. Named batch prefix (Evren, 2nd)
      3. Date prefix (031616, 071116, etc.) if only one per gel

    Without Metadata-Table.xlsx we cannot map these to the 4 iPSC-CM lines.
    Run parse_metadata.py to get the ground-truth mapping.
    """
    # Look for Exp<N> pattern anywhere in the column name
    exp_match = re.search(r"[Ee][Xx][Pp][_\s]?(\d+)", col_suffix)
    if exp_match:
        return f"Exp{exp_match.group(1)}"

    # Named prefixes
    if tokens and tokens[0].lower() == "evren":
        return "Evren"
    if tokens and tokens[0] == "2nd":
        return "2nd"

    # Date-style prefix (MMDDYY or MMDDYYYY)
    if tokens and re.fullmatch(r"\d{6,8}", tokens[0]):
        return tokens[0]

    # Fallback: first token
    return tokens[0] if tokens else "unknown"


def _parse_sample(col_suffix: str) -> dict | None:
    """
    Parse one LFQ intensity column suffix into sample metadata.
    Returns None to skip the sample (HeLa QC, unrecognised).
    """
    s = col_suffix.strip()
    # Split on underscore, space, or digit-letter boundaries
    tokens = re.split(r"[_\s]+", s)
    tokens_lower = [t.lower() for t in tokens]

    # Skip HeLa QC samples
    if any(t in HELA_TOKENS for t in tokens_lower):
        return None

    is_ctrl = any(t in CTRL_TOKENS for t in tokens_lower)

    if is_ctrl:
        drug = "CONTROL"
    else:
        drug = _extract_drug_from_tokens(tokens)
        if drug is None:
            # Cannot identify drug -- log and skip
            logger.debug(f"  Cannot identify drug in: '{s}' -- skipping")
            return None

    donor = _extract_donor(tokens, s)

    return {
        "sample":     s,
        "donor":      donor,
        "drug":       drug,
        "is_control": is_ctrl,
    }


def load_pxd014791(
    protein_groups_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PXD014791 proteinGroups.txt.

    Parameters
    ----------
    protein_groups_path : path to proteinGroups.txt
    metadata_path       : path to Metadata-Table.xlsx (optional but recommended)
                          Run `python parse_metadata.py` to download it.

    Returns
    -------
    matrix : DataFrame (n_samples, n_proteins), log2 LFQ intensities, NaN = missing
    meta   : DataFrame (n_samples,) with columns: donor, drug, is_control
    """
    if protein_groups_path is None:
        protein_groups_path = Path(DATA_DIR) / PXD014791_ACCESSION / "proteinGroups.txt"

    if not protein_groups_path.exists():
        raise FileNotFoundError(
            f"File not found: {protein_groups_path}\n"
            f"Run: pipenv run python remote_extract.py PXD014791"
        )

    logger.info(f"Loading {protein_groups_path} ...")
    pg = pd.read_csv(protein_groups_path, sep="\t", low_memory=False)
    logger.info(f"  Raw: {pg.shape[0]} protein groups, {pg.shape[1]} columns")

    # --- MaxQuant filters ---
    mask = pd.Series(True, index=pg.index)
    for col, bad in MQ_FILTER_COLS.items():
        if col in pg.columns:
            mask &= pg[col].fillna("").astype(str) != bad
    pg = pg[mask].copy()
    logger.info(f"  After MQ filters: {pg.shape[0]} protein groups")

    # --- Protein ID ---
    if "Gene names" in pg.columns:
        pg["_id"] = pg["Gene names"].fillna("").str.split(";").str[0].str.strip()
    else:
        pg["_id"] = pg["Protein IDs"].str.split(";").str[0].str.strip()
    pg = pg[pg["_id"] != ""].drop_duplicates("_id").set_index("_id")

    # --- LFQ columns ---
    lfq_cols = [c for c in pg.columns if c.startswith(PXD014791_LFQ_PREFIX)]
    logger.info(f"  Found {len(lfq_cols)} LFQ intensity columns")

    # --- Metadata-first parsing ---
    # Load metadata once; it is the authoritative source for Drug Treatment and Cell Line.
    # Token-based parsing is only the fallback for samples absent from the metadata.
    meta_df_xl = None
    meta_key_lookup = {}
    if metadata_path is None:
        metadata_path = Path(DATA_DIR) / PXD014791_ACCESSION / "Metadata-Table.xlsx"
    if metadata_path.exists():
        meta_df_xl = pd.read_excel(metadata_path)
        meta_df_xl["_key"] = (
            meta_df_xl["Raw Mass Spectrometry File"]
            .str.replace(r"\.raw$", "", regex=True, case=False)
            .str.strip()
            .str.lower()
        )
        meta_key_lookup = meta_df_xl.set_index("_key").to_dict("index")
    else:
        logger.warning("  Metadata-Table.xlsx not found -- using token parsing only")

    CTRL_UPPER = {"CTRL", "CTROL", "CONTROL", "DMSO", "VEH"}

    records, valid_cols = [], []
    skipped_hela = skipped_no_drug = skipped_meta_only = 0

    for col in lfq_cols:
        suffix = col[len(PXD014791_LFQ_PREFIX):].strip()

        # Skip HeLa QC (token-based -- same convention everywhere)
        if any(t.lower() == "hela" for t in re.split(r"[_\s]+", suffix)):
            skipped_hela += 1
            continue

        # --- Primary path: metadata lookup by exact filename ---
        sl = suffix.lower()
        if sl in meta_key_lookup:
            row = meta_key_lookup[sl]
            drug_code = str(row.get("Drug Treatment", "")).strip().upper()
            cell_line = str(row.get("Cell Line", "")).strip()
            is_ctrl   = drug_code in CTRL_UPPER
            if is_ctrl:
                drug = "CONTROL"
            else:
                drug = DRUG_MAP.get(drug_code, f"UNKNOWN_{drug_code}")
            records.append({
                "sample": suffix, "donor": cell_line,
                "drug": drug, "is_control": is_ctrl,
            })
            valid_cols.append(col)
            continue

        # --- Fallback path: token-based parsing (samples not in metadata) ---
        meta = _parse_sample(suffix)
        if meta is None:
            skipped_no_drug += 1
            continue
        records.append(meta)
        valid_cols.append(col)

    logger.info(
        f"  Parsed: {len(records)} samples kept, "
        f"{skipped_hela} HeLa skipped, {skipped_no_drug} no-drug-match skipped"
    )

    meta_df = pd.DataFrame(records).set_index("sample")

    # --- Infer donor for fallback samples using gel/experiment context ---
    # (Metadata-matched samples already have correct cell line; only fallback needs this)
    if meta_df_xl is not None:
        meta_df = _apply_metadata(meta_df, metadata_path, preloaded_xl=meta_df_xl)
    else:
        meta_df = _apply_metadata(meta_df, metadata_path)

    # --- Build matrix ---
    intensity = pg[valid_cols].astype(float).replace(0, np.nan)
    log_mat   = np.log2(intensity)
    log_mat.columns = meta_df.index
    log_mat   = log_mat.T   # samples x proteins

    _log_summary(meta_df)
    return log_mat, meta_df


def _apply_metadata(meta_df: pd.DataFrame, metadata_path: Path | None, preloaded_xl: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Apply Metadata-Table.xlsx to set:
      - donor   = Cell Line (PMC-A / PMC-B / PMC-D / PMC-E)
      - drug    = canonical name from metadata Drug Treatment abbreviation
      - is_control = True when Drug Treatment is a control keyword

    Column format confirmed from the actual file:
      Raw Mass Spectrometry File | Drug Treatment | Cell Line | Replicate Number
      031616_Gel2_Exp20_AFA_Tube_17.raw | AFA | PMC-A | 1
    """
    if metadata_path is None:
        metadata_path = Path(DATA_DIR) / PXD014791_ACCESSION / "Metadata-Table.xlsx"

    if not metadata_path.exists():
        logger.warning(
            "  Metadata-Table.xlsx not found -- using experiment-number donor proxy.\n"
            "  Run `pipenv run python parse_metadata.py` to download it."
        )
        return meta_df

    logger.info(f"  Applying {metadata_path.name} ...")
    xl = preloaded_xl if preloaded_xl is not None else pd.read_excel(metadata_path)

    # Strip .raw extension -> matches the LFQ column suffix exactly
    xl["key"] = (
        xl["Raw Mass Spectrometry File"]
        .str.replace(r"\.raw$", "", regex=True, case=False)
        .str.strip()
        .str.lower()          # normalise case for matching
    )
    meta_lookup = xl.set_index("key")

    # --- Pass 1: exact case-insensitive match ---
    matched, missing_idx = 0, []
    for idx in meta_df.index:
        if idx.lower() in meta_lookup.index:
            row = meta_lookup.loc[idx.lower()]
            cell_line = str(row["Cell Line"]).strip()
            drug_abbrev = str(row["Drug Treatment"]).strip()
            meta_df.at[idx, "donor"] = cell_line
            is_ctrl = drug_abbrev.upper() in {"CTRL","CTROL","CONTROL","DMSO","VEH"}
            meta_df.at[idx, "is_control"] = is_ctrl
            if is_ctrl:
                meta_df.at[idx, "drug"] = "CONTROL"
            else:
                canonical = DRUG_MAP.get(drug_abbrev.upper())
                meta_df.at[idx, "drug"] = canonical if canonical else f"UNKNOWN_{drug_abbrev}"
            matched += 1
        else:
            missing_idx.append(idx)

    # --- Pass 2: for unmatched, infer cell line from same gel+experiment ---
    # Build: experiment_key -> cell_line from already-matched samples
    exp_to_cell: dict[str, str] = {}
    for idx in meta_df.index:
        if idx in missing_idx:
            continue
        donor = meta_df.at[idx, "donor"]
        if not donor.startswith("PMC-"):
            continue
        # Experiment key = date + gel + optional exp number (case-insensitive prefix)
        key = _gel_exp_key(idx)
        if key:
            exp_to_cell.setdefault(key, donor)

    inferred = 0
    for idx in missing_idx:
        key = _gel_exp_key(idx)
        if key and key in exp_to_cell:
            meta_df.at[idx, "donor"] = exp_to_cell[key]
            inferred += 1

    missing = len(missing_idx) - inferred
    logger.info(
        f"  Metadata matched {matched} samples, "
        f"{inferred} inferred from gel/experiment, "
        f"{missing} still unmatched (using experiment-number proxy)"
    )
    donors = sorted(meta_df["donor"].unique())
    logger.info(f"  Cell lines in data: {donors}")
    return meta_df


def _gel_exp_key(filename: str) -> str | None:
    """
    Extract a gel+experiment key to match drug samples without metadata
    to the controls from the same gel run.
    E.g. '09022016_Gel21_EXP59_PRE_tube6' -> '09022016_gel21_exp59'
         '031616_Gel3_Exp22_ERL_Tube_5'    -> '031616_gel3_exp22'
         '031616_Gel1_BOS_Tube_11'          -> '031616_gel1'  (no exp number)
    """
    s = filename.lower()
    # Try date + gel + optional exp number
    m = re.match(r"^(\d{6,8}[_\s]gel\d+(?:[_\s]exp[_\s]?\d+)?)", s)
    if m:
        return m.group(1)
    # Evren prefix: Evren_GelN_...
    m = re.match(r"^(evren[_\s]gel\d+(?:[_\s]exp[_\s]?\d+)?)", s)
    if m:
        return m.group(1)
    return None


def _log_summary(meta_df: pd.DataFrame) -> None:
    drugs = sorted(meta_df[~meta_df["is_control"]]["drug"].unique())
    donors = sorted(meta_df["donor"].unique())
    n_ctrl = meta_df["is_control"].sum()
    n_drug = (~meta_df["is_control"]).sum()
    logger.info(
        f"  Samples: {n_drug} drug-treated, {n_ctrl} control\n"
        f"  Unique donors (proxy): {len(donors)} -- {donors[:10]}"
        f"{'...' if len(donors) > 10 else ''}\n"
        f"  Unique drugs: {len(drugs)} -- {drugs}"
    )
    # Check coverage of held-out drug list
    held_found    = PXD014791_HELD_OUT_DRUGS & set(drugs)
    held_missing  = PXD014791_HELD_OUT_DRUGS - set(drugs)
    training_found = PXD014791_TRAINING_DRUGS & set(drugs)
    logger.info(
        f"  Held-out drugs found: {len(held_found)}/21 -- "
        f"missing: {held_missing if held_missing else 'none'}"
    )
    logger.info(
        f"  Training drugs found: {len(training_found)}/1 -- "
        f"{training_found if training_found else 'MISSING -- check PRE/TRA/TRS mapping'}"
    )
