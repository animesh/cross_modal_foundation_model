"""
Find the exact (drug, cell_line) pair that gives us 57 instead of 58.

Approach:
  1. Load Metadata-Table.xlsx -> enumerate all held-out (drug, cell_line) pairs
  2. Load proteinGroups.txt LFQ columns -> enumerate what we actually have
  3. Diff the two to find what's missing and why

Run: pipenv run python find_missing_pair.py
"""

import re
from pathlib import Path
import pandas as pd

from config import PXD014791_HELD_OUT_DRUGS, PXD014791_TRAINING_DRUGS
from parse_lfq import DRUG_MAP, _gel_exp_key

DATA_DIR = Path("data/PXD014791")
LFQ_PREFIX = "LFQ intensity "


# ------------------------------------------------------------------
# Step 1: what the Metadata-Table says should exist
# ------------------------------------------------------------------
xl = pd.read_excel(DATA_DIR / "Metadata-Table.xlsx")
xl["drug_canonical"] = xl["Drug Treatment"].str.upper().map(
    lambda x: DRUG_MAP.get(x, x)
)
xl["cell_line"] = xl["Cell Line"].str.strip()

# All drug samples in metadata
drug_meta = xl[~xl["Drug Treatment"].str.upper().isin(
    {"CTRL", "CTROL", "CONTROL", "DMSO", "VEH"}
)]

print("=== Held-out drug × cell_line pairs in Metadata-Table.xlsx ===")
held_in_meta = drug_meta[drug_meta["drug_canonical"].isin(PXD014791_HELD_OUT_DRUGS)]
meta_pairs = held_in_meta.groupby(["drug_canonical", "cell_line"]).size().reset_index()
meta_pairs.columns = ["drug", "cell_line", "n_samples"]
print(f"Total held-out pairs in metadata: {len(meta_pairs)}")
print(meta_pairs.to_string(index=False))


# ------------------------------------------------------------------
# Step 2: what we actually parsed from proteinGroups.txt
# ------------------------------------------------------------------
print("\n=== Loading LFQ column names from proteinGroups.txt ===")
pg = pd.read_csv(DATA_DIR / "proteinGroups.txt", sep="\t", low_memory=False, nrows=1)
lfq_cols = [c for c in pg.columns if c.startswith(LFQ_PREFIX)]

# Rebuild the donor map using the same logic as parse_lfq._apply_metadata
xl2 = xl.copy()
xl2["key_lower"] = (
    xl2["Raw Mass Spectrometry File"]
    .str.replace(r"\.raw$", "", regex=True, case=False)
    .str.strip()
    .str.lower()
)
meta_lookup = xl2.set_index("key_lower")

CTRL_UPPER = {"CTRL", "CTROL", "CONTROL", "DMSO", "VEH"}
HELA = {"hela"}

parsed_samples = []
for col in lfq_cols:
    suffix = col[len(LFQ_PREFIX):].strip()
    tokens = re.split(r"[_\s]+", suffix)
    tokens_lower = [t.lower() for t in tokens]
    if any(t in HELA for t in tokens_lower):
        continue

    # Direct metadata lookup
    sl = suffix.lower()
    if sl in meta_lookup.index:
        row = meta_lookup.loc[sl]
        drug_abbrev = str(row["Drug Treatment"]).strip().upper()
        is_ctrl = drug_abbrev in CTRL_UPPER
        drug = "CONTROL" if is_ctrl else DRUG_MAP.get(drug_abbrev, f"UNKNOWN_{drug_abbrev}")
        cell_line = str(row["Cell Line"]).strip()
        parsed_samples.append({"sample": suffix, "drug": drug, "cell_line": cell_line, "is_control": is_ctrl})
        continue

    # Drug identification (fallback)
    is_ctrl = any(t.lower() in {"ctrl","ctrol","control","dmso"} for t in tokens)
    drug = None if not is_ctrl else "CONTROL"
    if not is_ctrl:
        for t in tokens:
            if t.upper() in DRUG_MAP:
                drug = DRUG_MAP[t.upper()]
                break
    if drug is None:
        continue

    # Cell line from gel/exp key
    exp_key = _gel_exp_key(suffix)
    cell_line = "unknown"
    # Try to find from already-matched samples (simplified: just mark as inferred)
    if exp_key:
        cell_line = f"inferred({exp_key})"

    parsed_samples.append({"sample": suffix, "drug": drug, "cell_line": cell_line, "is_control": is_ctrl})

parsed_df = pd.DataFrame(parsed_samples)
drug_parsed = parsed_df[~parsed_df["is_control"]]
held_parsed = drug_parsed[drug_parsed["drug"].isin(PXD014791_HELD_OUT_DRUGS)]

parsed_pairs = held_parsed.groupby(["drug", "cell_line"]).size().reset_index()
parsed_pairs.columns = ["drug", "cell_line", "n_samples"]

print(f"Total held-out pairs in parsed data: {len(parsed_pairs)}")


# ------------------------------------------------------------------
# Step 3: diff
# ------------------------------------------------------------------
print("\n=== Missing pairs (in metadata but NOT in parsed data) ===")
meta_set   = set(zip(meta_pairs["drug"],   meta_pairs["cell_line"]))
parsed_set = set(zip(parsed_pairs["drug"], parsed_pairs["cell_line"]))

missing = meta_set - parsed_set
extra   = parsed_set - meta_set

if missing:
    for drug, cell_line in sorted(missing):
        # Find the actual sample names in metadata for this pair
        rows = held_in_meta[
            (held_in_meta["drug_canonical"] == drug) &
            (held_in_meta["cell_line"] == cell_line)
        ]
        files = rows["Raw Mass Spectrometry File"].tolist()
        print(f"  MISSING: ({drug}, {cell_line})")
        for f in files:
            key = f.replace(".raw", "").replace(".RAW", "")
            in_lfq = any(col[len(LFQ_PREFIX):].lower() == key.lower() for col in lfq_cols)
            print(f"    file: {f}  in_proteinGroups: {in_lfq}")
else:
    print("  None -- metadata and parsed data have identical pair sets")

if extra:
    print("\n=== Extra pairs (in parsed data but NOT in metadata) ===")
    for drug, cell_line in sorted(extra):
        print(f"  EXTRA: ({drug}, {cell_line})")


# ------------------------------------------------------------------
# Step 4: full coverage matrix
# ------------------------------------------------------------------
print("\n=== Coverage matrix: drugs × cell lines (metadata) ===")
pivot = meta_pairs.pivot(index="drug", columns="cell_line", values="n_samples").fillna(0).astype(int)
print(pivot.to_string())
