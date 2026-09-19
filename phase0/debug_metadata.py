"""
Print the 9 unmatched sample names and compare them to metadata keys.
Run: pipenv run python debug_metadata.py
"""
import re
from pathlib import Path
import pandas as pd

pg_path = Path("data/PXD014791/proteinGroups.txt")
xl_path = Path("data/PXD014791/Metadata-Table.xlsx")
LFQ_PREFIX = "LFQ intensity "

DRUG_MAP = {
    "AFA":"Afatinib","AXI":"Axitinib","BOS":"Bosutinib","CAB":"Cabozantinib",
    "DAB":"Dabrafenib","DAS":"Dasatinib","ERL":"Erlotinib","EST":"Estaurosporine",
    "GEF":"Gefitinib","IMA":"Imatinib","LAP":"Lapatinib","NIL":"Nilotinib",
    "PAZ":"Pazopanib","PON":"Ponatinib","PRE":"Trametinib","REG":"Regorafenib",
    "RUX":"Ruxolitinib","SOR":"Sorafenib","SUN":"Sunitinib","TOF":"Tofacitinib",
    "TRA":"Trastuzumab","TRS":"Trastuzumab","USN":"Sunitinib","VAN":"Vandetanib","VEM":"Vemurafenib",
}
CTRL_TOKENS = {"ctrl","ctrol","control","dmso","vehicle"}
HELA_TOKENS = {"hela"}

def has_drug(suffix):
    tokens = re.split(r"[_\s]+", suffix)
    if any(t.lower() in HELA_TOKENS for t in tokens): return False
    if any(t.lower() in CTRL_TOKENS for t in tokens): return True
    return any(t.upper() in DRUG_MAP for t in tokens)

# Load LFQ column suffixes
print("Loading proteinGroups.txt ...")
pg = pd.read_csv(pg_path, sep="\t", low_memory=False, nrows=1)
lfq_cols = [c for c in pg.columns if c.startswith(LFQ_PREFIX)]
suffixes = [c[len(LFQ_PREFIX):].strip() for c in lfq_cols if has_drug(c[len(LFQ_PREFIX):])]
suffixes = [s for s in suffixes if not re.search(r"\bhela\b", s, re.I)]
print(f"  {len(suffixes)} non-HeLa LFQ samples")

# Load metadata
xl = pd.read_excel(xl_path)
xl["key_orig"] = xl["Raw Mass Spectrometry File"].str.replace(r"\.raw$", "", regex=True, case=False).str.strip()
xl["key_lower"] = xl["key_orig"].str.lower()
meta_keys_lower = set(xl["key_lower"])

print(f"\n  Metadata has {len(xl)} rows, {len(meta_keys_lower)} unique keys")

# Find unmatched
unmatched = [s for s in suffixes if s.lower() not in meta_keys_lower]
print(f"\n  {len(unmatched)} unmatched LFQ samples:\n")
for s in unmatched:
    print(f"  LFQ suffix : {s!r}")
    # Find closest metadata key
    s_lower = s.lower()
    candidates = [k for k in meta_keys_lower if s_lower[:15] in k or k[:15] in s_lower]
    if candidates:
        print(f"  Near match : {candidates[0]!r}")
    else:
        print(f"  No near match found")
    print()

# Also check a few matched ones for comparison
matched = [s for s in suffixes if s.lower() in meta_keys_lower][:3]
print(f"\nFor comparison -- 3 successfully matched samples:")
for s in matched:
    row = xl[xl["key_lower"] == s.lower()].iloc[0]
    print(f"  LFQ: {s!r} -> Cell Line: {row['Cell Line']}, Drug: {row['Drug Treatment']}")
