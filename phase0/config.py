"""
All constants from the Tesorai paper (Supplementary Information).
Held-out drug lists copied verbatim from SI to avoid transcription errors.
"""

DATA_DIR = "data"

# ------------------------------------------------------------------
# PXD014791  Cardiomyocyte kinase inhibitor LFQ dataset
# Xiong et al. Scientific Data 2022
# 4 iPSC-derived cardiomyocyte donor lines, 21 kinase inhibitors
# 237 drug / 71 control samples, LFQ quantification
# PRIDE FTP: ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2022/01/PXD014791/
# ------------------------------------------------------------------
PXD014791_ACCESSION = "PXD014791"

PXD014791_HELD_OUT_DRUGS = {
    "Afatinib", "Axitinib", "Bosutinib", "Cabozantinib", "Dabrafenib",
    "Dasatinib", "Erlotinib", "Gefitinib", "Imatinib", "Lapatinib",
    "Nilotinib", "Pazopanib", "Ponatinib", "Regorafenib", "Ruxolitinib",
    "Sorafenib", "Sunitinib", "Tofacitinib", "Trametinib", "Vandetanib",
    "Vemurafenib",
}

PXD014791_TRAINING_DRUGS = {"Trastuzumab"}

# Expected: 21 held-out drugs x ~4 donors = ~58 held-out pairs (paper: n=58)
PXD014791_EXPECTED_PAIRS = 58

# Column prefix in MaxQuant proteinGroups.txt
PXD014791_LFQ_PREFIX = "LFQ intensity "

# Keywords identifying control samples in sample names
PXD014791_CONTROL_KEYWORDS = {"dmso", "vehicle", "control", "ctrl", "veh", "0"}

# Protein filter flags in MaxQuant
MQ_FILTER_COLS = {
    "Only identified by site": "+",
    "Reverse": "+",
    "Potential contaminant": "+",
}

# ------------------------------------------------------------------
# ProTargetMiner  TMT dataset
# Saei et al. Nature Communications 2019
# A549, MCF7, RKO cancer cell lines; 56 anticancer drugs; TMT-10
# 237 drug / 35 control samples
# PRIDE accessions: PXD009775 (main), PXD009644 (deep A549),
#                   PXD013134 (deep MCF7+RKO)
# ------------------------------------------------------------------
PTM_ACCESSIONS = ["PXD009775", "PXD009644", "PXD013134"]

PTM_CONTEXTS = ["A549", "MCF7", "RKO"]

PTM_HELD_OUT_DRUGS = {
    "2-methoxyestradiol", "5-fluorouracil", "Afatinib", "Apatinib",
    "Auranofin", "Axitinib", "Azacitidine", "Bortezomib", "Bosutinib",
    "Cabozantinib", "Camptothecin", "Carmofur", "Crizotinib", "Dasatinib",
    "Docetaxel", "Doxorubicin", "Enzalutamide", "Epirubicin", "Etoposide",
    "Everolimus", "Floxuridine", "Fludarabine", "Gefitinib", "Genistein",
    "Idarubicin", "Irinotecan", "Lapatinib", "Methotrexate", "Mitotane",
    "Nilotinib", "OSI-420", "Oxaliplatin", "Paclitaxel", "Pazopanib",
    "Ponatinib", "RITA", "Raltitrexed", "Regorafenib", "Ruxolitinib",
    "Sorafenib", "Sunitinib", "Temsirolimus", "Teniposide", "Topotecan",
    "Vemurafenib", "Vincristine", "Vismodegib",
}

PTM_TRAINING_DRUGS = {
    "8-azaguanine", "Azaguanine", "Bleomycin", "Lomustine", "Nutlin",
    "OSW-1", "Tri-1", "Tri-2", "b-AP15",
}

PTM_EXPECTED_PAIRS = 61

# MaxQuant TMT column prefix (absolute reporter intensities, not ratios)
# Tesorai paper uses "isotope-corrected reporter intensities" = these columns:
PTM_REPORTER_PREFIX = "Reporter intensity corrected "

# Number of TMT channels per plex
PTM_CHANNELS = 10

PTM_CONTROL_KEYWORDS = {"dmso", "vehicle", "control", "ctrl"}

# ------------------------------------------------------------------
# Benchmark parameters
# ------------------------------------------------------------------
K_NEIGHBORS = 5          # k for cosine-kNN predictor
N_HVG = 2000             # high-variance genes for HVG baseline
N_PCA = 512              # PCA components
TOP_N_DE = 100           # top-N differentially expressed for primary metric
MIN_PROTEINS_PER_PAIR = 500  # pairs with fewer detected proteins are skipped
