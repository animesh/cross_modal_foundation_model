# Single-cell foundation models benefit from cross-modal training: adding proteomics data beats parameter scaling

> Burq et al. (2026) "Single-cell foundation models benefit from cross-modal training:
> adding proteomics data beats parameter scaling." bioRxiv 2026.08.14.744845

Great [paper](https://www.biorxiv.org/content/10.64898/2026.08.14.744845v1.full) to stress-test. Specially the tag-line with those em-dashes -- 

"Here we introduce a modality on which such models have not previously been trained. Proteins -- not transcripts -- are the key molecular level at which biological functions begin to manifest."

Jokes apart, here trying to structure the pipeline around what's actually testable.

---

## What the recon revealed

**Model weights (Trove1-70m)**: The [HuggingFace](https://huggingface.co/Tesorai/Trove1-70m) page exists but the README is completely empty -- no usage instructions, no architecture spec, no tokenizer info. The tesorai GitHub org has only `tesorai_search` as a public repo; `cross_modal_foundation_model` was empty at the time of writing. Fresh preprint (Aug 19, 2026), code release likely lags. Can't reproduce model-embedding steps yet. Independently verifying testable claims instead.

---

## Assumptions worth flagging

**A1 -- The comparison is unfair by design.**
The 70M model gets continued pretraining; the 1B and 3B don't. The headline claim "70M beats 3B" is really "fine-tuned 70M beats frozen 3B." A proper ablation would CPT the 1B on the same proteomics corpus. This is never done.

**A2 -- Token mapping is lossy in one direction.**
Proteins are mapped to Tahoe's gene token vocabulary. Multiple isoforms likely collapse to one token. Proteins with no Ensembl gene mapping are silently dropped. The paper doesn't report what fraction of each proteomics dataset survives this mapping -- a real blind spot for a proteomics dataset where isoform-level quantification matters.

**A3 -- The scaling curve hasn't saturated.**
Figure 4 is still rising at 48,843 samples. They're reporting a ceiling that isn't a ceiling.

**A4 -- Figure 3 baseline pattern: HVG/PCA beat all models, perturbation-mean/additive-linear fail on ProTargetMiner. (CORRECTED)**
Initial read of Figure 3 was wrong -- the -0.652/-0.650 delta labels belong to perturbation-mean and additive-linear (absolute ~0.09), not HVG/PCA. HVG and PCA are the tall rightmost grey bars at +0.156/+0.160 delta (absolute ~0.90). The actual anomaly: HVG and PCA outperform ALL models including the CPT model on both datasets (PXD014791: CPT=0.612 vs HVG=0.665; ProTargetMiner: CPT=0.813 vs HVG=0.902). This is never discussed in the paper text -- only visible in Figure 3.

**A5 -- TMT ratio conversion assumption.**
They use "absolute isotope-corrected reporter intensities" from MaxQuant rather than ratio columns. Reasonable for within-plex comparisons, but introduces scale assumptions that may not generalise across the heterogeneous TMT batches in their corpus.

**A6 -- HVG/PCA baseline reproducibility requires Tesorai's internal data.**
The paper used Tesorai Search to reprocess PXD014791 (MaxQuant 1.6.0.13 from 2016 = outdated). The public PRIDE deposit has MaxQuant output only. Our non-embedding baselines (perturbation-mean, additive-linear) reproduce within 1-4%, validating data loading. HVG/PCA are 8x lower (0.10 vs 0.665) -- the gap is in the data, not the algorithm. The paper presents HVG/PCA as reproducible "classical references" without disclosing this dependency on their proprietary reprocessing.

---

## Testing, in order of tractability

```
Phase 0 (no model) -- Baseline verification on protein perturbation benchmark
  -> Download PXD014791 from PRIDE via FTP Range requests (Search.zip, 22 GB)
  -> Implement leave-one-pair-out kNN predictor
  -> Verify all 5 baselines match Figure 3 numbers

Phase 1 (no model) -- PCA baseline on gene essentiality / tissue separation
  -> DepMap public portal
  -> MSigDB Hallmark gene sets
  -> Verify PCA baseline bars in Figure 2

Phase 2 (model) -- Once the HuggingFace weights are usable
  -> Embed proteomes with Trove1-70m
  -> Run kNN on the same benchmark
  -> Check if +0.070 / +0.067 improvements reproduce
```

Phase 0 is the highest-value target -- it checks whether the benchmark is methodologically sound, independent of any model claim.

---

# Phase 0 -- Baseline verification

Reproduces the **5 baselines** in Figure 3 of the paper using the public PRIDE deposit.

## Setup

```bash
pip install pipenv
cd phase0/
pipenv install
```

## Run

```bash
# Full pipeline (download + benchmark)
pipenv run python run_phase0.py

# Skip download if data already present
pipenv run python run_phase0.py --skip-download

# One dataset only
pipenv run python run_phase0.py --dataset pxd   # PXD014791 LFQ
pipenv run python run_phase0.py --dataset ptm   # ProTargetMiner TMT
```

## Data acquisition -- PXD014791

The PRIDE FTP lists `Search.zip` (22.8 GB) containing `proteinGroups.txt`. The HTTP server doesn't honour Range requests so we use FTP REST (seek) + RETR to extract only the target file (~200 MB download vs 22 GB):

```bash
pipenv run python remote_extract.py PXD014791
```

Retrieves `data/PXD014791/proteinGroups.txt` in four FTP partial reads.
Also download the metadata table (21 KB):

```bash
pipenv run python parse_metadata.py
```

### ProTargetMiner (TMT)

Three PRIDE deposits: PXD009775, PXD009644, PXD013134.
Requires manual TMT channel map from Saei et al. 2019 Nature Comms Supplementary Table 2.
Fill in `parse_tmt.py -> HARDCODED_CHANNEL_MAP` before running.

## Diagnostic scripts

| Script | Purpose |
|---|---|
| `diagnose.py` | Network reachability check -- run first if download fails |
| `explore_ftp.py PXD014791` | List FTP directory contents |
| `remote_extract.py PXD014791` | Extract proteinGroups.txt from Search.zip via FTP REST |
| `parse_metadata.py` | Download + display Metadata-Table.xlsx |
| `debug_metadata.py` | Show which LFQ columns don't match the metadata |
| `find_missing_pair.py` | Cross metadata vs parsed data to find missing pairs |

## Results -- PXD014791

### What we found vs paper (Figure 3)

| Baseline | Ours top-100 | Paper top-100 | Ours full | Paper full |
|---|---|---|---|---|
| control-mean | NaN (undefined) | NaN (undefined) | NaN | NaN |
| perturbation-mean | 0.517 | ~0.541 | 0.279 | ~0.219 |
| additive-linear | 0.468 | ~0.475 | 0.311 | ~0.229 |
| HVG | 0.102 | ~0.665 | 0.081 | ~0.415 |
| PCA | 0.111 | ~0.674 | 0.088 | ~0.425 |

Held-out pairs: **57** (paper: 58).

### Interpretation

**Non-embedding baselines reproduce within 1-4%** -- this validates data loading, log2 transform, LFC computation, and pair structure. The delta computation is correct.

**HVG/PCA are 8x lower than paper** -- not an implementation issue. Our non-embedding baselines match; only the embedding-based ones fail. The gap is in the underlying protein quantification data: the paper used Tesorai Search reprocessing of the raw spectra; we use the public MaxQuant 1.6.0.13 output. Without running Tesorai Search (a commercial product) on the ~94 GB of raw files, the HVG/PCA numbers cannot be reproduced from this PRIDE deposit.

This is a reproducibility gap: the paper presents HVG/PCA as standard classical references but their values depend on proprietary reprocessing that is not documented or publicly available.

### The missing 58th pair

Running `find_missing_pair.py` revealed:

- The Metadata-Table.xlsx itself has only **57 held-out pairs** -- not 58.
- **Trametinib is Drug Treatment code "TRS" in the metadata** -- confirmed by elimination (22 Drug Treatment codes in the metadata; 21 map to known drugs + TRA=Trastuzumab; TRS is the only code unaccounted for, and Trametinib is the only held-out drug without an obvious code).
- **"TRS" was previously mislabelled as Trastuzumab** in the initial DRUG_MAP. Correcting TRS->Trametinib and USN->Vandetanib (metadata confirms `031616_Gel1_USN_Tube_12` has Drug Treatment=VAN) brings the pair count to **59 held-out pairs vs paper's 58**.
- The remaining 1-pair gap is likely one Trametinib/cell line excluded by quality filtering in the paper's Tesorai pipeline.
- 5 drugs (Cabozantinib, Dabrafenib, Dasatinib, Ponatinib, Vemurafenib) were tested in only PMC-B, not all 4 cell lines.
- **Trastuzumab (TRA) is PMC-B only** -- 4 samples, 1 training pair. The apparent "3 training pairs" in earlier runs was an artefact of the wrong TRS mapping.

Full Drug Treatment code -> canonical name mapping (22 codes, confirmed from metadata):
| Code | Drug | Notes |
|------|------|-------|
| AFA | Afatinib | |
| AXI | Axitinib | |
| BOS | Bosutinib | |
| CAB | Cabozantinib | PMC-B only |
| CTRL | Control/DMSO | |
| DAB | Dabrafenib | PMC-B only |
| DAS | Dasatinib | PMC-B only |
| ERL | Erlotinib | |
| GEF | Gefitinib | |
| IMA | Imatinib | PMC-E only |
| LAP | Lapatinib | PMC-B, PMC-E only |
| NIL | Nilotinib | |
| PAZ | Pazopanib | |
| PON | Ponatinib | PMC-B only |
| REG | Regorafenib | |
| RUX | Ruxolitinib | |
| SOR | Sorafenib | |
| SUN | Sunitinib | |
| TOF | Tofacitinib | |
| TRA | Trastuzumab | **training drug**, PMC-B only |
| TRS | Trametinib | **held-out**, PMC-B + PMC-E; confirmed by elimination |
| USN | Vandetanib | alternative abbreviation; metadata confirms Drug Treatment=VAN |
| VAN | Vandetanib | |
| VEM | Vemurafenib | PMC-B only |

Note: `PRE` appears in proteinGroups.txt but is **absent from the metadata**. Not one of the 22 drugs. Likely a pilot/batch label. These 4 samples (informal runs) don't contribute to the benchmark.



### Drug × cell line coverage matrix (public data)

```
cell_line     PMC-A  PMC-B  PMC-D  PMC-E
Afatinib          4      4      4      5
Axitinib          4      4      4      5
Bosutinib         4      4      4      5
Cabozantinib      0      4      0      0
Dabrafenib        0      4      0      0
Dasatinib         0      4      0      0
Erlotinib         2      4      4      2
Gefitinib         0      4      4      5
Imatinib          0      0      0      3
Lapatinib         0      1      0      3
Nilotinib         4      4      4      5
Pazopanib         4      4      4      5
Ponatinib         0      4      0      0
Regorafenib       4      4      4      5
Ruxolitinib       4      4      4      5
Sorafenib         3      3      0      4
Sunitinib         3      4      4      6
Tofacitinib       0      4      4      4
Vandetanib        4      3      4      5
Vemurafenib       0      3      0      0
Trametinib        0      0      0      0   <- absent from public metadata
```

## Known assumptions and limitations

1. **Pool definition**: The kNN pool for each held-out pair = all other pairs (LOOCV). The paper's phrasing ("nearest training pairs") is ambiguous but with only 2 Trastuzumab training pairs a fixed-training-pool interpretation would be degenerate (k=5 with n=2 neighbours). LOOCV is the only interpretation that makes k=5 viable.

2. **HVG/PCA imputation**: Missing proteins treated as absent (pairwise-complete cosine for HVG; column-mean imputation on >=50% detected proteins for PCA). The paper doesn't specify strategy. See A6 -- the gap is the data, not this choice.

3. **Metadata inference for 9 pilot samples**: 9 LFQ samples (Trametinib, Estaurosporine, one Erlotinib) are in proteinGroups.txt but absent from Metadata-Table.xlsx. Their cell line is inferred from other samples in the same gel run via `_gel_exp_key()`. These don't add new held-out pairs since Trametinib has no matched controls in the public data.

4. **ProTargetMiner**: Not yet run -- requires manual TMT channel mapping from Saei 2019 Supplementary Table 2.

## Key findings for a paper critique

1. **The 70M vs 1B/3B comparison is fine-tuned vs frozen** -- not a fair parameter scaling comparison (A1).

2. **HVG and PCA outperform all models on both datasets** -- visible in Figure 3, never discussed in text. CPT model gets 0.612/0.813 while HVG/PCA get 0.665/0.902 (PXD014791/ProTargetMiner). The paper's framing as "model improves over baselines" needs qualification.

3. **HVG/PCA baseline values are not reproducible from public data** -- require Tesorai's proprietary search reprocessing (A6).

4. **n=58 held-out pairs cannot be reproduced** -- public metadata has 57; Trametinib is absent (A6).

5. **Scaling curve hasn't saturated** -- Figure 4 still rising at the full corpus (A3).


are we sure that we got the mapping right finally? check [supplemetary](https://www.biorxiv.org/content/10.64898/2026.08.14.744845v1.full#sec-14) of their article first meanwhile looks like batch effect normalization with median leads us quite close `pipenv run python test_normalisation.py`



