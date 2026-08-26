# Single-cell foundation models benefit from cross-modal training: adding proteomics data beats parameter scaling

> Burq et al. (2026) "Single-cell foundation models benefit from cross-modal training:
> adding proteomics data beats parameter scaling." bioRxiv 2026.08.14.744845

Great [paper](https://www.biorxiv.org/content/10.64898/2026.08.14.744845v1.full) to stress-test. Specially the tag-line with those em-dashes 😜 

"Here we introduce a modality on which such models have not previously been trained. Proteins — not transcripts—are the key molecular level at which biological functions begin to manifest."

Jokes apart, here trying to structure the pipeline around what's actually testable?

---

## What the recon revealed

**Model weights (Trove1-70m)**: The [HuggingFace](https://huggingface.co/Tesorai/Trove1-70m) page exists but the README is completely empty, no usage instructions, no architecture spec, no tokenizer info. The tesorai GitHub org has only `tesorai_search` as a public repo -- `cross_modal_foundation_model` is empty-ish. This is a fresh preprint (Aug 19, 2026) so the code release likely lags. Check [BriVL-Brain](https://github.com/RERV/BriVL-Brain)? Probably can't reproduce the model-embedding steps yet. Lets independently verify the some important claims 🤞

---

## Assumptions worth flagging

**A1 The comparison is probably unfair by design.**
The 70M model gets continued pretraining it seems and the 1B and 3B don't. The claim "70M beats 3B" is probably "fine-tuned 70M beats frozen 3B." A proper ablation would CPT the 1B on the same proteomics corpus. They never do this AFAICS?

**A2 Token mapping is lossy in one direction.**
Proteins are mapped to Tahoe's gene token vocabulary it seems likely that multiple isoforms collapse to one token. Proteins with no Ensembl gene mapping are silently dropped or? The paper doesn't report what fraction of each proteomics dataset survives this mapping...

**A3 The scaling curve hasn't saturated.**
Figure 4 is still rising at 48,843 samples. Are they reporting a ceiling that isn't a ceiling?

**A4 The ProTargetMiner baseline numbers look confusing.**
HVG and PCA show *negative* correlation relatively on perturb-mean,additive-lines (Figure 3, full-proteome: -0.658, -0.623) warrants independent verification 

**A5 TMT ratio conversion assumption.**
They convert TMT ratios to absolute intensities using isotope-corrected MaxQuant reporter intensities it seems. Does it generalize to other data?

---

## Testing, in order of tractability

```
Phase 0 (no model) -- Baseline verification on protein perturbation benchmark
  -> Download PXD014791 from PRIDE
  -> Download ProTargetMiner (PRIDE PXD020065? / Nature Comms supplementary)
  -> Implement leave-one-pair-out kNN predictor
  -> Verify all 5 baselines match Figure 3 numbers

Phase 1 (no model) -- PCA baseline on gene essentiality / tissue separation
  -> DepMap public portal
  -> MSigDB Hallmark gene sets
  -> Verify PCA baseline bars in Figure 2

Phase 2 (model) -- Once the HuggingFace weights are usable
  -> Embed proteomes with Trove1-70m
  -> Run kNN on the same benchmark
  -> Check if improvements (+0.070 / +0.067?) reproduce
```

---

Phase 0 is probably the highest-value target because it seems to check whether their benchmark is methodologically sound, independent of any model claim?

---

# Phase 0 -- Baseline verification of Tesorai cross-modal pretraining paper

Reproduces the **5 baselines** in Figure 3
## What this tests

verify the baseline numbers in Figure 3 *before* trusting the model-dependent claims.
If our baselines don't match the paper's baselines, something is wrong with either:
- our data parsing, or
- the paper's benchmark implementation.

The relatvie **negative** HVG/PCA values on ProTargetMiner (-0.65) are a primary target.

## Setup

```bash
git clone https://github.com/animesh/cross_modal_foundation_model
pip install pipenv 
cd phase0/
pipenv install
```

## Run

### Full pipeline (download + benchmark)
```bash
pipenv run python run_phase0.py
```

### Skip download (if you already have the data files)
```bash
pipenv run python run_phase0.py --skip-download
```

### One dataset only
```bash
pipenv run python run_phase0.py --dataset pxd   # PXD014791 only (LFQ, easier)
pipenv run python run_phase0.py --dataset ptm   # ProTargetMiner only (TMT) ?
```

## Data files required

### PXD014791 (cardiomyocyte LFQ) -- auto-downloaded
File: `data/PXD014791/proteinGroups.txt`
Source: https://www.ebi.ac.uk/pride/archive/projects/PXD014791
MaxQuant combined/txt/proteinGroups.txt (~50-150 MB)

If auto-download fails (FTP blocked), download manually:
1. Go to https://www.ebi.ac.uk/pride/archive/projects/PXD014791
2. Find the MaxQuant output zip / combined folder
3. Extract proteinGroups.txt to data/PXD014791/proteinGroups.txt

### ProTargetMiner (TMT) -- requires manual channel map step
Files needed:
  data/PXD009775/proteinGroups.txt
  data/PXD009644/proteinGroups.txt
  data/PXD013134/proteinGroups.txt

Sources:
  https://www.ebi.ac.uk/pride/archive/projects/PXD009775
  https://www.ebi.ac.uk/pride/archive/projects/PXD009644
  https://www.ebi.ac.uk/pride/archive/projects/PXD013134

**CRITICAL MANUAL STEP: TMT channel map**

The Tesorai paper uses "absolute isotope-corrected reporter intensities from the
deposited MaxQuant output" with "the published TMT-10 design (Supplementary Table 2)".
That Supplementary Table 2 is from:

  Saei et al. (2019) Nature Communications 10:5715
  https://doi.org/10.1038/s41467-019-13582-8

Download Supplementary Data 2 (the Excel file at the Nature Comms page).
It contains the TMT-10 channel assignments for each experiment batch.

Then fill in `parse_tmt.py` -> `HARDCODED_CHANNEL_MAP`:

```python
HARDCODED_CHANNEL_MAP = {
    # Format: (folder_name, channel_index_0based): {cell_line, drug, is_control}
    ("PXD009775", 0): {"cell_line": "A549", "drug": "Doxorubicin", "is_control": False},
    ("PXD009775", 9): {"cell_line": "A549", "drug": "DMSO",        "is_control": True},
    # ... one entry per channel per batch file
}
```

If the PRIDE deposits contain an SDRF or experimental design file,
the pipeline will try to auto-detect the mapping. Check phase0.log to see
whether auto-detection succeeded.

## Output files

```
results/
  phase0_results.csv          -- per-(dataset, baseline, pair) Pearson values
  phase0_summary.csv          -- mean ± std per baseline, aggregated across pairs
  pxd014791_per_pair.csv      -- per-pair detail for PXD014791
  protargetminer_per_pair.csv -- per-pair detail for ProTargetMiner
  phase0_figure3.png          -- our bars vs paper's red dashes (Figure 3 recreation)
phase0.log                    -- full run log
```

## Comparison targets (paper Figure 3)

### PXD014791 (n=58 held-out pairs)

## Known assumptions and limitations

1. **Sample name parsing (LFQ)**: `parse_lfq.py` infers donor/drug/replicate from
   MaxQuant column names using `_` separators. If the actual column naming differs
   from `<Donor>_<Drug>_<Rep>`, edit `_parse_sample_name()` in parse_lfq.py.
   Run `--skip-download` and check phase0.log for the actual column names.

2. **TMT absolute intensities**: The paper uses "Reporter intensity corrected" columns
   from MaxQuant (NOT the ratio columns). This is probably correct as implemented.

3. **Pool definition**: We treat the pool for each held-out pair as ALL other pairs
   (including other held-out pairs). The paper's phrasing is ambiguous. If the paper
   uses only "training drug" pairs as the pool, the kNN baseline numbers will differ.
   This is a testable assumption -- check which definition reproduces the paper's values.

4. **NaN imputation for PCA/HVG**: Missing proteins are imputed with column means
   before embedding. The paper does not specify imputation strategy.

5. **Pair count discrepancy**: If our held-out pair count differs from paper's n=58/n=61,
   it usually means a drug name mismatch. Check phase0.log for the drug list.

## Interpreting the results

Match within ±0.02 Pearson on most baselines  --> parsing is correct
Systematic offset on all baselines              --> check log2 transform / missing value handling
All baselines at 0                             --> everything is being predicted as the control
Pair count wrong by a lot                      --> drug name normalisation issue in parse_lfq.py
ProTargetMiner HVG/PCA positive (or negative) --> pool definition or embedding difference vs paper
