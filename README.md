# Single-cell foundation models benefit from cross-modal training: adding proteomics data beats parameter scaling

Independent baseline checks for Burq et al. (2026), *Single-cell foundation models benefit from cross-modal training: adding proteomics data beats parameter scaling*. [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.08.14.744845v1.full), DOI: **10.64898/2026.08.14.744845**.

The aim is to stress-test what is actually reproducible, starting with classical baselines before evaluating foundation-model claims. The results below distinguish observed data, implementation choices, and unresolved author-protocol details.

## Current status

| Stage | Completed | Remaining |
| --- | --- | --- |
| Phase 0: protein perturbation | PXD014791 input audit; 156-configuration classical-baseline sweep; local Windows rerun and output verification | Exact author preprocessing and held-out manifest; ProTargetMiner; model embeddings |
| Phase 1: Figure 2 baselines | Five-fold PCA/logistic DepMap diagnostic; local execution and independent output verification | Upstream random-forest protocol; actual MSigDB pathway benchmark; tissue separation |
| Phase 2: foundation models | Not evaluated | Original versus proteomics-trained checkpoints under identical evaluation conditions |

**This is not yet a complete reproduction of Figures 2 or 3.** Numerical proximity to a figure does not establish identical processing or validate the paper's model-training claims.

## Questions worth testing

### Continued pretraining versus parameter scaling

The paper compares a proteomics-continued-pretrained 70M model with larger RNA-only checkpoints. This tests the benefit of adding proteomics relative to those existing checkpoints, not a controlled comparison in which every model size receives identical continued pretraining. Continuing the larger models on the same corpus would address a different, complementary question.

### Protein-to-gene mapping

The paper maps proteins to gene tokens. Our current Phase 0 parser takes the first listed gene name and keeps the first protein-group row per gene. This is an explicit reconstruction choice, not verified equivalence to the authors' mapping. Mapping coverage and handling of isoforms, ambiguous identifiers, and duplicate genes remain relevant audit questions.

### Classical baselines versus model improvements

Improvement over an original checkpoint is distinct from superiority over HVG or PCA. Both comparisons must be reported. Approximate Figure 3 values recorded in the original code are useful orientation, but are not author-provided numerical outputs.

### Preprocessing and missing values

The initial conclusion that HVG/PCA required proprietary reprocessing was unsupported. The current experiments show that changing missing-value treatment and aggregation can recover close numerical agreement from the supplied public MaxQuant data. This does not prove the authors used our reconstruction choices.

### TMT and data scale

The paper describes using absolute isotope-corrected reporter intensities for ProTargetMiner rather than ratio columns. That branch has not been evaluated here. Likewise, this project has not tested the paper's training-data scaling claims or established whether its learning curve saturates.

# Phase 0: PXD014791 baseline verification

## Data and input audit

Local inputs:

```text
D:\data\phase0\data\PXD014791\proteinGroups.txt
D:\data\phase0\data\PXD014791\Metadata-Table.xlsx
```

Direct input inspection established:

- 7,193 protein-group rows and 3,466 columns in the original table.
- 381 LFQ sample columns: 308 metadata-matched formal samples, 64 HeLa QC samples, and nine unmatched non-QC samples.
- 237 treated samples, 71 controls, four biological contexts, and 60 drug-context pairs among the formal samples.
- 6,413 protein groups after the MaxQuant site-only, reverse, and potential-contaminant filters, before gene selection/deduplication.

The current sweep uses only the 308 metadata-matched samples. It does not infer donor or drug metadata for auxiliary samples.

## Drug mapping and pair selection

The source study explicitly identifies **`TRA = Trametinib`**. The earlier README's assignment of `TRA` to Trastuzumab was incorrect. `TRA` contributes one formal pair in PMC-B, with four samples.

**`TRS` remains unresolved.** It contributes two formal pairs, in PMC-B and PMC-E, with eight samples. The current runner retains drug codes and does not assign `TRS` a drug identity by elimination.

Two scoring scenarios are reported:

- **All 60 pairs.**
- **Provisional 58 pairs:** exclude `TRS` pairs from scoring only. They remain eligible neighbours for other queries.

The provisional exclusion matches the reported pair count but is not a verified author manifest. Trametinib is present in the formal metadata; the previous claim that it was absent is superseded.

## Current implementation

From the repository's `phase0` directory:

```bat
python test_pxd_methods.py
```

Dependencies: NumPy, pandas, SciPy, scikit-learn, openpyxl, and threadpoolctl. The metadata workbook requires openpyxl. No GPU, model checkpoint, or new download is needed for this sweep.

`test_pxd_methods.py` is the implementation used for the current results. The older `run_phase0.py`, `benchmark.py`, `parse_lfq.py`, and normalization/debug scripts are retained as legacy code and are not equivalent to the current sweep. Their output must not be substituted for the results below.

### Response and prediction

For each treated drug-context pair, aggregate treated log-abundances and controls from the same biological context. Given measured log-proteome `r`, predicted log-proteome `p`, and matched control `c`:

```text
Measured response:  delta     = r - c
Predicted response: delta_hat = p - c
```

HVG/PCA predictions use a cosine-similarity-weighted average of the five nearest other pairs' log-proteomes. Individual sample embeddings are averaged within each pair for sample-level variants. The query's measured treated profile is used to construct its embedding; this is not prospective prediction from drug identity and untreated controls alone.

Pearson correlation is calculated across proteins between the two response vectors. Top-100 scoring selects the 100 proteins with the largest absolute measured response. **There is no t-test, p-value, or FDR-based DE selection in this metric.**

### Method sweep

The runner tests:

- Mean of detected log-intensities, log of mean detected intensities, and mean of zero-filled `log2(intensity + 1)`.
- Finite-response intersection scoring and detected-union scoring with missing log values explicitly zero-filled.
- Raw and sample-median-centred embedding inputs.
- Unscaled and standardized features.
- Sample-global, sample-fold, and pair-fold embeddings.
- HVG/PCA predictors, perturbation-mean, and additive-linear baselines.

Sample-global fitting includes evaluated samples and is transductive. Sample-fold fitting excludes the query's treated samples, while retaining controls and other treated samples. Pair-fold fitting excludes the query pair mean.

PCA is capped by available sample count and numerical rank. Nominal caps are 307 for sample-global fitting, 301–306 for sample-fold fitting, and 58 for pair-fold fitting. **Literal 512-component PCA on 308 samples is not claimed.** A Gram-matrix implementation was checked against SVD training/query inner products.

Selecting HVGs after standardization equalizes variances; these variants are sensitivity diagnostics, not the recommended highest-variance selection protocol. The main comparison uses unscaled features.

## Verified results

Configuration: zero-filled `log2(intensity + 1)`, raw/unscaled sample-fold embeddings, detected-union scoring, and provisional 58-pair scoring. Non-embedding baselines use the same response/scoring conventions.

| Baseline | Local top-100 Pearson | Approximate figure target* | Local full Pearson | Approximate figure target* |
| --- | ---: | ---: | ---: | ---: |
| Perturbation-mean | 0.540237 | 0.541 | 0.215501 | 0.219 |
| Additive-linear | 0.471802 | 0.475 | 0.218536 | 0.229 |
| HVG | 0.666826 | 0.665 | 0.412999 | 0.415 |
| PCA | 0.673363 | 0.674 | 0.422155 | 0.425 |

*Targets are approximate values recorded in the original code, not verified author numerical outputs. Control-mean predicts zero response and has undefined Pearson correlation because its prediction has zero variance; it is not a scored configuration in this sweep.

The local Windows run reproduced the four reported rows to floating-point precision. All three output files were read completely and independently checked:

- **156 configurations and 9,360 unique per-pair records.**
- **312 scenario summaries**, with all means, counts, and PCA component ranges recomputed from per-pair records.
- All reported correlations finite and within [-1, 1].
- All unscaled configurations agree with the earlier Linux results within approximately **7e-16**.
- **72 standardized-HVG scenario summaries** differ by more than 1e-8; the maximum absolute mean-correlation difference is **0.0098334088**. The cause has not been established.

The main comparison is reproduced across environments, but the entire sweep is not numerically identical. Exact author preprocessing and the `TRS` exclusion remain unverified.

Outputs relative to `phase0`:

```text
results/method_audit/per_pair.csv
results/method_audit/summary.csv
results/method_audit/assumptions.json
```

## Exploratory VAE

A NumPy VAE with 512 posterior-mean coordinates was trained for 50 epochs across three seeds on all 308 formal samples, including evaluated samples. The posterior mean follows a 128-unit hidden layer, so 512 coordinates do not imply 512 independent features.

Using the same five-neighbour predictor and provisional 58-pair scoring, mean top-100/full correlations were **0.531447 / 0.267139**, below the classical PCA comparison. This was an exploratory transductive trial, not held-out model validation or a test of continued pretraining.

## ProTargetMiner

The ProTargetMiner branch has not been run or validated. Its reporter-channel mapping and absolute-intensity processing must be checked against the source experimental design before evaluation. Legacy acquisition and TMT parsing utilities are not evidence that this branch is ready.

# Phase 1: DepMap PCA/logistic diagnostic

## Data and preflight

Benchmark root:

```text
D:\data\phase1\benchmarks
```

Inputs used by the checker and current diagnostic:

```text
depmap/raw/ccle-counts.gct
depmap/raw/depmap-gene-dependencies.csv
depmap/raw/depmap-metadata.csv
depmap/misc/split-cls.csv
depmap/misc/split-genes-lt5gt70.csv
msigdb/sigs/genesets.v2023.2.Hs_hallmark.gmt
```

`gene-mapping.csv` is also available. The expression file is plain TSV with `Name` and `Description`, despite the `.gct` suffix. The parser also supports GCT #1.2.

From `phase1`:

```bat
python check_phase1_data.py
python run_phase1.py --eval
```

The preflight produces `results/phase1_preflight/report.json` and `sample_mapping.csv`; the evaluator consumes the mapping.

Verified local input properties:

- 56,202 expression rows and 1,019 sample columns.
- 661 unique one-to-one expression-to-DepMap mappings, covering all five sample folds.
- 14,285 split genes present in both expression descriptions and dependency columns.
- Dependency probabilities range from 0 to 1; 739,493 nonfinite labels are excluded from labelled evaluation.
- 50 Hallmark sets; 4,244 Hallmark genes overlap expression descriptions.

## Corrected diagnostic

The current implementation:

- Preserves every expression row and maps samples explicitly to DepMap IDs.
- Uses expression gene descriptions and strips numeric suffixes from dependency gene columns.
- Sums 1,931 duplicate gene-symbol rows at count level, yielding 54,271 features. This is a local aggregation choice, not verified upstream equivalence.
- Fits standardized-count PCA with 512 components on each fold's training cell lines and uses gene loadings as features.
- Trains the original logistic classifier on training genes, using mean dependency probability over training cell lines and a positive label of **`> 0.5`**.
- Evaluates held-out genes in validation cell lines. Missing labels are excluded, not assigned a negative class.
- Postpones type-annotation evaluation to avoid the observed older-Python import error.

## Verified local results

| Fold | Validation cell lines | Mean AUPRC |
| --- | ---: | ---: |
| 0 | 133 | 0.6259 |
| 1 | 132 | 0.6673 |
| 2 | 132 | 0.6386 |
| 3 | 132 | 0.6460 |
| 4 | 132 | 0.6495 |

Mean of fold means: **0.6454576962**. Sample SD across five fold means: **0.0152016393**. This SD is not variability across training seeds.

All seven returned CSVs were read, and fold means, overall mean/SD, and 661 unique validation cell lines were independently verified. A subsequent local rerun produced the same displayed results.

Outputs under `phase1/results/phase1`:

```text
depmap_fold0_per_cell.csv
depmap_fold1_per_cell.csv
depmap_fold2_per_cell.csv
depmap_fold3_per_cell.csv
depmap_fold4_per_cell.csv
phase1_results.csv
phase1_summary.csv
```

**This remains a PCA/logistic diagnostic, not exact Figure 2 reproduction.** The upstream DepMap implementation uses random forests and labelled embedding tables. Matching its classifier alone would not establish matching feature construction, labels, splits, or aggregation.

MSigDB is disabled by default. `--msigdb-proxy` enables a held-out-gene CCLE/logistic Hallmark proxy, not the actual pathway benchmark. The MSigDB and tissue-separation protocols remain to be implemented and verified.

# Next steps

1. Preserve the verified Phase 0 and Phase 1 diagnostics as comparison baselines.
2. Reproduce the upstream DepMap random-forest protocol without substituting the current logistic diagnostic.
3. Implement the actual MSigDB and tissue-separation evaluations.
4. Resolve the Phase 0 author preprocessing and exact held-out manifest.
5. Evaluate original versus proteomics-trained checkpoints under identical inputs, predictor, and scoring.

No foundation-model checkpoint has been evaluated in this project. Current baseline agreement does not validate or refute the paper's continued-pretraining claims.

## Sources

- [Target preprint](https://www.biorxiv.org/content/10.64898/2026.08.14.744845v1.full), DOI **10.64898/2026.08.14.744845**.
- [PXD014791 source study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8776854/), DOI **10.1038/s41597-021-01114-3**; Table 1 identifies `TRA` as Trametinib.
- [Upstream DepMap random-forest implementation](https://github.com/tahoebio/tahoe-x1/blob/main/scripts/depmap/rf.py).

The earlier README's claims that Trametinib was absent, that the formal data contained only 57 pairs, and that proprietary reprocessing was required are superseded. The verified data contain 60 formal pairs. `TRS` identity and the exact author split remain unresolved.
