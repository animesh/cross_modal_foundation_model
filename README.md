# Single-cell foundation models benefit from cross-modal training: adding proteomics data beats parameter scaling

Independent baseline checks for Burq et al. (2026), *Single-cell foundation models benefit from cross-modal training: adding proteomics data beats parameter scaling*. [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.08.14.744845v1.full), DOI: **10.64898/2026.08.14.744845**.

The aim is to stress-test what is actually reproducible, starting with classical baselines before evaluating foundation-model claims. The results below distinguish observed data, implementation choices, and unresolved author-protocol details.

## Current status

| Stage | Completed | Remaining |
| --- | --- | --- |
| Phase 0: protein perturbation | PXD014791 input audit; 156-configuration classical-baseline sweep; local Windows rerun and output verification | Exact author preprocessing and held-out manifest; ProTargetMiner; model embeddings |
| Phase 1: Figure 2 baselines | Five-fold PCA/logistic diagnostic; upstream marginal-essentiality PCA/RF artifact evaluation; raw-count PCA reconstruction; seven deposited MSigDB embeddings evaluated with independent output audit; tissue separation on 15 deposited cell embeddings with host verification and local displayed-score agreement | Exact target-paper configuration and numerical equivalence; MSigDB PCA baseline; checkpoint provenance; independent audit of local tissue-separation output files |
| Phase 2: foundation models | Deposited gene and cell embeddings evaluated in Phase 1; no foundation-model checkpoint inference or training performed | Original versus proteomics-trained checkpoint evaluation with verified provenance and identical conditions |

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

# Phase 1: DepMap diagnostics and upstream PCA/RF reconstruction

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

MSigDB is disabled by default. `--msigdb-proxy` enables a held-out-gene CCLE/logistic Hallmark proxy, not the actual pathway benchmark. The separate MSigDB MLP evaluation below has now been run on deposited embeddings; it does not use this proxy. The separate tissue-separation evaluation below has now been executed on deposited cell embeddings.

## Upstream marginal-essentiality PCA/RF reconstruction

The original 512-component PCA/logistic diagnostic above remains unchanged. The upstream marginal-essentiality task is a different evaluation: classify broadly essential versus inessential **genes**, using 15-component PCA loadings and five gene folds. Its AUPRC is not a mean of per-cell-line AUPRC values.

### Deposited-artifact evaluation

`run_depmap_upstream.py` reads `gene-embs/pcloading-lt5gt70-bin.h5ad`, the existing gene splits, and the deposited random-forest predictions under `gene-embs/results`. It verifies artifact labels against the raw dependency matrix and retrains the random forest using seed 42.

The upstream label convention differs from the earlier diagnostic:

1. Fill missing dependency probabilities with zero.
2. Discretize each cell-line probability: `> 0.5` is dependent.
3. Average those binary labels across all dependency rows.
4. Label a gene broadly essential when its dependent fraction is `>= 0.5`.

The supplied gene split selects the upstream broad-essentiality extremes. Existing splits are retained rather than regenerated. The local random forest uses scikit-learn defaults, eight jobs, and `random_state=42`; the upstream constructor does not specify a seed.

### Reconstruction from raw counts

`reconstruct_depmap_pca.py` additionally uses `raw/depmap-gene-effects.csv` and `raw/gene-mapping.csv` to reconstruct the upstream count matrix:

- Retain CCLE samples whose metadata ModelIDs occur in the gene-effect matrix.
- Retain genes shared by the vocabulary mapping, gene-effect columns, and CCLE descriptions.
- Strip Ensembl version suffixes and sort Ensembl IDs, preserving separate rows for duplicate gene symbols.
- Preserve raw float32 counts, without log transformation, library-size normalization, standardization, or gene-symbol summation.
- Fit 15-component PCA and compute loadings as `components_.T * sqrt(explained_variance_)`.
- For each marginal gene, select the first matching Ensembl row, following the inspected upstream implementation.

The reconstructed matrix contains **661 samples × 17,186 unique Ensembl features**, with **29 duplicate-symbol rows preserved**. This differs intentionally from the earlier diagnostic's 54,271 summed-symbol features.

The reconstruction uses deterministic **full SVD**. Upstream uses an unseeded automatic PCA solver. This is a documented implementation difference, not an established explanation for every observed numerical discrepancy. Sign alignment is used only to compare loadings, not to modify the RF inputs.

From `phase1`:

```bat
python run_depmap_upstream.py
python reconstruct_depmap_pca.py
```

The reconstruction imports `read_artifact` and `metrics` from the existing `run_depmap_upstream.py`. Both scripts require h5py in addition to NumPy, pandas, SciPy, and scikit-learn. Inputs and the earlier diagnostic outputs are not modified.

### Verified local results

| Source | Mean AUPRC | SD across gene folds | Mean AUROC |
| --- | ---: | ---: | ---: |
| Deposited RF predictions | 0.656246 | 0.024077 | 0.937212 |
| Deposited PCA features, local RF seed 42 | 0.654332 | 0.025978 | 0.935515 |
| Reconstructed raw-count PCA features, local RF seed 42 | **0.655672** | **0.028509** | **0.935769** |

Each gene is evaluated once: **14,285 unique validation genes**, with **2,857 genes in each of five folds**. Fold SD describes variation across these splits, not across training seeds.

All eight reconstruction output files were read completely. Independent checks confirmed:

- Unique sample names, ModelIDs, and sorted Ensembl IDs.
- Complete validation-gene coverage and agreement between loading and prediction gene sets.
- Finite reconstructed loadings and valid probability bounds.
- All **15 fold AUPRC/AUROC pairs** and all **three summary rows**, recomputed from the uploaded prediction labels and probabilities to floating-point precision.
- Minimum absolute component correlation in the uploaded feature report: **0.9990495824**.

The local runner reports a sign-aligned relative feature error of **0.0056448808**. Its report was inspected, but that error could not be independently recomputed from the results-only archive because the deposited loadings were absent. Likewise, raw dependency-label reconstruction could not be independently repeated from that archive because the dependency matrix was absent; label checks executed locally are distinct from independent verification of the uploaded metrics.

**Conclusion:** raw-count reconstruction produces close feature and RF agreement, not numerical identity. This validates the implemented upstream marginal-essentiality reconstruction and its output consistency; it does not establish the target paper's exact Figure 2 configuration or reproduce the foundation-model claims. The remaining feature discrepancy has not been fully explained.

Outputs relative to `phase1`:

```text
results/depmap_upstream/
  labels.csv
  metrics.csv
  predictions.csv
  summary.csv
  assumptions.json

results/depmap_raw_pca/
  count_genes.csv
  count_samples.csv
  feature_comparison.csv
  reconstructed_loadings.csv
  metrics.csv
  predictions.csv
  summary.csv
  assumptions.json
```

# Phase 1: MSigDB evaluation of deposited gene embeddings

## Scope and inputs

`run_msigdb_models.py` evaluates seven deposited gene embeddings with a multilabel MLP on the common **3,874-gene × 50-Hallmark-pathway** table. This is the upstream-style pathway-membership task, not the earlier CCLE/logistic proxy. No foundation-model checkpoint is loaded, no embeddings are regenerated, and no continued pretraining is performed.

Inputs under `D:\data\phase1\benchmarks\msigdb`:

```text
gene_embs/embs_adata.h5ad.gz
sigs/genesets.v2023.2.Hs_hallmark.gmt
```

The `.h5ad.gz` artifact is readable directly with h5py despite its suffix. The runner preserves its gene and pathway order, checks embedding row alignment, and compares the binary membership matrix with the GMT before training. These checks passed in the local run. The original artifact and GMT were not included in the later results-only audit, so their provenance and membership agreement were not independently rechecked there.

The seven embeddings are evaluated on the same deposited gene universe. Special tokens in the separate full-vocabulary NPZ files are not added to the evaluation. The large source expression dataset is not used by this branch.

**PCA is excluded.** Neither the supplied MSigDB source archive nor the reported repository searches established the MSigDB PCA-generation contract, and no PCA embedding was present in the local inventory. DepMap PCA is not substituted. Embedding filenames are retained as identifiers; their exact correspondence to target-paper checkpoints and training histories remains unverified.

## Evaluation protocol

For each seed **0, 1, and 2**, permute the deposited gene order and partition it into:

| Partition | Genes | Role |
| --- | ---: | --- |
| Train | 2,711 | Fit MLP weights |
| Validation | 194 | Select best epoch, schedule learning rate, and stop training |
| Test | 194 | Select the MLP configuration by lowest pooled log loss |
| Benchmark | 775 | Report pooled AUPRC after configuration selection |

These are the upstream **70/5/5/20%** partitions with integer cut points. The partition called `test` is used for hyperparameter selection; it is not the final untouched evaluation set. Benchmark genes are held out from weight fitting and configuration selection. Across seeds, partitions may overlap; the three evaluations are not disjoint folds.

Each embedding/seed evaluates **24 configurations**: hidden sizes 64, 512, 1,024, and 2,048; one or two hidden layers; and dropout 0.0, 0.1, or 0.5. The network uses LeakyReLU hidden activations, sigmoid outputs, binary cross-entropy, Adam with learning rate 0.001, batch size 256, ReduceLROnPlateau patience 7, early-stopping patience 15, and a maximum of 10,000 epochs.

The final metric is average precision over the flattened **775 × 50 gene-pathway probabilities and labels**, not the mean of per-pathway AUPRCs. Summary SD is the sample SD across three seeded runs, combining gene-partition and training-seed variation; it is not an uncertainty interval or variability across training seeds with fixed partitions.

### Implementation differences

The local runner replaces Lightning and multiprocessing with a sequential plain-PyTorch loop. It copies best-validation weights rather than retaining the upstream mutable `state_dict()` reference, explicitly handles CPU/GPU selection, and omits Lightning's sanity-validation pass. These differences are documented in `assumptions.json`; exact upstream numerical equivalence is not claimed.

The full local run completed **7 embeddings × 3 seeds × 24 configurations = 504 fits**. No correction to the completed run was indicated by the output-consistency audit.

## Local runtime

The successful run used a separate uv-managed **Python 3.12.12** environment, **PyTorch 2.5.1+cu121**, and an **NVIDIA GeForce RTX 2070 SUPER**. CUDA forward/backward execution and the three-epoch smoke test passed before the full evaluation. Scientific-package installation used NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1, scikit-learn 1.5.2, and h5py 3.11.0. The results archive records PyTorch and the CUDA device; it does not contain a complete independently verified environment lockfile.

From `phase1`, using the working environment executable:

```bat
"%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe" run_msigdb_models.py --smoke
"%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe" run_msigdb_models.py
```

Smoke outputs are separate and are not benchmark results. The runner requires NumPy, pandas, SciPy, scikit-learn, h5py, and PyTorch; it does not require Lightning, AnnData, or OmegaConf.

## Independently verified results

| Deposited embedding identifier | Mean pooled AUPRC | Sample SD, three seeds |
| --- | ---: | ---: |
| `mosaicfm-3b-prod_TE` | 0.337325 | 0.021470 |
| `mosaicfm-3b-prod-cont-MFMv2_TE` | 0.334581 | 0.024112 |
| `mosaicfm-v2-1_3b-merged_TE` | 0.327023 | 0.028082 |
| `mosaicfm-70m-merged_TE` | 0.274141 | 0.013837 |
| `state_TE` | 0.222807 | 0.004884 |
| `transcriptformer` | 0.200267 | 0.007333 |
| `geneformer` | 0.169982 | 0.013366 |

The original `mosaicfm-3b-prod_TE` identifier has a slightly higher mean than `mosaicfm-3b-prod-cont-MFMv2_TE` in this run. The latter scores lower in seeds 0 and 1 and higher in seed 2. This is an observed comparison of deposited embeddings under the local evaluator, not evidence establishing the effect of proteomics continued pretraining. No statistical significance or verified checkpoint identity is claimed.

## Output audit and limits

All **554 files** in the full results archive were read completely: **552 CSVs and two JSON files**. Independent checks confirmed:

- The binary **3,874 × 50** label table has unique gene and pathway identifiers.
- Exact seeded partition assignments and counts for all three seeds, with no within-seed overlap.
- All **21 benchmark prediction matrices**, each **775 × 50**, match the expected gene order and pathway columns and contain finite probabilities within [0, 1].
- All **21 pooled AUPRCs** recomputed from predictions and uploaded labels, with maximum discrepancy **8.326672684688674e-17**. A second tie-aware average-precision calculation agreed with scikit-learn.
- All seven summary means and sample SDs recomputed from the seeded results.
- All **504 configuration records** match the complete grid; selected configurations match the lowest recorded test log loss.
- All **504 histories covering 22,081 epochs** have consistent epoch sequences, finite nonnegative losses, reported best epochs, patience-15 stopping, and positive non-increasing learning rates.

**These are output-consistency checks, not an independent training rerun.** Test-set probabilities and checkpoints were excluded from the archive, so recorded test losses and saved best-model weights could not be independently recomputed or verified. The original H5AD embeddings and GMT were also absent, so original-input provenance and GMT membership were not independently rechecked. Exact target-paper reproduction and the PCA comparison remain unresolved.

Outputs relative to `phase1`:

```text
results/msigdb_models/
  assumptions.json
  labels.csv
  metrics.csv
  run_identity.json
  summary.csv
  seed_0/                         # same structure for seed_1 and seed_2
    partitions.csv
    <embedding_identifier>/
      benchmark_predictions.csv
      config_scores.csv
      config_00_history.csv      # through config_23_history.csv
      best_model.pt

results/msigdb_models_smoke/      # diagnostic only; separate from full results
```

Checkpoint files remain local outputs and were not included in the CSV/JSON audit archive.

# Phase 1: tissue separation of deposited cell embeddings

## Scope and inputs

`run_cell_line_separation.py` evaluates the upstream DepMap tissue-of-origin separation task on **15 deposited cell-embedding files**, including PCA. All files contain the same **661 unique ModelIDs in the same order**, with identical `OncotreeLineage` labels spanning **24 tissues**. This is not the Tabula Sapiens cell-type classification benchmark.

Input directory:

```text
D:\data\phase1\benchmarks\depmap\cell-embs
```

The uploaded H5AD matrices and relevant metadata were read directly. Three TranscriptFormer artifacts use numeric observation names rather than CCLE names; their ModelIDs and tissue labels nevertheless match the other artifacts. The runner checks cross-model alignment using ModelIDs and tissue labels, preserves original observation names in per-cell output, and uses ModelIDs for neighbour-edge identifiers.

No foundation-model checkpoint is loaded, no embeddings are regenerated, and no model training is performed. The deposited PCA artifact has **15 dimensions**; it is not the earlier 512-component PCA/logistic diagnostic or a newly reconstructed tissue-separation PCA.

## Metric and implementation

The inspected upstream `compute_lisi_scores` function in `scripts/depmap/evaluate-models.py` computes:

```text
same_tissue_fraction = fraction of directed neighbour edges sharing OncotreeLineage
frequency_baseline   = sum over tissues of (tissue sample count / total samples)^2
tissue_enrichment    = same_tissue_fraction / frequency_baseline
```

Neighbours are the **20 nearest other cell lines in Euclidean distance**, computed from the original `X` matrix with scikit-learn's `kneighbors_graph` defaults. No feature scaling, log transformation, dimension reduction, or UMAP coordinates are used for scoring. Original matrix dtypes are retained. The common frequency baseline is **0.07043149676943887**.

**Higher scores indicate greater same-tissue neighbour enrichment.** Although upstream calls this score “LISI”, it is not conventional inverse-Simpson LISI and is not silhouette scoring. The denominator uses squared observed tissue frequencies exactly as upstream; it is not a finite-population, self-excluded random-neighbour correction.

This is a descriptive evaluation of deposited embedding geometry on all 661 cell lines, not a held-out classifier or prospective prediction task. No fold or training-seed SD is reported. The metric is sample-weighted, not a macro-average giving every tissue equal weight.

The local runner preserves the upstream score formula but does not compute or write UMAPs back into input H5AD files. It adds metadata, finite-value, neighbour-count, and input-hash checks and exports auditable per-cell and neighbour-edge records. These changes do not alter the scored matrices or tissue labels.

## Executed results

| Deposited embedding identifier | Dimensions | Tissue enrichment |
| --- | ---: | ---: |
| `pca` | 15 | 6.708163 |
| `state` | 2058 | 5.702905 |
| `tx1-3b-alt` | 2560 | 5.695387 |
| `uce-33l` | 1280 | 5.610542 |
| `tx1-3b` | 2560 | 5.374263 |
| `tx1-70m` | 512 | 5.158390 |
| `tx1-1b` | 2048 | 5.157316 |
| `uce-4l` | 1280 | 5.139059 |
| `transcriptformer-sapiens` | 2048 | 4.786789 |
| `transcriptformer-exemplar` | 2048 | 4.752421 |
| `transcriptformer-metazoa` | 2048 | 4.741681 |
| `nvidia-gf-100m` | 768 | 4.595618 |
| `geneformer` | 256 | 4.427001 |
| `scgpt` | 512 | 4.304566 |
| `nvidia-gf-10m` | 256 | 4.094063 |

**PCA scores highest among these 15 deposited artifacts**, with a same-tissue neighbour fraction of **0.472466** and enrichment **6.708163**. The next scores are `state` at **5.702905** and `tx1-3b-alt` at **5.695387**. These are observed artifact comparisons under this metric, not a general claim that PCA is superior on other tasks.

Embedding filenames are identifiers, not independently verified checkpoint provenance. In particular, `tx1-3b-alt` is not assumed to be the target paper's proteomics-trained checkpoint. These results alone do not establish or refute a continued-pretraining benefit or a controlled parameter-scaling comparison.

## Host checks and local execution

The host runner was executed on all 15 real uploaded H5AD files. Checks confirmed:

- Unique ModelIDs, identical cross-model ModelID order, and identical tissue labels.
- Finite embedding matrices and the expected sample counts.
- **20 neighbours per cell**, no self-edges, and no duplicate query-neighbour pairs within a model.
- All **15 scores** agree with the original upstream function executed on the same matrices and labels within **1e-12**.
- Output CSVs were read back: **15 summary rows**, **9,915 per-cell records**, **198,300 directed neighbour edges**, and **24 tissue-count rows**. Summary scores were independently recomputed from edge records.
- Input hashes remained unchanged after execution.

A subsequent local run completed and displayed all **15 scores matching the host results to six decimal places**. This establishes local execution and displayed-score agreement. The newly generated local CSV/JSON files have not been uploaded and independently audited; full-precision cross-environment identity is not claimed.

Exact target-paper configuration, checkpoint identity, original embedding construction, and any overlap between pretraining data and evaluated cell lines remain unresolved. Agreement with the inspected upstream scoring function does not establish exact Figure 2 reproduction.

From `phase1`:

```bat
python run_cell_line_separation.py
```

Dependencies: NumPy, pandas, h5py, and scikit-learn. No GPU, PyTorch, Scanpy, UMAP, or new download is required. The runner reads the existing H5AD inputs without modifying them.

Outputs relative to `phase1`:

```text
results/cell_line_separation/
  summary.csv
  per_cell.csv
  neighbors.csv
  tissue_counts.csv
  assumptions.json
```

`assumptions.json` records the metric, scope, host/runtime versions available to the runner, and input SHA-256 hashes. Existing Phase 0, DepMap, and MSigDB implementations and outputs remain unchanged.

# Next steps

1. Preserve the verified Phase 0 and Phase 1 diagnostics as comparison baselines.
2. Preserve the upstream marginal PCA/RF reconstruction and resolve its remaining feature discrepancy and exact target-paper configuration.
3. Preserve the audited MSigDB and tissue-separation evaluations; resolve the missing MSigDB PCA construction and checkpoint provenance, and independently audit the local tissue-separation output files.
4. Resolve the Phase 0 author preprocessing and exact held-out manifest.
5. Evaluate original versus proteomics-trained checkpoints under identical inputs, predictor, and scoring.

No foundation-model checkpoint has been loaded for inference or trained in this project. Deposited gene embeddings have been evaluated through the MSigDB MLP task, and deposited cell embeddings through the tissue-separation task. Current results do not validate or refute the paper's continued-pretraining claims.

## Sources

- [Target preprint](https://www.biorxiv.org/content/10.64898/2026.08.14.744845v1.full), DOI **10.64898/2026.08.14.744845**.
- [PXD014791 source study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8776854/), DOI **10.1038/s41597-021-01114-3**; Table 1 identifies `TRA` as Trametinib.
- [Upstream DepMap random-forest implementation](https://github.com/tahoebio/tahoe-x1/blob/main/scripts/depmap/rf.py).

The earlier README's claims that Trametinib was absent, that the formal data contained only 57 pairs, and that proprietary reprocessing was required are superseded. The verified data contain 60 formal pairs. `TRS` identity and the exact author split remain unresolved.
