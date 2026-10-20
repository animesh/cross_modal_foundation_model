# Single-cell foundation models benefit from cross-modal training: adding proteomics data beats parameter scaling

Independent baseline checks and small-corpus continued-training experiments for Burq et al. (2026), *Single-cell foundation models benefit from cross-modal training: adding proteomics data beats parameter scaling*. [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.08.14.744845v1.full), DOI: **10.64898/2026.08.14.744845**.

The aim is to stress-test what is actually reproducible, starting with classical baselines before evaluating foundation-model claims. The results below distinguish observed data, implementation choices, and unresolved author-protocol details.

## Current status

| Stage | Completed | Remaining |
| --- | --- | --- |
| Phase 0: protein perturbation | PXD014791 input audit; 156-configuration classical-baseline sweep; local Windows rerun and output verification | Exact author preprocessing and held-out manifest; ProTargetMiner; uncontaminated checkpoint-based protein-transfer evaluation |
| Phase 1: Figure 2 baselines | Five-fold PCA/logistic diagnostic; upstream marginal-essentiality PCA/RF artifact evaluation; raw-count PCA reconstruction; seven deposited MSigDB embeddings evaluated with independent output audit; independent MSigDB PCA reconstruction and three-seed MLP evaluation; tissue separation on 15 deposited cell embeddings with host verification and local displayed-score agreement | Exact target-paper configuration and numerical equivalence, including author MSigDB PCA construction; deposited-embedding provenance; independent audit of local tissue-separation output files |
| Phase 2: checkpoint and continued training | Published original 70M checkpoint acquired at a pinned revision; strict tensor loading; actual GPU forward/backward tests; real protein input preparation; one protein epoch and matched 308-update RNA control completed; paired reconstruction evaluation on 1,024 reserved RNA cells completed | Fresh RNA-derived contextual embeddings and downstream DepMap/MSigDB/tissue evaluation for the three checkpoints; full-corpus author-recipe reproduction; multiple seeds and independent output audit |
| Paper-method review | Uploaded 11-page PDF, including supplementary information, reviewed; distinctions between paper benchmarks and local reconstruction experiment documented | Exact training configuration, batch-correction implementation, corpus/augmentation manifests, and numerical evaluation equivalence |

**This is not yet a complete reproduction of Figures 2 or 3.** Numerical proximity to a figure does not establish identical processing or validate the paper's model-training claims.

**Current experimental finding:** protein continued training slightly lowers masked RNA reconstruction MSE relative to the original checkpoint, but the matched RNA-only control performs substantially better on this diagnostic. This does not establish a protein-specific advantage and is not the downstream representation test used for the paper's central claim. Actual checkpoint inference and continued training have now been performed; older statements that no checkpoint was loaded or trained are superseded.

Status updated after the completed paired RNA evaluation. Existing classical-baseline and deposited-artifact results are preserved. Newly reported training/evaluation values below come from local console output unless an independent uploaded-file audit is explicitly stated.

## Questions worth testing

### Continued pretraining versus parameter scaling

The paper compares a proteomics-continued-pretrained 70M model with larger RNA-only checkpoints. This tests the benefit of adding proteomics relative to those existing checkpoints, not a controlled comparison in which every model size receives identical continued pretraining. Continuing the larger models on the same corpus would address a different, complementary question.

### Protein-to-gene mapping

The paper maps proteins to gene tokens. Our current Phase 0 parser takes the first listed gene name and keeps the first protein-group row per gene. This is an explicit reconstruction choice, not verified equivalence to the authors' mapping. The separate continued-training input preparer now uses unambiguous Ensembl mapping and sums same-token protein groups; it does not change the historical Phase 0 parser or results. Mapping coverage and handling of isoforms, ambiguous identifiers, and duplicate genes remain relevant audit questions.

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

**`TRS` remains unresolved as a local metadata code.** It contributes two formal pairs, in PMC-B and PMC-E, with eight samples. The current runner retains drug codes and does not assign `TRS` a drug identity by elimination.

Two scoring scenarios are reported:

- **All 60 pairs.**
- **Provisional 58 pairs:** exclude `TRS` pairs from scoring only. They remain eligible neighbours for other queries.

The provisional exclusion matches the reported pair count but is not a verified author manifest. The subsequently reviewed paper supplementary information lists excluded cardiomyocyte drugs and states that trastuzumab is retained; it does not by itself establish the local `TRS` code identity. Therefore the historical provisional exclusion must not be relabelled as the author split. Trametinib is present in the formal metadata; the previous claim that it was absent is superseded.

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

**PCA is excluded from this seven-deposited-embedding run, but a separate independent PCA reconstruction and evaluation has now completed**, as documented below. No deposited author MSigDB PCA embedding or exact author PCA-generation contract has been established. DepMap PCA is not substituted. Embedding filenames are retained as identifiers; their exact correspondence to target-paper checkpoints and training histories remains unverified.

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

**These are output-consistency checks, not an independent training rerun.** Test-set probabilities and checkpoints were excluded from the archive, so recorded test losses and saved best-model weights could not be independently recomputed or verified. The original H5AD embeddings and GMT were also absent, so original-input provenance and GMT membership were not independently rechecked. Exact target-paper reproduction remains unresolved. The separate independent PCA comparison below is complete; equivalence to the author PCA remains unverified.

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

# Phase 1: independent MSigDB PCA reconstruction

## Completed implementation and scope

`run_msigdb_pca.py` reconstructs a PCA gene embedding from the local MSigDB RNA expression source and evaluates it with the existing multilabel MLP protocol. This is a separate branch: the seven deposited-embedding results above remain unchanged. It is not a deposited author PCA artifact or an established exact reconstruction of the paper's PCA.

The recorded source matrix has **524,043 cells x 60,664 features**, in CSR format. The reconstruction uses **50,000 sampled cells**, with sampling seed 42, and the same **3,874 benchmark genes** used by the deposited-embedding evaluation. All sampled cells had nonzero library size. The build reports integer-like sampled counts and zero duplicate-symbol rows summed for this resolved gene universe.

Recorded preprocessing and PCA settings:

- Resolve source feature identifiers to the benchmark gene symbols; retain the recorded resolution evidence in `gene_resolution.json`.
- Sum duplicate-symbol counts where present; normalize using the all-feature library size to 10,000, then natural `log1p`.
- Center genes without variance scaling.
- Fit **512 components**, with SVD seed 42 and two power iterations.
- Construct a **3,874 x 512** loading matrix. Recorded explained-variance ratio: **0.5757755238414842**.

The PCA feature universe includes all benchmark genes, including genes later held out from MLP fitting, without using their pathway labels. This feature construction is **transductive**, not a PCA fitted only to MLP training genes. Source size and modification time are recorded, but the large source RNA file was not fully SHA-256 hashed. Author PCA preprocessing, dimension, feature universe, and sampling equivalence remain unresolved.

## Three-seed MLP results

The reconstructed embedding uses the same 70/5/5/20% gene partitions, 24-configuration grid, and pooled benchmark AUPRC definition as the deposited-embedding evaluator. The completed branch contains **3 seeds x 24 configurations = 72 fits**, with 775 benchmark genes per seed.

| Seed | Pooled benchmark AUPRC | Selected hidden size | Hidden layers | Dropout |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.25946286760652737 | 2,048 | 1 | 0.5 |
| 1 | 0.23418525748359878 | 2,048 | 1 | 0.5 |
| 2 | 0.23161234738549097 | 1,024 | 1 | 0.5 |

**Mean pooled AUPRC: 0.2417534908252057. Sample SD across three seeds: 0.015390629814284636.** This independent PCA scores below the deposited `mosaicfm-70m-merged_TE` mean of 0.274141 under the documented local evaluator. This is a comparison of these feature constructions, not evidence establishing a continued-pretraining benefit or exact agreement with the paper.

## Direct output inspection and limits

The available PCA results archive contains **92 files: 86 CSVs, five JSON files, and one NPZ loading file**. Archive integrity was checked, every member was read, and all CSV/JSON contents were parsed. The three **775 x 50** benchmark probability matrices were checked for finite values in [0, 1], aligned to the uploaded label table, and their pooled AUPRCs were independently recomputed. All three matched the recorded metrics; the summary mean and sample SD were also recomputed.

These are output-consistency checks, not a fresh PCA build or independent MLP training rerun. Original source RNA, the deposited input embedding artifact, and the GMT were not part of this results archive. Saved MLP weights were absent. No independent verification of the original RNA preprocessing, external gene-resolution evidence, or saved best-model weights is claimed. Reading the NPZ archive member does not by itself establish independent verification of every loading value or its recorded hash.

Outputs relative to `phase1`:

```text
results/msigdb_pca_reconstruction/
  assumptions.json
  gene_resolution.json
  labels.csv
  metrics.csv
  pca_build.json
  pca_cells.csv
  pca_gene_mapping.csv
  pca_identity.json
  pca_loadings.npz
  run_identity.json
  summary.csv
  seed_0/                         # same structure for seed_1 and seed_2
    partitions.csv
    pca_reconstructed/
      benchmark_predictions.csv
      config_scores.csv
      config_00_history.csv       # through config_23_history.csv
      best_model.pt               # local only; absent from inspected archive

results/msigdb_pca_reconstruction_smoke/  # diagnostic, not full results
```

From `phase1`, the completed full branch was implemented by `run_msigdb_pca.py`. Existing completed outputs should be preserved, not regenerated merely to update documentation.

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

# Phase 2: original checkpoint, continued training, and paired RNA evaluation

## Scope and evidence

This phase is an **independent one-study experiment**, not a reconstruction of the paper's 440-source continued-pretraining corpus. It preserves the original checkpoint and all Phase 0/Phase 1 outputs. Three checkpoint states are now available: original 70M, protein-continued 70M, and RNA-control-continued 70M.

Local console logs establish successful execution of both training arms and the paired evaluator. The real protein-input archive was independently inspected: its NPZ, report, and sample metadata were read and checked for shape, finite nonnegative values, gene uniqueness, measured-gene counts, and sample alignment. The final trained checkpoint files, complete training CSVs, and paired per-cell evaluation outputs have not been independently audited here. Runtime assertions performed locally are distinct from independent inspection of those final artifacts.

## Working runtime and hardware feasibility

The existing Windows environment is used:

```text
%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe
PyTorch 2.5.1+cu121
CUDA build 12.1; CUDA available
NVIDIA GeForce RTX 2070 SUPER, 8 GiB VRAM
```

Hugging Face Hub, safetensors, PyYAML and their dependencies were installed during checkpoint acquisition; h5py was already available, and openpyxl was installed for metadata preparation. This is not a complete environment lockfile. The implementation uses a plain-PyTorch adapter rather than requiring the upstream training stack.

`test_gpu_70m.py` first tested a synthetic 70,000,000-parameter proxy with batch size 1, FP32 parameters, FP16 autocast, gradient checkpointing, and two successful Adam updates at sequence lengths 256, 512, 1,024, and 2,048. All four passed using efficient SDPA; peak allocated memory was approximately 1.32 GiB. An earlier unnamed assertion was corrected to permit gradient-scaler adaptation and provide explicit diagnostics; the final run required no overflow retries. This proxy is not Tahoe checkpoint validation.

The RTX 2070 SUPER was not ruled out by parameter memory. Standard upstream FlashAttention-2 support and PyTorch efficient SDPA are separate issues: the latter executed successfully on this GPU. No hardware replacement was required for the completed experiment.

## Original checkpoint acquisition and loading

Published source: `tahoebio/Tahoe-x1`, revision `d218a580b9c2500ae9dfc8367a398545e6f017a8`, subdirectory `70m-model`. Only the 70M safetensors, vocabulary, model configuration, and collator configuration were downloaded, not the larger checkpoints.

```text
phase1/models/tahoe_original/70m-model/
  model.safetensors
  model_config.yml
  collator_config.yml
  vocab.json
```

The checkpoint contract contains **163 F32 tensors**, **70,996,993 parameters**, and a **62,720-token vocabulary**. The adapter checks the starting checkpoint SHA-256:

```text
ee1cdfa841066b9bce9c3705b112b17a2b459750d20696cbfedc6a3c397686c9
```

The architecture has 12 pre-normalized transformer layers, width 512, eight query/key/value heads, feed-forward expansion 4, ReLU transformer activation, continuous expression encoding, CLS cell pooling, an expression decoder, and a scaled inner-product MVC decoder. The downloaded configuration requests FlashAttention and `amp_bf16`; the local adapter instead uses PyTorch efficient SDPA and FP16 autocast. This is a documented runtime difference, not original-backend equivalence.

`test_tahoe_original.py` strictly loads all tensor names/shapes and checks loaded values. It reimplements the required transformer operations using SDPA while retaining the inspected encoder/decoder operations. No upstream files are modified. Efficient SDPA is compared against PyTorch math SDPA on the same loaded adapter; this comparison does **not** compare against the upstream FlashAttention implementation.

| Sequence length | Successful Adam updates | Attempts | Peak allocated GiB | Expression-prediction maximum absolute SDPA difference | MVC maximum absolute difference | CLS maximum absolute difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 2 | 2 | 1.323085 | 0.062500 | 0.031250 | 0.016329 |
| 1,024 | 2 | 2 | 1.329936 | 0.046875 | 0.031250 | 0.010788 |

Both tests passed the implemented `rtol=0.03`, `atol=0.03` comparisons and finite-gradient/update checks. Nonzero backend differences remain documented. The test uses synthetic binned inputs and does not establish biological performance.

## Real protein training inputs

`prepare_protein_training.py` uses the existing PXD014791 protein table, formal metadata workbook, checkpoint vocabulary, and the local MSigDB expression dataset's feature identifiers for symbol-to-Ensembl mapping. It strips Ensembl version suffixes and removes a feature-name suffix only when it exactly matches that row's Ensembl ID.

The complete mapping contract contained 6,413 filtered protein groups, 381 LFQ columns, and 60,664 expression feature records. Unambiguous mapping retained **5,924 protein groups**, excluded **489 groups**, and yielded **5,763 unique gene tokens**. Same-token protein-group intensities are summed. Multi-gene groups are retained only when all listed symbols resolve uniquely to the same Ensembl gene; unresolved or ambiguous groups are excluded. This differs from the historical first-gene/first-row Phase 0 reconstruction.

Only **308 metadata-matched formal samples** are used. The 73 auxiliary LFQ columns are excluded by metadata matching. Missing LFQ measurements remain zero in storage and are excluded from training targets; there is no imputation. The CSV field limit was increased to 2,147,483,647 after the original table exceeded Python's default 131,072-character field limit.

```text
phase1/results/protein_mapping_contract.json
phase1/results/protein_training_inputs/
  profiles.npz                 # gene_ids, lfq (308 x 5763), samples
  sample_metadata.csv
  report.json
```

The uploaded protein-input archive was independently checked in full. This does not imply that the original large MaxQuant table or workbook was independently reread during that archive audit.

## Completed protein continued-training arm

`train_protein_epoch.py` starts from the original checkpoint, not from the diagnostic test's transient updated model. It completes one pass over the 308 real protein profiles with seed 42.

- Batch size 1; maximum sequence length 1,024, including CLS.
- Positive measured LFQ values only; per-profile 51-bin quantiles computed before subsampling.
- Uniform sampling of up to 1,023 measured gene tokens without replacement, then sorting selected indices.
- 50% masking, excluding CLS; masked input value -1.
- Mean of masked expression-decoder MSE and masked MVC MSE.
- AdamW, learning rate 0.00001, weight decay 0.01, `foreach=False`.
- FP32 parameters, FP16 autocast, efficient SDPA, gradient checkpointing, GradScaler initial scale 1.0.
- Finite-loss/gradient checks, successful-update checks, final finite weights, and exact saved-checkpoint reload checks.

All **308 successful updates** completed. Displayed loss was **154.414154** at step 1 and **63.906990** at step 308; these are different sampled/masked profiles, not a fixed-validation learning curve. All displayed checkpoints used one attempt, scale 1.0, and approximately **1.3292 GiB** peak allocated memory. Complete retry counts must be verified from the full training CSV before claiming that every unprinted step used one attempt.

```text
phase1/results/protein_continued_seed42/
  model.safetensors
  model_config.yml
  collator_config.yml
  vocab.json
  training.csv
  manifest.json
```

The copied YAML remains the published configuration and is not a standalone record of the local SDPA/FP16 training settings; the local runtime recipe is recorded in `manifest.json` and the script. The manifest records source-profile, adapter, starting-checkpoint and trained-checkpoint hashes. Existing output directories are not overwritten. Interrupted-run resume is not implemented.

**Held-out protein evaluation restriction:** all formal PXD014791 samples were used for this arm, including treated samples. This checkpoint cannot be used to reproduce the paper's unseen-drug test on PXD014791. A training corpus with the paper's drug exclusions is required for that claim.

## Completed RNA-only continued-training control

`prepare_rna_control.py` uses the existing `rep_1.h5ad.gz` source under `phase1/benchmarks/msigdb/cellxgene_ds_data`. The file was read as a CSR HDF5 matrix in the successful local run; the suffix alone is not treated as proof of gzip compression. The recorded source size is **11,510,596,115 bytes**.

The preparer maps version-stripped Ensembl IDs to **60,529 unique vocabulary genes**, sums duplicate mapped IDs, checks finite nonnegative integer-like RNA counts, and selects cells with at least 1,023 positive mapped genes. With split seed **20261019**, it reserves **308 training cells** and **1,024 distinct evaluation cells**, sampled without replacement. Evaluation rows are disjoint from this continued-training control, but original model-pretraining overlap and donor/study separation are unknown.

```text
phase1/results/rna_control_inputs/
  profiles.npz                 # gene_ids, lfq field storing RNA counts, samples
  cell_split.npz               # train_rows, evaluation_rows
  manifest.json
```

The `lfq` array name is reused for interface compatibility; in this directory it stores RNA counts, not protein measurements.

`train_rna_control.py` independently starts from the original checkpoint and completes **308 updates**, matching the protein arm's optimizer, learning rate, weight decay, masking, maximum length, batch size, precision, and training seed. The profile-sampling function is identical. Count ties and differing measured-gene counts mean the sampled genes/masks need not remain identical across modalities. This is an equal-update small-corpus control, not the paper's million-sample bulk-RNA control.

Displayed loss was **154.649200** at step 1 and **161.671204** at step 308; this is not a fixed-profile validation comparison. All displayed records used one attempt and scale 1.0, with approximately **1.3292 GiB** peak allocation. Both arms completed and saved separate checkpoints. A stray terminal backtick produced an error only after RNA training finished and did not invalidate the completed run.

```text
phase1/results/rna_continued_seed42/
  model.safetensors
  model_config.yml
  collator_config.yml
  vocab.json
  training.csv
  manifest.json
```

## Completed paired RNA reconstruction evaluation

`evaluate_paired_rna.py` performs no training. It verifies checkpoint hashes and completed 308-update manifests, checks source size/mtime and train/evaluation row disjointness, and loads each checkpoint strictly into the same adapter.

All three models see the **same 1,024 reserved cells**, selected genes, quantile-binned targets, and masks, generated once with mask/sampling seed **73021**. Evaluation uses batch size 1, sequence length 1,024, FP16 autocast, efficient SDPA, and evaluation mode. The metric is per-cell masked expression/MVC MSE, averaged equally across the two decoders and then across cells. Lower is better.

| Checkpoint | Cells | Mean expression MSE | Mean MVC MSE | Mean combined MSE |
| --- | ---: | ---: | ---: | ---: |
| Original | 1,024 | 199.1997450515628 | 202.5993195027113 | 200.89953212440014 |
| Protein continued | 1,024 | 198.45608840882778 | 201.66026303172112 | 200.05817591398954 |
| RNA-only control | 1,024 | 188.60444156825542 | 193.05515705794096 | **190.82979918271303** |

| Paired comparison | Mean combined-MSE difference | Fraction of cells where protein MSE is lower |
| --- | ---: | ---: |
| Protein minus original | -0.8413562104105949 | 0.5830078125 (597/1,024) |
| Protein minus RNA control | +9.228376731276512 | 0.03515625 (36/1,024) |

**Observed conclusion:** protein training slightly improves reconstruction relative to the original checkpoint, but the RNA-only control performs better. This experiment does not support a protein-specific advantage for this reconstruction diagnostic. It does not confirm or refute the paper's downstream representation claim. No significance test, confidence interval, multiple-training-seed result, or independent final-output audit is claimed.

The evaluator also saves CLS embeddings generated from **masked** inputs. These are diagnostic outputs, not the unmasked contextual embeddings required for standard downstream representation evaluation; they must not be silently substituted for Figure 2 embeddings.

```text
phase1/results/paired_rna_evaluation/
  evaluation_inputs.npz
  original_per_cell.csv
  protein_per_cell.csv
  rna_control_per_cell.csv
  original_masked_cls.npz
  protein_masked_cls.npz
  rna_control_masked_cls.npz
  summary.csv
  report.json
```

All three 1,024-cell passes completed locally. The values above are transcribed from the displayed evaluator output. Final per-cell files and checkpoint hashes have not been independently recomputed from an uploaded final-results archive.

## Executable sequence for the completed experiment

From the repository root, using the existing environment:

```bat
"%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe" phase1/test_gpu_70m.py
"%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe" phase1/test_tahoe_original.py
"%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe" phase1/prepare_protein_training.py
"%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe" phase1/train_protein_epoch.py
"%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe" phase1/prepare_rna_control.py
"%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe" phase1/train_rna_control.py
"%LOCALAPPDATA%\venvs\msigdb-managed312\Scripts\python.exe" phase1/evaluate_paired_rna.py
```

These document completed stages, not instructions to rerun or overwrite them. `prepare_protein_training.py` consumes the existing mapping contract; checkpoint acquisition and contract generation are prerequisites. Training and paired-evaluation scripts refuse existing output directories. The tested adapter is shared unchanged by both training arms and evaluation.

# Paper protocol and differences from this experiment

The uploaded 11-page preprint, including supplementary information, was reviewed after the independent training/reconstruction experiment. The paper's Methods, Figure 2 caption, and supplementary sections are the source for the distinctions below; the earlier abstract-only account was incomplete.

## Author training and controls

The paper describes continued pretraining of the published 70M checkpoint for one epoch on **48,843 proteomic samples from 440 sources**, using the original Tahoe training recipe. Proteins are mapped to gene tokens, abundances to expression values, with per-sample quantile binning. Raw and batch-corrected views form a format-aware augmentation, with uniform sampling. The headline corpus excludes sources with fewer than ten samples and excludes drugs designated for held-out perturbation evaluation.

The RNA controls use **ARCHS4 bulk RNA-seq**, not our 308 single cells. The paper reports 877,715 successfully encoded samples from 888,821 source samples, and an RNA-only training arm of **1,003,000 rows, batch size 4, 250,750 steps**, for one epoch. The precise relationship between the encoded-source count and training-row count remains unresolved. A combined proteomics-plus-bulk-RNA arm is also evaluated. Their budget comparison is a one-epoch/per-sample exposure comparison, not our equal-308-update control.

The paper reports multiple continued-training seeds, generally three for the Figure 2 arms. Our experiment uses one seed, one protein study, raw measured LFQ only, no batch-corrected view, and independently chosen optimizer/loss/runtime settings. The PDF does not explicitly supply a complete proteomics optimizer, learning-rate, batch-size, loss, or batch-correction implementation contract. Referencing the original recipe does not establish that our selected settings are equivalent.

## Author representation evaluation

Figure 2 evaluates RNA-derived contextual gene/cell representations using the original Tahoe evaluation code and datasets:

- DepMap gene-essentiality prediction, AUPRC.
- MSigDB Hallmark pathway-membership prediction, AUPRC.
- Tissue-of-origin separation, the upstream LISI-labelled enrichment score.

Comparisons include original 70M, 1B, and 3B checkpoints, proteomics-continued 70M, bulk-RNA-continued 70M, combined-data-continued 70M, and classical references. The authors report improvements in essentiality and tissue separation with proteomics; pathway membership approximately matches the larger models within seed variability. Bulk RNA improves pathway membership and tissue separation but not essentiality in their reported comparison. These are author-reported results, not results reproduced by our trained checkpoints.

**Masked RNA reconstruction MSE is not the Figure 2 claim metric.** Existing deposited-artifact evaluations and our masked CLS diagnostic outputs do not replace fresh contextual embeddings from the three actual checkpoints under identical RNA input conditions.

## Author held-out protein evaluation and exclusions

The paper separately evaluates PXD014791 and ProTargetMiner with leave-one-drug-context-pair-out prediction using five cosine-nearest pair embeddings. It scores predicted versus measured log-fold changes relative to matched controls, primarily across the 100 proteins with largest absolute changes. The reported evaluated counts are **58 cardiomyocyte pairs** and **61 ProTargetMiner pairs**.

Supplementary information lists 21 excluded cardiomyocyte drugs and states that trastuzumab is retained. This establishes author exclusion intent, not the local metadata-code mapping or exact scoring manifest. Historical `TRS` exclusion remains provisional. Our all-sample PXD014791 training violates the unseen-drug condition for evaluation on that dataset, regardless of whether a downstream neighbour predictor excludes the query pair.

No successful ProTargetMiner reporter-channel reconstruction or benchmark run has been established in this project. This branch is not a prerequisite for the central Figure 2 representation experiment.

## Remaining reproducibility limits

- Original 70M checkpoint provenance is now pinned and locally hash-checked; provenance of older deposited embedding identifiers remains unresolved.
- The full author proteomics corpus, exact raw/corrected-view construction, drug-exclusion manifest, and complete training configuration have not been reproduced.
- The small-corpus RNA control is matched by updates/settings, not by source distribution or the paper's bulk-RNA sample budget.
- RNA evaluation rows are disjoint from our continued-training control, but original pretraining overlap and donor/study-level separation are unknown.
- Final training/evaluation output files need independent audit; console execution and local assertions are not that audit.
- Fresh unmasked/contextual gene and cell embeddings and downstream scores for our three checkpoints have not yet been generated.

# Next steps

1. Preserve completed classical baselines, deposited-artifact evaluations, original checkpoint, both trained checkpoints, and paired reconstruction outputs.
2. Independently audit final training manifests/logs, checkpoint identities, and all paired per-cell reconstruction outputs without retraining.
3. Generate fresh RNA-derived contextual gene and unmasked cell embeddings from original, protein-continued, and RNA-control checkpoints using identical input cells, feature mapping, preprocessing, and aggregation. Do not substitute static vocabulary embeddings or masked CLS diagnostics for contextual gene/cell representations.
4. Run the established DepMap, MSigDB, and tissue-separation evaluations on those fresh embeddings, with common labels/splits/classifiers. This is the next test relevant to the paper's central claim.
5. Resolve equivalence of the completed independent MSigDB PCA to the author construction, remaining DepMap feature differences, exact author evaluation configuration, deposited-embedding provenance, and local tissue-output audit limits.
6. For exact paper reproduction, establish the full proteomics corpus, format-aware augmentation, training configuration, ARCHS4 controls, exclusions, and multi-seed protocol before retraining.
7. Treat protein-perturbation transfer as a separate branch. Do not evaluate the PXD014791-trained checkpoint as unseen-drug transfer on PXD014791; ProTargetMiner remains unvalidated.

**Where we stand:** actual 70M checkpoint loading, GPU inference/backpropagation, protein continued training, an equal-update RNA-only control, and paired RNA reconstruction evaluation are complete locally. The RNA control wins the reconstruction diagnostic. The central downstream representation claim remains untested with our newly trained checkpoints, and Figures 2 and 3 are not fully reproduced.

## Sources

- [Target preprint](https://www.biorxiv.org/content/10.64898/2026.08.14.744845v1.full), DOI **10.64898/2026.08.14.744845**.
- [PXD014791 source study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8776854/), DOI **10.1038/s41597-021-01114-3**; Table 1 identifies `TRA` as Trametinib.
- Published original checkpoint: `tahoebio/Tahoe-x1`, pinned revision `d218a580b9c2500ae9dfc8367a398545e6f017a8`, `70m-model` subdirectory.
- Uploaded target-paper PDF: Methods, Figure 2 caption, protein-transfer protocol, and supplementary information; author claims are distinguished from local results above.
- [Upstream DepMap random-forest implementation](https://github.com/tahoebio/tahoe-x1/blob/main/scripts/depmap/rf.py).

The earlier README's claims that Trametinib was absent, that the formal data contained only 57 pairs, and that proprietary reprocessing was required are superseded. The verified data contain 60 formal pairs. The local `TRS` identity and exact author scoring split remain unresolved. The paper now establishes author drug-exclusion intent, while our all-sample PXD014791 training remains unsuitable for its held-out-drug claim.
