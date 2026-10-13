# SpaCompass

SpaCompass is a two-stage foundation model for spatial transcriptomics. Stage 1
encodes the gene-expression profile of a spot or cell with a gene-attention
encoder; stage 2 models a local niche with a cell-attention encoder over the
spatial neighbourhood graph. The learned representations are used zero-shot:
the released encoder is applied to a new sample without any fine-tuning.

This repository contains the model code, the released pretrained weights, and
the scripts used for zero-shot embedding extraction, spatial clustering and
batch integration.

## Repository layout

```
spacompass/
├── spacompass/                     # model package
│   ├── model/                      # gene attention, cell attention, MoE, graph/bias attention
│   ├── data/                       # tokenisation, preprocessing, patch construction
│   ├── infer.py                    # batched zero-shot inference
│   ├── infer_two_stage.py          # patch-level (two-stage) inference
│   └── utils/                      # clustering helpers
├── scripts/
│   ├── extract_embeddings.py             # zero-shot embeddings (stage 1)
│   ├── extract_embeddings_two_stage.py   # patch-level embeddings (stages 1+2)
│   ├── clustering_test.py                # clustering benchmark, one dataset
│   ├── run_clustering_benchmark.py       # clustering benchmark, all datasets
│   ├── batch_integration.py              # scIB evaluation from a manifest
│   └── run_batch_integration.py          # scIB benchmark, five sections in one command
├── configs/
│   ├── spacompass.json             # released architecture configuration
│   └── datasets.json               # per-dataset benchmark settings
├── data/                           # dataset documentation and checksums (files distributed separately)
├── resources/                      # gene vocabularies and platform means (see resources/README.md)
├── weights/                        # released pretrained weights (see weights/README.md)
└── examples/  LICENSE  THIRD_PARTY_NOTICES.md  environment.yml  requirements.txt
```

## Installation

```bash
conda env create -f environment.yml
conda activate spacompass
```

A pip-only alternative is `pip install -r requirements.txt`.

The `mclust` clustering protocol needs R with `mclust` plus
`rpy2` (both in `environment.yml`); `leiden`/`louvain` are pure-Python
alternatives available via `--method`. Activate the environment before running
so that `R` is on `PATH` — otherwise `rpy2` cannot locate `R_HOME`.

The released environment was validated with Python 3.10, PyTorch 2.4.1,
scanpy 1.11.0, anndata 0.11.3, scIB 1.1.7, scikit-learn 1.5.2, and R 4.4 with
mclust 6.1.1.

## Pretrained weights

The released weights are distributed as `weights/spacompass_pretrained.pt`
(1.9 GB). Both stages live in one state dict:

```python
checkpoint["model"]["first_stage_model.*"]   # stage 1, gene attention
checkpoint["model"]["second_stage_model.*"]  # stage 2, cell attention
```

`spacompass.model.builder` rebuilds both modules from
`configs/spacompass.json` and loads the checkpoint, so no fairseq installation
is required for inference. The checksum is listed in `weights/README.md`.

## Pretraining data

The large-scale single-cell and spatial-transcriptomics corpus used for
pretraining is not distributed with this release. The manuscript describing the
corpus is currently under review; the download address will be published here
once it has been accepted.

**Download address: to be added.**

## Zero-shot embeddings

```bash
python scripts/extract_embeddings.py \
    --input-h5ad data/ST8059048.h5ad \
    --output-dir embeddings/ST8059048 \
    --checkpoint weights/spacompass_pretrained.pt \
    --config configs/spacompass.json \
    --species mouse --platform visium --genes-are-ensembl --normalise
```

Outputs per sample: `<sample>_spacompass_embeddings.npy` (float32, `n_obs x 1536`,
L2-normalised), `<sample>_obs_names.tsv` (row order) and `<sample>_manifest.json`
(all preprocessing and inference settings, for auditability).

Embeddings are always taken from the **first-stage gene-attention output**,
which is the representation used by the released benchmark scripts.

### Input conventions

| Flag | Meaning |
| --- | --- |
| `--expression-layer` | AnnData layer holding the expression matrix (omit to use `.X`) |
| `--normalise` | Input is raw counts: normalise to 10,000 counts, `log1p`, then select HVGs |
| `--genes-are-ensembl` | `var_names` hold Ensembl ids instead of gene symbols |
| `--gene-column` | Read gene ids from a `var` column, for files whose `var_names` are positional indices |
| `--species`, `--platform` | Vocabulary and platform token, e.g. `human`/`visium`, `mouse`/`stereo` |
| `--platform-mean-file` | Optional platform-mean table for platform-mean normalisation |
| `--n-token` | Gene tokens per spot (default 1200, as released) |

Omitting `--normalise` treats the matrix as already normalised and log-transformed,
which is the setting used for the released low-count spatial assays.

Tokenisation fails loudly when no gene matches the vocabulary: that always means
either the gene naming or `--species` is wrong, and silently continuing would
return the same embedding for every spot. Files that store positional
`var_names` and keep the real ids in a `var` column need, for example,
`--gene-column gene --genes-are-ensembl --species human`.

## Spatial clustering

```bash
python scripts/clustering_test.py \
    --input-h5ad data/ST8059048.h5ad \
    --output-dir results/ST8059048 --name ST8059048 \
    --annotation-key annotation --n-clusters 26 \
    --checkpoint weights/spacompass_pretrained.pt \
    --config configs/spacompass.json \
    --species mouse --platform visium --normalise
```

The protocol is PCA (20 components) followed by mclust; ARI and NMI are reported
against the supplied annotation column. Pass `--embedding-path` to score
embeddings produced earlier.

## Clustering benchmark

```bash
python scripts/run_clustering_benchmark.py --data-dir /path/to/datasets
```

Every dataset in `configs/datasets.json` is processed with its own settings
(platform token, gene-id convention, whether the matrix holds raw counts, the
number of clusters and the annotation column), and the per-dataset ARI/NMI are
written to `results/clustering/clustering_benchmark_summary.csv` next to the
reference values. Add `--datasets E9.5 ST8059048` to run a subset.

## Batch integration

```bash
python scripts/run_batch_integration.py --data-dir /path/to/datasets
```

The five Visium sections are concatenated into a single sample, the two-stage
(patch-level) embeddings are extracted once for that sample, and the stage-1
representations are pooled along the spot axis for the scIB evaluation. Metrics
follow the shared scIB protocol (`NMI`, `ARI`, `ASW_label`, `ASW_label/batch`,
`graph_conn`, `PCR`), with `avg_bio` and `avg_batch` summarising biological
conservation and batch mixing.

## Single-sample and per-section scripts

`scripts/extract_embeddings.py`, `scripts/clustering_test.py` and
`scripts/batch_integration.py` expose the same pipeline one dataset at a time
(for example to score embeddings that were produced elsewhere, or to evaluate a
different set of sections through a manifest).

## Resources

`resources/` contains the gene vocabularies
(`{human,mouse}_merged_gene_to_ensembl.json`, `{human,mouse}_vocab.json`) and the
platform mean tables (`{human,mouse}_platform_mean.tsv`) that the tokeniser and
optional platform-mean normalisation need. Set `SPACOMPASS_RESOURCES` to load
them from another directory.

## Scope and limitations

- This release covers inference and evaluation. Pretraining and fine-tuning
  entry points are not included.
- Embeddings are intended for zero-shot use. Fine-tuned or study-specific
  variants are outside the scope of this release.
- The pretraining corpus is not included (see _Pretraining data_).
- Task-specific downstream evaluations (for example niche-query benchmarks) are
  not part of this release.

## Reproducibility checks

The released code was checked against the published results. Reference values
for the released checkpoint:

| Check | Expected |
| --- | --- |
| Clustering, mouse Visium `ST8059048`, 26 clusters, mclust | ARI 0.5850983016348106, NMI 0.7459886855081829 |
| Batch integration, five mouse brain Visium sections (14,965 spots, 5 batches, 29 labels) | avg_bio 0.593613, avg_batch 0.985939, graph_conn 0.986838 |

Batch-integration metrics are compared with `--hvg-score`, which is the setting
used for the reported runs. Note that these values are sensitive to the
`numba`/`llvmlite`/`pynndescent` versions listed in `environment.yml`, because
scIB's clustering-based metrics operate on an approximate kNN graph.

Expression binning breaks ties with `np.random`; the released scripts seed it
(`--seed 42`, the same convention as the embedding extraction), so every run of
`run_clustering_benchmark.py` and `run_batch_integration.py` is deterministic.

Some public datasets store positional `var_names` (`'0'`, `'1'`, …) with the real
gene ids in a `var` column; encode those with `--gene-column <column>` plus
`--genes-are-ensembl` or the symbols setting that matches the file. The
tokeniser raises an error when no gene matches the vocabulary, because that
always means the gene naming or `--species` is wrong.

## Licence

SpaCompass is released under the MIT License (see `LICENSE`).

The model implementation is derived from the CellPLM code base and has been
substantially modified; the derived parts keep the original BSD-2-Clause terms,
which are reproduced in `THIRD_PARTY_NOTICES.md`. Both licences are permissive
and compatible, so the BSD-2 notice must simply be retained alongside the MIT
licence.
