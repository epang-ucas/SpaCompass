# SpaCompass benchmark datasets

This file documents the nine `.h5ad` files used by the released clustering and
batch-integration benchmarks. They are distributed separately from the code
because of their size (~12 GB in total). Keep a copy of this file inside the
dataset archive so the download is self-describing.

| File | Technology | Spots/cells | Genes | Labels | Classes | Size |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `E9.5_E1S1.MOSTA.h5ad` | Stereo-seq | 5,913 | 25,568 | `annotation` | 12 | 0.5 GB |
| `E14.5_E1S3.MOSTA.h5ad` | Stereo-seq | 92,928 | 18,582 | `annotation` | 16 | 2.4 GB |
| `E16.5_E2S5.MOSTA.h5ad` | Stereo-seq | 132,481 | 28,555 | `annotation` | 24 | 4.8 GB |
| `merfish-updated.h5ad` | MERFISH | 280,327 (all sections) | 254 | `subclass` | 25 | 1.5 GB |
| `ST8059048.h5ad` | 10x Visium | 2,987 | 12,809 | `annotation` | 26 | 0.5 GB |
| `ST8059049.h5ad` | 10x Visium | 3,499 | 12,809 | `annotation` | 28 | 0.6 GB |
| `ST8059050.h5ad` | 10x Visium | 3,497 | 12,809 | `annotation` | 28 | 0.7 GB |
| `ST8059051.h5ad` | 10x Visium | 2,409 | 12,809 | `annotation` | 28 | 0.6 GB |
| `ST8059052.h5ad` | 10x Visium | 2,576 | 12,809 | `annotation` | 29 | 0.6 GB |

## Usage

Put the nine files in one directory and pass that directory to the benchmarks:

```bash
python scripts/run_clustering_benchmark.py --data-dir /path/to/datasets
python scripts/run_batch_integration.py --data-dir /path/to/datasets
```

The file names are part of the interface: they are listed in
`configs/datasets.json`, which also holds every per-dataset setting (platform
token, gene-id convention, whether the matrix holds raw counts, the number of
clusters and the annotation column).

Notes

- The three embryo sections come from the MOSTA mouse organogenesis atlas
  (Stereo-seq): Chen et al., *Cell* (2022), doi:10.1016/j.cell.2022.04.003.
- `merfish-updated.h5ad` holds many sections; the clustering benchmark uses the
  `mouse1_slice71` section, selected inside the released script through the
  `slice_id` column. Its spatial coordinates live in the `center_x` / `center_y`
  columns, which the script copies into `obsm["spatial"]`.
- The five Visium sections are the public 10x Genomics mouse brain sections
  `ST8059048`–`ST8059052` (10x Genomics); the batch-integration benchmark
  concatenates them in that order.
- Every file stores the raw count matrix in `X`. The released scripts normalise,
  log-transform and select HVGs themselves, so no extra preprocessing is needed.
- When using these datasets, cite the corresponding original studies.
