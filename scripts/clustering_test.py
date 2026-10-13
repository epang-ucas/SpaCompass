#!/usr/bin/env python
"""Spatial clustering benchmark for SpaCompass embeddings.

Spatial clustering uses PCA(20) followed by mclust, and reports the adjusted
Rand index (ARI) and normalised mutual information (NMI) against the curated
annotations.

Example
-------
python scripts/clustering_test.py \
    --input-h5ad data/ST8059048.h5ad \
    --output-dir results/ST8059048 \
    --name ST8059048 \
    --annotation-key annotation --n-clusters 26 \
    --checkpoint weights/spacompass_pretrained.pt \
    --config configs/spacompass.json \
    --species mouse --platform visium --normalise
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from sklearn import metrics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacompass.infer import build_dataset, encode_dataset  # noqa: E402
from spacompass.model.builder import load_gene_model, read_config  # noqa: E402
from spacompass.utils.clustering import clustering  # noqa: E402

PALETTE = [
    "#F56867", "#556B2F", "#C798EE", "#59BE86", "#006400", "#8470FF",
    "#CD69C9", "#EE7621", "#B22222", "#FFD700", "#CD5555", "#DB4C6C",
    "#8B658B", "#1E90FF", "#AF5F3C", "#CAFF70", "#F9BD3F", "#DAB370",
    "#877F6C", "#268785", "#82EF2D", "#B4EEB4", "#00CED1", "#8B0000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--annotation-key", required=True)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--method", choices=("mclust", "leiden", "louvain"), default="mclust")
    parser.add_argument("--point-size", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    # Either reuse precomputed embeddings, or extract them on the fly.
    parser.add_argument("--embedding-path", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--species", choices=("human", "mouse"), default="mouse")
    parser.add_argument("--platform", choices=("scRNA", "visium", "stereo", "merfish", "xenium"), default="visium")
    parser.add_argument("--expression-layer", default=None)
    parser.add_argument("--genes-are-ensembl", action="store_true")
    parser.add_argument("--normalise", action="store_true")
    parser.add_argument("--platform-mean-file", type=Path, default=None)
    parser.add_argument("--n-token", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--subset-column", default=None)
    parser.add_argument("--subset-value", default=None)
    parser.add_argument("--spatial-columns", nargs=2, default=None, metavar=("X", "Y"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-plot", action="store_true")
    return parser.parse_args()


def load_embeddings(args: argparse.Namespace, adata: ad.AnnData) -> np.ndarray:
    if args.embedding_path is not None:
        embeddings = np.load(args.embedding_path).astype(np.float32)
        if embeddings.shape[0] != adata.n_obs:
            raise ValueError(
                f"Embedding rows ({embeddings.shape[0]}) do not match observations ({adata.n_obs})"
            )
        return embeddings
    if args.checkpoint is None or args.config is None:
        raise ValueError("Provide either --embedding-path, or both --checkpoint and --config")
    config = read_config(args.config)
    model, source = load_gene_model(config, args.checkpoint)
    print(f"Loaded stage-1 weights from '{source}' of {args.checkpoint}")
    dataset = build_dataset(
        adata,
        species=args.species,
        platform=args.platform,
        n_token=args.n_token,
        data_is_raw=bool(args.normalise),
        is_symbol=not args.genes_are_ensembl,
        mean_file=str(args.platform_mean_file) if args.platform_mean_file else None,
    )
    embeddings, _ = encode_dataset(
        dataset, model, batch_size=args.batch_size, device=args.device
    )
    return embeddings


def plot(adata: ad.AnnData, args: argparse.Namespace, path: Path, scores: dict) -> None:
    truth = adata.obs[args.annotation_key]
    predicted = adata.obs["domain"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    for axis, values, title, legend in (
        (axes[0], truth, "Ground truth", True),
        (axes[1], predicted, f"SpaCompass ({args.method})", True),
    ):
        categories = list(values.cat.categories) if hasattr(values, "cat") else list(pd.unique(values))
        colours = {label: PALETTE[index % len(PALETTE)] for index, label in enumerate(categories)}
        axis.scatter(
            adata.obsm[args.spatial_key][:, 0],
            -adata.obsm[args.spatial_key][:, 1],
            s=args.point_size,
            c=[colours[value] for value in values],
            linewidths=0,
        )
        axis.set_title(title + (f"\nARI={scores['ARI']:.3f}  NMI={scores['NMI']:.3f}" if legend and title.startswith("SpaCompass") else ""))
        axis.set_aspect("equal")
        axis.axis("off")
        if legend:
            handles = [
                Line2D([0], [0], marker="o", color="w", label=str(label),
                       markerfacecolor=colours[label], markersize=5)
                for label in categories
            ]
            axis.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                        fontsize=7, frameon=False, ncol=1)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    adata = ad.read_h5ad(args.input_h5ad)
    if args.subset_column is not None:
        if args.subset_value is None:
            raise ValueError("--subset-column requires --subset-value")
        if args.subset_column not in adata.obs:
            raise KeyError(f"Missing obs['{args.subset_column}']")
        mask = adata.obs[args.subset_column].astype(str).to_numpy() == str(args.subset_value)
        if not mask.any():
            raise ValueError(f"No rows with obs['{args.subset_column}'] == '{args.subset_value}'")
        adata = adata[mask].copy()
    if args.spatial_columns is not None and "spatial" not in adata.obsm:
        x, y = args.spatial_columns
        adata.obsm["spatial"] = adata.obs[[x, y]].to_numpy()
    if args.expression_layer is not None:
        if args.expression_layer not in adata.layers:
            raise KeyError(f"Missing expression layer '{args.expression_layer}'")
        adata.X = adata.layers[args.expression_layer]
    if args.annotation_key not in adata.obs:
        raise KeyError(f"Missing annotation column '{args.annotation_key}'")
    if args.spatial_key not in adata.obsm:
        raise KeyError(f"Missing spatial coordinates in obsm['{args.spatial_key}']")

    embeddings = load_embeddings(args, adata)
    adata.obsm["emb"] = embeddings
    try:
        clustering(adata, n_clusters=args.n_clusters, method=args.method, emb_key="emb")
    except Exception as error:
        if args.method == "mclust":
            raise RuntimeError(
                "mclust needs R with the 'mclust' package plus rpy2. Make sure R is on "
                "PATH (for example `export PATH=$CONDA_PREFIX/bin:$PATH`) and R_HOME is "
                "set, or use --method leiden / --method louvain."
            ) from error
        raise

    scores = {
        "ARI": float(metrics.adjusted_rand_score(adata.obs[args.annotation_key], adata.obs["domain"])),
        "NMI": float(metrics.normalized_mutual_info_score(adata.obs[args.annotation_key], adata.obs["domain"])),
    }
    if not args.skip_plot:
        plot(adata, args, args.output_dir / f"{args.name}_spacompass_clustering.png", scores)

    payload = {
        "name": args.name,
        "input_h5ad": str(args.input_h5ad.resolve()),
        "n_obs": int(adata.n_obs),
        "n_clusters": args.n_clusters,
        "annotation_key": args.annotation_key,
        "clustering": {"embedding": "emb (stage-1 gene attention)",
                       "pca_components": 20, "method": args.method, "seed": args.seed},
        "metrics": scores,
    }
    (args.output_dir / f"{args.name}_spacompass_clustering.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame([{"dataset": args.name, **scores}]).to_csv(
        args.output_dir / f"{args.name}_spacompass_clustering.csv", index=False
    )
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
