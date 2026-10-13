#!/usr/bin/env python
"""Batch-integration benchmark for SpaCompass embeddings across sections.

Per-section SpaCompass embeddings are pooled along the spot axis into one
representation space and scored with the scIB integration metrics.  There is no
target-specific adaptation, no re-encoding and no dimension alignment, so the
reported numbers reflect the frozen pretrained encoder.

Example
-------
python scripts/batch_integration.py \
    --manifest examples/batch_integration_manifest.json \
    --output-dir results/batch_integration \
    --name mouse_brain_5_sections \
    --label-key annotation --batch-key batch_ids --organism mouse
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc
import scib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path,
                        help="JSON list of {name, h5ad, embedding} records.")
    parser.add_argument("--input-dir", type=Path, help="Directory of *.h5ad files (convenience mode).")
    parser.add_argument("--embedding-dir", type=Path,
                        help="Directory of '<sample>_spacompass_embeddings.npy' files (convenience mode).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--label-key", default="annotation")
    parser.add_argument("--batch-key", default="batch_ids")
    parser.add_argument("--organism", choices=("human", "mouse"), default="mouse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hvg-score", action="store_true",
                        help="Enable scIB's per-batch HVG score (unstable on some sparse data).")
    return parser.parse_args()


def resolve_records(args: argparse.Namespace) -> list[dict]:
    if args.manifest is not None:
        records = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not records:
            raise ValueError(f"Empty manifest: {args.manifest}")
        return records
    if args.input_dir is None or args.embedding_dir is None:
        raise ValueError("Provide --manifest, or both --input-dir and --embedding-dir")
    records = []
    for h5ad_path in sorted(args.input_dir.glob("*.h5ad")):
        sample = h5ad_path.name.removesuffix(".h5ad")
        embedding = args.embedding_dir / f"{sample}_spacompass_embeddings.npy"
        if not embedding.exists():
            raise FileNotFoundError(embedding)
        records.append({"name": sample, "h5ad": str(h5ad_path), "embedding": str(embedding)})
    if not records:
        raise FileNotFoundError(f"No *.h5ad files in {args.input_dir}")
    return records


def load_inputs(args: argparse.Namespace):
    datasets, provenance = [], []
    for record in resolve_records(args):
        sample = record["name"]
        adata = ad.read_h5ad(record["h5ad"])
        embeddings = np.load(record["embedding"]).astype(np.float32)
        if embeddings.shape != (adata.n_obs, 1536):
            raise ValueError(f"{sample}: unexpected embedding shape {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{sample}: embedding contains non-finite values")
        for key in (args.label_key, args.batch_key):
            if key not in adata.obs:
                raise KeyError(f"{sample}: missing obs['{key}']")
        # Visium barcodes repeat across sections, so make them globally unique.
        adata.obs_names = pd.Index([f"{sample}:{name}" for name in adata.obs_names.astype(str)])
        adata.obs["celltype"] = adata.obs[args.label_key].astype(str)
        adata.obs["batch_ids"] = adata.obs[args.batch_key].astype(str)
        adata.obsm["X_cell"] = embeddings
        datasets.append(adata)
        provenance.append({
            "sample": sample,
            "h5ad": str(Path(record["h5ad"]).resolve()),
            "embedding": str(Path(record["embedding"]).resolve()),
            "n_obs": int(adata.n_obs),
        })
    combined = ad.concat(datasets, join="inner", merge="same", index_unique=None)
    combined.obs["celltype"] = combined.obs["celltype"].astype("category")
    combined.obs["batch_ids"] = combined.obs["batch_ids"].astype("category")
    return combined, provenance


def evaluate(adata: ad.AnnData, organism: str, hvg_score: bool) -> dict:
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key="batch_ids",
        label_key="celltype",
        embed="X_cell",
        isolated_labels_asw_=True,
        silhouette_=True,
        hvg_score_=hvg_score,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=True,
        trajectory_=False,
        nmi_=True,
        ari_=True,
        cell_cycle_=True,
        kBET_=False,
        ilisi_=False,
        clisi_=False,
        organism=organism,
    )[0].to_dict()
    results["avg_bio"] = float(np.mean([
        results["NMI_cluster/label"], results["ARI_cluster/label"], results["ASW_label"],
    ]))
    results["avg_batch"] = float(np.mean([results["ASW_label/batch"], results["graph_conn"]]))
    return {key: float(value) for key, value in results.items() if not np.isnan(value)}


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adata, provenance = load_inputs(args)
    metrics = evaluate(adata, args.organism, args.hvg_score)

    pd.DataFrame([metrics]).to_csv(args.output_dir / f"{args.name}_metrics.csv", index=False)
    payload = {
        "name": args.name,
        "n_obs": int(adata.n_obs),
        "n_batches": int(adata.obs["batch_ids"].nunique()),
        "n_labels": int(adata.obs["celltype"].nunique()),
        "embedding": "stage-1 gene attention, pooled along the spot axis, no re-encoding",
        "metrics": metrics,
        "inputs": provenance,
    }
    (args.output_dir / f"{args.name}_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / f"{args.name}_adata_and_result.pkl").open("wb") as handle:
        pickle.dump((adata, metrics), handle)

    sc.pp.neighbors(adata, use_rep="X_cell", random_state=args.seed)
    sc.tl.umap(adata, min_dist=0.3, random_state=args.seed)
    for colour, suffix, score in (
        ("batch_ids", "batch_umap", "avg_batch"),
        ("celltype", "label_umap", "avg_bio"),
    ):
        figure = sc.pl.umap(
            adata, color=colour, frameon=False, return_fig=True, show=False,
            title=f"{args.name}: {score}={metrics[score]:.4f}",
        )
        figure.savefig(args.output_dir / f"{args.name}_{suffix}.png", dpi=300, bbox_inches="tight")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
