#!/usr/bin/env python
"""Extract zero-shot SpaCompass embeddings for one spatial dataset.

The released model is applied to the first-stage gene-attention encoder: every
spot/cell is tokenised into gene vocabulary ids plus binned expression values,
and the encoder returns a 1536-dimensional embedding that is L2-normalised
before storage.

Example
-------
python scripts/extract_embeddings.py \
    --input-h5ad data/ST8059048.h5ad \
    --output-dir embeddings/ST8059048 \
    --checkpoint weights/spacompass_pretrained.pt \
    --config configs/spacompass.json \
    --species mouse --platform visium --genes-are-ensembl --normalise
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacompass.infer import build_dataset, encode_dataset  # noqa: E402
from spacompass.model.builder import load_gene_model, read_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--species", choices=("human", "mouse"), default="human")
    parser.add_argument(
        "--platform",
        choices=("scRNA", "visium", "stereo", "merfish", "xenium"),
        default="visium",
        help="Platform token; must match how the reference data was encoded.",
    )
    parser.add_argument(
        "--expression-layer",
        default=None,
        help="AnnData layer to use as the expression matrix; omit to use .X",
    )
    parser.add_argument(
        "--genes-are-ensembl",
        action="store_true",
        help="Set when var_names are Ensembl ids instead of gene symbols.",
    )
    parser.add_argument(
        "--gene-column",
        default=None,
        help=(
            "AnnData var column holding the gene symbols/Ensembl ids. Use this when "
            "var_names are positional indices (e.g. '0', '1', ...)."
        ),
    )
    parser.add_argument(
        "--normalise",
        action="store_true",
        help=(
            "Input is raw counts: normalise to 10,000 counts, log1p, then select "
            "HVGs. Use this for raw count matrices."
        ),
    )
    parser.add_argument(
        "--platform-mean-file",
        type=Path,
        default=None,
        help="Optional platform-mean table used for platform-mean normalisation.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--n-token", type=int, default=1200)
    parser.add_argument("--subset-column", default=None,
                        help="Keep only rows whose obs[--subset-column] equals --subset-value.")
    parser.add_argument("--subset-value", default=None)
    parser.add_argument("--spatial-columns", nargs=2, default=None, metavar=("X", "Y"),
                        help="Build obsm['spatial'] from these two obs columns when it is absent.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-un-normalised",
        action="store_true",
        help="Skip the final L2 normalisation (not part of the released protocol).",
    )
    return parser.parse_args()


def load_matrix(adata: ad.AnnData, layer: str | None) -> ad.AnnData:
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Missing expression layer '{layer}' in {adata.filename}")
        adata.X = adata.layers[layer]
    return adata


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.min_batch_size < 1:
        raise ValueError("Batch sizes must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = args.input_h5ad.name.removesuffix(".h5ad")
    embedding_path = args.output_dir / f"{sample}_spacompass_embeddings.npy"
    if embedding_path.exists() and not args.overwrite:
        print(f"Output already exists: {embedding_path} (use --overwrite to redo)")
        return

    started = time.time()
    config = read_config(args.config)
    model, source = load_gene_model(config, args.checkpoint)
    print(f"Loaded stage-1 weights from '{source}' of {args.checkpoint}")

    device = torch.device(args.device)
    model.to(device).eval()

    adata = ad.read_h5ad(args.input_h5ad)
    if args.subset_column is not None:
        if args.subset_value is None:
            raise ValueError("--subset-column requires --subset-value")
        if args.subset_column not in adata.obs:
            raise KeyError(f"Missing obs['{args.subset_column}'] in {args.input_h5ad}")
        mask = adata.obs[args.subset_column].astype(str).to_numpy() == str(args.subset_value)
        if not mask.any():
            raise ValueError(f"No rows with obs['{args.subset_column}'] == '{args.subset_value}'")
        adata = adata[mask].copy()
    if args.spatial_columns is not None and "spatial" not in adata.obsm:
        x, y = args.spatial_columns
        adata.obsm["spatial"] = adata.obs[[x, y]].to_numpy()
    original_names = adata.obs_names.astype(str).to_numpy(copy=True)
    original_n_obs = adata.n_obs
    if args.gene_column is not None:
        if args.gene_column not in adata.var:
            raise KeyError(f"Missing var column '{args.gene_column}' in {args.input_h5ad}")
        adata.var_names = adata.var[args.gene_column].astype(str).to_numpy()
    adata = load_matrix(adata, args.expression_layer)

    dataset = build_dataset(
        adata,
        data_is_raw=bool(args.normalise),
        is_symbol=not args.genes_are_ensembl,
        species=args.species,
        platform=args.platform,
        n_token=args.n_token,
        mean_file=str(args.platform_mean_file) if args.platform_mean_file else None,
    )
    if dataset.adata.n_obs != original_n_obs:
        raise RuntimeError(f"Spot count changed from {original_n_obs} to {dataset.adata.n_obs}")
    output_names = dataset.adata.obs_names.astype(str).to_numpy(copy=True)
    if not np.array_equal(output_names, original_names):
        raise RuntimeError("Spot order changed during preprocessing")

    embeddings, actual_batch_size = encode_dataset(
        dataset,
        model,
        batch_size=args.batch_size,
        min_batch_size=args.min_batch_size,
        device=device,
        l2_normalise=not args.save_un_normalised,
    )
    if embeddings.shape != (original_n_obs, config["d_model"]):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Embedding matrix contains non-finite values")
    np.save(embedding_path, embeddings, allow_pickle=False)
    (args.output_dir / f"{sample}_obs_names.tsv").write_text(
        "barcode\n" + "\n".join(output_names) + "\n", encoding="utf-8"
    )
    manifest = {
        "sample": sample,
        "input_h5ad": str(args.input_h5ad.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_component": source,
        "config": str(args.config.resolve()),
        "embedding_stage": "first_stage_gene_attention",
        "embedding_dim": int(embeddings.shape[1]),
        "n_obs": int(embeddings.shape[0]),
        "species": args.species,
        "platform": args.platform,
        "expression_layer": args.expression_layer,
        "normalised_counts": bool(args.normalise),
        "genes_are_symbols": not args.genes_are_ensembl,
        "gene_column": args.gene_column,
        "platform_mean_file": str(args.platform_mean_file) if args.platform_mean_file else None,
        "l2_normalised": not args.save_un_normalised,
        "spot_filtering": False,
        "n_token": args.n_token,
        "inference_batch_size": actual_batch_size,
        "device": str(device),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.output_dir / f"{sample}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {embeddings.shape} embeddings to {embedding_path}")


if __name__ == "__main__":
    main()
