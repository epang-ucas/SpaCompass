#!/usr/bin/env python
"""Extract patch-level (two-stage) SpaCompass embeddings.

Stage 1 encodes every cell of a spatial patch and stage 2 refines the
representation with the patch's neighbourhood graph; the two outputs are
pooled over patches to give one vector per cell.

Several sections can be concatenated into a single sample by passing more than
one ``--input-h5ad``; the AnnData file names are used as the barcode prefix, as
in the released batch-integration benchmark.

Example
-------
python scripts/extract_embeddings_two_stage.py \
    --input-h5ad data/ST8059048.h5ad data/ST8059049.h5ad \
    --name visium_mouse_brain --output-dir embeddings/visium_mouse_brain \
    --checkpoint weights/spacompass_pretrained.pt \
    --config configs/spacompass.json \
    --species mouse --platform visium --genes-are-ensembl --normalise --knn 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacompass.data.patch_dataset import PatchDataset  # noqa: E402
from spacompass.infer_two_stage import encode_patches  # noqa: E402
from spacompass.model.builder import read_config  # noqa: E402
from spacompass.model.two_stage import load_two_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-h5ad", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default=None, help="Sample name (default: first input stem).")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--species", choices=("human", "mouse"), default="mouse")
    parser.add_argument("--platform", choices=("scRNA", "visium", "stereo", "merfish", "xenium"),
                        default="visium")
    parser.add_argument("--expression-layer", default=None)
    parser.add_argument("--genes-are-ensembl", action="store_true")
    parser.add_argument("--normalise", action="store_true")
    parser.add_argument("--n-token", type=int, default=1200)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--aggregation", choices=("first", "last", "mean"), default="first")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_inputs(paths: list[Path], layer: str | None, prefix_names: bool) -> ad.AnnData:
    frames = []
    for path in paths:
        adata = ad.read_h5ad(path)
        if layer is not None:
            if layer not in adata.layers:
                raise KeyError(f"Missing expression layer '{layer}' in {path}")
            adata.X = adata.layers[layer]
        if prefix_names and len(paths) > 1:
            prefix = f"{path.stem}_"
            adata.obs_names = [
                name if name.startswith(prefix) else f"{prefix}{name}"
                for name in adata.obs_names.astype(str)
            ]
        frames.append(adata)
    if len(frames) == 1:
        return frames[0]
    return ad.concat(frames, join="inner")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or args.input_h5ad[0].stem
    first_path = args.output_dir / f"{name}_spacompass_first_stage_embeddings.npy"
    if first_path.exists() and not args.overwrite:
        print(f"Output already exists: {first_path} (use --overwrite to redo)")
        return

    started = time.time()
    # The expression binning draws np.random for tie-breaking, so the seed is
    # part of the preprocessing contract (same convention as the other scripts).
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = read_config(args.config)
    adata = read_inputs(args.input_h5ad, args.expression_layer, prefix_names=True)
    original_n_obs = adata.n_obs

    dataset = PatchDataset(
        adata,
        data_is_raw=bool(args.normalise),
        is_symbol=not args.genes_are_ensembl,
        species=args.species,
        batch=args.platform,
        n_token=args.n_token,
        shuffle=False,
        knn=args.knn,
        patch_size=args.patch_size,
    )
    model = load_two_stage(config, args.checkpoint)
    result = encode_patches(dataset, model, strategy=args.aggregation, device=args.device)

    kept = result["covered_indices"]
    names = dataset.adata.obs_names.astype(str).to_numpy()
    first = result["first_stage"][kept]
    second = result["second_stage"][kept]
    np.save(first_path, first, allow_pickle=False)
    np.save(args.output_dir / f"{name}_spacompass_second_stage_embeddings.npy", second, allow_pickle=False)
    (args.output_dir / f"{name}_obs_names.tsv").write_text(
        "barcode\n" + "\n".join(names[kept]) + "\n", encoding="utf-8"
    )
    manifest = {
        "name": name,
        "inputs": [str(path) for path in args.input_h5ad],
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "embedding_stage": "two_stage_patch_aggregated",
        "aggregation": args.aggregation,
        "knn": args.knn,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "n_patches": int(dataset.data_len),
        "n_obs_input": int(original_n_obs),
        "n_obs_after_preprocessing": int(dataset.adata.n_obs),
        "n_obs_kept": int(len(kept)),
        "embedding_dim": int(first.shape[1]),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (args.output_dir / f"{name}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {first.shape} first-stage and {second.shape} second-stage embeddings to {args.output_dir}")


if __name__ == "__main__":
    main()
