#!/usr/bin/env python
"""Run the batch-integration benchmark on the five released Visium sections.

The sections are concatenated into one sample (as in the benchmark), the
two-stage SpaCompass embeddings are extracted once for the whole sample, and the
scIB integration metrics are computed on the pooled stage-1 representations.

Example
-------
python scripts/run_batch_integration.py --data-dir /path/to/datasets
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from batch_integration import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REPO / "configs/datasets.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "weights/spacompass_pretrained.pt")
    parser.add_argument("--config", type=Path, default=REPO / "configs/spacompass.json")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/batch_integration")
    parser.add_argument("--knn", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reuse-embeddings", action="store_true",
                        help="Skip extraction when the embedding file already exists.")
    return parser.parse_args()


def extract(args: argparse.Namespace, spec: dict) -> tuple[Path, Path, Path]:
    """Extract two-stage embeddings for the concatenated sample."""
    name = spec["name"]
    embedding_dir = args.output_dir / "embeddings" / name
    embedding_dir.mkdir(parents=True, exist_ok=True)
    first_path = embedding_dir / f"{name}_spacompass_first_stage_embeddings.npy"
    names_path = embedding_dir / f"{name}_obs_names.tsv"
    if args.reuse_embeddings and first_path.exists() and names_path.exists():
        print(f"[spacompass] reusing embeddings in {embedding_dir}", flush=True)
        return first_path, names_path, embedding_dir
    command = [
        sys.executable, str(HERE / "extract_embeddings_two_stage.py"),
        "--input-h5ad", *[str(args.data_dir / spec["files"][s]) for s in spec["samples"]],
        "--name", name,
        "--output-dir", str(embedding_dir),
        "--checkpoint", str(args.checkpoint),
        "--config", str(args.config),
        "--species", spec["species"],
        "--platform", spec["platform"],
        "--knn", str(args.knn),
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    if spec.get("genes_are_ensembl"):
        command.append("--genes-are-ensembl")
    if spec.get("normalise"):
        command.append("--normalise")
    print(f"[spacompass] extracting two-stage embeddings for {len(spec['samples'])} sections", flush=True)
    if subprocess.run(command).returncode != 0:
        raise SystemExit("embedding extraction failed")
    return first_path, names_path, embedding_dir


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    spec = registry["batch_integration"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first_path, names_path, _ = extract(args, spec)
    embeddings = np.load(first_path).astype(np.float32)
    keep_names = pd.read_csv(names_path, sep="\t")["barcode"].astype(str).to_numpy()

    frames = [ad.read_h5ad(args.data_dir / spec["files"][sample]) for sample in spec["samples"]]
    combined = ad.concat(frames, join="inner")
    combined = combined[keep_names].copy()
    combined.obs["celltype"] = combined.obs[spec["label_key"]].astype(str).astype("category")
    combined.obs["batch_ids"] = combined.obs[spec["batch_key"]].astype(str).astype("category")
    if embeddings.shape[0] != combined.n_obs:
        raise ValueError("Embedding rows do not match the evaluated spots")
    combined.obsm["X_cell"] = embeddings

    print(f"[spacompass] running scIB evaluation on {combined.n_obs} spots", flush=True)
    metrics = evaluate(combined, spec["organism"], True)
    payload = {
        "name": spec["name"],
        "n_obs": int(combined.n_obs),
        "n_batches": int(combined.obs["batch_ids"].nunique()),
        "n_labels": int(combined.obs["celltype"].nunique()),
        "protocol": "two-stage patch embeddings, pooled along the spot axis",
        "knn": args.knn,
        "seed": args.seed,
        "metrics": metrics,
    }
    (args.output_dir / f"{spec['name']}_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame([metrics]).to_csv(args.output_dir / f"{spec['name']}_metrics.csv", index=False)
    print("\n--- batch integration ---")
    for key in ("avg_bio", "avg_batch", "graph_conn", "NMI_cluster/label", "ARI_cluster/label",
                "ASW_label", "ASW_label/batch"):
        if key in metrics:
            print(f"  {key:20s} {metrics[key]:.6f}")
    reference = {k: spec[k] for k in spec if k.startswith("reference_")}
    if reference:
        print("\n  reference:", json.dumps(reference))


if __name__ == "__main__":
    main()
