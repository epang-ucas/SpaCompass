#!/usr/bin/env python
"""Run the spatial-clustering benchmark on every released dataset.

Per-dataset hyperparameters (platform token, gene-id convention, whether the
matrix holds raw counts, the number of clusters and the annotation column) are
declared in ``configs/datasets.json``, so a single command reproduces the whole
benchmark table.

Example
-------
python scripts/run_clustering_benchmark.py --data-dir /path/to/datasets
python scripts/run_clustering_benchmark.py --data-dir /path/to/datasets --datasets E9.5 ST8059048
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory holding the released .h5ad files.")
    parser.add_argument("--registry", type=Path, default=REPO / "configs/datasets.json")
    parser.add_argument("--checkpoint", type=Path,
                        default=REPO / "weights/spacompass_pretrained.pt")
    parser.add_argument("--config", type=Path, default=REPO / "configs/spacompass.json")
    parser.add_argument("--output-dir", type=Path, default=REPO / "results/clustering")
    parser.add_argument("--method", choices=("mclust", "leiden", "louvain"), default="mclust")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Optional subset of dataset names (default: all).")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, entry: dict, output_dir: Path) -> list[str]:
    command = [
        sys.executable, str(HERE / "clustering_test.py"),
        "--input-h5ad", str(args.data_dir / entry["file"]),
        "--output-dir", str(output_dir),
        "--name", entry["name"],
        "--annotation-key", entry["annotation_key"],
        "--n-clusters", str(entry["n_clusters"]),
        "--method", args.method,
        "--checkpoint", str(args.checkpoint),
        "--config", str(args.config),
        "--species", entry["species"],
        "--platform", entry["platform"],
        "--device", args.device,
        "--point-size", str(entry.get("point_size", 20.0)),
    ]
    if entry.get("genes_are_ensembl"):
        command.append("--genes-are-ensembl")
    if entry.get("normalise"):
        command.append("--normalise")
    if entry.get("subset_column"):
        command += ["--subset-column", entry["subset_column"],
                    "--subset-value", entry["subset_value"]]
    if entry.get("spatial_columns"):
        command += ["--spatial-columns", *entry["spatial_columns"]]
    if args.skip_plots:
        command.append("--skip-plot")
    return command


def main() -> None:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    entries = registry["clustering_benchmark"]
    if args.datasets:
        wanted = set(args.datasets)
        entries = [entry for entry in entries if entry["name"] in wanted]
        missing = wanted - {entry["name"] for entry in entries}
        if missing:
            raise SystemExit(f"Unknown dataset name(s): {sorted(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        dataset_dir = args.output_dir / entry["name"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(args, entry, dataset_dir)
        print(f"[spacompass] {entry['name']}: running", flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit(f"{entry['name']}: clustering_test.py failed")
        payload = json.loads((dataset_dir / f"{entry['name']}_spacompass_clustering.json").read_text())
        rows.append({
            "dataset": entry["name"],
            "ARI": payload["metrics"]["ARI"],
            "NMI": payload["metrics"]["NMI"],
            "reference_ARI": entry.get("reference_ari"),
            "reference_NMI": entry.get("reference_nmi"),
        })
        print(f"[spacompass] {entry['name']}: ARI={rows[-1]['ARI']:.4f} NMI={rows[-1]['NMI']:.4f}",
              flush=True)

    summary = args.output_dir / "clustering_benchmark_summary.csv"
    header = "dataset,ARI,NMI,reference_ARI,reference_NMI\n"
    with summary.open("w", encoding="utf-8") as handle:
        handle.write(header)
        for row in rows:
            handle.write("{dataset},{ARI:.6f},{NMI:.6f},{reference_ARI},{reference_NMI}\n".format(**row))
    mean_ari = sum(row["ARI"] for row in rows) / len(rows)
    mean_nmi = sum(row["NMI"] for row in rows) / len(rows)
    print(f"\nmean ARI={mean_ari:.4f}  mean NMI={mean_nmi:.4f}  ({len(rows)} datasets)")
    print(f"summary table: {summary}")


if __name__ == "__main__":
    main()
