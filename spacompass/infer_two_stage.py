"""Patch-level inference for the two-stage SpaCompass model."""

from __future__ import annotations

import numpy as np
import torch


def encode_patches(
    dataset,
    model,
    *,
    strategy: str = "first",
    device: torch.device | str = "cuda",
    l2_normalise: bool = True,
) -> dict:
    """Return per-cell stage-1 and stage-2 embeddings for a patch dataset.

    Every patch contributes the representations of its centre cells; a cell
    appears in several patches, and ``strategy`` decides which occurrence is
    kept (``first``, ``last`` or ``mean``).
    """
    if strategy not in ("first", "last", "mean"):
        raise ValueError(f"Unknown aggregation strategy: {strategy}")
    device = torch.device(device)
    model = model.to(device).eval()

    n_cells = dataset.adata.n_obs
    dim = model.d_model
    first_chunks: dict[int, list[np.ndarray]] = {}
    second_chunks: dict[int, list[np.ndarray]] = {}

    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for i_patch in range(dataset.data_len):
            batch = dataset.collater([dataset[i_patch]])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out_dict = model(batch)
            first_stage_emb = out_dict["cell_emb"].detach().float().cpu().numpy().squeeze(0)
            second_stage_emb = out_dict["spot_emb"].detach().float().cpu().numpy().squeeze(0)
            is_center = batch["is_center_cell"].squeeze(0).detach().cpu().numpy()
            original_indices = batch["original_index"].squeeze(0)
            for position, (index, is_centre) in enumerate(zip(original_indices, is_center)):
                if not is_centre:
                    continue
                first_chunks.setdefault(int(index), []).append(first_stage_emb[position])
                second_chunks.setdefault(int(index), []).append(second_stage_emb[position])

    first = np.zeros((n_cells, dim), dtype=np.float32)
    second = np.zeros((n_cells, dim), dtype=np.float32)
    for index in first_chunks:
        if strategy == "first":
            first[index] = first_chunks[index][0]
            second[index] = second_chunks[index][0]
        elif strategy == "last":
            first[index] = first_chunks[index][-1]
            second[index] = second_chunks[index][-1]
        else:
            first[index] = np.mean(first_chunks[index], axis=0)
            second[index] = np.mean(second_chunks[index], axis=0)

    covered = np.array(sorted(first_chunks.keys()), dtype=int)
    if l2_normalise:
        for matrix in (first, second):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.divide(matrix, norms, out=matrix, where=norms > 0)
    return {
        "first_stage": first,
        "second_stage": second,
        "covered_indices": covered,
        "n_obs_total": int(n_cells),
        "aggregation": strategy,
    }
