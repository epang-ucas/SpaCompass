"""Zero-shot inference helpers shared by the released scripts."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch

from spacompass.data import SinglecellDataset


def build_dataset(
    adata,
    *,
    species: str,
    platform: str,
    n_token: int = 1200,
    data_is_raw: bool = True,
    is_symbol: bool = True,
    mean_file: Optional[str] = None,
    vocab_path: Optional[str] = None,
) -> SinglecellDataset:
    """Tokenise an AnnData object exactly as in the released protocol."""
    return SinglecellDataset(
        adata,
        data_is_raw=data_is_raw,
        is_symbol=is_symbol,
        species=species,
        batch=platform,
        n_token=n_token,
        shuffle=False,
        vocab_path=vocab_path,
        mean_file=mean_file,
    )


def _move_batch(batch: Mapping[str, Any], device: torch.device, pad_id: int, freeze: bool) -> dict:
    moved = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    moved["src_key_padding_mask"] = moved["gene_list"].eq(pad_id)
    moved["freeze_first_attn"] = freeze
    return moved


def _infer_once(dataset: SinglecellDataset, model, batch_size: int, device: torch.device) -> np.ndarray:
    chunks: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, dataset.data_len, batch_size):
            samples = [
                dataset[index]
                for index in range(start, min(start + batch_size, dataset.data_len))
            ]
            batch = _move_batch(
                dataset.collater(samples), device, dataset.pad_id, model.freeze_first_attn
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                embeddings = model(batch)["cell_emb"]
            chunks.append(embeddings.float().cpu().numpy())
            del samples, batch, embeddings
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def encode_dataset(
    dataset: SinglecellDataset,
    model,
    *,
    batch_size: int = 64,
    min_batch_size: int = 1,
    device: torch.device | str = "cuda",
    l2_normalise: bool = True,
) -> tuple[np.ndarray, int]:
    """Return ``(embeddings, batch_size_used)`` for a tokenised dataset."""
    device = torch.device(device)
    model = model.to(device).eval()
    current = min(batch_size, dataset.data_len)
    while True:
        try:
            print(f"Inference batch size: {current}", flush=True)
            embeddings = _infer_once(dataset, model, current, device)
            break
        except torch.OutOfMemoryError:
            if current <= min_batch_size:
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current = max(min_batch_size, current // 2)
            print(f"CUDA OOM; retrying with batch size {current}", flush=True)
    if l2_normalise:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms != 0)
    return embeddings, current
