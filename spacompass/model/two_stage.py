"""Two-stage SpaCompass model: gene attention followed by cell attention."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from spacompass.model.builder import build_cell_model, build_gene_model, load_checkpoint


class TwoStageModel(nn.Module):
    """Stage 1 encodes every cell in a patch; stage 2 models the patch."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        self.config = dict(config)
        self.d_model = config["d_model"]
        self.pad_id = 0
        self.use_batch_label = bool(config["use_batch_labels"])
        self.gene_mask_ratio = 0
        self.cell_mask_ratio = 0
        self.freeze_first_attn = bool(config.get("freeze_first_stage", False))
        self.first_stage_model = build_gene_model(config)
        self.second_stage_model = build_cell_model(config)

    def forward(self, batch: Mapping[str, Any]) -> dict:
        batch["src_key_padding_mask"] = batch["gene_list"].eq(self.pad_id)
        bsz, n_cells, n_genes = batch["x_seq"].shape
        stage_one_input = {
            "gene_list": batch["gene_list"].reshape(bsz * n_cells, n_genes),
            "src_key_padding_mask": batch["src_key_padding_mask"].reshape(bsz * n_cells, n_genes),
            "x_seq": batch["x_seq"].reshape(bsz * n_cells, n_genes),
            "batch_id": (
                batch["batch_id"].unsqueeze(1).repeat(1, n_cells).reshape(-1)
                if self.use_batch_label else None
            ),
            "freeze_first_attn": self.freeze_first_attn,
        }
        out_dict = self.first_stage_model(stage_one_input)
        batch["cell_emb"] = out_dict["cell_emb"].reshape(bsz, n_cells, -1)
        batch["gene_vocab_emb"] = out_dict["gene_vocab_emb"].reshape(bsz, n_cells, n_genes, -1)
        return self.second_stage_model(batch)


def load_two_stage(config: Mapping[str, Any], checkpoint: str | Path) -> TwoStageModel:
    """Build the two-stage model and load the released checkpoint."""
    model = TwoStageModel(config)
    state_dict = load_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Incomplete checkpoint load: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model
