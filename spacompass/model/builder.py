"""Build and load SpaCompass models without the fairseq training stack.

The pretraining code registers its architectures with fairseq, which is only
needed for *training*.  Inference, embedding extraction and evaluation only
need the parameter layout, so this module reconstructs the same modules from
the released configuration file and loads the released checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
from torch import nn

from spacompass.model import OmicsFormer
from spacompass.model.gene_model import GeneModel

STAGE1_PREFIXES = ("first_stage_model.", "model.")


def build_gene_model(config: Mapping[str, Any]) -> GeneModel:
    """Stage 1: within-cell / within-spot gene attention encoder."""
    return GeneModel(
        config["vocab_len"],
        config["d_model"],
        config["n_gene_head"],
        config["d_model"],
        config["n_gene_layers"],
        config["model_dropout"],
        0,
        True,
        True,
        norm_scheme=config["norm_scheme"],
        use_fast_transformer=config["use_flashAttn"],
        use_batch_labels=config["use_batch_labels"],
        num_batch_labels=config["num_batches"],
        use_moe=config["use_moe"],
        num_experts=config["num_batches"],
        do_cls=False,
        n_cls=0,
        nlayers_cls=0,
    )


def build_cell_model(config: Mapping[str, Any]) -> OmicsFormer:
    """Stage 2: niche-level cell attention encoder."""
    return OmicsFormer(
        config["d_model"],
        config["n_cell_layers"],
        config["n_cell_head"],
        config["norm_scheme"],
        config["model_dropout"],
        config["use_flashAttn"],
        config["use_batch_labels"],
        config["num_batches"],
        n_gat_layers=config.get("n_gat_layers", 0),
        n_gat_head=config.get("n_gat_head", 0),
        n_bias_layers=config.get("n_bias_layers", 0),
        n_bias_head=config.get("n_bias_head", 0),
    )


def read_config(conf_path: str | Path) -> Dict[str, Any]:
    with open(conf_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, torch.Tensor]:
    """Return the raw ``model`` state dict stored in a SpaCompass checkpoint."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(blob, dict) and "model" in blob:
        return blob["model"]
    raise ValueError(f"{path} does not look like a SpaCompass checkpoint (no 'model' entry)")


def extract_stage1_state_dict(state_dict: Mapping[str, torch.Tensor]):
    """Split a stage-1 or two-stage state dict into stage-1 parameters."""
    for prefix in STAGE1_PREFIXES:
        stage1 = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if stage1:
            return stage1, prefix.rstrip(".")
    raise ValueError(
        "Checkpoint contains neither 'model.*' (stage-1) nor "
        "'first_stage_model.*' (two-stage) parameters"
    )


def load_gene_model(config: Mapping[str, Any], checkpoint: str | Path):
    """Build the stage-1 encoder and load the released pretrained weights."""
    model = build_gene_model(config)
    model.gene_mask_ratio = 0
    model.cell_mask_ratio = 0
    model.freeze_first_attn = bool(config.get("freeze_first_stage", False))
    state_dict = load_checkpoint(checkpoint)
    stage1_state, source = extract_stage1_state_dict(state_dict)
    model.load_state_dict(stage1_state, strict=True)
    model.eval()
    return model, source

