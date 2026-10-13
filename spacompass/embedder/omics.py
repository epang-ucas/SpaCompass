import torch
from torch import nn
import torch.nn.functional as F
from ..utils.pe import select_pe_encoder
from ..utils import create_norm, create_activation
import numpy as np
from ..utils.sparse import sparse_normalize, sparse_tpm
from ..model.gene_model import GeneModel

import json

class OmicsEmbeddingLayer(nn.Module):
    def __init__(self, n_gene, d_model, n_gene_layers, n_gene_head, norm_scheme="post", dropout=0.2, 
                 pe_type=None, cat_pe=True, use_flashAttn=False, scFound=False, freeze_scFound=False, scFound_path=None,vocab_map=None):   
        super().__init__()

        self.pe_type = pe_type
        self.cat_pe = cat_pe
        self.act = nn.ReLU() 
        self.norm0 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.extra_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)
        if pe_type is not None:
            if cat_pe:
                num_emb = d_model // 2
            else:
                num_emb = d_model
            self.pe_enc = select_pe_encoder(pe_type)(num_emb)
        else:
            self.pe_enc = None
            num_emb = d_model

        

    def forward(self, x, mask_ratio=0., no_pe=False):

        x = self.extra_linear(x)
        if self.pe_enc is not None:
            pe_input = x_dict['coord']

  
        x_dict['h'] = x
        return x_dict
    
    def forward_with_emb(self, cell_embs, coords, no_pe=False):
        # cell_embs: [n_cells, n_emb]
        x = self.extra_linear(cell_embs)
        if no_pe:
            pe = 0
        else:
            pe = self.pe_enc(coords) # position embedding
        if self.cat_pe:
            x = torch.cat([x, pe], 1)
        else:
            x = x + pe
        # x = self.extra_linear(x)
        return x
    
    def apply_cell_mask(self, x_dict, mask_ratio=0.4):
        cell_embeds = x_dict['cell_emb']
        B, D = cell_embeds.shape
        mask = torch.rand(B) < mask_ratio
        masked_input = cell_embeds.clone()
        masked_input[mask] = 0
        x_dict['masked_cell_emb'] = masked_input
        x_dict['cell_mask'] = mask
        return x_dict
