import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Mapping, Optional, Tuple, Any, Union


from ..encoder import setup_encoder
from ..latent import LatentModel, PreLatentNorm
from ..latent.adversarial import AdversarialLatentLayer
from ..head import setup_head
from .gene_model import MVCDecoder, FlashTransformerEncoderLayer, FlashTransformerEncoder, BatchLabelEncoder
from .graph_attention import GraphAttentionEncoder
from .bias_attention import BiasTransformer

class OmicsFormer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_cell_layers: int,
        n_cell_head: int, 
        norm_scheme: str = 'post', 
        model_dropout: float = 0.2, 
        use_flashAttn: bool = True,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        mode: str = 'train',
        activation:str = 'gelu', 
        n_gat_layers: int = 0,
        n_gat_head: int = 0,
        n_bias_layers: int = 0,
        n_bias_head: int = 0,
    ):
        super(OmicsFormer, self).__init__()
        self.use_batch_labels = use_batch_labels
        self.mode = mode
        self.use_gat = n_gat_layers > 0
        self.use_bias_attn = n_bias_layers > 0

        self.extra_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(model_dropout),
        )
        if n_gat_layers > 0:
            self.gat_encoder = GraphAttentionEncoder(d_model, n_gat_layers, n_gat_head, model_dropout)
            
        if n_bias_layers > 0:
            self.bias_transformer = BiasTransformer(d_model, n_bias_layers, n_bias_head, n_bias_layers, model_dropout)

        if mode == "flowformer":
            self.cell_encoder = setup_encoder("flowformer", 512, 4, model_dropout, activation, "groupnorm", 8)
            self.pre_latent_norm = PreLatentNorm('ln', 512)
        else:
            if use_flashAttn:
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    n_cell_head,
                    model_dropout,
                    norm_scheme=norm_scheme,
                )
                self.cell_encoder = FlashTransformerEncoder(encoder_layers, n_cell_layers)
            else:
                encoder_layers = nn.TransformerEncoderLayer(
                    d_model, n_cell_head, d_model, model_dropout, batch_first=True
                )
                self.cell_encoder = nn.TransformerEncoder(encoder_layers, n_cell_layers, enable_nested_tensor=False)

        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        self.decoder = MVCDecoder(    
                d_model,
                # arch_style=mvc_decoder_style,
                arch_style="inner product, detach",
                explicit_zero_prob=False,
                use_batch_labels=use_batch_labels,
                )

    def forward(self, x_dict, no_pe=False):
        
        cell_emb = self._encode(x_dict)
        x_dict['spot_emb'] = cell_emb # [bsz, ncells, d_model]
        bsz, n_cells, d_emb = x_dict['spot_emb'].shape
        
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(x_dict['batch_id'])  # (batch, embsize)
            batch_emb = batch_emb.unsqueeze(1).repeat(1,  n_cells, 1).reshape((bsz* n_cells, d_emb))

       
        spot_embedding = x_dict['spot_emb'].reshape((bsz* n_cells, d_emb))
        gene_embedding = x_dict['gene_vocab_emb'].reshape((bsz* n_cells, -1, d_emb))
        # import pdb;pdb.set_trace()
        mvc_output = self.decoder(
                spot_embedding 
                if not self.use_batch_labels
                else torch.cat([spot_embedding, batch_emb], dim=-1),
                gene_embedding,
            )

        x_dict['spot_output'] = mvc_output["pred"].reshape((bsz, n_cells, -1))
        return x_dict
        
    def nondisc_parameters(self):
        other_params = []
        for pname, p in self.named_parameters():
            if 'discriminator' not in pname:
                other_params += [p]
            else:
                print(pname)
        return other_params
    
    def apply_gene_mask(self, x_dict, mask_ratio=0.4, mask_value=0):

        values = x_dict.get('x_masked_seq', x_dict['x_seq'])  # [B, G]
        masked_values = values.clone()
        B, G = values.shape

        # 忽略CLS (第0列)，只对[:, 1:]操作
        gene_values = values[:, 1:]  # [B, G-1]

        # 找出非0表达的基因（即可以被遮蔽的位置）
        nonzero_mask = gene_values != 0  # [B, G-1]

        # 为每个细胞（行）计算要遮蔽的基因数量
        num_mask_per_row = (nonzero_mask.sum(dim=1).float() * mask_ratio).long()
        num_mask_per_row = num_mask_per_row.clamp(min=1)  # 保证每个细胞至少遮一个

        # 为每行生成随机分数用于采样遮蔽位置
        rand_scores = torch.rand_like(gene_values, dtype=torch.float)
        rand_scores[~nonzero_mask] = 2.0  # 非表达基因不参与遮蔽（给个高值，排除）

        # 对每行按分数升序排序，选出最小的前 num_mask 个
        sorted_indices = rand_scores.argsort(dim=1)  # [B, G-1]

        # 构造遮蔽位置的 mask
        mask = torch.zeros_like(gene_values, dtype=torch.bool)  # [B, G-1]
        for i in range(B):
            mask[i, sorted_indices[i, :num_mask_per_row[i]]] = True

        # 在原矩阵上应用遮蔽：注意保留CLS不动
        full_mask = torch.zeros_like(values, dtype=torch.bool)  # [B, G]
        full_mask[:, 1:] = mask  # CLS (第0列) 不遮蔽
        masked_values[full_mask] = mask_value

        x_dict['masked_x_seq'] = masked_values
        x_dict['gene_mask'] = full_mask.int()  # 可以用于 loss 计算或 attention mask
        return x_dict
    
    def _encode(self, x_dict):
        cell_emb = x_dict.get("masked_cell_emb", x_dict["cell_emb"]) # [bsz, ncells, d_model]
        adj = x_dict.get("masked_adj", x_dict["adj"])
        if self.use_gat:
            assert 'adj' in x_dict
            cell_emb =  self.gat_encoder(cell_emb, adj)
        if self.use_bias_attn:
            assert 'coords' in x_dict
            cell_emb =  self.bias_transformer(cell_emb, x_dict['coords'])
        if self.mode == "flowformer":
            x_dict['h'] = self.cell_encoder(x_dict)['hidden']
            x_dict['h'] = self.pre_latent_norm(x_dict)
        else:
            cell_emb = self.cell_encoder(cell_emb) # (bsz, n_cells, -1)

        return cell_emb
    
 
class SpotDecoder(nn.Module):
    def __init__(
        self,
        d_model: int, 
        use_batch_labels: bool = False,
        num_batchs: int = 0,
        dropout: float=0.2,
    ):
        super().__init__()
        self.use_batch_labels = use_batch_labels
        if use_batch_labels:
            assert num_batchs != 0
            self.batch_encoder = nn.Sequential(
                nn.Embedding(num_batchs, d_model),
                nn.LayerNorm(d_model),
            )
            d_in = d_model * 2
        else:
            d_in = d_model 
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout), 
            nn.Linear(d_model, d_model),
        )
        self.mvc_decoder = MVCDecoder(d_model, arch_style="inner product")
        # self.mvc_decoder = MVCDecoder(d_model, arch_style="inner product, detach")
        

    def forward(self, x, gene_embs, batch_labels=None):
        """x is the output of the cell , (num_cells, d_model)"""
        if self.use_batch_labels:
            if batch_labels is not None:
                # import pdb;pdb.set_trace()
                batch_embs = self.batch_encoder(batch_labels) # (num_cells, d_model)
            else:
                embedding_layer = self.batch_encoder[0]  # nn.Embedding(num_batches, d_model)
                all_batch_ids = torch.arange(embedding_layer.num_embeddings, device=embedding_layer.weight.device)  # [num_batches]
                all_embeddings = embedding_layer(all_batch_ids)
                mean_embedding = all_embeddings.mean(dim=0, keepdim=True)  # [1, d_model]
                layernorm = self.batch_encoder[1]
                # import pdb;pdb.set_trace()
                batch_embs = layernorm(mean_embedding).expand(x.shape[0], -1)  # [1, d_model]
            cell_embs = self.fc(torch.cat([x, batch_embs], dim=1))  # (num_cells, d_model)
        else:
            cell_embs = self.fc(x)
        spot_output = self.mvc_decoder(cell_embs, gene_embs)
        spot_preds = spot_output['pred']
        return spot_preds
    
