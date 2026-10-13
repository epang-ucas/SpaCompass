import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from torch.distributions import Bernoulli
from torch.utils.checkpoint import checkpoint_sequential
from tqdm import trange

import numpy as np
import copy
from typing import Dict, Mapping, Optional, Tuple, Any, Union

from .moe import MoELayer


def _load_flash_mha():
    """Import flash-attn lazily: it is only needed when use_flashAttn is enabled."""
    try:
        from flash_attn.modules.mha import MHA
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            'use_flashAttn requires the "flash-attn" package. Install it, or set '
            '"use_flashAttn": false in the configuration to use the standard '
            "PyTorch attention path."
        ) from error
    return MHA

# from .gene_gnn import GeneConvLayer

# from flash_attn.flash_attention import FlashMHA
# from .dsbn import DomainSpecificBatchNorm1d
# from .grad_reverse import grad_reverse

class GeneModel(nn.Module):

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        # nlayers_cls: int = 3,
        # n_cls: Optional[int] = None,
        dropout: float = 0.2,
        padding_idx: int = 0,
        # pad_token: str = "<pad>",
        # pad_value: int = 0,
        do_mvg: bool = False,
        do_mvc: bool = False,
        # do_cce: bool = False,
        # do_ecs: bool = False,
        # do_dab: bool = False,
        do_cls: bool = False,
        n_cls: int = 0,
        nlayers_cls: int = 3,
        do_sample: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        domain_spec_batchnorm: Union[bool, str] = False,
        input_emb_style: str = "continuous",
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        norm_scheme: str = "post",
        
        use_moe: bool = False,
        num_experts: int = 5,

        use_gnn: bool = False,
        use_embed: bool = False,
        nlayers_gnn: int = 3,
        n_message: int = 3,
        n_edge_layers: int = 3,

        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
    ):
        super().__init__()

        self.d_model = d_model
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = norm_scheme

        self.do_mvg = do_mvg
        self.do_mvc = do_mvc
        # self.do_cce = do_cce
        # self.do_ecs = do_ecs
        # self.do_dab = do_dab
        self.do_cls = do_cls
        self.do_sample = do_sample
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.use_gnn = use_gnn
        self.use_embed = use_embed
        self.dtype = torch.float
        self.use_moe_dec = use_moe
        self.num_experts = num_experts

        ## Encoder
        # Gene Tokens Encoder
        self.gene_encoder = GeneEncoder(ntoken, d_model, padding_idx=padding_idx)  # tmp padding_idx=vocab[pad_token]
        # Gene Expression Encoder
        self.value_encoder = ValueEncoder(d_model, dropout)
        # Batch Id Encoder

        self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        # Transformer Encoder
        if use_fast_transformer:
            encoder_layers = FlashTransformerEncoderLayer(
                d_model,
                nhead,
                dropout,
                norm_scheme=self.norm_scheme,
            )
            self.transformer_encoder = FlashTransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = nn.TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers, enable_nested_tensor=False)
        
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        ## Decoder
        # Gene Expression Predicition 
        if self.do_mvg:
            if self.use_moe_dec:
                self.decoder = MoeDecoder(
                    d_model,
                    num_experts=num_experts,
                    use_batch_labels=use_batch_labels,
                )
            else:
                self.decoder = ExprDecoder(
                    d_model,
                    explicit_zero_prob=explicit_zero_prob,
                    use_batch_labels=use_batch_labels,
                )
        # Genetype Classification 
        if self.do_cls:
            self.cls_decoder = ClsDecoder(
                d_model, 
                n_cls, 
                nlayers=nlayers_cls, 
                # use_batch_labels=use_batch_labels
            )
        # Gene Expression Predicition with Cell Embedding 
        if self.do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                # arch_style="inner product, detach",
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )
        # Domain Adaptation with Reverse Back-propagation 
        # if do_dab:
        #     self.grad_reverse_discriminator = AdversarialDiscriminator(
        #         d_model,
        #         n_cls=num_batch_labels,
        #         reverse_grad=True,
        #     )
        # # Elastic Cell Similarity (ECS)
        # if self.do_cce:
        #     self.sim = Similarity(temp=0.5)
        #     self.creterion_cce = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.gene_encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        # embed: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        # batch_labels: Optional[Tensor] = None,  # (batch,)
        # edge_index: Tensor = None,
        # edge_attr: Tensor = None,
        freeze_first_attn: bool = False,
    ) -> Tensor:
        # self._check_batch_labels(batch_labels)
        # import pdb;pdb.set_trace()
        
        src = self.gene_encoder(src)  # (batch, seq_len, embsize)
        
        self.gene_embedding = src
        values = self.value_encoder(values)  # (batch, seq_len, embsize)
        total_embs = src + values
        total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)
        if freeze_first_attn:
            layers = self.transformer_encoder.layers
            # y = checkpoint_sequential(layers, 4, y)
            output = checkpoint_sequential(
                [lambda x, layer=layer: layer(x, src_key_padding_mask=src_key_padding_mask) for layer in layers],
                8,
                total_embs
            )
        else:
            output = self.transformer_encoder(
                total_embs, src_key_padding_mask=src_key_padding_mask
            )
        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output[:, 1:, :], dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        # elif batch_labels is not None:
        #     raise ValueError(
        #         "batch_labels should only be provided when `self.use_batch_labels`"
        #         " or `self.domain_spec_batchnorm` is True"
        #     )

    def forward(self, x_dict) -> Mapping[str, Tensor]:
        # import pdb;pdb.set_trace()
        if 'masked_x_seq' in x_dict and x_dict['masked_x_seq'] is not None:
            x = x_dict['masked_x_seq']
        else:
            x = x_dict['x_seq']
        gene_list = x_dict['gene_list']
        src_key_padding_mask = x_dict.get('src_key_padding_mask', None) 
        batch_labels = x_dict.get('batch_id', None)

        transformer_output = self._encode(gene_list, x, src_key_padding_mask, x_dict['freeze_first_attn'])     
        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        if self.do_mvg:
            # import pdb;pdb.set_trace()
            mlm_output = self.decoder(
                transformer_output
                if not self.use_batch_labels
                else torch.cat([transformer_output, batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1)],dim=2,),
                # else transformer_output + batch_emb.unsqueeze(1),
            )
            x_dict["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        cell_emb = self._get_cell_emb_from_layer(transformer_output, x)
        x_dict["cell_emb"] = cell_emb
        if self.do_mvc:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.gene_embedding,
            )
            x_dict["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            x_dict["gene_vocab_emb"] = self.gene_embedding
        if self.do_cls:
            x_dict["cls_output"] = self.cls_decoder(
                cell_emb
                # if not self.use_batch_labels
                # else torch.cat([cell_emb, batch_emb], dim=1),
            )  # (batch, n_cls)
        return x_dict

   
class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2, norm_scheme="post"):
        super().__init__()
 
        MHA = _load_flash_mha()
        self.self_attn = MHA(embed_dim, num_heads, dropout=dropout, causal=False)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    def forward(self, x):
        if self.norm_scheme == "pre":
            x = x + self.dropout(self.self_attn(self.norm1(x)))
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(x + self.dropout(self.self_attn(x)))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
    
class FlashTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(encoder_layer.norm1.normalized_shape)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.norm(x)
        return x
    
class DynamicGeneEncoder(nn.Module):
    def __init__(
        self,
        num_full_vocab: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.full_embedding = nn.Embedding(
            num_full_vocab, embedding_dim, padding_idx=padding_idx
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        # x = self.embedding(x)  # (batch, seq_len, embsize)
        # x = self.norm(x)
        
        input_ids_flat = input_ids.view(-1)
        unique_ids, inverse_indices = torch.unique(input_ids_flat, return_inverse=True)

        # 子词表 embedding：从 full_embedding 中选取需要的部分
        sub_embeddings = self.full_embedding(unique_ids)  # [U, D]

        # 映射 input 到 [B, L] 的索引
        sub_input = inverse_indices.view(input_ids.shape)

        # 通过子 embedding 查值
        out = sub_embeddings[sub_input]  # [B, L, D]
        
        return self.norm(out)

class ValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(
        self, 
        d_model: int, 
        dropout: float = 0.1, 
        max_value: int = 512
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.norm(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class MoeDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.moe = MoELayer(d_in, d_model, 1, num_experts)

    def forward(self, x: Tensor, topn: int = 2) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.moe(x, topn).squeeze(-1)
        return dict(pred=pred_value)
    
class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        else:
            zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)

class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_in, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        # import pdb;pdb.set_trace()
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.

    There are actually three ways of making this, all start with gene_embs -> query_vecs,
    and then:
    1. cell_emb x W x query vecs.
       This one makes the most sense, since in the query space, the query look at
       different dimensions of cel_emb and sync them. This one has explicit interaction.
    2. FC([cell_emb, query_vecs]).
       This one has the benifit to have smaller query_vecs and makes them like bottle
       neck layer. For example 64 dims.
    3. FC(cell_emb + query_vecs).

    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            if query_vecs.ndim != 3:
                query_vecs = query_vecs.unsqueeze(0).repeat(cell_emb.shape[0], 1, 1) # (batch, gene, emb)
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)

class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
