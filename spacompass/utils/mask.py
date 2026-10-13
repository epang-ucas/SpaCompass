import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from scipy.stats import expon
from scipy.sparse import csr_matrix
from .sparse import simple_mask
import torch.distributions as td
import math


def drop_nodes(x_dict, drop_node_rate=0., max_batch_size=2000, inplace=True):
    if inplace == False:
        raise NotImplementedError('Only support inplace drop nodes')

    if drop_node_rate > 0:  # cell_idx is the index of the nodes that are not dropped
        cell_idx = torch.randperm(x_dict['x_seq'].shape[0], device=x_dict['x_seq'].device)[
                   :min(max_batch_size, int(x_dict['x_seq'].shape[0] * (1 - drop_node_rate)))]
        x_dict['x_seq'] = x_dict['x_seq'].index_select(0, cell_idx)
        if 'batch' in x_dict:
            x_dict['batch'] = x_dict['batch'][cell_idx]
        if 'h' in x_dict:
            x_dict['h'] = x_dict['h'][cell_idx]
        if 'g' in x_dict:
            x_dict['g'] = x_dict['g'][cell_idx][:, cell_idx]
        if 'coord' in x_dict:
            x_dict['coord'] = x_dict['coord'][cell_idx]
        if 'label' in x_dict:
            x_dict['label'] = x_dict['label'][cell_idx]
        if 'lib_size' in x_dict:
            x_dict['lib_size'] = x_dict['lib_size'][cell_idx]
        if 'x_masked_seq' in x_dict:
            x_dict['x_masked_seq'] = x_dict['x_masked_seq'].index_select(0, cell_idx)
        if 'dataset' in x_dict:
            x_dict['dataset'] = x_dict['dataset'][cell_idx]
        if 'loss_mask' in x_dict:
            x_dict['loss_mask'] = x_dict['loss_mask'][cell_idx]

class NullMaskBuilder(nn.Module):
    def __init__(self, drop_node_rate, max_batch_size=2000):
        super().__init__()
        self._drop_node_rate = drop_node_rate
        self._max_batch_size = max_batch_size

    def apply_mask(self, x_dict):
        if self._drop_node_rate > 0 and self.training:
            drop_nodes(x_dict, self._drop_node_rate, self._max_batch_size)
        # x_dict['mask'] = torch.arange(x_dict['h'].shape[0], device=x_dict['h'].device)
        x_dict['input_mask'] = torch.ones(*x_dict['x_seq'].shape, device=x_dict['x_seq'].device).int()
        return x_dict

class MaskBuilder(nn.Module):
    def __init__(self, mask_node_rate, mask_feature_rate, drop_node_rate=0, max_batch_size=2000, edge_mask=None, mask_beta=False):
        super().__init__()
        self._mask_node_rate = mask_node_rate
        self._mask_feature_rate = mask_feature_rate
        self._edge_mask = edge_mask
        self._drop_node_rate = drop_node_rate
        self._max_batch_size = max_batch_size
        if self._mask_node_rate > 0 and self._mask_feature_rate and mask_beta:
            alpha = 5
            beta = 4 / self._mask_feature_rate + 2 - alpha
            self.beta_dist = td.Beta(alpha, beta)

        self.mask_beta = mask_beta

    def update_mask_ratio(self, mask_node_rate, mask_feature_rate):
        self._mask_node_rate = mask_node_rate
        self._mask_feature_rate = mask_feature_rate

    # This function mask parts of the nodes, and only the masked nodes will be used in the loss function
    def apply_mask(self, x_dict):
        # if self.training and self._drop_node_rate > 0:
        #     drop_nodes(x_dict, self._drop_node_rate, self._max_batch_size)
        import pdb;pdb.set_trace()

        if self.training and self._mask_node_rate > 0:
            if 'x_masked_seq' in x_dict:
                x = x_dict['x_masked_seq']
            else:
                x = x_dict['x_seq']

            if self.mask_beta:
                mask_ratio = self.beta_dist.sample((x.shape[0],)).to(x.device)
                mask_ratio[mask_ratio > 0.9] = 0.9
                num_nodes = x.shape[0]
                perm = np.random.permutation(num_nodes)
                num_mask_nodes = int(self._mask_node_rate * num_nodes)
                keep_nodes = perm[num_mask_nodes:]
                mask = torch.rand(*x.shape, device=x.device) <= mask_ratio.unsqueeze(-1)
                mask[keep_nodes] = False
            else:
                num_nodes = x.shape[0]
                perm = np.random.permutation(num_nodes)
                num_mask_nodes = int(self._mask_node_rate * num_nodes)
                keep_nodes = perm[num_mask_nodes:]  # keep_nodes is the index of the nodes that are not masked
                mask = torch.rand(*x.shape, device=x.device) <= self._mask_feature_rate
                mask[keep_nodes] = False

            # x = x.coalesce()
            # masked_x_seq = simple_mask(x, mask)
            masked_x_seq = x * (~mask)  # 用 0 替换被 mask 的元素
            x_dict['masked_x_seq'] = masked_x_seq
            x_dict['input_mask'] = mask.int()        
        else:
            x_dict['input_mask'] = torch.ones(*x_dict['x_seq'].shape, device=x_dict['x_seq'].device).int()
        return x_dict
    

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
    
    def apply_cell_mask(self, x_dict, mask_ratio=0.4):
        cell_embeds = x_dict['cell_emb']
        B, D = cell_embeds.shape
        mask = torch.rand(B) < mask_ratio
        masked_input = cell_embeds.clone()
        masked_input[mask] = 0
        x_dict['masked_cell_emb'] = masked_input
        x_dict['cell_mask'] = mask
        return x_dict


class HiddenMaskBuilder(nn.Module):
    def __init__(self, mask_node_rate, mask_countsure_rate, drop_node_rate=0, max_batch_size=2000, edge_mask=None):
        super().__init__()
        self._mask_node_rate = mask_node_rate
        self._mask_countsure_rate = mask_countsure_rate
        self._edge_mask = edge_mask
        self._drop_node_rate = drop_node_rate
        self._max_batch_size = max_batch_size

    def update_mask_ratio(self, mask_node_rate, mask_feature_rate):
        self._mask_node_rate = mask_node_rate
        self._mask_feature_rate = mask_feature_rate

    # This function mask parts of the nodes, and only the masked nodes will be used in the loss function
    def apply_mask(self, x_dict):
        if self._drop_node_rate > 0 and self.training:
            drop_nodes(x_dict, self._drop_node_rate, self._max_batch_size)
        if self._mask_node_rate > 0 and self.training:
            num_nodes = x_dict['h'].shape[0]
            perm = np.random.permutation(num_nodes)
            num_mask_nodes = int(self._mask_node_rate * num_nodes)
            keep_nodes = perm[num_mask_nodes:] # keep_nodes is the index of the nodes that are not masked

            out_x = F.dropout(x_dict['h'], p=self._mask_countsure_rate) # mask the countsures of all nodes
            out_x[keep_nodes] = x_dict['h'][keep_nodes] # keep the countsures of the nodes that are not masked
            # x_dict['h'] = out_x
            x_dict['input_mask'] = torch.zeros(x_dict['h'].shape[0], device=x_dict['h'].device).unsqueeze(-1)
            x_dict['input_mask'][perm[: num_mask_nodes]] = 1.
        else:
            x_dict['input_mask'] = torch.ones(x_dict['h'].shape[0], device=x_dict['h'].device).unsqueeze(-1)
        return x_dict


class InputDropoutMaskBuilder(nn.Module):
    def __init__(self, input_drop_type="mar", valid_drop_rate=0.1, test_drop_rate=0.1, seed=10,
                 min_gene_counts=5):
        super().__init__()
        assert 0 <= valid_drop_rate < 1, "valid_drop_rate should be in [0, 1)"
        assert 0 < test_drop_rate < 1, "test_drop_rate should be in (0, 1)"
        assert 0 < valid_drop_rate + test_drop_rate < 1, "Total masking rate should be in (0, 1)"
        self._input_drop_type = input_drop_type
        self._valid_drop_rate = valid_drop_rate
        self._test_drop_rate = test_drop_rate
        self._min_gene_counts = min_gene_counts
        self._seed = seed
        if input_drop_type == "mcar":
            self.distr = "uniform"
        elif input_drop_type == "mar":
            self.distr = "exp"
        else:
            raise NotImplementedError(f"Expect mask_type in ['mar', 'mcar'], but found {self.mask_type}")

    def _get_probs(self, vec):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(self.distr)
    
    def apply_mask(self, x_seq):
        counts = x_seq.to_dense()
        train_mask = np.ones(counts.shape, dtype=bool)
        valid_mask = np.zeros(counts.shape, dtype=bool)
        test_mask = np.zeros(counts.shape, dtype=bool)
        rng = np.random.default_rng(self._seed)

        for c in range(counts.shape[0]):
            # Retrieve indices of positive values
            ind_pos = torch.nonzero(counts[c], as_tuple=True)[0]
            cells_c_pos = counts[c, ind_pos]

            # Get masking probability of each value
            if len(cells_c_pos) > self._min_gene_counts:
                mask_prob = self._get_probs(cells_c_pos)
                mask_prob = mask_prob / sum(mask_prob)
                n_test = int(np.floor(len(cells_c_pos) * self._test_drop_rate))
                n_valid = int(np.floor(len(cells_c_pos) * self._valid_drop_rate))
                if n_test + n_valid >= len(cells_c_pos):
                    print(f"Too many genes masked for cell {c} ({n_test + n_valid}/{len(cells_c_pos)})")
                    n_test -= 1
                    n_valid -= 1

                idx_mask = np.ones(len(ind_pos), dtype=bool)
                test_idx = rng.choice(np.arange(len(ind_pos)), n_test, p=mask_prob, replace=False)
                train_mask[c, ind_pos[test_idx]] = False
                test_mask[c, ind_pos[test_idx]] = True
                if self._valid_drop_rate > 0:
                    idx_mask[test_idx] = False
                    masked_mask_prob = mask_prob[idx_mask] / sum(mask_prob[idx_mask])
                    valid_idx = rng.choice(np.arange(len(ind_pos))[idx_mask], n_valid, p=masked_mask_prob, replace=False)
                    train_mask[c, ind_pos[valid_idx]] = False
                    valid_mask[c, ind_pos[valid_idx]] = True

        return train_mask, valid_mask, test_mask
