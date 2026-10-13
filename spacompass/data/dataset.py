import os, logging, pickle, json, scipy
from typing import *
import numpy as np
import pandas as pd
import anndata as ad

import torch
import torch.nn.functional as F

from spacompass.data.preprocess import Preprocessor, make_var_unique, select_token_indices
from spacompass.resources import resource_path

logger = logging.getLogger(__file__)

class SinglecellDataset:
    def __init__(
        self,
        adata,
        data_is_raw: bool = True,
        is_symbol: bool = True,
        species: str = "mouse",
        batch: str = "scRNA",
        n_token=1200,
        shuffle: bool = True,
        human_vocab_len: int = 42677,
        vocab_path: str = None,
        mean_file: str = None,
        
    ):
        self.shuffle = shuffle
        self.human_vocab_len = human_vocab_len
        
        mapping = {"scRNA":0, "visium":1, "stereo":2, "merfish":3, "xenium":4}
        self.batch_id = mapping[batch]
        if species == "human":
            self.species_id = 0
        elif species == "mouse":
            self.species_id = 1
        else:
            raise ValueError(f"Species {species} is invalid!")
        if vocab_path is not None:
            vocab1_path = vocab_path + "/symbol2ensem.json"
            vocab2_path = vocab_path + "/new_ensem2id.json"
        else:
            vocab1_path = resource_path("vocab", f"{species}_merged_gene_to_ensembl.json")
            vocab2_path = resource_path("vocab", f"{species}_vocab.json")
        with open(vocab1_path, "r", encoding="utf-8") as f:
            symbol2ensem = json.load(f)  # 解析 JSON 内容为字典
        assert all(isinstance(v, list) and len(v) == 1 for v in symbol2ensem.values())
        with open(vocab2_path, "r", encoding="utf-8") as f:
            ensem2id = json.load(f)  # 解析 JSON 内容为字典
        cls_id = ensem2id['<cls>']
        pad_id = ensem2id['<pad>']
        
        # mean norm
        if mean_file is not None:
            df = pd.read_csv(mean_file, sep="\t", index_col=0)
            df.columns = df.columns.str.lower()
            gene_mean_mapping = df[batch]
        else:
            print("No platform mean file supplied: skipping platform-mean normalisation.")

        # preprocess
        if adata.var.index.duplicated().any():
            print(f"Detect duplicated genes in adata")
            adata = make_var_unique(adata)
        # 去掉不在词表基因
        genes = adata.var_names.tolist()
        if is_symbol:
            if species == 'mouse':
                genes = [g.lower() for g in genes]
            else:
                genes = [g.upper() for g in genes]
            gene_flags = np.array([True if g in symbol2ensem else False for g in genes])
        else:
            gene_flags = np.array([True if g in ensem2id else False for g in genes])
        print(f"{gene_flags.sum()}/{len(gene_flags)} have been matched!")
        if not gene_flags.any():
            raise ValueError(
                "No gene in the input matched the SpaCompass vocabulary. Check that "
                "var_names hold gene symbols/Ensembl ids (for files whose var_names are "
                "positional indices, remap them first, e.g. "
                "`adata.var_names = adata.var['gene']`) and that --species matches the "
                "sample organism; otherwise the encoder would receive empty inputs and "
                "return identical embeddings for every spot."
            )
        adata = adata[:, gene_flags].copy()
        genes = np.array(genes)[gene_flags]
 
        gene_ensem_ids = np.array([symbol2ensem[g][0] for g in genes]) if is_symbol else genes
        gene_vocab_ids = np.array([ensem2id[g] for g in gene_ensem_ids])  
        # mean norm 
        if mean_file is not None:
            gene_mean_exp = np.array([gene_mean_mapping[g] for g in gene_ensem_ids])  
            gene_flags = gene_mean_exp!=0 
            print(f"corpus norming ...")
            print(f"{gene_flags.sum()}/{len(gene_flags)} have been matched!")
            adata = adata[:, gene_flags].copy()
            gene_ensem_ids = gene_ensem_ids[gene_flags]
            gene_vocab_ids = gene_vocab_ids[gene_flags]
            gene_mean_exp = gene_mean_exp[gene_flags]
            adata.X = adata.X / gene_mean_exp[np.newaxis, :]
        
        adata.var["vocab_ids"] = gene_vocab_ids
        # import pdb;pdb.set_trace()
        adata_hvg = self.scPipeline(
            adata, 
            filter_gene_by_counts = 3 if data_is_raw else False, 
            # filter_cell_by_counts= 50 if data_is_raw else False,
            # filter_gene_by_counts = False,
            filter_cell_by_counts = False,
            data_is_raw=data_is_raw, ishvg=data_is_raw) #, ishvg=data_is_raw
        X = adata_hvg.layers["X_binned"]
        gene_vocab_ids = adata_hvg.var["vocab_ids"].values

        dataset = []
        for id,data in enumerate(X):
            id_selected = select_token_indices(data, n_token) # num_gene <= n_tokens
            gene_ids = gene_vocab_ids[id_selected]
            values = data[id_selected]
            # 增加<cls>
            gene_ids = np.append(cls_id, gene_ids) # num_gene <= n_tokens+1
            values = np.append(0, values)
            # pad
            if len(gene_ids) > n_token+1:
                raise ValueError(f"Impossible!")
            elif len(gene_ids) < n_token+1:
                pad_len = n_token + 1 - len(gene_ids)        
                gene_ids = np.concatenate([gene_ids, np.full(pad_len, pad_id, dtype=gene_ids.dtype)])
                values = np.concatenate([values, np.zeros(pad_len, dtype=values.dtype)])
            feature_dict = {
                'gene_list': gene_ids, 
                'values': values,
                # 'coord': coords[id],
            }
            dataset.append(feature_dict)
        self.dataset = dataset
        self.data_len = len(dataset)
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.adata = adata_hvg
        logger.info(f"[*]Total number of cells is {self.data_len}")
            
    def scPipeline(self, adata, ishvg=True,data_is_raw=True,n_hvg=10000,filter_gene_by_counts=3,filter_cell_by_counts=50, n_bins=51):
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=filter_gene_by_counts,  # step 1 过滤掉表达count小于3的基因
            filter_cell_by_counts=filter_cell_by_counts,  # step 2 过滤掉表达基因数小于10的细胞
            normalize_total=data_is_raw,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=data_is_raw,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=n_hvg if ishvg else None,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        preprocessor(adata)
        return adata
   
   
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.data_len
    
    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()

    def num_tokens(self, idx):
        return 1201

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        res = np.arange(len(self), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(res)
        return res
    
    def sparse_scipy_to_tensor(self, x: scipy.sparse.csr_matrix):
        return torch.sparse_csr_tensor(x.indptr, x.indices, x.data, (x.shape[0], x.shape[1])).to_sparse().float().coalesce()

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        gene_list = data_dict['gene_list'].astype(int)
        value = data_dict['values'].astype(float)
        if self.species_id == 1:
            gene_list = np.where(gene_list < 3, gene_list, gene_list + self.human_vocab_len-3)
        data_dict = {
            'gene_list': torch.from_numpy(gene_list).int(),    # 基因词表ID
            'value': torch.from_numpy(value).float(),
            'batch': self.batch_id,
            'species': self.species_id,
        }
        
        return data_dict
    

    @staticmethod
    def collater(samples):
        # import pdb;pdb.set_trace()
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return None

        batch = {}
        batch['gene_list'] = torch.stack([s['gene_list'] for s in samples], dim=0)
        batch['x_seq'] = torch.stack([s['value'] for s in samples], dim=0)
        batch['batch_id'] = torch.tensor([s['batch'] for s in samples], dtype=torch.int)
        batch['species'] = torch.tensor([s['species'] for s in samples], dtype=torch.int)


        return batch
