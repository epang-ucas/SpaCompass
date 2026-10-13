"""Dataset for two-stage (patch-level) SpaCompass inference.

A spatial sample is cut into patches of ``patch_size`` neighbouring cells. A
cell is a *centre* of its patch when its ``knn`` nearest neighbours all fall
inside that patch, so aggregating the per-patch outputs over centre cells yields
exactly one representation per cell.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import anndata as ad
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from spacompass.data.preprocess import Preprocessor, make_var_unique, select_token_indices
from spacompass.resources import resource_path

logger = logging.getLogger(__name__)


class PatchDataset:
    """Tokenise a spatial sample into gene tokens and patch neighbourhoods."""

    def __init__(
        self,
        adata,
        data_is_raw: bool = True,
        is_symbol: bool = True,
        species: str = "mouse",
        batch: str = "scRNA",
        n_token: int = 1200,
        shuffle: bool = True,
        human_vocab_len: int = 42677,
        use_coords: bool = True,
        knn: int = 8,
        patch_size: int = 192,
        vocab_path: Optional[str] = None,
        mean_file: Optional[str] = None,
    ):
        self.shuffle = shuffle
        self.human_vocab_len = human_vocab_len
        self.patch_size = patch_size
        self.knn = knn

        mapping = {"scRNA": 0, "visium": 1, "stereo": 2, "merfish": 3, "xenium": 4}
        self.batch_id = mapping[batch]
        if self.batch_id == 0 and use_coords:
            raise ValueError("Single-cell batches do not provide spatial coordinates")

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
        with open(vocab1_path, "r", encoding="utf-8") as handle:
            symbol2ensem = json.load(handle)
        assert all(isinstance(v, list) and len(v) == 1 for v in symbol2ensem.values())
        with open(vocab2_path, "r", encoding="utf-8") as handle:
            ensem2id = json.load(handle)
        cls_id = ensem2id["<cls>"]
        pad_id = ensem2id["<pad>"]

        if adata.var.index.duplicated().any():
            print("Detected duplicated genes in adata")
            adata = make_var_unique(adata)
        genes = adata.var_names.tolist()
        if is_symbol:
            if species == "mouse":
                genes = [g.lower() for g in genes]
            else:
                genes = [g.upper() for g in genes]
            gene_flags = np.array([g in symbol2ensem for g in genes])
        else:
            gene_flags = np.array([g in ensem2id for g in genes])
        print(f"{gene_flags.sum()}/{len(gene_flags)} have been matched!")
        if not gene_flags.any():
            raise ValueError(
                "No gene in the input matched the SpaCompass vocabulary; check the "
                "gene-id convention (--genes-are-ensembl / --gene-column) and --species."
            )
        adata = adata[:, gene_flags].copy()
        genes = np.array(genes)[gene_flags]
        if is_symbol:
            gene_vocab_ids = np.array([ensem2id[symbol2ensem[g][0]] for g in genes])
        else:
            gene_vocab_ids = np.array([ensem2id[g] for g in genes])
        adata.var["vocab_ids"] = gene_vocab_ids

        adata_hvg = self.scPipeline(
            adata,
            filter_gene_by_counts=3 if data_is_raw else False,
            filter_cell_by_counts=50 if data_is_raw else False,
            data_is_raw=data_is_raw,
            ishvg=data_is_raw,
        )
        X = adata_hvg.layers["X_binned"]
        gene_vocab_ids = adata_hvg.var["vocab_ids"].values

        patches = self.slice_adata_with_adj(adata_hvg, patch_size=patch_size, knn_k=knn, seed=42)
        dataset = []
        for patch in patches:
            X = patch.layers["X_binned"]
            gene_vocab_ids = patch.var["vocab_ids"].values
            coords = patch.obsm["spatial"] if use_coords else None
            gene_list_batch, value_batch = [], []
            for data in X:
                id_selected = select_token_indices(data, n_token)
                gene_ids = gene_vocab_ids[id_selected]
                values = data[id_selected]
                gene_ids = np.append(cls_id, gene_ids)
                values = np.append(0, values)
                if len(gene_ids) > n_token + 1:
                    raise ValueError("Impossible!")
                pad_len = n_token + 1 - len(gene_ids)
                gene_ids = np.concatenate([gene_ids, np.full(pad_len, pad_id, dtype=gene_ids.dtype)])
                values = np.concatenate([values, np.zeros(pad_len, dtype=values.dtype)])
                gene_list_batch.append(gene_ids)
                value_batch.append(values)
            dataset.append({
                "gene_list": np.array(gene_list_batch),
                "values": np.array(value_batch),
                "coord": coords,
                "adj": patch.uns["adj"],
                "is_center_cell": patch.obs["is_center"].values,
                "original_index": patch.obs["original_index"].values,
                "batch": self.batch_id,
            })
        self.dataset = dataset
        self.data_len = len(dataset)
        self.pad_id = pad_id
        self.cls_id = cls_id
        self.adata = adata_hvg
        logger.info(f"[*]Total number of patches is {self.data_len}")

    def scPipeline(
        self,
        adata,
        ishvg=True,
        data_is_raw=True,
        n_hvg=10000,
        filter_gene_by_counts=3,
        filter_cell_by_counts=50,
        n_bins=51,
    ):
        preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=filter_gene_by_counts,
            filter_cell_by_counts=filter_cell_by_counts,
            normalize_total=data_is_raw,
            result_normed_key="X_normed",
            log1p=data_is_raw,
            result_log1p_key="X_log1p",
            subset_hvg=n_hvg if ishvg else None,
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,
            result_binned_key="X_binned",
        )
        preprocessor(adata)
        return adata

    def __len__(self) -> int:
        return self.data_len

    def num_tokens(self, idx) -> int:
        return self.patch_size

    def ordered_indices(self):
        res = np.arange(len(self), dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(res)
        return res

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        gene_list = data_dict["gene_list"].astype(int)
        value = data_dict["values"].astype(float)
        coord = data_dict["coord"]
        if coord is not None and coord.dtype != "float64":
            coord = coord.astype(np.float64)
        if self.species_id == 1:
            gene_list = np.where(gene_list < 3, gene_list, gene_list + self.human_vocab_len - 3)
        return {
            "gene_list": torch.from_numpy(gene_list).int(),
            "value": torch.from_numpy(value).float(),
            "coord": torch.from_numpy(coord).float() if coord is not None else None,
            "adj": torch.from_numpy(data_dict["adj"].toarray()).bool(),
            "is_center_cell": torch.from_numpy(data_dict["is_center_cell"]).bool(),
            "original_index": data_dict["original_index"],
            "batch": self.batch_id,
        }

    @staticmethod
    def collater(samples):
        samples = [s for s in samples if s is not None]
        if len(samples) == 0:
            return None
        batch = {
            "gene_list": torch.stack([s["gene_list"] for s in samples], dim=0),
            "x_seq": torch.stack([s["value"] for s in samples], dim=0),
            "adj": torch.stack([s["adj"] for s in samples], dim=0),
            "is_center_cell": torch.stack([s["is_center_cell"] for s in samples], dim=0),
            "original_index": np.stack([s["original_index"] for s in samples], axis=0),
            "batch_id": torch.tensor([s["batch"] for s in samples], dtype=torch.int),
        }
        if samples[0]["coord"] is not None:
            batch["coords"] = torch.stack([s["coord"] for s in samples], dim=0)
        return batch

    def slice_adata_with_adj(
        self,
        adata: ad.AnnData,
        patch_size: int = 192,
        knn_k: int = 8,
        max_patches: Optional[int] = None,
        seed: int = 42,
        verbose: bool = False,
    ):
        """Cut the sample into KNN patches and annotate centre cells."""
        np.random.seed(seed)
        try:
            coords = adata.obsm["spatial"]
        except KeyError:
            coords = adata.obs[["x", "y"]].values
        n_cells = adata.n_obs

        full_knn = NearestNeighbors(n_neighbors=knn_k + 1, algorithm="auto").fit(coords)
        _, global_knn_indices = full_knn.kneighbors(coords)
        global_knn_sets = [set(nbs) for nbs in global_knn_indices]

        patch_knn = NearestNeighbors(n_neighbors=patch_size, algorithm="auto").fit(coords)
        all_indices = np.arange(n_cells)
        patches, used = [], set()

        while len(used) < n_cells:
            i = np.random.choice(np.setdiff1d(all_indices, list(used)))
            _, patch_indices = patch_knn.kneighbors([coords[i]])
            patch_indices = patch_indices[0]
            patch_set = set(patch_indices)
            valid_centers = [idx for idx in patch_indices if global_knn_sets[idx].issubset(patch_set)]

            patch_adata = adata[patch_indices].copy()
            patch_adata.obs["original_index"] = patch_indices
            patch_adata.obs["is_center"] = [idx in valid_centers for idx in patch_indices]

            patch_coords = patch_adata.obsm["spatial"]
            local_knn = NearestNeighbors(n_neighbors=knn_k + 1).fit(patch_coords)
            _, patch_knn_indices = local_knn.kneighbors(patch_coords)
            row_idx, col_idx = [], []
            for row, neighbors in enumerate(patch_knn_indices):
                for neighbour in neighbors[1:]:
                    row_idx.append(row)
                    col_idx.append(neighbour)
            adj = csr_matrix(
                (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
                shape=(patch_adata.n_obs, patch_adata.n_obs),
            )
            patch_adata.uns["adj"] = adj
            patches.append(patch_adata)
            used.update(valid_centers)
            if max_patches and len(patches) >= max_patches:
                break

        if verbose:
            print(f"Generated {len(patches)} patches")
        return patches
