from typing import Dict, Optional, Union

import numpy as np
import torch
import pickle, anndata
from scipy.sparse import issparse
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData
import logging
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        use_key: Optional[str] = None,
        filter_gene_by_counts: Union[int, bool] = False,
        filter_cell_by_counts: Union[int, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_normed_key: Optional[str] = "X_normed",
        log1p: bool = False,
        result_log1p_key: str = "X_log1p",
        subset_hvg: Union[int, bool] = False,
        hvg_use_key: Optional[str] = None,
        hvg_flavor: str = "seurat_v3",
        binning: Optional[int] = None,
        result_binned_key: str = "X_binned",
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        filter_gene_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, filter genes with counts
        filter_cell_by_counts (:class:`int` or :class:`bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, filter cells with counts
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        result_normed_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log1p (:class:`bool`, default: ``True``):
            Whether to apply log1p transform to the normalized data.
        result_log1p_key (:class:`str`, default: ``"X_log1p"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or :class:`bool`, default: ``False``):
            Whether to subset highly variable genes.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat_v3"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details.
        binning (:class:`int`, optional):
            Whether to bin the data into discrete values of number of bins provided.
        result_binned_key (:class:`str`, default: ``"X_binned"``):
            The key of :class:`~anndata.AnnData` to store the binned data.
        """
        self.use_key = use_key
        self.filter_gene_by_counts = filter_gene_by_counts
        self.filter_cell_by_counts = filter_cell_by_counts
        self.normalize_total = normalize_total
        self.result_normed_key = result_normed_key
        self.log1p = log1p
        self.result_log1p_key = result_log1p_key
        self.subset_hvg = subset_hvg
        self.hvg_use_key = hvg_use_key
        self.hvg_flavor = hvg_flavor
        self.binning = binning
        self.result_binned_key = result_binned_key

    def __call__(self, adata: AnnData, batch_key: Optional[str] = None) -> Dict:
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        key_to_process = self.use_key
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
        # is_logged = self.check_logged(adata, obs_key=key_to_process)
        is_logged = False
        # import pdb;pdb.set_trace()
        # step 1: filter genes
        if self.filter_gene_by_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_by_counts
                if isinstance(self.filter_gene_by_counts, int)
                else None,
            )

        # step 2: filter cells
        if isinstance(self.filter_cell_by_counts, int):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_genes=self.filter_cell_by_counts
                if isinstance(self.filter_cell_by_counts, int)
                else None,
            )
        # import pdb;pdb.set_trace()
        # step 3: normalize total
        if self.normalize_total:
            logger.info("Normalizing total counts ...")
            normed_ = sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                layer=key_to_process,
                inplace=False,
            )["X"]
            key_to_process = self.result_normed_key or key_to_process
            _set_obs_rep(adata, normed_, layer=key_to_process)

        # step 4: log1p
        if self.log1p:
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log1p_key:
                _set_obs_rep(
                    adata,
                    _get_obs_rep(adata, layer=key_to_process),
                    layer=self.result_log1p_key,
                )
                key_to_process = self.result_log1p_key
            sc.pp.log1p(adata, layer=key_to_process)
        # import pdb;pdb.set_trace()
        # step 5: subset hvg
        if self.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            # if batch_key is None:
            #     logger.warning(
            #         "No batch_key is provided, will use all cells for HVG selection."
            #     )
            sc.pp.highly_variable_genes(
                adata,
                layer=self.hvg_use_key,
                n_top_genes=self.subset_hvg
                if isinstance(self.subset_hvg, int)
                else None,
                batch_key=batch_key,
                flavor=self.hvg_flavor,
                subset=True,
            )
        # import pdb;pdb.set_trace()

        # step 6: binning
        if self.binning:
            logger.info("Binning data ...")
            if not isinstance(self.binning, int):
                raise ValueError(
                    "Binning arg must be an integer, but got {}.".format(self.binning)
                )
            n_bins = self.binning  # NOTE: the first bin is always a spectial for zero
            binned_rows = []
            bin_edges = []
            all_zero_row = []
            layer_data = _get_obs_rep(adata, layer=key_to_process)
            layer_data = layer_data.toarray()  if issparse(layer_data) else layer_data
            for row in layer_data:
                if np.all(row==0):
                    print(f"warning: find all zero row!")
                    binned_row = np.zeros_like(row).astype(int)
                    binned_rows.append(binned_row)
                    bin_edges.append(np.zeros(n_bins))
                    all_zero_row.append(True)
                else:
                    non_zero_ids = row.nonzero()
                    non_zero_row = row[non_zero_ids]

                    bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
                    # bins = np.sort(np.unique(bins))
                    # NOTE: comment this line for now, since this will make the each category
                    # has different relative meaning across datasets
                    non_zero_digits = self._digitize(non_zero_row, bins)
                    assert non_zero_digits.min() >= 1
                    assert non_zero_digits.max() <= n_bins - 1
                    binned_row = np.zeros_like(row, dtype=np.int64)
                    binned_row[non_zero_ids] = non_zero_digits
                    binned_rows.append(binned_row)
                    bin_edges.append(np.concatenate([[0], bins]))
                    all_zero_row.append(False)
            # import pdb;pdb.set_trace()
            adata.layers[self.result_binned_key] = np.stack(binned_rows)
            adata.obsm["bin_edges"] = np.stack(bin_edges)
            
            adata.obs["all_zero_row"] = np.array(all_zero_row)
            

    def _digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Digitize the data into bins. This method spreads data uniformly when bins
        have same values.

        Args:

        x (:class:`np.ndarray`):
            The data to digitize.
        bins (:class:`np.ndarray`):
            The bins to use for digitization, in increasing order.

        Returns:

        :class:`np.ndarray`:
            The digitized data.
        """
        assert x.ndim == 1 and bins.ndim == 1

        left_digits = np.digitize(x, bins)
        right_difits = np.digitize(x, bins, right=True)

        rands = np.random.rand(len(x))  # uniform random numbers

        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True


def binning(
    row: Union[np.ndarray, torch.Tensor], n_bins: int
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins."""
    dtype = row.dtype
    return_np = False if isinstance(row, torch.Tensor) else True
    row = row.cpu().numpy() if isinstance(row, torch.Tensor) else row
    # TODO: use torch.quantile and torch.bucketize

    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
        non_zero_digits = np.digitize(non_zero_row, bins)
        binned_row = np.zeros_like(row, dtype=np.int64)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = np.digitize(row, bins)
    return torch.from_numpy(binned_row) if not return_np else binned_row.astype(dtype)

def tokenize(
    vocab,
    genes: np.ndarray,
    values: np.ndarray,
    include_zero_gene: bool = True,
    append_cls: bool = True,
    cls_token: bool = "<cls>",
    return_pt: bool = True,
):
    """
    Tokenize a sample of data. Returns tokenized values and gene names.

    Args:
        data (array-like): A sample of data, with shape (n_features,).
            n_features equals the number of all genes.
        gene_ids (array-like): A sample of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A tuple (gene_id, count) of non zero gene expressions.
    """
    if len(genes) != len(values):
        raise ValueError(
            f"Number of features in data ({len(genes)}) does not match "
            f"number of gene_ids ({len(values)})."
        )
    gene_ids = np.array(vocab(genes.tolist()))
    cls_id = vocab[cls_token]
    if not include_zero_gene:
        idx = np.nonzero(values)
        values = values[idx]
        gene_ids = gene_ids[idx]
    if append_cls:
        gene_ids = np.append(cls_id,gene_ids)
        values = np.append(0,values)
        # gene_ids = np.insert(gene_ids, 0, cls_id)
        # values = np.insert(values, 0, 0)
    if return_pt:
        gene_ids = torch.from_numpy(gene_ids).long()
        values = torch.from_numpy(values)
    return gene_ids, values

def pad(
    vocab,
    gene_ids,
    values,
    max_len: int,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
):
    """
    Pad a sample of data. Returns padded values and gene names.

    """
    pad_id = vocab[pad_token]
    idx =  np.array(range(len(gene_ids)))
    if len(gene_ids) > max_len:
        if not cls_appended:
            idx = np.random.choice(len(gene_ids), max_len, replace=False)
        else:
            idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
            idx = idx + 1
            idx = np.append(0, idx)
        gene_ids = gene_ids[idx]
        values = values[idx]
    if len(gene_ids) < max_len:
        gene_ids = torch.cat(
            [
                gene_ids,
                torch.full(
                    (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                ),
            ]
        )
        values = torch.cat(
            [
                values,
                torch.full(
                    (max_len - len(values),), pad_value, dtype=values.dtype
                ),
            ]
        )
        
    return gene_ids, values, idx

def select_token_indices(arr: np.ndarray, n_token: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)

    nonzero_indices = np.flatnonzero(arr)

    if len(nonzero_indices) > n_token:
        # 非0元素大于n_token：按表达值概率采样
        values = arr[nonzero_indices]
        probs = values / values.sum()
        selected_indices = np.random.choice(nonzero_indices, size=n_token, replace=False, p=probs)
    else:
        selected_indices = nonzero_indices

    return np.array(selected_indices)

def make_var_unique(adata, method="mean"):
    assert method == "mean"

    # 利用 pd.factorize 将 var_names 分组，返回重复时的组号以及唯一基因名数组
    # import pdb;pdb.set_trace()
    group_indices, unique_genes = pd.factorize(adata.var_names)

    rows = np.arange(len(group_indices))
    cols = group_indices
    data = np.ones(len(group_indices), dtype=adata.X.dtype)
    G = sp.csr_matrix((data, (rows, cols)), shape=(adata.n_vars, len(unique_genes)))
    X_sum = adata.X * G  # shape (n_obs, n_unique_genes)
    counts = np.array(G.sum(axis=0)).ravel()
    D = sp.diags(1.0 / counts)
    X_new = X_sum * D

    adata_new = anndata.AnnData(
        X=X_new,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=pd.Index(unique_genes)),
        uns=adata.uns,
        obsm=adata.obsm
    )
    return adata_new