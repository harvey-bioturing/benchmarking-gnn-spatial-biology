import scanpy as sc
import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.spatial import cKDTree
from scipy.stats import mode
from collections import Counter
from sklearn.neighbors import kneighbors_graph
import torch



# New utils functions
def cKD_refine_label(coords, labels, k):
    """
    Refine labels using majority voting of k-nearest neighbors.
    
    Args:
        coords (np.ndarray): Coordinates of the nodes, shape (N, 2)
        labels (torch.Tensor or np.ndarray): Labels for each node, shape (N,)
        k (int): Number of nearest neighbors to consider
    
    Returns:
        np.ndarray: Refined labels of shape (N,)
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.detach().cpu().numpy()
    else:
        labels_np = np.array(labels)

    # Build KD-Tree and query neighbors
    tree = cKDTree(coords)
    _, neighbors = tree.query(coords, k=k+1)  # include self
    neighbors = neighbors[:, 1:]  # exclude self

    # Refine labels
    new_labels = labels_np.copy()
    for i, nbrs in enumerate(neighbors):
        neighbor_labels = labels_np[nbrs]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        new_labels[i] = most_common_label

    return new_labels

def cKD_refine_logits(coords, logits, k):
    """
    Refine logits using PyTorch ops to preserve autograd.
    
    Args:
        coords (np.ndarray): (N, 2)
        logits (torch.Tensor): (N, C), requires_grad = True
        k (int): number of neighbors

    Returns:
        torch.Tensor: (N, C), refined logits with grad_fn
    """
    coords_np = np.array(coords)
    tree = cKDTree(coords_np)
    _, neighbors = tree.query(coords_np, k=k+1)
    neighbors = neighbors[:, 1:]

    N, C = logits.shape
    device = logits.device
    refined_logits = torch.zeros_like(logits)

    for i, nbrs in enumerate(neighbors):
        nbrs_tensor = torch.tensor(nbrs, device=device, dtype=torch.long)
        refined_logits[i] = logits[nbrs_tensor].mean(dim=0)

    return refined_logits


def SelfGraphConv(adata, n_neighbors = 4):
    connectivity = kneighbors_graph(adata.obsm['spatial'], n_neighbors=n_neighbors, include_self=False)
    connectivity = 0.5 * (connectivity + connectivity.T)
    adata.X = connectivity.dot(adata.X)
    return (adata)


# This is from https://github.com/JinmiaoChenLab/GraphST/blob/main/GraphST/GraphST.py
def preprocess(adata, n_hvg):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_hvg)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    adata.obsm['hvg_feature'] = feat


def construct_interaction_KNN(adata: AnnData, n_neighbors: int = 3, store_key: str = 'adj'):
    """
    Constructs a symmetric KNN graph based on spatial coordinates.
    Stores the binary adjacency matrix in `adata.obsm[store_key]`.

    Parameters:
    - adata: AnnData object with .obsm['spatial'] coordinates
    - n_neighbors: Number of neighbors to connect each node to
    - store_key: Key under adata.obsm to store the adjacency matrix
    """
    if 'spatial' not in adata.obsm:
        raise ValueError("Missing `adata.obsm['spatial']` coordinates.")

    position = adata.obsm['spatial']
    n_spot = position.shape[0]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)

    x = np.repeat(np.arange(n_spot), n_neighbors)
    y = indices[:, 1:].flatten()

    interaction = np.zeros((n_spot, n_spot), dtype=np.float32)
    interaction[x, y] = 1

    # Make symmetric
    adj = interaction + interaction.T
    adj[adj > 1] = 1  # Avoid double edges

    adata.obsm[store_key] = adj
    print(f"Graph constructed with {n_neighbors} neighbors and stored in adata.obsm['{store_key}'].")

def normalize_adj(adj: np.ndarray) -> np.ndarray:
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj_normalized.toarray()

def preprocess_adj(adj: np.ndarray) -> np.ndarray:
    """
    Normalize and add self-connections for GCN models.
    """
    adj_normalized = normalize_adj(adj)
    adj_normalized += np.eye(adj.shape[0])
    return adj_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
   """Convert a scipy sparse matrix to a torch sparse tensor."""
   sparse_mx = sparse_mx.tocoo().astype(np.float32)
   indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
   values = torch.from_numpy(sparse_mx.data)
   shape = torch.Size(sparse_mx.shape)
   return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)  

def dense_to_sparse_edge_index(adj: np.ndarray):
    sparse_adj = sp.coo_matrix(adj)
    edge_index, _ = from_scipy_sparse_matrix(sparse_adj)
    return edge_index
