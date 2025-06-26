import numpy as np
from scipy.stats import entropy
from nilearn.connectome import ConnectivityMeasure
from typing import Sequence 
import bct


def compute_correlation_matrix(time_series, kind='correlation', standardize='zscore_sample'):
    """
    Calculates the functional connectivity matrix from time series.
    Args:
        time_series: array de shape (n_timepoints, n_regions)
        kind: 'correlation' | 'partial_correlation' | 'tangent'
        standardize: method for standarization (only for correlation matrix)
    Returns:
        np.ndarray de shape (n_regions, n_regions)
    """
    conn = ConnectivityMeasure(kind=kind, standardize=standardize)
    mat = conn.fit_transform(
        time_series if isinstance(time_series, list) else [time_series]
    )
    return mat[0]


def detect_communities(adj_matrix, seed=None, gamma=1.0, method='louvain'):
    """
    Detects communities in a structural connectivity matrix using Louvain.
    Args:
        adj_matrix: simetric (n x n)
        seed: seed for reproducibility
        gamma: resolution for the modularity
        method: 'louvain'
    Returns:
        Tuple (ci, Q) where:
          - Ci: array with the community tags (length: n)
          - Q: modularity
    """
    # bct.modularity_louvain_und returns Ci (an array of module assignments for each node) and Q (modularity coefficient)
    if method == 'louvain':
        #seed for reproducibility
        #gamma parameter that can be adjusted to tune the granularity of the detected modules
        ci, Q = bct.modularity_louvain_und(adj_matrix, gamma=gamma, seed=seed)
        return np.array(ci), Q
    else:
        raise ValueError(f"Method of detecting  communities {method} does not supported.")

def mean_intramod_positive(cm, nodes):
    #extract the submatrix with the nodes passed as parameter
    sub = cm[np.ix_(nodes, nodes)]
    vals = sub[np.triu_indices(len(nodes), k=1)] #takes the upper triangle without the diagonal
    pos = vals[vals > 0] #takes only the positive values
    return pos.mean() if pos.size else 0 

def mean_intramod_negative(cm, nodes):
    sub = cm[np.ix_(nodes, nodes)]
    vals = sub[np.triu_indices(len(nodes), k=1)]
    neg = vals[vals < 0]
    return neg.mean() if neg.size else 0

def mean_intrainter_connectivity(mat, idx1, idx2):
    """
    Computes the average conenctivity within or between two node sets
    """
    if idx1 is idx2 or np.array_equal(idx1, idx2):
        #Intramodule: extract submatrix for ROIs in idx1 and average its upper triangle
        vals = mat[np.ix_(idx1, idx1)] #extract submatrix with the ROIs that belong to module idx1
        return np.mean(vals[np.triu_indices_from(vals, k=1)]) #average of the upper triangle
    else:
        #Inter-module: extract connections between ROIs in idx1 and idx2 and average them
        return np.mean(mat[np.ix_(idx1, idx2)]) #average between the ROIs that belong to module idx1 and idx2

def compute_entropy(ts: np.ndarray, n_bins: int = 30) -> float:
    """
    Shannon Entropy of a 1D time series.
    """
    p, _ = np.histogram(ts, bins=n_bins, density=True)
    p = p[p > 0]
    return entropy(p)
