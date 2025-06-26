# src/io.py

import h5py
import nibabel as nib
import numpy as np
from nilearn import image
from pathlib import Path

from .config import DTI_MAT, LABELS_TXT, AFFECTED_ROIS_FILE

def load_mat(path: str | Path = DTI_MAT, var_names=('CM','DTI_CM')) -> np.ndarray:
    """
    Loads the structural matrix from a .mat and transposes it.
    """
    with h5py.File(path, 'r') as f:
        for name in var_names:
            if name in f and f[name].ndim == 2:
                return f[name][()].T #Python and MATLAB index 2D arrays differently for adjacency matrix, so because of that
        #if any, takes the first variable
        key = list(f.keys())[0]
        return f[key][()].T

def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """
    Load a NIfTI image using nibabel.
    """
    return nib.load(path)

def load_labels(path: str | Path = LABELS_TXT) -> list[str]:
    """
    Reads the tag file and returns the list of region names.
    """
    with open(path, 'r') as f:
        return [line.strip().split(maxsplit=1)[1] for line in f]

def load_indices_file(path= AFFECTED_ROIS_FILE) -> list[int]:
    """
    Reads a .txt with indexes (1-based or 0-based) separated by spaces or lines.
    Returns list of 0-based ints.
    """
    text = open(path).read().strip()
    nums = [int(tok) for tok in text.split()]
    #if all the index are >=1, 1-based is assumed
    if all(n >= 1 for n in nums):
        nums = [n - 1 for n in nums]
    return nums

def load_vta_labels(path: str | Path) -> list[str]:
    """
    Reads a text file with one label per line.
    """
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]
