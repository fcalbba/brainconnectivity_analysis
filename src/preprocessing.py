# src/preprocessing.py
import numpy as np
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image

def resample_atlas(atlas_img, target_img, interpolation='nearest'):
    """
    Resampling atlas → target_img.
    """
    return image.resample_to_img(atlas_img, target_img, interpolation=interpolation)

def extract_timeseries(fmri_img, atlas_img, standardize=True):
    """
    Extract and standarize the timeseries of each ROI.
    """
    masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=standardize)
    return masker.fit_transform(fmri_img)


def clean_matrix(matrix: np.ndarray,
                 drop_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Clean the matrix (0-based).
    Returns (matrix_clean, keep_indices).
    """
    mask = np.ones(matrix.shape[0], dtype=bool)
    mask[drop_indices] = False
    keep = np.where(mask)[0]
    return matrix[np.ix_(keep, keep)], keep

