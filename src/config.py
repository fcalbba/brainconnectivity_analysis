from pathlib import Path

# Folt
BASE_DIR    = Path(__file__).resolve().parent.parent
# Data folder
DATA_DIR    = BASE_DIR / "data"
# Preprocessed data folder
PREPROC_DIR = DATA_DIR / "preproc_data"

# Paths to main files
DTI_MAT            = DATA_DIR / "DTI_CM.mat" #dti matrix obtained from Lead DBS
LABELS_TXT         = DATA_DIR / "Schaefer2018_200Parcels_17Networks_order.txt" #labels of Schaefer atlas
AFFECTED_ROIS_FILE = DATA_DIR / "affected_rois.txt" #IDs with the susceptibility affected ROIs
VTA_LABELS_FILE    = DATA_DIR / "vta_labels.txt" #labels of the Volume of Affected Tissue

# Atlas parameters
N_ROIS        = 200 #number of regions
YEO_NETWORKS  = 17 #number of networks
RESOLUTION_MM = 2

# Templates for preprocessed fMRI data obtained from the pipeline of Biocruces (https://github.com/compneurobilbao/compneuro-fmriproc)
PREP_TEMPLATES = {
    cond: PREPROC_DIR / f"sub-{{subject_id}}/sub-{{subject_id}}_preprocessed.nii.gz"
    for cond in ('pre','post_on','post_off')
}

# Seed for reproducibility
RANDOM_SEED = 20
