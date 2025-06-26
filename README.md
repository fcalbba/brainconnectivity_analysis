# brainconnectivity_analysis
This repository contains a complete pipeline for analyzing brain functional and structural connectivity in subjects undergoing Deep Brain Stimulation (DBS). It evaluates tree experimental conditions: Pre-DBS, Post-DBS ON, and Post-DBS OFF.

# Project structure:
├── src/                              # Source code
│   ├── main.py                       # Main script with 3 analysis approaches
│   ├── io.py                         # Data loading functions (.mat, NIfTI, labels)
│   ├── preprocessing.py              # Atlas resampling and matrix cleaning
│   ├── connectivity.py               # Connectivity computation and entropy measures
│   ├── visualization.py              # Plotting and visualization utilities
│   ├── config.py                     # Global configuration and paths
│
├── data/                             # Input data
│   ├── DTI_cm.mat                    # Structural connectivity matrix (from Lead-DBS)
│   ├── affected_rois.txt             # Indices of ROIs affected by susceptibility artifacts
│   ├── Schaefer2018_200_17net.nii.gz # Schaefer atlas (200 parcels, 17 networks)
│   ├── Schaefer_labels.txt           # Labels of the Schaefer atlas
│   ├── vta_labels.txt                # Labels of VTA ROIs (affected by DBS)
│   └── preproc_data/                 # Preprocessed fMRI data
│       ├── sub-01/                   # Preoperative
│       │   └── fmri_preprocessed.nii.gz
│       ├── sub-02/                   # Postoperative ON
│       │   └── fmri_preprocessed.nii.gz
│       └── sub-03/                   # Postoperative OFF
│           └── fmri_preprocessed.nii.gz
├── results/                          # Output folder (auto-generated)
│   ├── approach1/                    # Modular analysis results
│   ├── approach2/                    # VTA-based analysis (entropy, mean connectivity)
│   └── approach3/                    # Motor network connectivity

# Analysis Approaches:
  1. Modular Analysis
  2. VTA Region Analysis
  3. Motor Network Analysis

# How to run:
python src/main.py \
  --atlas path/to/atlas.nii.gz \
  --pre-subj <ID pre DBS> \
  --post_on-subj <ID post DBS ON> \
  --post_off-subj <ID post DBS OFF> \
  --out-dir results/

  # Requirements:
- Python >= 3.12
- WSL

# Context
This pipeline was developed as part of a Master's thesis project to investigate how DBS modulates brain connectivity patterns.

