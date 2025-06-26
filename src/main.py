#!/usr/bin/env python
"""
Three different approaches to analyze the brain connectivity:
    1. Modular Analysis
    2. VTA Region Analysis
    3. Motor Network Analysis
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from src.config import (
    RANDOM_SEED,
    DTI_MAT,
    LABELS_TXT,
    AFFECTED_ROIS_FILE,
    VTA_LABELS_FILE,
    PREP_TEMPLATES,
)
from src.io import load_mat, load_nifti, load_labels, load_indices_file, load_vta_labels
from src.preprocessing import extract_timeseries, resample_atlas, clean_matrix

from src.connectivity import (
    compute_correlation_matrix, #
    detect_communities, #
    compute_entropy, #
    mean_intramod_positive, #
    mean_intramod_negative, #
    mean_intrainter_connectivity,#
)

from src.visualization import (
    plot_reordered_matrix, #
    plot_reordered_fc, #
    plot_motor_blocks_simple, #
    plot_positive_mean_per_roi, #
    plot_negative_mean_per_roi, #
    plot_entropy, #
    plot_absolute_change_matrix #
)

def main(args):
    np.random.seed(RANDOM_SEED)
    # Load DTI matrix and labels of the atlas
    sCM    = load_mat(args.mat or DTI_MAT)
    labels = load_labels(args.labels or LABELS_TXT)

    # Load the susceptibility-affected ROIS and delete it from the DTI matrix
    affected = load_indices_file(args.affected_rois or AFFECTED_ROIS_FILE)
    print("Deleted ROIs (affected by susceptibility artifacts):")
    for idx in affected:
        print(f"  • {labels[idx]}")
    print(f"Total ROIs removed: {len(affected)}\n")
    sCM_clean, keep_idx = clean_matrix(sCM, affected)

    # Detect the communities in the clean SC
    Ci, Q = detect_communities(sCM_clean, seed=RANDOM_SEED)
    print(f"Modularity Q={Q:.3f}, Number of communities={len(np.unique(Ci))}\n")

    # Which nodes belong to each module
    labels_mod  = np.unique(Ci)
    idx_regions = {lab: np.where(Ci == lab)[0] for lab in labels_mod}

    # Functional network dominant in each module
    mod_networks = {}
    for mod, roi_indices in idx_regions.items():
        redes = [labels[i].split('_')[2] for i in roi_indices]
        red_principal = Counter(redes).most_common(1)[0][0]
        mod_networks[mod] = red_principal

    print("\nDominant functional network per module:")
    for mod, net in mod_networks.items():
        print(f"  • Module {mod}: {net}")

    # Clean functional matrix for each conditiom
    conds = {}
    for cond in ('pre', 'post_on', 'post_off'):
        subj = getattr(args, f"{cond}_subj")
        tpl  = str(PREP_TEMPLATES[cond])
        fmri = load_nifti(tpl.format(subject_id=subj))
        atlas_res = resample_atlas(load_nifti(args.atlas), fmri)
        ts        = extract_timeseries(fmri, atlas_res)
        mat       = compute_correlation_matrix(ts)
        mat_cln, _= clean_matrix(mat, affected)
        conds[cond] = mat_cln

    # Output folders
    p1 = os.path.join(args.out_dir, 'approach1')
    p2 = os.path.join(args.out_dir, 'approach2')
    p3 = os.path.join(args.out_dir, 'approach3')
    for d in (p1, p2, p3):
        os.makedirs(d, exist_ok=True)

    # ===== APPROACH 1 =====
    # Clean reordered DTI according to the modules
    fig = plot_reordered_matrix(sCM_clean, Ci, title="sCM Reordered")
    fig.savefig(os.path.join(p1, "scm_reordered.png"))
    plt.close(fig)

    #FC in all conditions reordered according to th e modules
    for cond, M in conds.items():
        fig = plot_reordered_fc(M, Ci, title=f"FC {cond.upper()} Reordered")
        fig.savefig(os.path.join(p1, f"fc_{cond}_reordered.png"))
        plt.close(fig)

    # 1.3 Intramodular +/-
    labels_mod  = np.unique(Ci)
    idx_regions = {lab: np.where(Ci == lab)[0] for lab in labels_mod}

    intra_pos = {
        c: [mean_intramod_positive(conds[c], idx) for idx in idx_regions.values()]
        for c in conds
    }
    intra_neg = {
        c: [mean_intramod_negative(conds[c], idx) for idx in idx_regions.values()]
        for c in conds
    }

    mods = np.arange(1, len(labels_mod) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    markers = ['o', 's', 'x']
    colors  = ['b', 'r', 'g']
    for i, cond in enumerate(conds):
        axs[0].plot(mods, intra_pos[cond],    markers[i]+'-'+colors[i], label=cond.upper())
        axs[1].plot(mods, intra_neg[cond],    markers[i]+'-'+colors[i], label=cond.upper())
    axs[0].set(title='Intramodular +', xlabel='Module')
    axs[0].legend(); axs[0].grid(True)
    axs[1].set(title='Intramodular –', xlabel='Module')
    axs[1].legend(); axs[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(p1, "intramodular.png"))
    plt.close(fig)

    # Intramodularity
    n_mod = len(labels_mod)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    inter_mats = {}
    for cond in conds:
        M = conds[cond]
        mat = np.zeros((n_mod, n_mod))
        for ii, li in enumerate(labels_mod):
            for jj, lj in enumerate(labels_mod):
                mat[ii, jj] = mean_intrainter_connectivity(
                    M, idx_regions[li], idx_regions[lj]
                )
        inter_mats[cond] = mat

    vmin = min(m.min() for m in inter_mats.values())
    vmax = max(m.max() for m in inter_mats.values())

    for ax, (cond, mat) in zip(axs, inter_mats.items()):
        im = ax.imshow(mat, cmap='coolwarm',
                       vmin=vmin, vmax=vmax, aspect='equal')
        # white grid between cells
        ax.set_xticks(np.arange(-.5, n_mod, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_mod, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        # numbers
        for i in range(n_mod):
            for j in range(n_mod):
                ax.text(j, i, f"{mat[i,j]:.2f}",
                        ha='center', va='center', fontsize=8)
        mods = np.arange(1, n_mod + 1)
        ax.set_xticks(mods - 1); ax.set_xticklabels(mods, fontsize=8)
        ax.set_yticks(mods - 1); ax.set_yticklabels(mods, fontsize=8)
        ax.set_xlabel("Module")
        if ax is axs[0]:
            ax.set_ylabel("Module")
        ax.set_title(cond.upper())

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax, label="Mean connectivity")
    fig.subplots_adjust(left=0.08, right=0.9, wspace=0.3)
    fig.savefig(os.path.join(p1, "intermodular_annotated.png"), dpi=300)
    plt.close(fig)
    # --------------------------------------------------
    # === APPROACH 2: VTA ROIs + Entropy + Mean ± per ROI ===
    # --------------------------------------------------
    # Load VTA tags
    vta_labels = load_vta_labels(args.vta_labels or VTA_LABELS_FILE)
    orig_idx   = [labels.index(lbl) for lbl in vta_labels]
    vta_idx    = [np.where(keep_idx == i)[0][0] for i in orig_idx]
    # Positive / negative mean per ROI
    fig = plot_positive_mean_per_roi(conds, labels, keep_idx, vta_idx, vta_labels)
    fig.savefig(os.path.join(p2, "positive_mean_per_roi_vta.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    fig = plot_negative_mean_per_roi(conds, labels, keep_idx, vta_idx, vta_labels)
    fig.savefig(os.path.join(p2, "negative_mean_per_roi_vta.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Extracting temporal series and compute entropies
    conds_ts = {}
    for cond in ('pre','post_on','post_off'):
        tpl       = str(PREP_TEMPLATES[cond])
        fmri      = load_nifti(tpl.format(subject_id=getattr(args, f"{cond}_subj")))
        atlas_res = resample_atlas(load_nifti(args.atlas), fmri)
        ts        = extract_timeseries(fmri, atlas_res)
        conds_ts[cond] = ts[:, keep_idx].T

    entropies = {cond: [compute_entropy(conds_ts[cond][i]) for i in vta_idx]
                 for cond in ('pre','post_on','post_off')}

    # PNG per ROI with the three conditions
    entropy_dir = os.path.join(p2, "entropy-rois")
    os.makedirs(entropy_dir, exist_ok=True)
    for i, roi in enumerate(vta_labels):
        fig, ax = plt.subplots(figsize=(8,3))
        for cond, color in zip(('pre','post_on','post_off'), ('b','r','g')):
            h = entropies[cond][i]
            ax.plot(conds_ts[cond][i], color, label=f"{cond} (H={h:.2f})")
        ax.set_title(roi, fontsize=10)
        ax.set_xlabel("Timepoints")
        ax.set_ylabel("Signal (z-score)")
        ax.legend(fontsize=8, loc='upper right')
        fig.tight_layout()
        safe = roi.replace('/','_').replace(' ','_')
        fig.savefig(os.path.join(entropy_dir, f"{safe}.png"), dpi=300)
        plt.close(fig)

    # Barplot of entropies
    fig = plot_entropy(conds_ts, entropies, vta_labels, p2)
    fig.savefig(os.path.join(p2, "entropy_barplot.png"), dpi=300)
    plt.close(fig)

    # Matrix with the absolute difference between the conditions
    vta_mats = {cond: conds[cond][np.ix_(vta_idx, vta_idx)] for cond in conds}

    fig = plot_absolute_change_matrix(vta_mats['pre'], vta_mats['post_on'],
                                      labels=vta_labels,
                                      title="|DBS-ON − Pre-DBS|")
    fig.savefig(os.path.join(p2, "vta_abs_change_on_pre.png"), dpi=300)
    plt.close(fig)
    fig = plot_absolute_change_matrix(vta_mats['pre'], vta_mats['post_off'],
                                      labels=vta_labels,
                                      title="|DBS-OFF − Pre-DBS|")
    fig.savefig(os.path.join(p2, "vta_abs_change_off_pre.png"), dpi=300)
    plt.close(fig)
    fig = plot_absolute_change_matrix(vta_mats['post_off'], vta_mats['post_on'],
                                      labels=vta_labels,
                                      title="|DBS-OFF − DBS-ON|")
    fig.savefig(os.path.join(p2, "vta_abs_change_off_on.png"), dpi=300)
    plt.close(fig)

    # Top 5 ROIs with the highets global change |ON−PRE|
    delta_global = []
    prem = vta_mats['pre']; onm = vta_mats['post_on']
    for i in range(len(vta_idx)):
        dif = np.abs(onm[i,:] - prem[i,:])
        dif = np.delete(dif, i)
        delta_global.append(dif.mean())
    delta_global = np.array(delta_global)
    top5 = np.argsort(delta_global)[-5:][::-1]
    print("\nTop 5 ROIs with the highest global change |ON−PRE|:")
    for idx in top5:
        print(f"  • {vta_labels[idx]}: Δ={delta_global[idx]:.4f}")

    # --------------------------------------------------
    # === APPROACH 3: Motor networks ===
    # --------------------------------------------------
    motor_idx = {
        'SomMotA_LH': np.arange(12,20),
        'SomMotB_LH': np.arange(20,28),
        'SomMotA_RH': np.arange(112,123),
        'SomMotB_RH': np.arange(123,130),
    }
    for cond, M in conds.items():
        fig = plot_motor_blocks_simple(M, cond, motor_idx, list(motor_idx.keys()))
        fig.savefig(os.path.join(p3, f"motor_{cond}.png"), dpi=300)
        plt.close(fig)

    # Table with the mean of the subnetworks for each condition
    blocks = list(motor_idx.keys())
    print("\nMean positive intra-block connectivity (motor networks):")
    print("Condition        " + "  ".join(f"{b:12s}" for b in blocks))
    name_map = {'pre':'Pre DBS','post_on':'Post DBS (ON)','post_off':'Post DBS (OFF)'}
    for cond in ('pre','post_on','post_off'):
        row = [name_map[cond].ljust(15)]
        for blk in blocks:
            m = mean_intramod_positive(conds[cond], motor_idx[blk])
            row.append(f"{m:10.6f}")
        print("  ".join(row))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mat', help='Path to a DTI .mat')
    p.add_argument('--atlas', required=True, help='Atlas NIfTI')
    p.add_argument('--labels', help='Labels ROI .txt')
    p.add_argument('--affected-rois', help='IDs of affected ROIs .txt')
    p.add_argument('--vta-labels', help='Labels VTA .txt')
    p.add_argument('--pre-subj', required=True)
    p.add_argument('--post_on-subj', required=True)
    p.add_argument('--post_off-subj', required=True)
    p.add_argument('--out-dir', default='results')
    args = p.parse_args()
    main(args)
