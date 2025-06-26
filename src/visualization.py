"""
Functions for displaying connectivity matrices and modular results.
"""
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
import os

def plot_reordered_matrix(matrix, communities, title=None):
    """
    Plots the rearranged matrix according to communities and draws edges.
    """
    order = np.argsort(communities)
    mat_ord = matrix[np.ix_(order, order)]
    comm_ord = communities[order]
    boundaries = np.where(np.diff(comm_ord) != 0)[0] + 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mat_ord, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    for b in boundaries:
        ax.axhline(b - 0.5, color='black', lw=1.5)
        ax.axvline(b - 0.5, color='black', lw=1.5)
    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_reordered_fc(mat, communities, title):
    """
    Wrapper para plot_reordered_matrix con FC.
    """
    return plot_reordered_matrix(mat, communities, title)

def plot_motor_blocks_simple(FC, title, region_indices: dict, region_names: list[str]):
    """
    Draw blocks of motor subnetworks with averages per block.
    Now return the figure so that you can save it later.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    #calculation of size and boundaries
    sizes = [len(region_indices[n]) for n in region_names]
    boundaries = np.concatenate([[0], np.cumsum(sizes)])
    #concatenate all the indexes to extract the sub-matrix
    idx = np.concatenate([region_indices[n] for n in region_names])
    sub = FC[np.ix_(idx, idx)]

    # creation of fig and axis
    fig, ax = plt.subplots(figsize=(7, 7))

    #show the submatrx
    im = ax.imshow(sub, interpolation='nearest', vmin=-1, vmax=1)

    #centered tags
    ticks = [(boundaries[i] + boundaries[i+1] - 1) / 2 for i in range(len(region_names))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(region_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(ticks)
    ax.set_yticklabels(region_names, fontsize=12)

    #separated lines between modules
    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color='k', lw=2)
        ax.axvline(b - 0.5, color='k', lw=2)

    #mean within each module
    for i in range(len(region_names)):
        for j in range(len(region_names)):
            si, ei = int(boundaries[i]), int(boundaries[i+1])
            sj, ej = int(boundaries[j]), int(boundaries[j+1])
            block = sub[si:ei, sj:ej]
            if i == j:
                # upper triangle without the diagonal
                vals = block[np.triu_indices(ei-si, k=1)]
            else:
                vals = block.flatten()
            mv = vals.mean() if vals.size else 0
            ax.text((sj+ej-1)/2, (si+ei-1)/2, f"{mv:.2f}",
                    ha='center', va='center', fontsize=15, color='k')

    ax.set_title(title)
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation')

    plt.tight_layout()
    return fig


def plot_entropy(vta_ts: dict[str, np.ndarray],
                 entropies: dict[str, list[float]],
                 labels: list[str],
                 out_dir: str):
    """
    Shannon entropy plots:
      1) Time series by ROI (stored in out_dir/entropy-roi/).
      2) Comparative bar-plot (fig is returned)
    """
    # Temporal timeseries
    ts_dir = os.path.join(out_dir, 'entropy-rois')
    os.makedirs(ts_dir, exist_ok=True)

    for i, roi in enumerate(labels):
        fig, ax = plt.subplots(figsize=(8, 2))
        for cond, ts in vta_ts.items():
            ax.plot(ts[i], label=f"{cond} (H={entropies[cond][i]:.2f})")
        ax.set_title(roi)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fn = roi.replace('/', '_').replace(' ', '_') + '.png'
        fig.savefig(os.path.join(ts_dir, fn), dpi=300)
        plt.close(fig)
    # Comparative barplot
    x = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x - width, entropies['pre'],      width, label='Pre-DBS')
    ax.bar(x,         entropies['post_on'],  width, label='Post-DBS ON')
    ax.bar(x + width, entropies['post_off'], width, label='Post-DBS OFF')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel('Shannon Entropy')
    ax.set_title('Entropy per ROI (VTA)')
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig


def plot_positive_mean_per_roi(conds, roi_labels, keep_idx, vta_idx, vta_labels) -> plt.Figure:

    """
    Plots the positive average connectivity for each ROI VTA in the three conditions (pre, post_on, post_off).
    """
    data_pos = {}
    for cond, M in conds.items():
        vals = []
        for i in vta_idx:
            row = M[i, :]
            row = np.delete(row, i)
            pos = row[row > 0]
            vals.append(pos.mean() if pos.size else 0)
        data_pos[cond] = np.array(vals)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(vta_idx))
    ax.plot(x, data_pos['pre'],     '-o', label='Pre-DBS')
    ax.plot(x, data_pos['post_on'], '-s', label='Post-DBS ON')
    ax.plot(x, data_pos['post_off'],'-^', label='Post-DBS OFF')

    ax.set_xticks(x)
    ax.set_xticklabels(vta_labels,
                       rotation=90, fontsize=8)
    ax.set_ylabel('Positive mean')
    ax.set_xlabel('ROIs VTA')
    ax.set_title('Positive mean connectivity per ROI (VTA)')
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    return fig


def plot_negative_mean_per_roi(conds, roi_labels, keep_idx, vta_idx, vta_labels) -> plt.Figure:
    """
    Plots the average negativ connectivity for each ROI VTA in the three conditions (pre, post_on, post_off).
    """
    data_neg = {}
    for cond, M in conds.items():
        vals = []
        for i in vta_idx:
            row = M[i, :]
            row = np.delete(row, i)
            neg = row[row < 0]
            vals.append(neg.mean() if neg.size else 0)
        data_neg[cond] = np.array(vals)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(vta_idx))
    ax.plot(x, data_neg['pre'],     '-o', label='Pre-DBS')
    ax.plot(x, data_neg['post_on'], '-s', label='Post-DBS ON')
    ax.plot(x, data_neg['post_off'],'-^', label='Post-DBS OFF')

    ax.set_xticks(x)
    ax.set_xticklabels(vta_labels,
                       rotation=90, fontsize=8)
    ax.set_ylabel('Negative mean')
    ax.set_xlabel('ROIs VTA')
    ax.set_title('Negative mean connectivity (VTA)')
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    return fig

def plot_absolute_change_matrix(mat1: np.ndarray, mat2: np.ndarray, labels: list | np.ndarray, title: str = "") -> plt.Figure:
    """
    Plot matrix |mat2 - mat1| with the numbers
    """
    diff = np.abs(mat2 - mat1)
    n = diff.shape[0]
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(diff, cmap='viridis', vmin=0, vmax=diff.max(), aspect='equal')

    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    #numeric annotations for each cell
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{diff[i,j]:.2f}",
                    ha='center', va='center', fontsize=10, color='k')

    #axis tags
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    #better visualization
    if isinstance(labels, (list, np.ndarray)) and all(isinstance(x, str) for x in labels):
        ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
    else:
        ax.set_xticklabels(labels, rotation=0, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)

    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|Δ connectivity|")

    fig.tight_layout()
    return fig

