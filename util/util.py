import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


def evaluate_gc(gc_estimated, gc_true, threshold=0.0, plot_roc=True):
    """
    Evaluate estimated Granger causality against binary ground truth.

    Parameters
    ----------
    gc_estimated : np.ndarray (N x N)
        Continuous-valued estimated GC matrix (positive values)
    gc_true : np.ndarray (N x N)
        Binary ground-truth adjacency (0/1), diagonal should be 0
    threshold : float
        Threshold to binarize gc_estimated for accuracy/precision/recall
    plot_roc : bool
        Whether to plot ROC curve

    Returns
    -------
    metrics : dict
        Dictionary containing 'accuracy', 'precision', 'recall', 'auc'
    """

    gc_estimated = np.asarray(gc_estimated)
    gc_true = np.asarray(gc_true)

    assert gc_estimated.shape == gc_true.shape, "Shape mismatch"

    # Ignore diagonal
    mask = ~np.eye(gc_true.shape[0], dtype=bool)
    y_true = gc_true[mask].ravel()
    y_score = gc_estimated[mask].ravel()

    # Ensure binary ground truth
    y_true = (y_true > 0).astype(int)

    # Binarize estimated GC for accuracy, precision, recall
    y_pred = (y_score > threshold).astype(int)

    # Compute metrics
    accuracy = np.mean(y_pred == y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }

    # Optional: ROC curve
    if plot_roc:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve – Granger Causality")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return metrics


def plot_dataset_structure(X, A, index):
    T, N = X.shape
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series
    ax_ts = axes[0]
    for i in range(N):
        ax_ts.plot(X[:, i], label=f"Var {i+1}")
    ax_ts.set_title(f"Dataset {index} - Time Series")
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Value")
    ax_ts.legend(loc='upper right', fontsize=8)
    
    # Causal adjacency heatmap
    ax_causal = axes[1]
    im = ax_causal.imshow(A, cmap='viridis', interpolation='none')
    ax_causal.set_title(f"Structure {index} - Causal Adjacency")
    ax_causal.set_xlabel("Target Variable")
    ax_causal.set_ylabel("Source Variable")
    ax_causal.set_xticks(range(N))
    ax_causal.set_yticks(range(N))
    fig.colorbar(im, ax=ax_causal, fraction=0.046, pad=0.04, label='Causality strength')
    
    plt.tight_layout()
    plt.show()

def batch_sequence(x, sequence_length=20, num_shift=1):
    num_points = x.shape[0]
    inputs = []
    targets = []
    # for p in np.arange(0, num_points, max(num_shift, sequence_length // 5)):
    for p in np.arange(0, num_points, num_shift):
        # prepare inputs (we're sweeping from left to right in steps sequence_length long)
        if p + sequence_length + num_shift >= num_points:
            break

        inputs.append(x[p: p + sequence_length, :])
        targets.append(x[p + sequence_length, :])
    inputs = np.array(inputs)
    targets = np.array(targets)
    idx = np.random.permutation(np.arange(inputs.shape[0]))
    inputs = inputs[idx]
    targets = targets[idx]

    return inputs, targets


def plot_final_average_results(linear, nonlinear, nonlinear_lag, save_dir, index):
    ground_truth = np.zeros((5, 5))
    ground_truth[0, 1] = 1
    ground_truth[0, 2] = 1
    ground_truth[0, 3] = 1
    ground_truth[3, 4] = 1
    ground_truth[4, 3] = 1

    plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(142)
    ax1.matshow(linear)
    ax1.axis('off')
    ax1.set_title('Linear')

    ax2 = plt.subplot(143)
    ax2.matshow(nonlinear)
    ax2.axis('off')
    ax2.set_title('Nonlinear')

    ax3 = plt.subplot(144)
    ax3.matshow(nonlinear_lag)
    ax3.axis('off')
    ax3.set_title('Nonlinear lag')

    ax4 = plt.subplot(141)
    ax4.matshow(ground_truth)
    ax4.axis('off')
    ax4.set_title('Ground Truth')

    plt.savefig(os.path.join(save_dir, str(index).rjust(2, '0') + 'all.png'))


def plot_save_intermediate_results(matrix, mode, index, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(matrix)
    plt.savefig(os.path.join(save_dir, mode + str(index).rjust(2, '0') + '.png'))
    np.savetxt(os.path.join(save_dir, mode + str(index).rjust(2, '0') + '.txt'), matrix)
