import numpy as np
import matplotlib.pyplot as plt

import os

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
    
def batch_sequence(x, sequence_length=10, num_shift=1):
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
