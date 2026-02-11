import numpy as np
import matplotlib.pyplot as plt
import os

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

def compare_est_to_true_structure(gc_matrix, true_matrix):
    
    size = gc_matrix.shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Estimated Granger causality
    im0 = axes[0].imshow(gc_matrix, cmap="viridis")
    axes[0].set_title(f"Estimated GC")
    axes[0].set_xticks(range(size))
    axes[0].set_yticks(range(size))
    axes[0].set_xticklabels(range(1, size + 1))
    axes[0].set_yticklabels(range(1, size + 1))
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    # True causal structure
    im2 = axes[1].imshow(true_matrix, cmap="viridis")
    axes[1].set_title("True structure")
    axes[1].set_xticks(range(size))
    axes[1].set_yticks(range(size))
    axes[1].set_xticklabels(range(1, size + 1))
    axes[1].set_yticklabels(range(1, size + 1))
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.show()

def compare_sims(matrices):
    num_sim = len(matrices)

    fig, axes = plt.subplots(2, num_sim, figsize=(4.5 * num_sim, 9), constrained_layout=True)

    # Ensure axes is always 2D (important if num_sim == 1)
    if num_sim == 1:
        axes = axes[:, None]

    for i in range(num_sim):
        gc_matrix, true_matrix = matrices[i]
        size = gc_matrix.shape[0]

        # ---------- Estimated Granger causality ----------
        im0 = axes[0, i].imshow(gc_matrix, cmap="viridis")
        axes[0, i].set_title(f"Sim {i+1}\nEstimated GC")
        fig.colorbar(im0, ax=axes[0, i])

        # ---------- True causal structure ----------
        im2 = axes[1, i].imshow(true_matrix, cmap="viridis")
        axes[1, i].set_title("True structure")
        fig.colorbar(im2, ax=axes[1, i])

        # ---------- Axis formatting ----------
        for r in range(2):
            axes[r, i].set_xticks(range(size))
            axes[r, i].set_yticks(range(size))
            axes[r, i].set_xticklabels(range(1, size + 1))
            axes[r, i].set_yticklabels(range(1, size + 1))

    plt.show()

def plot_loss_curves(hist_res):
    loss_values = hist_res.history["loss"]
    val_loss_values = hist_res.history.get("val_loss", None)
    
    plt.plot(loss_values, label="Training loss")
    if val_loss_values is not None:
        plt.plot(val_loss_values, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM training loss")
    plt.legend()
    plt.show()

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
