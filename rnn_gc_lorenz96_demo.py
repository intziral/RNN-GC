# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

from models.lorenz import Lorenz96
from util.util import plot_final_average_results, plot_save_intermediate_results
from options.base_options import BaseOptions
from models.rnn_gc import RNN_GC

NUM_HIDDEN = 30
NUM_EPOCHS = 100

# Data parameters
p = 20        # number of variables
T = 500         # time series length
F = 8.0       # Lorenz96 forcing constant
num_sim = 1    # number of 6 
seed = 42

if __name__ == "__main__":

    opt = BaseOptions().parse()

    # =====================
    # Generate Lorenz96 data
    # =====================

    datasets = []
    structures = []

    print(f"Running {num_sim} Lorenz96 simulations...")
    print(f"p = {p}, F = {F}, T = {T}")

    for i in range(num_sim):
        lorenz_system = Lorenz96(dim=p, force=F, seed=seed + i)
        data = lorenz_system.create_data(num_sim=1, t_len=T)[0]

        # Standardize each variable
        for j in range(p):
            data[:, j] = (data[:, j] - np.mean(data[:, j])) / np.std(data[:, j])

        a_true = lorenz_system.get_causal_structure()

        datasets.append(data)
        structures.append(a_true)

    # =====================
    # Save simulated data
    # =====================

    save_dir = f"./datasets/lorenz96"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_sim):
        np.savetxt(
            fname=f"{save_dir}/lorenz96_data_r_{i}.csv",
            X=datasets[i]
        )
        np.savetxt(
            fname=f"{save_dir}/lorenz96_struct_r_{i}.csv",
            X=structures[i]
        )

    print("Simulated data saved.")


    # =====================
    # TESTING
    # =====================
    matrices = []
    accuracies = []

    for i in range(len(datasets)):

        rnn_gc = RNN_GC(opt,
                        num_hidden = NUM_HIDDEN,
                        num_epochs = NUM_EPOCHS)
        x, y = rnn_gc.load_data(datasets[i])
        matrix = rnn_gc.nue(x, y)
        matrices.append(matrix)

        accuracy = np.mean(matrix==structures[i])
        accuracies.append(accuracy)
        print('Accuracy = %.2f%%' % ( 100 * accuracy))
        

    # =====================
    # PLOTTING
    # =====================
    for i, (gc_matrix, true_matrix) in enumerate(zip(matrices, structures)):
        size = gc_matrix.shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        # ---- Estimated Granger causality ----
        im0 = axes[0].imshow(gc_matrix, cmap="viridis")
        axes[0].set_title(f"Estimated GC (run {i+1})")
        axes[0].set_xticks(range(size))
        axes[0].set_yticks(range(size))
        axes[0].set_xticklabels(range(1, size + 1))
        axes[0].set_yticklabels(range(1, size + 1))
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        # ---- True causal structure ----
        im1 = axes[1].imshow(true_matrix, cmap="viridis")
        axes[1].set_title("True structure")
        axes[1].set_xticks(range(size))
        axes[1].set_yticks(range(size))
        axes[1].set_xticklabels(range(1, size + 1))
        axes[1].set_yticklabels(range(1, size + 1))
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

        plt.tight_layout()
        plt.show()