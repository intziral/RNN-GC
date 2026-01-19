# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from util.util import plot_dataset_structure
from options.base_options import BaseOptions
from models.rnn_gc_2 import RNN_GC

NUM_HIDDEN = 30
NUM_EPOCHS = 50
DATA_DIR = "./datasets/mackey_glass/"


if __name__ == "__main__":

    opt = BaseOptions().parse()

    num_pairs = len([entry for entry in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, entry))]) // 2

    datasets = []
    structures = []

    for i in range(1, num_pairs+1):
    
        dataset_file = DATA_DIR + f"dataset_{i}.csv"
        structure_file = DATA_DIR + f"structure_{i}.csv"
    
        data = pd.read_csv(dataset_file, header=None)
        structure = pd.read_csv(structure_file, header=None)

        datasets.append(data)
        structures.append(structure)
        

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