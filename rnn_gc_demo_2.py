# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import os
import matplotlib.pyplot as plt

from models.lorenz import Lorenz96
from util.util import plot_final_average_results, plot_save_intermediate_results
from options.base_options import BaseOptions
from models.rnn_gc_2 import RNN_GC


HIDDEN_LAYER_NUM = 10

# Data parameters
p = 5          # number of variables
T = 500         # time series length
F = 10.0        # Lorenz96 forcing constant
num_sim = 5     # number of simulations
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

    save_dir = f"./datasets/experiment_data/lorenz96"
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

    # Run tests and accumulate results
    for i in range(len(datasets)):
        rnn_gc = RNN_GC(opt, HIDDEN_LAYER_NUM)
        x, y = rnn_gc.load_data(datasets[i])
        matrix = rnn_gc.nue(x, y)

        # # Compute averages
        # matrix /= num_test
        print(matrix)
    
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(matrix)
        plt.show()