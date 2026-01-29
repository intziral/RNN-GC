# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt

from models.lorenz import Lorenz96
from util.util import compare_est_to_true_structure
from options.base_options import BaseOptions
from models.rnn_gc_2 import RNN_GC

SEQ_LENGTH = 10
NUM_HIDDEN = 30
NUM_EPOCHS = 100

# Data parameters
p = 5        # number of variables
T = 500      # time series length
F = 8.0      # Lorenz96 forcing constant
num_sim = 1
seed = 42

if __name__ == "__main__":

    # =====================
    # Generate Lorenz96 data
    # =====================

    save_dir = f"./datasets/lorenz96"
    os.makedirs(save_dir, exist_ok=True)

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

        # Save simulated data
        np.savetxt(
            fname=f"{save_dir}/lorenz96_data_r_{i+1}.csv",
            X=data
        )
        np.savetxt(
            fname=f"{save_dir}/lorenz96_struct_r_{i+1}.csv",
            X=a_true
        )

    print("Simulated data saved.")


    # =====================
    # TESTING
    # =====================

    results_dir = f"./results/lorenz96"
    os.makedirs(results_dir, exist_ok=True)

    opt = BaseOptions().parse()

    for i in range(len(datasets)):

        rnn_gc = RNN_GC(opt,
                        num_hidden = NUM_HIDDEN,
                        num_epochs = NUM_EPOCHS,
                        sequence_length = SEQ_LENGTH)
        x, y = rnn_gc.load_sequence_data(datasets[i])
        gc_est = rnn_gc.nue(x, y)

        compare_est_to_true_structure(gc_est, structures[i])

        # Save Granger estimations
        np.savetxt(
            fname=f"{results_dir}/gc_est{i+1}.csv",
            X=gc_est
        )
        print("Granger Causality estimation saved.")
        
