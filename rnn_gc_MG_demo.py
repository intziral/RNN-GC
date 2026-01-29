# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

from options.base_options import BaseOptions
from models.rnn_gc_2 import RNN_GC
from util.util import compare_est_to_true_structure, plot_loss_curves

SEQ_LENGTH = 10
NUM_HIDDEN = 30
NUM_EPOCHS = 100
DATA_DIR = "./datasets/mackey_glass/"


if __name__ == "__main__":

    # =====================
    # PREPARING DATA
    # =====================
    num_sim = len([entry for entry in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, entry))]) // 2

    datasets = []
    structures = []

    for i in range(1, num_sim+1):
    
        dataset_file = DATA_DIR + f"dataset_{i}.csv"
        structure_file = DATA_DIR + f"structure_{i}.csv"
    
        data = pd.read_csv(dataset_file, header=None)
        structure = pd.read_csv(structure_file, header=None)
        structure = np.array(structure)

        datasets.append(data)
        structures.append(structure)

    # =====================
    # TESTING
    # =====================
    save_dir = f"./results/mackey_glass"
    os.makedirs(save_dir, exist_ok=True)

    opt = BaseOptions().parse()

    for i in range(len(datasets)):
        
        matrix = np.zeros((5, 5))
        # for k in range(5):
        # Initialize model
        rnn_gc = RNN_GC(opt,
                        num_hidden = NUM_HIDDEN,
                        num_epochs = NUM_EPOCHS,
                        sequence_length = SEQ_LENGTH)
        
        # Training and causality extraction
        x, y = rnn_gc.load_sequence_data(datasets[i])
        # gc_est = rnn_gc.nue(x, y)
        gc_est, hist_res = rnn_gc.lstm_gc(x, y)
        # matrix += gc_est

        # gc_est = matrix / 5
        # Figures
        plot_loss_curves(hist_res)
        compare_est_to_true_structure(gc_est, structures[i])

        # Save Granger estimations
        np.savetxt(
            fname=f"{save_dir}/gc_est{i+1}.csv",
            X=gc_est
        )
        print("Granger Causality estimation saved.")

   