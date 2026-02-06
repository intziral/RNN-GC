# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

from options.base_options import BaseOptions
from models.rnn_gc_2 import RNN_GC
from util.util import compare_est_to_true_structure, stability_based_thresholding

SEQ_LENGTH = 20
NUM_HIDDEN = 30
NUM_EPOCHS = 50
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
        
        # Initialize model
        rnn_gc = RNN_GC(opt,
                        num_hidden = NUM_HIDDEN,
                        num_epochs = NUM_EPOCHS,
                        sequence_length = SEQ_LENGTH)
        
        # Train and estimate GC on each network
        x, y = rnn_gc.load_sequence_data(datasets[i])
        gc_est = rnn_gc.lstm_gc(x, y)

        # # Train and estimate GC on time-reversed data for each network
        # reversed_data = np.flip(np.array(datasets[i]), axis=0) 
        # x_r, y_r = rnn_gc.load_sequence_data(reversed_data)
        # gc_est_r = rnn_gc.lstm_gc(x_r, y_r)
        # gc_est_r = np.transpose(gc_est_r)   # A causal edge j → i in reversed time corresponds to i → j in forward time

        # compare_est_to_true_structure(gc_est, gc_est_r)

        # # gc_est = stability_based_thresholding(gc_est, gc_est_r)
        # gc_est_final, best_thr, best_bacc = threshold_gc_by_time_reversal(
        #                             gc_est,
        #                             gc_est_r,
        #                             n_thresholds=200)

        # print("Best threshold:", best_thr)
        # print("Best balanced accuracy:", best_bacc)

        compare_est_to_true_structure(gc_est, structures[i])

        # Save Granger estimations
        np.savetxt(
            fname=f"{save_dir}/gc_est{i+1}.csv",
            X=gc_est
        )
        print("Granger Causality estimation saved.")

   