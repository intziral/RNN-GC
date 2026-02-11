# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import copy
import datetime

import numpy as np
from sklearn import preprocessing
import scipy.io as sio
import matplotlib.pyplot as plt

from models.custom_lstm import CustomLSTM
from util.util import batch_sequence

class RNN_GC:
    def __init__(self, opt, num_hidden, sequence_length, mode):
        self.sequence_length = sequence_length
        self.batch_size = opt.batch_size
        self.num_shift = opt.num_shift
        self.num_hidden = num_hidden
        self.num_epoch = opt.num_epoch
        self.theta = opt.theta
        self.data_length = opt.data_length
        self.weight_decay = opt.weight_decay

        self.mode = mode

    def load_sequence_data(self):
        """Loads and preprocesses sequence data from .mat file."""
        simulation_name = f'realization_{self.mode}_{self.data_length}.mat'
        simulation_dir = 'simulation_difflen'
        simulation_path = os.path.join(simulation_dir, simulation_name)
        simulation_data = sio.loadmat(simulation_path)["data"]

        # Transpose and normalize
        simulation_data = np.array(simulation_data).T
        self.num_channel = simulation_data.shape[1]

        # Standardize and scale to [0, 1]
        scaler = preprocessing.StandardScaler().fit(simulation_data)
        scaled_data = scaler.transform(simulation_data)
        data = preprocessing.MinMaxScaler().fit_transform(scaled_data)

        x, y = batch_sequence(data, num_shift=self.num_shift, sequence_length=self.sequence_length)
        return x, y
    
    def nue(self):
        """Computes Granger causality using RNN-based (LSTM) prediction errors.
        Returns:
            granger_matrix: A matrix where entry (j, k) indicates the causal influence of variable j on variable k."""
        x, y = self.load_sequence_data()

        # Initialize the Granger causality matrix
        granger_matrix = np.zeros((self.num_channel, self.num_channel))

        # Denominator for later normalization (variance of prediction errors)
        var_denominator = np.zeros((1, self.num_channel))

        all_candidate = []
        hist_result = []
        error_model = []
        error_all = []

        start_time = datetime.datetime.now()
        
        # Loop over each channel as target (i.e., trying to predict y[:, k])
        for k in range(self.num_channel):
            tmp_y = y[:, k].reshape(-1, 1)  # Reshape target to 2D (samples x 1)
            channel_set = list(range(self.num_channel))  # All possible inputs
            input_set = []
            last_error = 0

            # Step-by-step input channel selection
            for i in range(self.num_channel):
                min_error = float("inf")
                min_idx = None

                # Try adding each remaining channel to the input set
                for x_idx in channel_set:
                    tmp_set = input_set + [x_idx]
                    tmp_x = x[:, :, tmp_set]  # Select current input set

                    # Train LSTM model
                    lstm = CustomLSTM(num_hidden=self.num_hidden,
                                    num_channel=len(tmp_set),
                                    weight_decay=self.weight_decay)
                    lstm.fit(tmp_x, tmp_y, batch_size=self.batch_size, epochs=self.num_epoch)

                    # Compute prediction error
                    tmp_error = np.mean((lstm.predict(tmp_x) - tmp_y) ** 2)

                    # Keep the channel that gives lowest error
                    if tmp_error < min_error:
                        min_error = tmp_error
                        min_idx = x_idx

                    # Log the error for this trial
                    error_all.append([k, i, x_idx, tmp_error])

                # Store improvement between last and current error
                error_model.append([k, last_error, min_error])

                # Stop adding inputs if improvement is too small
                if i > 0 and (abs(last_error - min_error) / last_error < self.theta or last_error < min_error):
                    break

                # Update input set with the selected channel
                input_set.append(min_idx)
                channel_set.remove(min_idx)
                last_error = min_error

            # Store final input set for this output
            all_candidate.append(input_set)

            # Train LSTM with final input set to compute variance of residuals
            lstm = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
            hist_res = lstm.fit(x[:, :, input_set], tmp_y, batch_size=self.batch_size, epochs=self.num_epoch)
            hist_result.append(hist_res)

            # Compute residual variance for normalization
            var_denominator[0, k] = np.var(lstm.predict(x[:, :, input_set]) - tmp_y, axis=0)

            # Evaluate effect of removing each input channel j
            for j in range(self.num_channel):
                if j not in input_set:
                    # If j not in the model, its influence is same as baseline
                    granger_matrix[j, k] = var_denominator[0, k]
                elif len(input_set) == 1:
                    # If only one input, remove it completely
                    tmp_x = x[:, :, k][:, :, np.newaxis]
                    granger_matrix[j, k] = np.var(lstm.predict(tmp_x) - tmp_y, axis=0)
                else:
                    # Otherwise, zero-out channel j in the input
                    tmp_x = x[:, :, input_set]
                    tmp_x[:, :, input_set.index(j)] = 0
                    granger_matrix[j, k] = np.var(lstm.predict(tmp_x) - tmp_y, axis=0)

            print(f'Training model for output {k + 1} complete.')

        granger_matrix /= var_denominator
        np.fill_diagonal(granger_matrix, 1)
        granger_matrix[granger_matrix < 1] = 1
        granger_matrix = np.log(granger_matrix)

        print(f"Training completed in {datetime.timedelta(seconds=int((datetime.datetime.now()-start_time).total_seconds()))}")

        return granger_matrix

    def newe(self, nue=True, perm_testing=False):
        """Computes Granger causality using RNN-based prediction errors."""
        x, y = self.load_sequence_data()

        granger_matrix = np.zeros((self.num_channel, self.num_channel))
        all_candidate = []
        hist_result = []
        
        if (nue):
            error_model = []
            error_all = []
        if (perm_testing):
            num_perms = 50
            alpha = 0.05
            offset_frac = 0.1
            offset = int(offset_frac * self.sequence_length / 2)
            gc_perm = np.zeros((self.num_channel, self.num_channel, num_perms))
            p_values = np.ones((self.num_channel, self.num_channel))

        start_time = datetime.datetime.now()

        for j in range(self.num_channel):
            target_j = y[:, j].reshape(-1, 1)

            if (nue):
                channel_set = list(range(self.num_channel))
                input_set = []
                last_error = 0

                for i in range(self.num_channel):
                    min_error = float("inf")
                    min_idx = None

                    for x_idx in channel_set:
                        tmp_set = input_set + [x_idx]
                        tmp_x = x[:, :, tmp_set]

                        lstm = CustomLSTM(num_hidden=self.num_hidden,
                                        num_channel=len(tmp_set),
                                        weight_decay=self.weight_decay)
                        lstm.fit(tmp_x, target_j, batch_size=self.batch_size, epochs=self.num_epoch)
                        tmp_error = np.mean((lstm.predict(tmp_x) - target_j) ** 2)

                        if tmp_error < min_error:
                            min_error = tmp_error
                            min_idx = x_idx

                        error_all.append([j, i, x_idx, tmp_error])

                    error_model.append([j, last_error, min_error])

                    if i > 0 and (abs(last_error - min_error) / last_error < self.theta or last_error < min_error):
                        break

                    input_set.append(min_idx)
                    channel_set.remove(min_idx)
                    last_error = min_error
                
                all_candidate.append(input_set)

            else:
                input_set = list(range(self.num_channel))  # All possible inputs
            
            lstm = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
            hist_res = lstm.fit(x[:, :, input_set], target_j, batch_size=self.batch_size, epochs=self.num_epoch)
            hist_result.append(hist_res)

            var_full = np.var(lstm.predict(x[:, :, input_set]) - target_j)
            print(f"Full model residual variance {var_full}")

            for i in range(self.num_channel):
                if i not in input_set:
                    granger_matrix[i, j] = 0.0
                    continue
                
                idx = input_set.index(i)
                x_no_i = x[:, :, input_set].copy()
                x_no_i[:, :, idx] = 0.0

                lstm_no_i = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
                lstm_no_i.fit(x_no_i, target_j, batch_size=self.batch_size, epochs=self.num_epoch)
                var_no_i = np.var(lstm_no_i.predict(x_no_i) - target_j)

                gc = np.log(var_no_i / var_full)
                granger_matrix[i, j] = max(gc, 0.0)

                # Permutation testing for every i -> j
                if (perm_testing):
                    for p in range(num_perms):
                        # Permute input channel i
                        x_perm_i = x[:, :, input_set].copy()
                        x_perm_i[:, :, idx] = np.roll(x_perm_i[:, :, idx], shift=offset, axis=1)

                        # Train and calculate residual variance with permuted channel i
                        lstm_perm_i = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
                        lstm_perm_i.fit(x_perm_i, target_j, batch_size=self.batch_size, epochs=self.num_epoch)
                        var_perm_i = np.var(lstm_perm_i.predict(x_perm_i) - target_j)

                        gc_perm[i, j, p] = np.log(var_no_i / var_perm_i)

                    # p-value i->j
                    p_values[i, j] = np.mean(gc_perm[i, j] >= granger_matrix[i, j])

            print(f'Training model for output {j + 1} complete.')

        if (perm_testing):
            granger_matrix[p_values >= alpha] = 0.0
            
        np.fill_diagonal(granger_matrix, 0.0)

        print(f'Training completed in {(datetime.datetime.now() - start_time).seconds} seconds.')
        
        return granger_matrix
