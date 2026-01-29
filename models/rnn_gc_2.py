# -*- coding: utf-8 -*-

from __future__ import print_function, division
import datetime
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn import preprocessing

from models.custom_lstm import CustomLSTM
from util.util import batch_sequence


class RNN_GC:
    def __init__(self, opt, num_hidden, num_epochs, sequence_length):
        self.sequence_length = sequence_length
        self.batch_size = opt.batch_size
        self.num_shift = opt.num_shift
        self.num_hidden = num_hidden
        self.num_epoch = num_epochs
        self.theta = opt.theta
        self.data_length = opt.data_length
        self.weight_decay = opt.weight_decay
    
    def load_sequence_data(self, simulation_data):
        """Loads and preprocesses data"""

        # Normalize
        simulation_data = np.array(simulation_data)
        self.num_channel = simulation_data.shape[1]

        # Standardize and scale to [0, 1]
        scaler = preprocessing.StandardScaler().fit(simulation_data)
        scaled_data = scaler.transform(simulation_data)
        data = preprocessing.MinMaxScaler().fit_transform(scaled_data)

        # # Plot data
        # fig, axarr = plt.subplots(1, 2, figsize=(16, 5))
        # axarr[0].plot(data)
        # axarr[0].set_xlabel('T')
        # axarr[0].set_title('Entire time series')
        # axarr[1].plot(data[:50])
        # axarr[1].set_xlabel('T')
        # axarr[1].set_title('First 50 time points')
        # plt.tight_layout()
        # plt.show()

        x, y = batch_sequence(data, num_shift=self.num_shift, sequence_length=self.sequence_length)
        return x, y

    def lstm_gc(self, x, y):
        # Init GC matrix
        granger_matrix = np.zeros((self.num_channel, self.num_channel))

        start_time = datetime.datetime.now()

        # Train model
        lstm = CustomLSTM(num_hidden=self.num_hidden, num_channel=self.num_channel)
        hist_res = lstm.fit(x, y, batch_size=self.batch_size, epochs=self.num_epoch)

        # Calculate residual variance for full input
        error_full = y - lstm.predict(x)
        var_full = np.var(error_full, axis=0)

        # Calculate GC for each channel
        for k in range(self.num_channel):
            # Remove k channel from input
            x_no_k = x.copy()
            x_no_k[:, :, k] = 0.0

            # Calculate residual variance of prediction without channel k for each output
            error_no_k = y - lstm.predict(x_no_k)
            var_no_k = np.var(error_no_k, axis=0)

            granger_matrix[k,:] = np.log(var_no_k / var_full)
        
        np.fill_diagonal(granger_matrix, 0)

        # Permutation test
        B=100
        alpha=0.05
        GC_null = np.zeros((B, self.num_channel, self.num_channel))

        for b in range(B):
            print(f"Permutation {b+1}/{B}")

            for k in range(self.num_channel):
                x_perm = x.copy()
                for n in range(x.shape[0]):      # per sample
                    np.random.shuffle(x_perm[n, :, k])

                y_hat_perm = lstm.predict(x_perm)
                var_perm = np.var(y - y_hat_perm, axis=0)

                GC_null[b, k, :] = np.log(var_perm / var_full)

        pvals = np.mean(GC_null >= granger_matrix[None, :, :], axis=0)  # p-values
        granger_matrix = (pvals <= alpha).astype(int)                   # Threshold GC to 0/1 based on significance

        print(f"Training completed in {datetime.timedelta(seconds=int((datetime.datetime.now()-start_time).total_seconds()))}")

        return granger_matrix, hist_res
    
    def nue(self, x, y):
        """Computes Granger causality using RNN-based (LSTM) prediction errors.
        Returns:
            granger_matrix: A matrix where entry (j, k) indicates the causal influence of variable j on variable k."""

        # Initialize the Granger causality matrix
        granger_matrix = np.zeros((self.num_channel, self.num_channel))

        # Denominator for later normalization (variance of prediction errors)
        var_denominator = np.zeros((1, self.num_channel))

        all_candidate = []
        error_model = []
        error_all = []
        hist_result = []

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

            # Retrain LSTM with final input set to compute variance of residuals
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
