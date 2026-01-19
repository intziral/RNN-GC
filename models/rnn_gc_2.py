# -*- coding: utf-8 -*-

from __future__ import print_function, division
import datetime
import pandas as pd

import numpy as np
from sklearn import preprocessing

from models.custom_lstm import CustomLSTM
from util.util import batch_sequence


class RNN_GC:
    def __init__(self, opt, num_hidden, num_epochs):
        self.sequence_length = opt.sequence_length
        self.batch_size = opt.batch_size
        self.num_shift = opt.num_shift
        self.num_hidden = num_hidden
        self.num_epoch = num_epochs
        self.theta = opt.theta
        self.data_length = opt.data_length
        self.weight_decay = opt.weight_decay
    
    def load_data(self, simulation_data):
        """Loads and preprocesses data"""

        # Normalize
        simulation_data = np.array(simulation_data)
        self.num_channel = simulation_data.shape[1]

        # Standardize and scale to [0, 1]
        scaler = preprocessing.StandardScaler().fit(simulation_data)
        scaled_data = scaler.transform(simulation_data)
        data = preprocessing.MinMaxScaler().fit_transform(scaled_data)

        x, y = batch_sequence(data, num_shift=self.num_shift, sequence_length=self.sequence_length)
        return x, y
    
    def load_data_csv(self, simulation_data, data_path):
        """Loads and preprocesses data from .csv file."""
        simulation_data = pd.read_csv(data_path)

        # Normalize
        simulation_data = np.array(simulation_data)
        self.num_channel = simulation_data.shape[1]

        # Standardize and scale to [0, 1]
        scaler = preprocessing.StandardScaler().fit(simulation_data)
        scaled_data = scaler.transform(simulation_data)
        data = preprocessing.MinMaxScaler().fit_transform(scaled_data)

        x, y = batch_sequence(data, num_shift=self.num_shift, sequence_length=self.sequence_length)
        return x, y

    def nue(self, x, y):
        """Computes Granger causality using RNN-based (LSTM) prediction errors.
        Returns:
            granger_matrix: A matrix where entry (j, k) indicates the causal influence of variable j on variable k."""

        # Initialize the Granger causality matrix
        granger_matrix = np.zeros((self.num_channel, self.num_channel))

        # Denominator for later normalization (variance of prediction errors)
        var_denominator = np.zeros((1, self.num_channel))

        # Tracking structures
        all_candidate = []   # Stores selected input channels for each target
        error_model = []     # Stores error improvements per target
        error_all = []       # Stores all trial errors
        hist_result = []     # Stores training history

        # Start measuring execution time
        start_time = datetime.datetime.now()

        # Loop over each channel as target (i.e., trying to predict y[:, k])
        for k in range(self.num_channel):
            tmp_y = y[:, k].reshape(-1, 1)  # Reshape target to 2D (samples x 1)

            channel_set = list(range(self.num_channel))  # All possible inputs
            input_set = []
            last_error = 0  # Initialize prediction error tracker

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
                    tmp_x = x[:, :, input_set].copy()
                    tmp_x[:, :, input_set.index(j)] = 0
                    granger_matrix[j, k] = np.var(lstm.predict(tmp_x) - tmp_y, axis=0)

            print(f'Training model for output {k + 1} complete.')

        granger_matrix /= var_denominator
        np.fill_diagonal(granger_matrix, 1)
        granger_matrix[granger_matrix < 1] = 1
        granger_matrix = np.log(granger_matrix)

        print(f"Training completed in {datetime.timedelta(seconds=int((datetime.datetime.now()-start_time).total_seconds()))}")
        return granger_matrix
