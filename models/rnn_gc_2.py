# -*- coding: utf-8 -*-

from __future__ import print_function, division
import datetime

import torch
import numpy as np
from sklearn import preprocessing

#from models.custom_lstm import CustomLSTM
from models.torch_lstm import CustomLSTM
from util.util import batch_sequence


class RNN_GC:
    def __init__(self, opt, num_hidden, num_epochs, sequence_length, theta):
        self.sequence_length = sequence_length
        self.batch_size = opt.batch_size
        self.num_shift = opt.num_shift
        self.num_hidden = num_hidden
        self.num_epoch = num_epochs
        self.theta = theta
        self.data_length = opt.data_length
        self.weight_decay = opt.weight_decay
    
    def load_sequence_data(self, simulation_data):
        """Loads and preprocesses data"""

        # Normalize
        simulation_data = np.array(simulation_data)
        self.num_channel = simulation_data.shape[1]

        # Standardize and scale to [0, 1]
        scaler = preprocessing.StandardScaler().fit(simulation_data)
        data = scaler.transform(simulation_data)
        # data = preprocessing.MinMaxScaler().fit_transform(data)

        x, y = batch_sequence(data, num_shift=self.num_shift, sequence_length=self.sequence_length)

        return x, y
    
    def nue(self, x, y, nue=True, permutation_testing=False):
        """Computes Granger causality using RNN-based (LSTM) prediction errors.
        Returns:
            granger_matrix: A matrix where entry (j, k) indicates the causal influence of variable j on variable k."""

        # Initialize the Granger causality matrix
        granger_matrix = np.zeros((self.num_channel, self.num_channel))

        if nue:
            error_model = []
            error_all = []
        if permutation_testing:
            num_perms = 50
            alpha = 0.05
            p_values = np.ones((self.num_channel, self.num_channel))

        start_time = datetime.datetime.now()
        
        # Loop over each channel as target (i.e., trying to predict y[:, k])
        for j in range(self.num_channel):
            target_j = y[:, j].reshape(-1, 1)  # Reshape target to 2D (samples x 1)
            
            if nue:
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
                        lstm.fit(tmp_x, target_j, batch_size=self.batch_size, epochs=self.num_epoch)

                        # Compute prediction error
                        tmp_error = np.mean((lstm.predict(tmp_x) - target_j) ** 2)

                        # Keep the channel that gives lowest error
                        if tmp_error < min_error:
                            min_error = tmp_error
                            min_idx = x_idx

                        # Log the error for this trial
                        error_all.append([j, i, x_idx, tmp_error])

                    # Store improvement between last and current error
                    error_model.append([j, last_error, min_error])

                    # Stop adding inputs if improvement is too small
                    if i > 0 and (abs(last_error - min_error) / last_error < self.theta or last_error < min_error):
                        break

                    # Update input set with the selected channel
                    input_set.append(min_idx)
                    channel_set.remove(min_idx)
                    last_error = min_error
            else:
                input_set = list(range(self.num_channel))  # All possible inputs

            # Train LSTM with final input set to compute variance of residuals
            lstm_full = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
            lstm_full.fit(x[:, :, input_set], target_j, batch_size=self.batch_size, epochs=self.num_epoch)

            # Compute residual variance for normalization
            var_full = np.var(lstm_full.predict(x[:, :, input_set]) - target_j)

            # Evaluate effect of removing each input channel j
            for i in range(self.num_channel):
                
                # If channel i is not in the input set, it cannot be a cause of j, so GC is zero
                if i not in input_set:
                    granger_matrix[i, j] = 0.0
                    continue
                
                # Remove channel i from input set
                idx = input_set.index(i)
                x_no_i = x[:, :, input_set].copy()
                x_no_i[:, :, idx] = 0.0

                # Train LSTM without channel i and compute residual variance
                lstm_no_i = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
                lstm_no_i.fit(x_no_i, target_j, batch_size=self.batch_size, epochs=self.num_epoch)
                var_no_i = np.var(lstm_no_i.predict(x_no_i) - target_j)

                gc_est = np.log(var_no_i / var_full)
                granger_matrix[i, j] = max(gc_est, 0.0)

                # Permutation testing for every i -> j
                if permutation_testing:
                    perm_stats = []

                    for p in range(num_perms):

                        # Permute input channel i by shuffling time indices
                        x_perm_i = x[:, :, input_set].copy()
                        perm_idx = np.random.permutation(x_perm_i.shape[1])
                        x_perm_i[:, :, idx] = x_perm_i[:, perm_idx, idx]

                        # Train and calculate residual variance with permuted channel i
                        lstm_perm_i = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
                        lstm_perm_i.fit(x_perm_i, target_j, batch_size=self.batch_size, epochs=self.num_epoch)
                        var_perm_i = np.var(lstm_perm_i.predict(x_perm_i) - target_j)

                        gc_perm = np.log(var_no_i / var_perm_i)
                        perm_stats.append(gc_perm)

                    # p-value i->j
                    p_values[i, j] = np.mean(np.array(perm_stats) >= gc_est)

            print(f'Training model for output {j + 1} complete.')

        # Set GC to zero for non-significant connections based on permutation testing
        if permutation_testing:
            granger_matrix[p_values >= alpha] = 0.0

        np.fill_diagonal(granger_matrix, 0)

        print(f"Training completed in {datetime.timedelta(seconds=int((datetime.datetime.now()-start_time).total_seconds()))}")

        return granger_matrix
    

    def nue_single_target(self, x, y, j, nue=True, permutation_testing=False, device="cuda"):
        """
        Compute one column (target j) of Granger causality matrix.
        Designed for 1 GPU per job.
        """
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)

        target_j = y[:, j].reshape(-1, 1)

        granger_col = torch.zeros(self.num_channel, device=device)

        if permutation_testing:
            num_perms = 50
            alpha = 0.05
            p_values = torch.ones(self.num_channel, device=device)

        # -------- NUE FEATURE SELECTION --------
        if nue:
            channel_set = list(range(self.num_channel))
            input_set = []
            last_error = float("inf")

            for _ in range(self.num_channel):
                min_error = float("inf")
                min_idx = None

                for x_idx in channel_set:
                    tmp_set = input_set + [x_idx]
                    tmp_x = x[:, :, tmp_set]

                    lstm = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(tmp_set), weight_decay=self.weight_decay)
                    lstm.fit(tmp_x, target_j, batch_size=self.batch_size, epochs=self.num_epoch, device=device)

                    with torch.no_grad():
                        pred = lstm.predict(tmp_x, device=device)
                        tmp_error = torch.mean((pred - target_j) ** 2).item()

                    if tmp_error < min_error:
                        min_error = tmp_error
                        min_idx = x_idx
                
                if len(input_set) > 0 and (abs(last_error - min_error) / (last_error + 1e-8) < self.theta or last_error < min_error):
                    break

                input_set.append(min_idx)
                channel_set.remove(min_idx)
                last_error = min_error
        else:
            input_set = list(range(self.num_channel))

        # -------- FULL MODEL --------
        lstm_full = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
        lstm_full.fit(x[:, :, input_set], target_j, batch_size=self.batch_size, epochs=self.num_epoch, device=device)

        with torch.no_grad():
            res_full = lstm_full.predict(x[:, :, input_set], device=device) - target_j
            var_full = torch.var(res_full)

        # -------- LEAVE-ONE-OUT + PERM TEST --------
        for i in range(self.num_channel):

            if i not in input_set:
                granger_col[i] = 0.0
                continue

            idx = input_set.index(i)

            x_no_i = x[:, :, input_set].clone()
            x_no_i[:, :, idx] = 0.0

            lstm_rest = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
            lstm_rest.fit(x_no_i, target_j, batch_size=self.batch_size, epochs=self.num_epoch, device=device)

            with torch.no_grad():
                res_rest = lstm_rest.predict(x_no_i, device=device) - target_j
                var_rest = torch.var(res_rest)

            gc_obs = torch.log(var_rest / var_full)
            gc_obs = torch.clamp(gc_obs, min=0.0)
            granger_col[i] = gc_obs

            # -------- PERMUTATION TEST --------
            if permutation_testing:
                perm_stats = []

                for _ in range(num_perms):

                    x_perm = x[:, :, input_set].clone()

                    perm_idx = torch.randperm(x_perm.shape[1], device=device)
                    x_perm[:, :, idx] = x_perm[:, perm_idx, idx]

                    lstm_perm = CustomLSTM(num_hidden=self.num_hidden, num_channel=len(input_set))
                    lstm_perm.fit(x_perm, target_j, batch_size=self.batch_size, epochs=self.num_epoch, device=device)

                    with torch.no_grad():
                        res_perm = lstm_perm.predict(x_perm, device=device) - target_j
                        var_perm = torch.var(res_perm)

                    gc_perm = torch.log(var_rest / var_perm)
                    perm_stats.append(gc_perm)

                perm_stats = torch.stack(perm_stats)
                p_val = torch.mean((perm_stats >= gc_obs).float())
                p_values[i] = p_val

        if permutation_testing:
            granger_col[p_values >= alpha] = 0.0

        return granger_col.detach().cpu().numpy()
