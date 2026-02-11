# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import os
import datetime

from util.util import plot_final_average_results, plot_save_intermediate_results
from options.base_options import BaseOptions
from models.rnn_gc import RNN_GC


def test(opt, num_hidden, sequence_length, mode, i):
    """Runs a test using the RNN_GC model and saves intermediate results."""
    rnn_gc = RNN_GC(opt, num_hidden, sequence_length, mode)
    matrix = rnn_gc.newe(nue=False, perm_testing=True)
    output_dir = "./inter_results"
    os.makedirs(output_dir, exist_ok=True)

    plot_save_intermediate_results(matrix, mode, i, output_dir)
    return matrix


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    num_test = 1
    opt = BaseOptions().parse()

    # Initialize matrices
    linear = np.zeros((5, 5))
    nonlinear = np.zeros((5, 5))
    nonlinear_lag = np.zeros((5, 5))

    # Run tests and accumulate results
    for i in range(num_test):
        linear += test(opt, num_hidden=30, sequence_length=20, mode="linear", i=i)
        nonlinear += test(opt, num_hidden=30, sequence_length=20, mode="nonlinear", i=i)
        nonlinear_lag += test(opt, num_hidden=30, sequence_length=20, mode="nonlinearlag", i=i)

    # Compute averages
    linear /= num_test
    nonlinear /= num_test
    nonlinear_lag /= num_test

    print(f"Testing completed in {datetime.timedelta(seconds=int((datetime.datetime.now()-start_time).total_seconds()))}")

    # Plot final results
    plot_final_average_results(linear, nonlinear, nonlinear_lag, save_dir="./", index=1)