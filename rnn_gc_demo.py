# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import os

from util.util import plot_final_average_results, plot_save_intermediate_results
from options.base_options import BaseOptions
from models.rnn_gc import RNN_GC


def test(opt, num_hidden, mode, i):
    """Runs a test using the RNN_GC model and saves intermediate results."""
    rnn_gc = RNN_GC(opt, num_hidden, mode)
    matrix = rnn_gc.nue()

    output_dir = "./inter_results"
    os.makedirs(output_dir, exist_ok=True)

    plot_save_intermediate_results(matrix, mode, i, output_dir)
    return matrix


if __name__ == "__main__":
    num_test = 10
    opt = BaseOptions().parse()

    # Initialize matrices
    linear = np.zeros((5, 5))
    nonlinear = np.zeros((5, 5))
    nonlinear_lag = np.zeros((5, 5))

    # Run tests and accumulate results
    for i in range(num_test):
        linear += test(opt, num_hidden=30, mode="linear", i=i)
        nonlinear += test(opt, num_hidden=13, mode="nonlinear", i=i)
        nonlinear_lag += test(opt, num_hidden=30, mode="nonlinearlag", i=i)

    # Compute averages
    linear /= num_test
    nonlinear /= num_test
    nonlinear_lag /= num_test

    # Plot final results
    plot_final_average_results(linear, nonlinear, nonlinear_lag, output_dir="./", plot_idx=1)
