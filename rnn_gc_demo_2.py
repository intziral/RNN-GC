# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import os
import matplotlib.pyplot as plt


from util.util import plot_final_average_results, plot_save_intermediate_results
from options.base_options import BaseOptions
from models.rnn_gc_2 import RNN_GC


def test(opt, num_hidden):
    """Runs a test using the RNN_GC model and saves intermediate results."""
    rnn_gc = RNN_GC(opt, num_hidden)
    matrix = rnn_gc.nue()

    return matrix


if __name__ == "__main__":
    num_test = 10
    opt = BaseOptions().parse()

    # Run tests and accumulate results
    matrix = test(opt, num_hidden=30)

    # Compute averages
    matrix /= num_test

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(matrix)
    plt.show()