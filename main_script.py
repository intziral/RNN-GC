import numpy as np
import pandas as pd
import torch
import os

from options.base_options import BaseOptions
from models.rnn_gc_2 import RNN_GC

SEQ_LENGTH = 30
NUM_HIDDEN = 15
NUM_EPOCHS = 100
THETA = 0.09
DATA_DIR = "./datasets/mackey_glass"

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(0)
torch.manual_seed(0)

opt = BaseOptions().parse()
j = opt.target
dataset_index = opt.dataset_index

print(f"Target variable index: {j} and dataset index: {dataset_index}")

dataset_file = f"{DATA_DIR}/dataset_{dataset_index}.csv"
data = pd.read_csv(dataset_file, header=None)

# Initialize model
rnn_gc = RNN_GC(opt,
                num_hidden = NUM_HIDDEN,
                num_epochs = NUM_EPOCHS,
                sequence_length = SEQ_LENGTH,
                theta = THETA)

# Train and estimate GC on each network
x, y = rnn_gc.load_sequence_data(data)

gc_col = rnn_gc.nue_single_target(x, y, j,
                                  nue=True,
                                  permutation_testing=True,
                                  device=device)

os.makedirs("./results/mackey_glass", exist_ok=True)
np.save(f"./results/mackey_glass/gc_col_{dataset_index}_{j}.npy", gc_col)

print(f"Finished processing dataset {dataset_index}, target {j}. GC values: {gc_col}")