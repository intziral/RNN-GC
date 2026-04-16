import numpy as np
import matplotlib.pyplot as plt
import os

import argparse

parser = argparse.ArgumentParser(description='Process GC results for Mackey-Glass datasets.')
parser.add_argument('--dataset_index', type=int, required=True, help='Index of the dataset to process (e.g., 1, 2, etc.)')
args = parser.parse_args()

dataset_index = args.dataset_index

save_dir = "./results/mackey_glass"

p = 5
gc = np.zeros((p, p))

for j in range(p):
    gc[:, j] = np.load(os.path.join(save_dir, f"gc_col_{dataset_index}_{j}.npy"))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
cax = ax.matshow(gc)
fig.colorbar(cax)

plt.savefig(os.path.join(save_dir, f'granger_matrix_{dataset_index}.png'))
np.save(os.path.join(save_dir, f'granger_matrix_{dataset_index}.npy'), gc)