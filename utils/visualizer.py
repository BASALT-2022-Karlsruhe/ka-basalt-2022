import numpy as np
import matplotlib.pyplot as plt

def visualize_loss(log_file_path, output_path="avrg_loss_findcave_full_dataset.png"):
    data = np.loadtxt(log_file_path, comments="Save", delimiter=" ", skiprows=4, usecols=8)

    plt.plot(100 * (np.arange(data.shape[0] - 1) + 1), data[1:])
    plt.yscale("log")
    plt.xlabel("num batches")
    plt.ylabel("loss")
    plt.savefig(output_path)