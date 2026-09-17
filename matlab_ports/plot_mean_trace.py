"""
Python port of plotmeantrace_regressed.m
Plots (optionally trial-averaged) pixel traces from EMXJ21-style .mat trial files.
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

FRATE = 200  # fixed acquisition rate, matches original MATLAB script


def plot_mean_trace(sessidstr, trials, pixels, sesspath):
    """
    sessidstr: session name string, e.g. 'EMXJ21_0406'
    trials: single trial number (int) or list of trial numbers to average
    pixels: 2xN array-like, MATLAB-style 1-indexed (row, col) pairs,
            e.g. [[20, 30, 40], [20, 30, 40]] for pixels (20,20), (30,30), (40,40)
    sesspath: full path to the folder containing the .mat trial files
    """
    if isinstance(trials, int):
        trials = [trials]

    pixels = np.array(pixels)
    n_pixels = pixels.shape[1]

    plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, n_pixels))

    for pidx in range(n_pixels):
        mean_trace = np.zeros(2048)  # matches original's fixed preallocation

        for trial in trials:
            fidx = f"{trial:03d}"
            filename = os.path.join(sesspath, f"{sessidstr}-{fidx}.mat")

            with h5py.File(filename, 'r') as f:
                raw = f['datBRLSreg'][:]  # h5py axis order is reversed vs MATLAB

            img_array = np.transpose(raw, (1, 2, 0))  # back to MATLAB's (Y, X, N)

            # Convert 1-indexed MATLAB coordinates to 0-indexed Python
            row_idx = pixels[0, pidx] - 1
            col_idx = pixels[1, pidx] - 1

            trace = img_array[row_idx, col_idx, :]
            mean_trace[:len(trace)] += trace

        mean_trace /= len(trials)
        ti = np.arange(len(mean_trace)) / FRATE

        # regressed data is inverted compared to others, per original script
        plt.plot(ti, -mean_trace, color=colors[pidx], linewidth=1.5,
                  label=f"Pixel {pixels[0, pidx]}x {pixels[1, pidx]}y")

    plt.legend()
    plt.title(f"Pixel traces for {sessidstr}, trials {trials}")
    plt.xlabel("time (s)")
    plt.ylabel("AU")
    plt.show()