import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = "EMXJ21_0406-013.mat"

with h5py.File(filename, 'r') as f:
    raw = f['datBRLSreg'][:]  # shape (2048, 50, 50) = (time, y, x)

# Reorder to (y, x, time) to match your surrogate data convention
real_data = np.transpose(raw, (1, 2, 0))  # now shape (50, 50, 2048)

print(real_data.shape)  # should print (50, 50, 2048)

# plot a single frame (snapshot of activity across the whole cortex at one moment)
plt.imshow(real_data[:, :, 500], cmap='viridis')
plt.title("Frame 500")
plt.colorbar()
plt.show()

# plot the time trace at one pixel, e.g. pixel (25,25)
plt.figure()
plt.plot(real_data[25, 25, :])
plt.title("Pixel (25,25) over time")
plt.xlabel("Frame")
plt.ylabel("Signal")
plt.show()