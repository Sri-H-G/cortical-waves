"""
Python port of PlayMiaoDataTrial_v2.m
Synchronized viewer for behavior video(s) and brain imaging data.
Simplified from the original MATLAB uifigure GUI using matplotlib widgets.
"""
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import imageio.v3 as iio

BEHAV_HZ = 100
IMAGING_HZ = 200
RATIO = IMAGING_HZ / BEHAV_HZ


def play_data_trial(video1=None, video2=None, imaging=None, clim_abs=None):
    has_video1 = video1 is not None
    has_video2 = video2 is not None
    has_imaging = imaging is not None

    if not (has_video1 or has_video2 or has_imaging):
        raise ValueError("At least one of video1, video2, or imaging must be provided.")
    if has_video2 and not has_video1:
        raise ValueError("video2 was provided without video1.")
    if has_imaging and clim_abs is None:
        raise ValueError("clim_abs must be provided when imaging is supplied.")

    for path in (video1, video2, imaging):
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    warn_msgs = []
    frames1, frames2 = [], []
    n_behav = 0

    if has_video1:
        frames1 = list(iio.imiter(video1))
        n_behav = len(frames1)
        if has_video2:
            frames2 = list(iio.imiter(video2))
            n_behav2 = len(frames2)
            if n_behav != n_behav2:
                warn_msgs.append(f"Video frame counts differ ({n_behav} vs {n_behav2}); trimming.")
            n_behav = min(n_behav, n_behav2)
            frames1, frames2 = frames1[:n_behav], frames2[:n_behav]

    dat = None
    total_imaging = total_behav = 0

    if has_imaging:
        with h5py.File(imaging, 'r') as f:
            raw = f['datBRLSreg'][:]
        dat = np.transpose(raw, (1, 2, 0))  # (Y, X, N)
        n_imaging = dat.shape[2]

        if has_video1:
            expected = n_behav * RATIO
            if n_imaging != expected:
                warn_msgs.append(f"Expected {expected:.0f} imaging frames, got {n_imaging}; trimming.")
            total_imaging = int(min(n_imaging, expected))
            total_behav = int(total_imaging // RATIO)
            frames1 = frames1[:total_behav]
            if has_video2:
                frames2 = frames2[:total_behav]
        else:
            total_imaging = n_imaging
        dat = dat[:, :, :total_imaging]
    else:
        total_behav = n_behav

    total_master = total_imaging if has_imaging else total_behav
    master_label = "Imaging frame" if has_imaging else "Behavior frame"

    if warn_msgs:
        print("WARNING: " + " | ".join(warn_msgs))

    n_panels = sum([has_video1, has_video2, has_imaging])
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    idx = 0
    im1 = im2 = im_img = None

    if has_video1:
        ax1 = axes[idx]; idx += 1
        im1 = ax1.imshow(frames1[0]); ax1.axis('off')
        ax1.set_title(os.path.basename(video1))
    if has_video2:
        ax2 = axes[idx]; idx += 1
        im2 = ax2.imshow(frames2[0]); ax2.axis('off')
        ax2.set_title(os.path.basename(video2))
    if has_imaging:
        ax_img = axes[idx]
        im_img = ax_img.imshow(-dat[:, :, 0], cmap='seismic', vmin=-clim_abs, vmax=clim_abs)
        ax_img.axis('off'); ax_img.set_title("Brain imaging")
        fig.colorbar(im_img, ax=ax_img)

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.15, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, master_label, 1, total_master, valinit=1, valstep=1)
    ax_button = plt.axes([0.8, 0.1, 0.1, 0.05])
    button = Button(ax_button, 'Play')
    state = {'playing': False}

    def show_frame(n_master):
        n_master = int(max(1, min(total_master, n_master)))
        if has_imaging and has_video1:
            n_beh = int(max(1, min(total_behav, (n_master - 1) // RATIO + 1)))
        elif has_video1:
            n_beh = n_master
        else:
            n_beh = 0
        if has_video1 and n_beh > 0:
            im1.set_data(frames1[n_beh - 1])
        if has_video2 and n_beh > 0:
            im2.set_data(frames2[n_beh - 1])
        if has_imaging:
            im_img.set_data(-dat[:, :, n_master - 1])
        fig.canvas.draw_idle()

    slider.on_changed(lambda val: show_frame(int(val)))

    def on_play(_event):
        state['playing'] = not state['playing']
        button.label.set_text('Pause' if state['playing'] else 'Play')

    button.on_clicked(on_play)

    timer = fig.canvas.new_timer(interval=1000 // IMAGING_HZ)

    def advance():
        if state['playing']:
            nxt = min(total_master, slider.val + 1)
            slider.set_val(nxt)
            if nxt == total_master:
                state['playing'] = False
                button.label.set_text('Play')

    timer.add_callback(advance)
    timer.start()
    plt.show()