"""
wave_overlay_real.py
Overlay viewer for REAL trials.

Top: band-passed movie, white wavefront lines, and the detector's arrow.
Below: two "kymographs" = the signal averaged along one axis, plotted against
time. TRAVELING wave = diagonal stripes. STANDING pattern (halves flipping in
place) = vertical stripes.

Folder layout:
  data/EMXJ21_0406/                          the real trial .mat files
  data/masks/EMXJ21_20220406_true_mask.png   Dieter's cortex mask
Run:  python wave_overlay_real.py
"""
import glob
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy import ndimage

from detection import (bandpass, get_phase, phase_gradient_detect, erode_mask,
                       load_true_mask)

FOLDER = "data/EMXJ21_0406"
MASK_PNG = "data/masks/EMXJ21_20220406_true_mask.png"
FS = 200
PIXEL_MM = 0.2            # UNCONFIRMED: ask Dieter
ARROW_MIN_PGD = 0.25      # show an arrow above this (green at >= 0.5)
SMOOTH_SIGMA = 1.0        # same preprocessing as run_real_data.py
KYMO_HALF_S = 1.5         # kymograph window shown around the current frame
PLAY_STEP = 2
PLAY_MS = 50


def load(path):
    with h5py.File(path, "r") as f:
        raw = f["datBRLSreg"][:]
    return np.nan_to_num(-np.transpose(raw, (1, 2, 0)).astype(float))


def preprocess(data, mask50):
    """Same as run_real_data.py: remove the signal shared across the whole
    cortex at each frame, then smooth lightly (masked)."""
    d = data - data[mask50].mean(axis=0)[None, None, :]
    if SMOOTH_SIGMA > 0:
        w = ndimage.gaussian_filter(mask50.astype(float), SMOOTH_SIGMA)
        num = ndimage.gaussian_filter(d * mask50[:, :, None],
                                      sigma=(SMOOTH_SIGMA, SMOOTH_SIGMA, 0))
        d = num / np.maximum(w, 1e-6)[:, :, None]
    return d * mask50[:, :, None]


def kymograph(bp, emask, axis_keep):
    """Average the band-passed signal over the OTHER axis.
    axis_keep=1 keeps position along x (columns); axis_keep=0 keeps y (rows)."""
    avg_axis = 0 if axis_keep == 1 else 1
    cnt = emask.sum(axis=avg_axis).astype(float)
    k = bp.sum(axis=avg_axis) / np.maximum(cnt, 1)[:, None]
    k[cnt == 0, :] = np.nan
    return k


files = sorted(glob.glob(f"{FOLDER}/*.mat"))
if not files:
    raise SystemExit(f"No .mat files in {FOLDER}")
if not os.path.isfile(MASK_PNG):
    raise SystemExit(f"Mask not found: {MASK_PNG}")
start = next((i for i, f in enumerate(files) if "182" in f), 0)

S = dict(file_idx=start, nog=True, first=True, bp=None, phase=None, pgd=None,
         dirs=None, speeds=None, kx=None, ky=None, contour=None, arrow=None,
         playing=False, busy=False, drawing=False)

fig = plt.figure(figsize=(12.5, 8.5))
ax = fig.add_axes([0.30, 0.43, 0.66, 0.51])
im = ax.imshow(np.zeros((50, 50)), cmap="RdBu", vmin=-1, vmax=1)
ax.axis("off")
title = ax.set_title("")

ax_kx = fig.add_axes([0.34, 0.275, 0.60, 0.11])
ax_ky = fig.add_axes([0.34, 0.135, 0.60, 0.11])
im_kx = ax_kx.imshow(np.zeros((50, 10)), aspect="auto", cmap="RdBu", vmin=-1, vmax=1,
                     extent=[0, 10, 50, 0])
im_ky = ax_ky.imshow(np.zeros((50, 10)), aspect="auto", cmap="RdBu", vmin=-1, vmax=1,
                     extent=[0, 10, 50, 0])
ax_kx.set_ylabel("x position", fontsize=8)
ax_ky.set_ylabel("y position", fontsize=8)
ax_ky.set_xlabel("time (s)", fontsize=8)
ax_kx.set_title("Signal along x vs time: diagonal stripes = traveling, "
                "vertical stripes = standing", fontsize=9)
vl_kx = ax_kx.axvline(0, color="k", lw=1)
vl_ky = ax_ky.axvline(0, color="k", lw=1)

ax_prev = fig.add_axes([0.03, 0.90, 0.09, 0.045]); b_prev = Button(ax_prev, "< Trial")
ax_next = fig.add_axes([0.14, 0.90, 0.09, 0.045]); b_next = Button(ax_next, "Trial >")
ax_hz = fig.add_axes([0.08, 0.83, 0.17, 0.03])
hz_sl = Slider(ax_hz, "Freq Hz", 3, 30, valinit=4, valstep=1)
ax_comp = fig.add_axes([0.03, 0.76, 0.22, 0.045]); b_comp = Button(ax_comp, "Recompute")
ax_play = fig.add_axes([0.03, 0.69, 0.22, 0.045]); b_play = Button(ax_play, "Play")
ax_burst = fig.add_axes([0.03, 0.62, 0.22, 0.045]); b_burst = Button(ax_burst, "Next burst")
ax_tog = fig.add_axes([0.03, 0.55, 0.22, 0.045])
b_tog = Button(ax_tog, "Shared signal: REMOVED")
ax_stat = fig.add_axes([0.03, 0.10, 0.24, 0.40]); ax_stat.axis("off")
stat = ax_stat.text(0, 1, "", va="top", fontsize=9)
ax_fr = fig.add_axes([0.30, 0.05, 0.66, 0.03])
fr_sl = Slider(ax_fr, "Frame", 0, 2047, valinit=0, valstep=1)


def clear_overlay():
    if S["contour"] is not None:
        try:
            S["contour"].remove()
        except Exception:
            for c in getattr(S["contour"], "collections", []):
                c.remove()
        S["contour"] = None
    if S["arrow"] is not None:
        S["arrow"].remove()
        S["arrow"] = None


def draw(frame):
    if S["bp"] is None or S["drawing"]:
        return
    S["drawing"] = True
    try:
        frame = int(frame)
        im.set_data(S["bp"][:, :, frame])
        clear_overlay()
        ph = S["phase"][:, :, frame]
        if np.isfinite(ph).sum() > 10:
            S["contour"] = ax.contour(ph, levels=[-np.pi / 2, 0, np.pi / 2],
                                      colors="white", linewidths=1.0)
        # kymograph window around the current frame
        tcur = frame / FS
        lo, hi = max(0.0, tcur - KYMO_HALF_S), min(S["bp"].shape[2] / FS, tcur + KYMO_HALF_S)
        for a in (ax_kx, ax_ky):
            a.set_xlim(lo, hi)
        vl_kx.set_xdata([tcur, tcur]); vl_ky.set_xdata([tcur, tcur])

        pgd, d, sp = S["pgd"][frame], S["dirs"][frame], S["speeds"][frame]
        lines = [os.path.basename(files[S["file_idx"]]),
                 f"Frame {frame}  ({tcur:.2f} s)",
                 f"Band: {hz_sl.val:.0f} Hz (+/- 2)",
                 f"Shared signal: {'removed' if S['nog'] else 'kept'}",
                 f"PGD now: {pgd:.2f}",
                 f"PGD trial max: {np.nanmax(S['pgd']):.2f}"]
        if pgd >= ARROW_MIN_PGD and np.isfinite(d):
            col = "lime" if pgd >= 0.5 else "orange"
            dx, dy = 10 * np.cos(np.deg2rad(d)), 10 * np.sin(np.deg2rad(d))
            S["arrow"] = ax.annotate("", xy=(25 + dx, 25 + dy), xytext=(25, 25),
                                     arrowprops=dict(arrowstyle="-|>", color=col, lw=3))
            lines += ["", "WAVE-LIKE" if pgd >= 0.5 else "weak organization",
                      f"direction: {d:.0f} deg (0/180 = along x)",
                      f"speed: {sp:.1f} mm/s"]
            stat.set_color("black")
        else:
            lines += ["", "no coherent wave"]
            stat.set_color("red")
        stat.set_text("\n".join(lines))
        fig.canvas.draw_idle()
    finally:
        S["drawing"] = False


def compute(_=None):
    if S["busy"]:
        return
    S["busy"] = True
    S["playing"] = False
    b_play.label.set_text("Play")
    name = os.path.basename(files[S["file_idx"]])
    stat.set_text(f"Loading and computing\n{name}\n(a few seconds)")
    stat.set_color("black")
    fig.canvas.draw_idle(); plt.pause(0.001)
    try:
        hz = hz_sl.val
        raw = load(files[S["file_idx"]])
        mask50 = load_true_mask(MASK_PNG, raw, verbose=S["first"])
        S["first"] = False
        emask = erode_mask(mask50, 2)
        data = preprocess(raw, mask50) if S["nog"] else raw

        bp = np.asarray(bandpass(data, hz, FS, bandwidth_hz=4.0), dtype=float)
        ph = get_phase(data, hz, FS).astype(float)
        ph[~emask, :] = np.nan
        pgd, dirs, speeds = phase_gradient_detect(data, hz, FS, PIXEL_MM, emask)
        bp[~emask, :] = 0

        kx = kymograph(bp, emask, 1)
        ky = kymograph(bp, emask, 0)
        vmax = 3 * bp[emask].std()
        kmax = 3 * np.nanstd(kx)
        S.update(bp=bp, phase=ph, pgd=np.asarray(pgd), dirs=np.asarray(dirs),
                 speeds=np.asarray(speeds), kx=kx, ky=ky)
        im.set_clim(-vmax, vmax)
        T = bp.shape[2] / FS
        for imk, k in ((im_kx, kx), (im_ky, ky)):
            imk.set_data(np.nan_to_num(k))
            imk.set_extent([0, T, k.shape[0], 0])
            imk.set_clim(-kmax, kmax)
        title.set_text(f"Real data, band-passed at {hz:.0f} Hz "
                       f"(white lines = wavefronts)")
        start_frame = int(np.nanargmax(S["pgd"]))
        fr_sl.eventson = False; fr_sl.set_val(start_frame); fr_sl.eventson = True
        draw(start_frame)
    except Exception as e:
        stat.set_text(f"ERROR: {e}"); stat.set_color("red")
        fig.canvas.draw_idle()
    finally:
        S["busy"] = False


def change_trial(step):
    S["file_idx"] = (S["file_idx"] + step) % len(files)
    compute()


def toggle_global(_=None):
    S["nog"] = not S["nog"]
    b_tog.label.set_text("Shared signal: REMOVED" if S["nog"] else "Shared signal: KEPT")
    compute()


def next_burst(_=None):
    if S["pgd"] is None or S["busy"]:
        return
    above = S["pgd"] >= ARROW_MIN_PGD
    starts = np.where(above & ~np.roll(above, 1))[0]
    if len(starts) == 0:
        target = int(np.nanargmax(S["pgd"]))
    else:
        later = starts[starts > int(fr_sl.val)]
        target = int(later[0] if len(later) else starts[0])
    fr_sl.eventson = False; fr_sl.set_val(target); fr_sl.eventson = True
    draw(target)


def on_play(_):
    S["playing"] = not S["playing"]
    b_play.label.set_text("Pause" if S["playing"] else "Play")


def tick():
    if not S["playing"] or S["bp"] is None or S["busy"] or S["drawing"]:
        return
    nxt = int(fr_sl.val) + PLAY_STEP
    if nxt > 2047:
        nxt = 0
    fr_sl.eventson = False; fr_sl.set_val(nxt); fr_sl.eventson = True
    draw(nxt)


fr_sl.on_changed(lambda v: draw(v))
b_comp.on_clicked(compute)
b_prev.on_clicked(lambda _e: change_trial(-1))
b_next.on_clicked(lambda _e: change_trial(+1))
b_play.on_clicked(on_play)
b_burst.on_clicked(next_burst)
b_tog.on_clicked(toggle_global)

timer = fig.canvas.new_timer(interval=PLAY_MS)
timer.add_callback(tick)
timer.start()
fig._timer = timer

compute()
plt.show()