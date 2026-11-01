"""
run_real_data.py  (self-contained apart from detection.py)

Folder layout:
  data/EMXJ21_0406/                          the real trial .mat files
  data/masks/EMXJ21_20220406_true_mask.png   Dieter's cortex mask

Run:  python run_real_data.py

For every trial it runs the wave detector at several frequencies in TWO modes:
  raw      : data as recorded
  no-global: shared signal across the cortex removed + light spatial smoothing
Each is compared to a FAIR shuffled control: pixel locations are shuffled
first and then given the SAME preprocessing, so the control has the same
spatial smoothness as the real data.
Also reports the direction of the wave-like frames (PGD >= 0.5).
"""
import os
import glob
import csv
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage

from detection import (erode_mask, phase_gradient_detect, estimate_frequency,
                       shuffle_pixels, load_true_mask)

FOLDER = "data/EMXJ21_0406"
MASK_PNG = "data/masks/EMXJ21_20220406_true_mask.png"
BANDS_HZ = [4, 6, 8, 10, 12, 20, 30]
FS = 200
PIXEL_MM = 0.2                  # UNCONFIRMED: ask Dieter (0.2 vs 0.1 mm)
SMOOTH_SIGMA = 1.0              # pixels of spatial smoothing in "no-global" mode
MODES = ("raw", "no-global")
MIN_WAVE_FRAMES = 20            # need >= 0.1 s of wave-like frames to report a direction


def load(path):
    """Stored (T, Y, X) -> (Y, X, T), un-inverted."""
    with h5py.File(path, "r") as f:
        raw = f["datBRLSreg"][:]
    return np.nan_to_num(-np.transpose(raw, (1, 2, 0)).astype(float))


def preprocess(data, mask50):
    """Remove the signal shared across the whole cortex at each frame, then
    smooth lightly in space (masked, so the edge doesn't leak zeros in)."""
    d = data - data[mask50].mean(axis=0)[None, None, :]
    if SMOOTH_SIGMA > 0:
        w = ndimage.gaussian_filter(mask50.astype(float), SMOOTH_SIGMA)
        num = ndimage.gaussian_filter(d * mask50[:, :, None],
                                      sigma=(SMOOTH_SIGMA, SMOOTH_SIGMA, 0))
        d = num / np.maximum(w, 1e-6)[:, :, None]
    return d * mask50[:, :, None]


def prepare(path, mode, verbose=False):
    """Returns (data, fair control data, eroded mask) for the chosen mode."""
    raw = load(path)
    mask50 = load_true_mask(MASK_PNG, raw, verbose=verbose)
    emask = erode_mask(mask50, 2)
    shuffled_raw = shuffle_pixels(raw, mask50, seed=0)
    if mode == "no-global":
        return preprocess(raw, mask50), preprocess(shuffled_raw, mask50), emask
    return raw, shuffled_raw, emask


def mean_direction(dirs, sel):
    """Circular mean direction (deg) of the selected frames, or nan."""
    if sel.sum() < MIN_WAVE_FRAMES:
        return np.nan
    a = np.deg2rad(dirs[sel])
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


files = sorted(glob.glob(f"{FOLDER}/*.mat"))
print(f"{len(files)} trials in {FOLDER}\n")
if not files:
    raise SystemExit(f"No .mat files found in {FOLDER}.")
if not os.path.isfile(MASK_PNG):
    raise SystemExit(f"Mask not found: {MASK_PNG}")

# ---------------------------------------------------------------- scan
rows = []
first = True
for path in files:
    name = os.path.basename(path)
    for mode in MODES:
        try:
            data, ctrl, emask = prepare(path, mode, verbose=first)
            first = False

            trial_rows = []
            for hz in BANDS_HZ:
                pgd, dirs, _ = phase_gradient_detect(data, hz, FS, PIXEL_MM, emask)
                pgd_c, _, _ = phase_gradient_detect(ctrl, hz, FS, PIXEL_MM, emask)
                pgd, dirs, pgd_c = np.asarray(pgd), np.asarray(dirs), np.asarray(pgd_c)
                wave = pgd >= 0.5
                trial_rows.append(dict(
                    trial=name, path=path, mode=mode, hz=hz,
                    mean_pgd=float(pgd.mean()),
                    control=float(pgd_c.mean()),
                    control_max=float(pgd_c.max()),
                    frac_wavelike=float(wave.mean()),
                    max_pgd=float(pgd.max()),
                    n_wave=int(wave.sum()),
                    dir_deg=mean_direction(dirs, wave)))
            rows += trial_rows

            b = max(trial_rows, key=lambda r: r["mean_pgd"])
            print(f"{name} [{mode:9s}]: best {b['hz']} Hz, mean PGD {b['mean_pgd']:.3f} "
                  f"(control {b['control']:.3f}, control max {b['control_max']:.2f}), "
                  f"max {b['max_pgd']:.2f}, {b['frac_wavelike'] * 100:.1f}% frames >= 0.5")
        except Exception as e:
            print(f"{name} [{mode}]: FAILED ({e})")

if not rows:
    raise SystemExit("No results. Check the error messages above.")

with open("real_batch_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "path"])
    w.writeheader()
    for r in rows:
        w.writerow({k: v for k, v in r.items() if k != "path"})

for mode in MODES:
    mrows = [r for r in rows if r["mode"] == mode]
    if not mrows:
        continue
    print(f"\nTop 8 [{mode}] (by mean PGD above control):")
    for r in sorted(mrows, key=lambda r: -(r["mean_pgd"] - r["control"]))[:8]:
        d = f"{r['dir_deg']:.0f} deg" if np.isfinite(r["dir_deg"]) else "n/a"
        print(f"  {r['trial']}  {r['hz']:>2} Hz  PGD={r['mean_pgd']:.3f}  "
              f"control={r['control']:.3f}  frames>=0.5: {r['frac_wavelike'] * 100:.1f}%  "
              f"max={r['max_pgd']:.2f}  direction={d}")

print("\nOverall averages across all trials and bands:")
for mode in MODES:
    mrows = [r for r in rows if r["mode"] == mode]
    if mrows:
        print(f"  {mode:9s}: mean PGD {np.mean([r['mean_pgd'] for r in mrows]):.3f}, "
              f"control {np.mean([r['control'] for r in mrows]):.3f}, "
              f"highest max PGD {max(r['max_pgd'] for r in mrows):.2f}, "
              f"highest CONTROL max {max(r['control_max'] for r in mrows):.2f}")

print("\nDirections of wave-like frames, no-global mode "
      f"(needs >= {MIN_WAVE_FRAMES} frames at PGD >= 0.5):")
dir_rows = [r for r in rows if r["mode"] == "no-global" and np.isfinite(r["dir_deg"])]
for r in sorted(dir_rows, key=lambda r: (r["trial"], r["hz"])):
    print(f"  {r['trial']}  {r['hz']:>2} Hz  {r['dir_deg']:5.0f} deg  "
          f"({r['n_wave']} frames)")
if dir_rows:
    a = np.deg2rad([r["dir_deg"] for r in dir_rows])
    R = float(np.hypot(np.sin(a).mean(), np.cos(a).mean()))
    print(f"  consistency across entries R = {R:.2f} "
          f"(1 = all the same direction, 0 = random)")
print("\nSaved real_batch_results.csv")

# ---------------------------------------------------------------- plot best
best = max(rows, key=lambda r: r["mean_pgd"] - r["control"])
print(f"\nPlotting best candidate: {best['trial']} at {best['hz']} Hz "
      f"[{best['mode']}]")
data, ctrl, emask = prepare(best["path"], best["mode"])

pgd, dirs, speeds = phase_gradient_detect(data, best["hz"], FS, PIXEL_MM, emask)
pgd_c, _, _ = phase_gradient_detect(ctrl, best["hz"], FS, PIXEL_MM, emask)
pgd, dirs, pgd_c = np.asarray(pgd), np.asarray(dirs), np.asarray(pgd_c)
wavelike = pgd >= 0.5
peak, freqs, power = estimate_frequency(data, FS, mask=emask)

t = np.arange(data.shape[2]) / FS
fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(freqs, power)
ax[0].axvline(best["hz"], color="r", ls="--", label=f"tested {best['hz']} Hz")
ax[0].set_xlim(0, 70); ax[0].set_xlabel("Hz"); ax[0].legend()
ax[0].set_title(f"Mean power spectrum: {best['trial']} [{best['mode']}]")
ax[1].plot(t, pgd, label="real")
ax[1].plot(t, pgd_c, label="fair control (shuffled, then same smoothing)", alpha=0.7)
ax[1].axhline(0.5, color="k", ls=":")
ax[1].set_xlabel("time (s)"); ax[1].legend(); ax[1].set_title("PGD over the trial")
if wavelike.any():
    ax[2].plot(t[wavelike], dirs[wavelike], ".", ms=3)
ax[2].set_ylim(0, 360); ax[2].set_xlabel("time (s)")
ax[2].set_title("Direction when PGD >= 0.5 (empty = none found)")
plt.tight_layout()
plt.savefig("real_data_result.png", dpi=150)
plt.show()