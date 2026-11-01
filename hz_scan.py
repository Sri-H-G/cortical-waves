"""
scan_frequencies.py

For every trial and every frequency, counts the BURSTS of wave-like activity
and compares them with a shuffled control. Prints, for each trial, the
frequency with the most bursts, and saves everything to a CSV.

  burst = a run of consecutive frames with PGD >= PGD_THRESHOLD lasting at
          least MIN_CYCLES cycles of that frequency (half a cycle = 125 ms at 4 Hz,
          62 ms at 8 Hz). Runs separated by less than MERGE_GAP_MS are
          merged into one burst (PGD flickers around the threshold at a
          burst's start and end). The first and last EDGE_S seconds are
          skipped because the band-pass filter is unreliable at the edges.

Ranking: among frequencies holding at least MIN_POWER_PCT of the power
(i.e. a real rhythm is there), the one with the most bursts; ties go to the
stronger rhythm. POWER is needed because PGD is not frequency-selective: a
strong wave at 8 Hz also gives high PGD in neighbouring bands, and a filter
at the wrong frequency chops one real burst into several short ones.

Preprocessing matches wave_overlay_real.py (shared signal removed, light
smoothing). The shuffled control is FAIR: pixel locations are shuffled
first, then the same preprocessing is applied, so it is just as smooth.

Folder layout (same as the viewer):
  data/EMXJ21_0406/                          the real trial .mat files
  data/masks/EMXJ21_20220406_true_mask.png   cortex mask
Run:  python scan_frequencies.py      (takes a few minutes)
"""
import csv
import glob
import os
import time

import numpy as np
import h5py
from scipy import ndimage

from detection import (phase_gradient_detect, erode_mask, load_true_mask,
                       shuffle_pixels, estimate_frequency)

FOLDER = "data/EMXJ21_0406"
MASK_PNG = "data/masks/EMXJ21_20220406_true_mask.png"
FS = 200
PIXEL_MM = 0.2                      # unconfirmed (0.1 or 0.2)
FREQS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 37]   # band = centre +/- 2 Hz
PGD_THRESHOLD = 0.5
MIN_CYCLES = 0.5                    # a burst must last >= half a cycle of its rhythm
MERGE_GAP_MS = 100                  # bursts closer than this are one burst
EDGE_S = 0.25                       # skip filter edge effects
REMOVE_SHARED = True                # same default as the viewer
SMOOTH_SIGMA = 1.0
OUT_CSV = "frequency_scan_results.csv"


def load(path):
    with h5py.File(path, "r") as f:
        raw = f["datBRLSreg"][:]
    return np.nan_to_num(-np.transpose(raw, (1, 2, 0)).astype(float))


def preprocess(data, mask50):
    """Same as wave_overlay_real.py."""
    d = data
    if REMOVE_SHARED:
        d = data - data[mask50].mean(axis=0)[None, None, :]
    if SMOOTH_SIGMA > 0:
        w = ndimage.gaussian_filter(mask50.astype(float), SMOOTH_SIGMA)
        num = ndimage.gaussian_filter(d * mask50[:, :, None],
                                      sigma=(SMOOTH_SIGMA, SMOOTH_SIGMA, 0))
        d = num / np.maximum(w, 1e-6)[:, :, None]
    return d * mask50[:, :, None]


def find_bursts(pgd, dirs, hz):
    """Runs of PGD >= threshold, at least MIN_CYCLES cycles long, short gaps merged, edges skipped."""
    n = len(pgd)
    edge = int(EDGE_S * FS)
    min_len = max(5, int(round(MIN_CYCLES / hz * FS)))
    above = np.nan_to_num(pgd, nan=0.0) >= PGD_THRESHOLD
    above[:edge] = False
    above[n - edge:] = False
    # close short gaps so one burst isn't split into several
    gap = int(round(MERGE_GAP_MS / 1000 * FS))
    idx = np.flatnonzero(above)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            above[a:b] = True
    bursts, i = [], 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_len:
                a = np.deg2rad(dirs[i:j][np.isfinite(dirs[i:j])])
                mdir = (np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360
                        if len(a) else np.nan)
                bursts.append({"start_s": i / FS, "dur_ms": (j - i) / FS * 1000,
                               "peak_pgd": float(np.nanmax(pgd[i:j])), "dir_deg": float(mdir)})
            i = j
        else:
            i += 1
    return bursts


def scan_trial(raw, mask50, emask):
    data = preprocess(raw, mask50)
    data_sh = preprocess(shuffle_pixels(raw, mask50, seed=0), mask50)
    peak_hz, freqs, power = estimate_frequency(data, FS, mask=emask, freq_range=(1, 45))
    rows = []
    for hz in FREQS:
        pgd, dirs, _ = phase_gradient_detect(data, hz, FS, PIXEL_MM, emask)
        pgd_s, dirs_s, _ = phase_gradient_detect(data_sh, hz, FS, PIXEL_MM, emask)
        b, b_s = find_bursts(pgd, dirs, hz), find_bursts(pgd_s, dirs_s, hz)
        band = (freqs >= hz - 2) & (freqs <= hz + 2)
        rows.append({"freq_hz": hz, "bursts": len(b), "shuffled_bursts": len(b_s),
                     "burst_time_s": sum(k["dur_ms"] for k in b) / 1000,
                     "power_pct": 0.0, "_power": float(power[band].mean()),
                     "mean_pgd": float(np.nanmean(pgd)), "shuffled_mean_pgd": float(np.nanmean(pgd_s)),
                     "max_pgd": float(np.nanmax(pgd)), "burst_list": b})
    total = sum(r["_power"] for r in rows)
    for r in rows:
        r["power_pct"] = 100 * r.pop("_power") / total
    return rows, peak_hz


MIN_POWER_PCT = 10                  # a frequency must hold this share of power to win


def best_row(rows):
    """Among frequencies with real power behind them (>= MIN_POWER_PCT),
    the one with the most bursts; ties go to the stronger rhythm.
    Why the power filter: tested on fake trials with a known frequency,
    'most bursts' alone picked the wrong band, because a filter tuned to the
    wrong frequency chops one real burst into several short ones."""
    cand = [r for r in rows if r["power_pct"] >= MIN_POWER_PCT] or rows
    return max(cand, key=lambda r: (r["bursts"], r["power_pct"]))


def main():
    files = sorted(glob.glob(os.path.join(FOLDER, "*.mat")))
    if not files:
        raise SystemExit(f"No .mat files in {FOLDER}")
    if not os.path.isfile(MASK_PNG):
        raise SystemExit(f"Mask not found: {MASK_PNG}")

    print(f"{len(files)} trials | {len(FREQS)} frequencies | burst = PGD >= {PGD_THRESHOLD} "
          f"for >= {MIN_CYCLES} cycle | shared signal {'removed' if REMOVE_SHARED else 'kept'}\n")
    all_rows, best = [], []
    first = True
    for path in files:
        name = os.path.basename(path)
        t0 = time.time()
        raw = load(path)
        mask50 = load_true_mask(MASK_PNG, raw, verbose=first)
        first = False
        emask = erode_mask(mask50, 2)
        rows, peak_hz = scan_trial(raw, mask50, emask)

        print("=" * 72)
        print(f"{name}   ({time.time() - t0:.0f} s)")
        print(f"  strongest rhythm in the power spectrum: {peak_hz:.1f} Hz")
        print(f"  {'Hz':>4} {'bursts':>7} {'shuffled':>9} {'time(s)':>8} {'power%':>7} "
              f"{'mean PGD':>9} {'shuffled':>9} {'max PGD':>8}")
        for r in rows:
            print(f"  {r['freq_hz']:>4} {r['bursts']:>7} {r['shuffled_bursts']:>9} {r['burst_time_s']:>8.2f} "
                  f"{r['power_pct']:>7.1f} {r['mean_pgd']:>9.3f} {r['shuffled_mean_pgd']:>9.3f} {r['max_pgd']:>8.2f}")
        b = best_row(rows)
        if b["bursts"] == 0:
            print("  -> no bursts at any frequency")
        else:
            print(f"  -> BEST: {b['freq_hz']} Hz, {b['bursts']} bursts, "
                  f"{b['burst_time_s']:.2f} s in total (shuffled control: {b['shuffled_bursts']})")
            for k in b["burst_list"]:
                print(f"       {k['start_s']:6.2f} s  {k['dur_ms']:5.0f} ms  "
                      f"peak PGD {k['peak_pgd']:.2f}  direction {k['dir_deg']:5.0f} deg")
        best.append((name, b, peak_hz))
        for r in rows:
            all_rows.append({"trial": name, **{k: v for k, v in r.items() if k != "burst_list"},
                             "burst_times_s": " ".join(f"{k['start_s']:.2f}" for k in r["burst_list"])})

    # summary
    print("\n" + "=" * 72)
    print(f"SUMMARY: best frequency per trial (most bursts among bands with >= {MIN_POWER_PCT}% of power)")
    print("=" * 72)
    print(f"  {'trial':28s} {'best':>12} {'bursts':>7} {'shuffled':>9} {'power peak':>11}")
    for name, b, pk in best:
        if b["bursts"] == 0:
            print(f"  {name:28s} {'none':>12} {'':>7} {'':>9} {pk:>9.1f} Hz")
        else:
            print(f"  {name:28s} {b['freq_hz']:>9} Hz {b['bursts']:>7} {b['shuffled_bursts']:>9} {pk:>9.1f} Hz")
    print("\nTotal bursts across all trials, per frequency:")
    print(f"  {'Hz':>4} {'real':>6} {'shuffled':>9} {'time(s)':>8}")
    for hz in FREQS:
        real = sum(r["bursts"] for r in all_rows if r["freq_hz"] == hz)
        sh = sum(r["shuffled_bursts"] for r in all_rows if r["freq_hz"] == hz)
        tt = sum(r["burst_time_s"] for r in all_rows if r["freq_hz"] == hz)
        print(f"  {hz:>4} {real:>6} {sh:>9} {tt:>8.2f}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()