"""
noise_sweep.py

Runs the phase-gradient detector across a range of SNR values, for both
the planar (direction/speed) and radial (source location) cases, and
plots how detection accuracy degrades as noise increases. This is the
concrete validation result the surrogate-data pipeline was built to
produce: not just "the method works," but "here is exactly how far
noise can go before it stops working."
"""
import numpy as np
import matplotlib.pyplot as plt

from wave_sim import make_block, make_mask, FS, PIXEL_MM
from detection import (detect_in_burst_window, estimate_radial_source,
                       erode_mask)

SNR_VALUES = [5.0, 3.0, 2.0, 1.0, 0.7, 0.5, 0.3, 0.2, 0.1]


def run_planar_sweep(direction_deg=45, speed_mm_s=20, freq_hz=8):
    mask = make_mask()
    direction_err, speed_err, pgd_vals = [], [], []
    for snr in SNR_VALUES:
        data, truth = make_block("planar", freq_hz=freq_hz, speed_mm_s=speed_mm_s,
                                  direction_deg=direction_deg, snr=snr, seed=1, mask=mask)
        r = detect_in_burst_window(data, truth, freq_hz, FS, PIXEL_MM, mask)
        if r["direction_deg"] is None:
            direction_err.append(np.nan)
            speed_err.append(np.nan)
        else:
            d_err = abs((r["direction_deg"] - direction_deg + 180) % 360 - 180)
            direction_err.append(d_err)
            speed_err.append(abs(r["speed_mm_s"] - speed_mm_s) / speed_mm_s * 100)
        pgd_vals.append(r["mean_pgd"])
    return direction_err, speed_err, pgd_vals


def run_radial_sweep(source_xy=(2.0, 7.0), speed_mm_s=20, freq_hz=8):
    # IMPORTANT: source_xy must NOT be the mask's centroid (5,5). With no
    # real signal at all, the fit degenerates toward the centroid, so an
    # on-center source hides that failure mode instead of revealing it.
    mask = make_mask()
    safe_mask = erode_mask(mask, pixels=2)
    loc_err, residuals = [], []
    for snr in SNR_VALUES:
        data, truth = make_block("radial", freq_hz=freq_hz, speed_mm_s=speed_mm_s,
                                  source_xy=source_xy, snr=snr, seed=1, mask=mask)
        start_f = int(truth["burst_start_s"] * FS)
        dur_f = int(truth["burst_dur_s"] * FS)
        est = estimate_radial_source(data, freq_hz, FS, PIXEL_MM, mask=safe_mask,
                                      frame_range=(start_f + 50, start_f + dur_f - 50))
        if est is None:
            loc_err.append(np.nan)
            residuals.append(np.nan)
        else:
            err = np.hypot(est[0] - source_xy[0], est[1] - source_xy[1])
            loc_err.append(err)
            residuals.append(est[2])
    return loc_err, residuals


def plot_sweep():
    d_err, s_err, pgd = run_planar_sweep()
    loc_err, residuals = run_radial_sweep()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(SNR_VALUES, pgd, 'o-', color='tab:blue')
    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1, label='detection threshold')
    axes[0].set_xlabel("SNR (signal/noise)")
    axes[0].set_ylabel("Mean PGD")
    axes[0].set_title("Planar: wave coherence vs noise")
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=8)

    axes[1].plot(SNR_VALUES, d_err, 'o-', color='tab:orange', label='direction error (deg)')
    ax2 = axes[1].twinx()
    ax2.plot(SNR_VALUES, s_err, 's--', color='tab:green', label='speed error (%)')
    axes[1].set_xlabel("SNR (signal/noise)")
    axes[1].set_ylabel("Direction error (deg)", color='tab:orange')
    ax2.set_ylabel("Speed error (%)", color='tab:green')
    axes[1].set_title("Planar: direction/speed accuracy vs noise")
    axes[1].invert_xaxis()

    axes[2].plot(SNR_VALUES, loc_err, 'o-', color='tab:red', label='location error (mm)')
    ax3b = axes[2].twinx()
    ax3b.plot(SNR_VALUES, residuals, 's--', color='gray', label='fit residual (mm)')
    axes[2].set_xlabel("SNR (signal/noise)")
    axes[2].set_ylabel("Source location error (mm)", color='tab:red')
    ax3b.set_ylabel("Fit residual (mm)", color='gray')
    axes[2].set_title("Radial: source localization error vs noise\n(off-center source, avoids mask-symmetry artifact)")
    axes[2].invert_xaxis()

    plt.tight_layout()
    plt.savefig("noise_sweep_results.png", dpi=120)
    print("Saved noise_sweep_results.png")

    print("\n--- Summary ---")
    print(f"{'SNR':>6} {'PGD':>6} {'dir_err':>8} {'speed_err%':>11} {'radial_err_mm':>14}")
    for i, snr in enumerate(SNR_VALUES):
        print(f"{snr:>6} {pgd[i]:>6.2f} {d_err[i]:>8.2f} {s_err[i]:>11.2f} {loc_err[i]:>14.3f}")

    return fig


if __name__ == "__main__":
    plot_sweep()
    plt.show()