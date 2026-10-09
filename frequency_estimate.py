"""
frequency_estimate.py

Estimates the dominant oscillation frequency directly from the data,
rather than assuming it's known in advance.

IMPORTANT FINDING: the existing PGD coherence metric (phase_gradient_detect)
is USELESS for this on radial waves -- it averages near zero even for a
perfect, high-SNR radial wave, because gradients point outward in every
direction depending on location and cancel out in a global mean. PGD
only works as a coherence check for planar waves.

This function sidesteps that entirely: it looks at the average power
spectrum across all valid pixels, which never looks at spatial pattern
at all -- so it works identically for planar, radial, and synchronous
waves.
"""
import numpy as np


def estimate_frequency(data, fs, mask=None, frame_range=None, freq_range=(2, 70)):
    """
    Finds the dominant oscillation frequency.

    data: (ny, nx, nt) array
    mask: boolean array, True = valid pixel (optional)
    frame_range: (start, end) frame indices to restrict analysis to
                 (e.g. the known burst window). Uses the full trial if
                 not given.
    freq_range: (low, high) Hz to search within -- excludes very low
                frequencies dominated by slow background drift.

    Returns: (peak_freq, freqs, mean_power) -- the detected frequency,
    and the full power spectrum for plotting/inspection.
    """
    if frame_range:
        seg = data[:, :, frame_range[0]:frame_range[1]]
    else:
        seg = data

    if mask is not None:
        pixels = seg[mask]  # (n_valid_pixels, T)
    else:
        pixels = seg.reshape(-1, seg.shape[2])

    pixels = pixels - pixels.mean(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(pixels.shape[1], d=1 / fs)
    power = np.abs(np.fft.rfft(pixels, axis=1)) ** 2
    mean_power = power.mean(axis=0)

    band = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    peak_idx = np.argmax(mean_power[band])
    peak_freq = freqs[band][peak_idx]
    return peak_freq, freqs, mean_power


if __name__ == "__main__":
    from wave_sim import make_block, make_mask, FS

    mask = make_mask()

    print("Frequency recovery test across all wave types (true freq = 8Hz):\n")
    for wtype in ["planar", "radial", "synchronous"]:
        kwargs = dict(freq_hz=8, snr=3, seed=0, mask=mask)
        if wtype == "planar":
            kwargs["speed_mm_s"] = 20
            kwargs["direction_deg"] = 45
        elif wtype == "radial":
            kwargs["speed_mm_s"] = 20
            kwargs["source_xy"] = (5, 5)
        data, truth = make_block(wtype, **kwargs)
        sf = int(truth["burst_start_s"] * FS)
        df = int(truth["burst_dur_s"] * FS)
        peak, freqs, power = estimate_frequency(data, FS, mask=mask, frame_range=(sf, sf + df))
        print(f"  {wtype:12s}: detected = {peak:.2f} Hz")

    print("\nGamma-band test (true freq = 55Hz, radial wave):")
    data, truth = make_block("radial", freq_hz=55, speed_mm_s=20, source_xy=(5, 5),
                              snr=3, seed=0, mask=mask)
    sf = int(truth["burst_start_s"] * FS)
    df = int(truth["burst_dur_s"] * FS)
    peak, freqs, power = estimate_frequency(data, FS, mask=mask, frame_range=(sf, sf + df))
    print(f"  radial (55Hz): detected = {peak:.2f} Hz")