"""
detection.py

ALL detection logic lives here -- this is the single source of truth.
Every other script (the overlay viewer, any future analysis) imports
FROM this file and never reimplements any of this logic separately.

The analysis side, kept separate from the simulator (wave_sim.py), per
Dieter's instruction: the simulator just produces data blocks, and
this script loads a block the same way real data gets loaded, then
analyzes whatever matrix comes out.

Core method: generalized phase / phase-gradient analysis (Rubino 2006,
refined by Davis 2020):
    bandpass filter -> Hilbert transform -> instantaneous phase
    -> spatial phase gradient -> Phase Gradient Directionality (PGD)

IMPORTANT FINDING: PGD (the mean-vector coherence check below) only
works for PLANAR waves. For a RADIAL wave, local gradients point
outward in every direction depending on location, so they cancel out
in a global mean -- PGD reads near zero even for a perfect, high-SNR
radial wave. Radial waves need estimate_radial_source() instead, and
frequency estimation needs estimate_frequency() (power-spectrum based,
works for every wave type) rather than scanning PGD.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import binary_erosion

try:
    import h5py
except ImportError:
    h5py = None


def load_block(path):
    """Loads a .mat data block the same way real trial files are loaded
    (matches plot_mean_trace.py / play_data_trial.py)."""
    with h5py.File(path, "r") as f:
        raw = f["datBRLSreg"][:]           # (T, 50, 50)
    return np.transpose(-raw, (1, 2, 0))    # back to (Y, X, T), sign restored


def erode_mask(mask, pixels=2):
    """Shrinks a mask inward by the given number of pixels, so pixels
    right at the mask boundary are excluded from the gradient
    calculation. Always use this on whatever mask the data was
    generated with before passing it to phase_gradient_detect."""
    return binary_erosion(mask, iterations=pixels)


def bandpass(data, center_hz, fs, bandwidth_hz=4.0, order=3):
    """Filters each pixel's time series to a band around center_hz."""
    low = max(0.5, center_hz - bandwidth_hz / 2) / (fs / 2)
    high = min(fs / 2 - 1, center_hz + bandwidth_hz / 2) / (fs / 2)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=2)


def get_phase(data, center_hz, fs):
    """Shared step used by every detector below: bandpass filter, then
    Hilbert transform, returning the instantaneous phase at every
    pixel, every frame. Exposed as its own function so the overlay
    viewer (or anything else) can get the SAME phase computation used
    internally by phase_gradient_detect, rather than recomputing it
    separately."""
    filtered = bandpass(data, center_hz, fs)
    analytic = hilbert(filtered, axis=2)
    return np.angle(analytic)


def phase_gradient_detect(data, center_hz, fs, pixel_mm, mask=None):
    """
    Returns, per frame:
      pgd: Phase Gradient Directionality, 0-1 (PLANAR waves only --
           see module docstring. Near-zero for radial waves even when
           they're real.)
      direction_deg: estimated propagation direction (where defined)
      speed_mm_s: estimated propagation speed (where defined)

    mask: boolean array, True = valid pixel. Pass an ERODED mask
    (erode_mask) if the data has a mask baked in.
    """
    phase = get_phase(data, center_hz, fs)

    dphi_x = np.angle(np.exp(1j * (phase[:, 1:, :] - phase[:, :-1, :])))
    dphi_y = np.angle(np.exp(1j * (phase[1:, :, :] - phase[:-1, :, :])))

    gx = dphi_x[:-1, :, :] / pixel_mm
    gy = dphi_y[:, :-1, :] / pixel_mm

    if mask is not None:
        m = mask[:-1, :-1]
        gx = np.where(m[:, :, None], gx, np.nan)
        gy = np.where(m[:, :, None], gy, np.nan)

    nt = data.shape[2]
    pgd = np.full(nt, np.nan)
    direction_deg = np.full(nt, np.nan)
    speed_mm_s = np.full(nt, np.nan)
    omega = 2 * np.pi * center_hz

    for ti in range(nt):
        vx, vy = gx[:, :, ti], gy[:, :, ti]
        valid = ~np.isnan(vx) & ~np.isnan(vy)
        if valid.sum() < 10:
            continue
        vx_v, vy_v = vx[valid], vy[valid]
        mean_vec = np.array([vx_v.mean(), vy_v.mean()])
        mean_mag = np.hypot(*mean_vec)
        mag_of_means = np.hypot(vx_v, vy_v).mean()
        if mag_of_means < 1e-9:
            continue
        pgd[ti] = mean_mag / mag_of_means
        direction_deg[ti] = np.degrees(np.arctan2(-mean_vec[1], -mean_vec[0])) % 360
        if mean_mag > 1e-9:
            speed_mm_s[ti] = omega / mean_mag

    return pgd, direction_deg, speed_mm_s


def summarize(pgd, direction_deg, speed_mm_s, pgd_threshold=0.5):
    """Collapses the per-frame estimates into one verdict, using only the
    frames where PGD clears the threshold (a coherent wave was present)."""
    good = ~np.isnan(pgd) & (pgd >= pgd_threshold)
    frac_wavelike = float(np.mean(~np.isnan(pgd) & (pgd >= pgd_threshold)))
    if good.sum() == 0:
        return dict(frac_wavelike=frac_wavelike, mean_pgd=float(np.nanmean(pgd)),
                    direction_deg=None, speed_mm_s=None)
    angles = np.deg2rad(direction_deg[good])
    mean_dir = np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))) % 360
    return dict(frac_wavelike=frac_wavelike, mean_pgd=float(np.nanmean(pgd)),
                direction_deg=float(mean_dir), speed_mm_s=float(np.nanmedian(speed_mm_s[good])))


def detect_in_burst_window(data, truth, center_hz, fs, pixel_mm, mask, trim_frames=50):
    """Convenience wrapper: runs phase_gradient_detect, then restricts
    the summary to the known burst window (trimmed slightly at each
    edge to avoid filter edge effects). Only usable on simulated data
    where the burst timing is known from the ground truth."""
    safe_mask = erode_mask(mask, pixels=2) if mask is not None else None
    pgd, dirs, speeds = phase_gradient_detect(data, center_hz, fs, pixel_mm, mask=safe_mask)
    start_f = int(truth["burst_start_s"] * fs)
    dur_f = int(truth["burst_dur_s"] * fs)
    lo, hi = start_f + trim_frames, start_f + dur_f - trim_frames
    return summarize(pgd[lo:hi], dirs[lo:hi], speeds[lo:hi])


def estimate_radial_source(data, center_hz, fs, pixel_mm, mask=None, frame_range=None):
    """
    Estimates the source location of a radial (point-origin) wave.

    Method: at each valid pixel, the local phase gradient points in the
    direction the wave is moving AWAY from the source (outward). The
    line through that pixel, running backward along its gradient
    direction, should pass through the source. We collect these lines
    from every valid pixel across several frames and find the point
    minimizing total perpendicular distance to all of them.

    IMPORTANT CAVEAT (found by testing against a pure-noise control):
    with no real signal at all, this still returns an answer -- it
    converges toward the centroid of the evaluated pixels. A low
    residual_mm means the lines genuinely converge (trustworthy); a
    high residual_mm means the answer is really just the centroid
    fallback and should not be trusted.

    Returns: (x0, y0, residual_mm) or None if not enough valid data.
    """
    phase = get_phase(data, center_hz, fs)

    dphi_x = np.angle(np.exp(1j * (phase[:, 1:, :] - phase[:, :-1, :])))
    dphi_y = np.angle(np.exp(1j * (phase[1:, :, :] - phase[:-1, :, :])))
    gx = dphi_x[:-1, :, :] / pixel_mm
    gy = dphi_y[:, :-1, :] / pixel_mm

    ny, nx = gx.shape[0], gx.shape[1]
    ys = (np.arange(ny) + 0.5) * pixel_mm
    xs = (np.arange(nx) + 0.5) * pixel_mm
    X, Y = np.meshgrid(xs, ys)

    m = mask[:-1, :-1] if mask is not None else np.ones((ny, nx), dtype=bool)
    frames = range(*frame_range) if frame_range else range(data.shape[2])

    A = np.zeros((2, 2))
    b = np.zeros(2)
    all_d, all_p = [], []

    for ti in frames:
        vx, vy = gx[:, :, ti], gy[:, :, ti]
        mag = np.hypot(vx, vy)
        valid = m & ~np.isnan(vx) & ~np.isnan(vy) & (mag > 1e-6)
        if valid.sum() < 10:
            continue
        dx = vx[valid] / mag[valid]
        dy = vy[valid] / mag[valid]
        px, py = X[valid], Y[valid]
        for dxi, dyi, pxi, pyi in zip(dx, dy, px, py):
            d = np.array([dxi, dyi])
            Pm = np.eye(2) - np.outer(d, d)
            A += Pm
            b += Pm @ np.array([pxi, pyi])
            all_d.append(d)
            all_p.append([pxi, pyi])

    if len(all_d) < 50:
        return None
    try:
        source = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    all_d, all_p = np.array(all_d), np.array(all_p)
    diffs = all_p - source
    proj = (diffs * all_d).sum(axis=1)[:, None] * all_d
    perp = diffs - proj
    residual_mm = float(np.sqrt((perp ** 2).sum(axis=1).mean()))

    return float(source[0]), float(source[1]), residual_mm


def estimate_frequency(data, fs, mask=None, frame_range=None, freq_range=(2, 70)):
    """
    Finds the dominant oscillation frequency, directly from the data,
    with no assumption about wave type -- unlike PGD, which only works
    for planar waves, this looks purely at each pixel's own temporal
    power spectrum and averages across pixels, so spatial pattern
    (planar, radial, synchronous) is irrelevant to this step.

    Returns: (peak_freq, freqs, mean_power) for plotting/inspection.
    """
    if frame_range:
        seg = data[:, :, frame_range[0]:frame_range[1]]
    else:
        seg = data

    if mask is not None:
        pixels = seg[mask]
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


def shuffle_pixels(data, mask, seed=0):
    """
    Randomly reassigns which spatial location each pixel's time trace
    sits at, among valid (masked) pixels only. This destroys any real
    neighbor-to-neighbor spatial relationship while keeping every
    individual pixel's own time series completely intact.

    Used as a control: if phase_gradient_detect (or any detector above)
    still reports a confident wave on shuffled data, that detector is
    finding an artifact of its own math, not real structure.
    """
    rng = np.random.default_rng(seed)
    shuffled = data.copy()
    ys, xs = np.where(mask)
    order = rng.permutation(len(ys))
    shuffled[ys, xs, :] = data[ys[order], xs[order], :]
    return shuffled


if __name__ == "__main__":
    from wave_sim import make_block, make_mask, FS, PIXEL_MM

    mask = make_mask()

    print("=" * 60)
    print("TEST 1: Planar wave, direction/speed detection")
    print("=" * 60)
    data, truth = make_block("planar", freq_hz=8, speed_mm_s=20,
                              direction_deg=45, snr=5, seed=0, mask=mask)
    result = detect_in_burst_window(data, truth, 8, FS, PIXEL_MM, mask)
    print("ground truth:", {k: truth[k] for k in ("direction_deg", "speed_mm_s")})
    print("detected:   ", result)

    print("\n" + "=" * 60)
    print("TEST 2: Synchronous wave (PGD should correctly find nothing)")
    print("=" * 60)
    data2, truth2 = make_block("synchronous", freq_hz=8, snr=5, seed=0, mask=mask)
    result2 = detect_in_burst_window(data2, truth2, 8, FS, PIXEL_MM, mask)
    print("ground truth: synchronous (no direction/speed)")
    print("detected:   ", result2)

    print("\n" + "=" * 60)
    print("TEST 3: Noise sweep (planar)")
    print("=" * 60)
    for snr in (3.0, 1.0, 0.5, 0.3, 0.15):
        d, tr = make_block("planar", freq_hz=8, speed_mm_s=20, direction_deg=45,
                            snr=snr, seed=0, mask=mask)
        r = detect_in_burst_window(d, tr, 8, FS, PIXEL_MM, mask)
        print(f"  SNR={snr:>4}: direction={r['direction_deg']}, "
              f"speed={r['speed_mm_s']}, pgd={r['mean_pgd']:.2f}")

    print("\n" + "=" * 60)
    print("TEST 4: Radial source localization")
    print("=" * 60)
    for true_source in [(5.0, 5.0), (2.0, 7.0)]:
        d, tr = make_block("radial", freq_hz=8, speed_mm_s=20, source_xy=true_source,
                            snr=5, seed=0, mask=mask)
        sf, df = int(tr["burst_start_s"] * FS), int(tr["burst_dur_s"] * FS)
        safe_mask = erode_mask(mask, pixels=2)
        est = estimate_radial_source(d, 8, FS, PIXEL_MM, mask=safe_mask,
                                      frame_range=(sf + 50, sf + df - 50))
        print(f"  true source={true_source} -> estimated={est}")

    print("\n" + "=" * 60)
    print("TEST 5: Frequency estimation, ALL wave types (true freq = 8Hz)")
    print("        (PGD cannot do this for radial -- this uses")
    print("         estimate_frequency's power-spectrum approach instead)")
    print("=" * 60)
    for wtype in ["planar", "radial", "synchronous"]:
        kwargs = dict(freq_hz=8, snr=3, seed=0, mask=mask)
        if wtype == "planar":
            kwargs["speed_mm_s"] = 20
            kwargs["direction_deg"] = 45
        elif wtype == "radial":
            kwargs["speed_mm_s"] = 20
            kwargs["source_xy"] = (5, 5)
        d, tr = make_block(wtype, **kwargs)
        sf, df = int(tr["burst_start_s"] * FS), int(tr["burst_dur_s"] * FS)
        peak, _, _ = estimate_frequency(d, FS, mask=mask, frame_range=(sf, sf + df))
        print(f"  {wtype:12s}: detected = {peak:.2f} Hz")

    print("\n" + "=" * 60)
    print("TEST 6: Shuffle control (detection.py's own detect_in_burst_window")
    print("        rerun on scrambled data -- should collapse to no detection)")
    print("=" * 60)
    data6, truth6 = make_block("planar", freq_hz=8, speed_mm_s=20, direction_deg=45,
                                snr=3, seed=0, mask=mask)
    real_result = detect_in_burst_window(data6, truth6, 8, FS, PIXEL_MM, mask)
    print(f"  Real data:     direction={real_result['direction_deg']}, "
          f"PGD={real_result['mean_pgd']:.3f}")
    shuffled = shuffle_pixels(data6, mask, seed=1)
    shuffled_result = detect_in_burst_window(shuffled, truth6, 8, FS, PIXEL_MM, mask)
    print(f"  Shuffled data: direction={shuffled_result['direction_deg']}, "
          f"PGD={shuffled_result['mean_pgd']:.3f}")