"""
wave_sim.py

Generates synthetic data blocks shaped like the real EMXJ21 trials
(50x50 pixels, 200 Hz, 10.24 s = 2048 frames), saves them in the same
'datBRLSreg' layout, and writes the ground truth to a JSON file next to
each one. The simulator is independent of any analysis code.
"""
import json
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    import h5py
except ImportError:
    h5py = None

FS = 200
PIXEL_MM = 0.2  # 10 mm field of view / 50 pixels. Confirm with Dieter.


def make_mask(ny=50, nx=50):
    """Placeholder brain-shaped mask (ellipse). Replace with the real mask
    once you have derived one from the data."""
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    return (((yy - cy) / (0.48 * ny)) ** 2 + ((xx - cx) / (0.48 * nx)) ** 2) <= 1.0


def background_noise(shape, rng, alpha=1.0, spatial_sigma_px=1.0, white_frac=0.5):
    """Background activity: temporal 1/f^alpha noise, smoothed across
    neighboring pixels, plus independent white sensor noise. Unit std."""
    ny, nx, nt = shape
    white = rng.normal(0, 1, shape)

    spec = np.fft.rfft(white, axis=2)
    f = np.fft.rfftfreq(nt, d=1 / FS)
    f[0] = f[1]
    spec = spec / (f ** (alpha / 2))
    colored = np.fft.irfft(spec, n=nt, axis=2)
    colored = gaussian_filter(colored, sigma=(spatial_sigma_px, spatial_sigma_px, 0))
    colored /= colored.std()

    sensor = rng.normal(0, 1, shape)
    mixed = np.sqrt(1 - white_frac) * colored + np.sqrt(white_frac) * sensor
    return mixed / mixed.std()


def make_block(wave_type="planar", freq_hz=8.0, speed_mm_s=20.0,
               direction_deg=45.0, source_xy=(5.0, 5.0),
               burst_start_s=3.0, burst_dur_s=3.0,
               snr=1.0, duration_s=10.24, mask=None, seed=0,
               alpha=1.0, spatial_sigma_px=1.0, white_frac=0.5, scale=1.0):
    """
    wave_type: 'planar', 'radial', or 'synchronous' (same oscillation
               everywhere, zero delay between pixels)
    snr: signal RMS (inside the burst and mask) divided by noise std
    Returns data (50, 50, nt) and a ground-truth dict.
    """
    rng = np.random.default_rng(seed)
    nt = int(round(duration_s * FS))
    t = np.arange(nt) / FS
    ny = nx = 50
    xs = np.arange(nx) * PIXEL_MM
    ys = np.arange(ny) * PIXEL_MM
    X, Y = np.meshgrid(xs, ys)
    if mask is None:
        mask = make_mask()

    omega = 2 * np.pi * freq_hz
    warnings = []
    if wave_type != "synchronous":
        wl_px = speed_mm_s / freq_hz / PIXEL_MM
        if wl_px < 2:
            warnings.append(f"wavelength is {wl_px:.2f} pixels (<2): aliased, "
                            f"needs speed >= {2 * PIXEL_MM * freq_hz:.1f} mm/s")

    if wave_type == "planar":
        th = np.deg2rad(direction_deg)
        k = omega / speed_mm_s
        space_phase = k * (np.cos(th) * X + np.sin(th) * Y)
    elif wave_type == "radial":
        dist = np.sqrt((X - source_xy[0]) ** 2 + (Y - source_xy[1]) ** 2)
        space_phase = omega * dist / speed_mm_s
    elif wave_type == "synchronous":
        space_phase = np.zeros_like(X)
    else:
        raise ValueError("wave_type must be planar, radial, or synchronous")

    # smooth burst envelope (fades in and out, like a sporadic oscillation)
    env = np.zeros(nt)
    on = (t >= burst_start_s) & (t < burst_start_s + burst_dur_s)
    env[on] = np.hanning(on.sum())

    signal = env[None, None, :] * np.cos(
        space_phase[:, :, None] - omega * t[None, None, :])
    signal *= mask[:, :, None]

    active = env > 0.5
    sig_rms = np.sqrt((signal[:, :, active][mask] ** 2).mean())
    noise = background_noise((ny, nx, nt), rng, alpha, spatial_sigma_px, white_frac)
    noise *= mask[:, :, None]

    data = (signal + noise * (sig_rms / snr)) * scale

    truth = dict(wave_type=wave_type, freq_hz=freq_hz, speed_mm_s=speed_mm_s,
                 direction_deg=direction_deg, source_xy=list(source_xy),
                 burst_start_s=burst_start_s, burst_dur_s=burst_dur_s,
                 snr=snr, fs=FS, pixel_mm=PIXEL_MM, seed=seed, warnings=warnings)
    return data, truth


def save_block(data, truth, path):
    """Saves in the same layout the real loader expects. The viewers negate
    the stored values (regressed data is inverted), so we store -data."""
    stored = np.transpose(-data, (2, 0, 1)).astype(np.float32)  # (T, 50, 50)
    with h5py.File(path, "w") as f:
        f.create_dataset("datBRLSreg", data=stored)
    with open(path.replace(".mat", "_truth.json"), "w") as f:
        json.dump(truth, f, indent=2)


if __name__ == "__main__":
    for i, snr in enumerate([5.0, 1.0, 0.3], start=1):
        d, tr = make_block("planar", snr=snr, seed=i)
        save_block(d, tr, f"data/SIM_0001-{i:03d}.mat")
    d, tr = make_block("synchronous", snr=1.0, seed=10)
    save_block(d, tr, "data/SIM_0001-004.mat")
    print("saved 4 blocks to data/")