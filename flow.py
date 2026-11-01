"""
flow.py

The FLOW method (Linden, Tabuena, Steinmetz, Moody, Brunton & Brunton 2021,
"Go with the FLOW", J. R. Soc. Interface 18: 20210523), implemented in
Python for our 200 Hz JEDI-1P voltage data.

This is a SECOND, INDEPENDENT detection method next to detection.py:

    detection.py (phase gradient / PGD)       flow.py (FLOW)
    ----------------------------------       -------------------------------
    narrow-band (filter around one Hz)        broadband (filter optional)
    uses PHASE of each pixel                  uses MOTION of the brightness pattern
    answers: is there a coherent wave         answers: where does activity start
    at this frame, which way, how fast?       and where does it end, over a window?

Pipeline (same three steps as the paper):
    1. Optical flow (Horn-Schunck): a velocity arrow at every pixel and frame.
       Arrows are scaled by how much the pixel's activity is changing
       (|dI/dt|), so flat regions don't produce strong arrows, then smoothed
       over time.
    2. FTLE (finite-time Lyapunov exponent): drop a particle on every pixel,
       let the arrows carry it for T frames, and measure how fast neighbours
       separate.  Forward in time -> ridges mark where activity STARTS
       (sources, orange).  Backward in time -> ridges mark where activity
       ENDS (sinks, purple).
    3. FLOW portrait: average the FTLE fields over a time window, keep the
       top few percent (default 90th percentile), thin to lines (skeletonize).
       Read it as: activity flows from ORANGE to PURPLE.

Extra, not in the paper (for Dieter's front-to-back sweep question):
    flow_timeline(): per frame, the mean flow direction, speed and a
    "flow coherence" (0 = arrows point every which way, 1 = all arrows agree;
    the flow analogue of PGD), plus the component along the anterior axis.

Units: arrows are in pixels/frame inside the module; speeds are converted to
mm/s with PIXEL_MM and FS.  PIXEL_MM is still the unconfirmed 0.2 (ask Dieter).
"""
import glob
import os

import numpy as np
from scipy import ndimage
from scipy.signal import butter, filtfilt

try:
    from skimage.morphology import skeletonize, remove_small_objects
except ImportError:          # portraits still work, just without thinning
    skeletonize = None
    remove_small_objects = None

try:
    import h5py
except ImportError:
    h5py = None

FS = 200            # Hz, JEDI-1P frame rate
PIXEL_MM = 0.2      # UNCONFIRMED (0.2 vs 0.1)


# =====================================================================
# Loading (identical to detection.py so both methods see the same data)
# =====================================================================

def load_trial(path):
    """Real trial .mat (v7.3/HDF5) -> (Y, X, T) array, sign restored."""
    if h5py is None:
        raise ImportError("pip install h5py")
    with h5py.File(path, "r") as f:
        raw = f["datBRLSreg"][:]                       # (T, 50, 50)
    return np.nan_to_num(-np.transpose(raw, (1, 2, 0)).astype(float))


def get_mask(data, png_path=None):
    """Use Dieter's true mask via detection.load_true_mask when available,
    otherwise fall back to the pixels that actually vary."""
    if png_path and os.path.exists(png_path):
        try:
            from detection import load_true_mask
            return load_true_mask(png_path, data)
        except ImportError:
            pass
    s = data.std(axis=2)
    m = s > 0.2 * np.median(s[s > 0]) if np.any(s > 0) else np.ones(s.shape, bool)
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def list_trials(folder="data/EMXJ21_0406"):
    return sorted(glob.glob(os.path.join(folder, "*.mat")))


# =====================================================================
# Preprocessing
# =====================================================================

def preprocess(data, mask, fs=FS, band=None, spatial_sigma=1.0,
               remove_global=False, data_time_sigma=2.0):
    """z-score each pixel, optional band-pass, optional removal of the
    cortex-wide shared signal, light spatial and temporal smoothing.
    Outside-mask = 0.

    band: None for broadband (as in the paper), or (lo, hi) in Hz.
    data_time_sigma: Gaussian smoothing over time in frames (2 frames =
        10 ms).  Needed at 200 Hz: from one frame to the next the real
        signal barely changes, so without it optical flow mostly tracks
        noise.  Ignored when a band-pass is used (the filter already
        smooths).
    """
    d = np.asarray(data, float).copy()
    d -= d.mean(axis=2, keepdims=True)
    sd = d.std(axis=2, keepdims=True)
    d /= np.where(sd > 0, sd, 1)
    if band is not None:
        lo, hi = band
        lo = max(lo, 0.3)
        hi = min(hi, 0.45 * fs)
        b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        d = filtfilt(b, a, d, axis=2)
    elif data_time_sigma and data_time_sigma > 0:
        d = ndimage.gaussian_filter1d(d, data_time_sigma, axis=2)
    if remove_global:
        g = d[mask].mean(axis=0)                       # (T,)
        d -= g[None, None, :]
    d[~mask] = 0.0
    if spatial_sigma and spatial_sigma > 0:
        # normalised convolution so the mask edge doesn't bleed zeros inward
        w = ndimage.gaussian_filter(mask.astype(float), spatial_sigma)
        d = ndimage.gaussian_filter(d, (spatial_sigma, spatial_sigma, 0))
        d /= np.where(w > 1e-6, w, 1)[:, :, None]
        d[~mask] = 0.0
    return d


def shuffle_pixels(data, mask, seed=0):
    """Fair control: move every pixel's time series to a random location
    inside the mask (one fixed arrangement for all frames). Same logic as
    detection.shuffle_pixels."""
    rng = np.random.default_rng(seed)
    out = data.copy()
    ys, xs = np.nonzero(mask)
    perm = rng.permutation(len(ys))
    out[ys, xs, :] = data[ys[perm], xs[perm], :]
    return out


# =====================================================================
# Step 1: Horn-Schunck optical flow (vectorised over all frames at once)
# =====================================================================

_HS_KERNEL = np.array([[1 / 12, 1 / 6, 1 / 12],
                       [1 / 6, 0.0, 1 / 6],
                       [1 / 12, 1 / 6, 1 / 12]])[:, :, None]


def horn_schunck(d, mask, alpha=1.0, n_iter=100):
    """Optical flow between every pair of consecutive frames.

    d: (Y, X, T) preprocessed movie.  Returns u, v of shape (Y, X, T-1),
    in pixels/frame; u = +x (columns), v = +y (rows).  alpha sets how smooth
    the field is (larger = smoother)."""
    d = d.astype(np.float32)
    f1, f2 = d[:, :, :-1], d[:, :, 1:]
    avg = 0.5 * (f1 + f2)
    Iy, Ix = np.gradient(avg, axis=(0, 1))
    It = f2 - f1
    u = np.zeros_like(It)
    v = np.zeros_like(It)
    denom = alpha ** 2 + Ix ** 2 + Iy ** 2
    for _ in range(n_iter):
        ub = ndimage.convolve(u, _HS_KERNEL, mode="nearest")
        vb = ndimage.convolve(v, _HS_KERNEL, mode="nearest")
        num = Ix * ub + Iy * vb + It
        u = ub - Ix * num / denom
        v = vb - Iy * num / denom
    u[~mask] = 0.0
    v[~mask] = 0.0
    return u, v, It


def compute_flow(d, mask, alpha=1.0, n_iter=100, scale_by_change=True,
                 time_sigma=2.0):
    """Step 1 of FLOW: optical flow, scaled by |dI/dt| and smoothed in time.
    Returns u, v (Y, X, T-1) in pixels/frame."""
    u, v, It = horn_schunck(d, mask, alpha, n_iter)
    a = np.abs(It)
    ref = np.percentile(a[mask], 99) if mask.any() else 1.0
    w = np.clip(a / (ref + 1e-12), 0, 1)
    if scale_by_change:
        u, v = u * w, v * w
    if time_sigma and time_sigma > 0:
        u = ndimage.gaussian_filter1d(u, time_sigma, axis=2)
        v = ndimage.gaussian_filter1d(v, time_sigma, axis=2)
        w = ndimage.gaussian_filter1d(w, time_sigma, axis=2)
    u[~mask] = 0.0
    v[~mask] = 0.0
    return u, v, w


# =====================================================================
# Step 2: FTLE from particle trajectories
# =====================================================================

def _vel_at(u, v, y, x, t):
    """Velocity at fractional positions (y, x) and fractional frame t:
    trilinear interpolation (bilinear in space, linear in time)."""
    T = u.shape[2]
    tt = np.full(y.size, float(np.clip(t, 0, T - 1)))
    coords = np.vstack([y.ravel(), x.ravel(), tt])
    uu = ndimage.map_coordinates(u, coords, order=1, mode="nearest")
    vv = ndimage.map_coordinates(v, coords, order=1, mode="nearest")
    return uu.reshape(y.shape), vv.reshape(y.shape)


def flow_map(u, v, t0, T_frames, substeps=1):
    """Where does a particle starting on each pixel at frame t0 end up after
    T_frames (negative = backward in time)?  RK4 integration, one step per
    frame (arrows are well under 1 pixel/frame at 200 Hz, so finer steps
    give the same answer)."""
    Y, X, _ = u.shape
    y, x = np.mgrid[0:Y, 0:X].astype(float)
    n = int(abs(T_frames) * substeps)
    if n == 0:
        return y, x
    h = np.sign(T_frames) / substeps
    t = float(t0)
    for _ in range(n):
        k1u, k1v = _vel_at(u, v, y, x, t)
        k2u, k2v = _vel_at(u, v, y + 0.5 * h * k1v, x + 0.5 * h * k1u, t + 0.5 * h)
        k3u, k3v = _vel_at(u, v, y + 0.5 * h * k2v, x + 0.5 * h * k2u, t + 0.5 * h)
        k4u, k4v = _vel_at(u, v, y + h * k3v, x + h * k3u, t + h)
        x = x + h * (k1u + 2 * k2u + 2 * k3u + k4u) / 6
        y = y + h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        x = np.clip(x, 0, X - 1)
        y = np.clip(y, 0, Y - 1)
        t += h
    return y, x


def ftle(u, v, t0, T_frames, mask, fs=FS):
    """FTLE field (1/s) at frame t0 for integration time T_frames.
    Positive T = forward (sources), negative T = backward (sinks)."""
    fy, fx = flow_map(u, v, t0, T_frames)
    dfx_dy, dfx_dx = np.gradient(fx)
    dfy_dy, dfy_dx = np.gradient(fy)
    # Cauchy-Green tensor C = J^T J, J = [[dfx/dx, dfx/dy], [dfy/dx, dfy/dy]]
    a = dfx_dx ** 2 + dfy_dx ** 2
    b = dfx_dx * dfx_dy + dfy_dx * dfy_dy
    d = dfx_dy ** 2 + dfy_dy ** 2
    lam = 0.5 * (a + d) + np.sqrt(0.25 * (a - d) ** 2 + b ** 2)
    out = np.log(np.sqrt(np.maximum(lam, 1e-12))) / (abs(T_frames) / fs)
    out[~mask] = np.nan
    return out


# =====================================================================
# Step 3: FLOW portrait
# =====================================================================

def _ridges(field, mask, percentile, min_ftle=0.0, min_size=3, rim=2):
    """Ridges = pixels in the top (100 - percentile)% of the field AND above
    an absolute floor (min_ftle, 1/s), thinned to lines.  The floor is what
    stops synchronous activity from producing ridges: a percentile alone
    always finds *something*.  The outer `rim` pixels of the mask are
    excluded, because particles stall at the edge of cortex and that
    creates fake separation there."""
    m = ndimage.binary_erosion(mask, iterations=rim) if rim else mask
    vals = field[m & np.isfinite(field)]
    if vals.size == 0:
        return np.zeros_like(mask)
    thr = max(np.percentile(vals, percentile), min_ftle)
    r = (np.nan_to_num(field, nan=-np.inf) >= thr) & m
    if skeletonize is not None and r.any():
        r = skeletonize(r)
        if remove_small_objects is not None and r.any():
            try:
                r = remove_small_objects(r, max_size=min_size - 1)
            except TypeError:            # older scikit-image
                r = remove_small_objects(r, min_size=min_size)
    return r


def flow_portrait(u, v, mask, t_start, t_end, T_frames=40, stride=5,
                  percentile=90, min_ftle=1.0, fs=FS):
    """Average forward and backward FTLE over frames t_start..t_end and
    extract ridges.

    T_frames: integration time.  40 frames = 0.2 s recovered the synthetic
        start/end lines best.  Longer T acts as a low-pass filter (smears
        fast events); shorter T gets noisier.  Sweep it.
    min_ftle: absolute floor (1/s) for a ridge.  Synthetic waves give
        FTLE ~2-6 /s, synchronous activity ~0.1-0.2 /s.
    """
    Tmax = u.shape[2] - 1
    t_start = int(np.clip(t_start, 0, Tmax))
    t_end = int(np.clip(t_end, t_start + 1, Tmax))
    fwd, bwd = [], []
    with np.errstate(all="ignore"):
        for t0 in range(t_start, t_end + 1, max(1, stride)):
            fwd.append(ftle(u, v, t0, +T_frames, mask, fs))
            bwd.append(ftle(u, v, t0, -T_frames, mask, fs))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            F = np.nanmean(fwd, axis=0)
            B = np.nanmean(bwd, axis=0)
    return {"ftle_fwd": F, "ftle_bwd": B,
            "ridge_fwd": _ridges(F, mask, percentile, min_ftle),
            "ridge_bwd": _ridges(B, mask, percentile, min_ftle),
            "window": (t_start, t_end), "T_frames": T_frames,
            "percentile": percentile, "min_ftle": min_ftle}


def _weighted_centre(field, mask, percentile, min_ftle, rim=2):
    """FTLE-weighted centre of the above-threshold region.  For a straight
    wavefront the FTLE is equally high all along the line, so a single
    peak would land at a random point on it; the weighted centre lands in
    the middle of the line."""
    m = ndimage.binary_erosion(mask, iterations=rim) if rim else mask
    f = np.nan_to_num(field)
    vals = f[m]
    if vals.size == 0:
        return None
    thr = max(np.percentile(vals, percentile), min_ftle)
    wgt = np.where(m & (f >= thr), f - thr + 1e-6, 0.0)
    if wgt.sum() <= 0:
        return None
    yy, xx = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    return float((wgt * yy).sum() / wgt.sum()), float((wgt * xx).sum() / wgt.sum())


def portrait_summary(p, mask, shuffled=None, coherence=None, pixel_mm=PIXEL_MM):
    """Plain-language readout of a portrait.

    start_yx / end_yx: FTLE-weighted centre of the forward (source) and
        backward (sink) regions.  None when nothing passes min_ftle.
    strength_fwd / strength_bwd: 95th percentile of FTLE inside cortex (1/s).
    shuffled: optional portrait_summary of the pixel-shuffled control; gives
        `ratio` = real strength / shuffled strength.  Synthetic waves give
        ~2.2, synchronous activity ~0.95.  We call >= 1.5 organised.
    coherence: optional mean flow coherence over the window (from
        flow_timeline).  High (>= 0.5) means all arrows agree = a planar
        sweep, so start->end direction is meaningful.  Low means radial or
        complex flow (e.g. a ring), where only the source/sink locations are.
    direction_deg: 0 = right, 90 = up on screen (same as detection.py).
    """
    inner = ndimage.binary_erosion(mask, iterations=2)
    F, B = p["ftle_fwd"], p["ftle_bwd"]
    sf = float(np.nanpercentile(F[inner], 95)) if inner.any() else np.nan
    sb = float(np.nanpercentile(B[inner], 95)) if inner.any() else np.nan
    s = _weighted_centre(F, mask, p["percentile"], p["min_ftle"]) if p["ridge_fwd"].any() else None
    e = _weighted_centre(B, mask, p["percentile"], p["min_ftle"]) if p["ridge_bwd"].any() else None
    out = {"start_yx": s, "end_yx": e, "direction_deg": None, "distance_mm": None,
           "strength_fwd": sf, "strength_bwd": sb, "ratio": None,
           "ridge_count_fwd": int(ndimage.label(p["ridge_fwd"])[1]),
           "ridge_count_bwd": int(ndimage.label(p["ridge_bwd"])[1])}
    if s is not None and e is not None:
        dy, dx = e[0] - s[0], e[1] - s[1]
        out["direction_deg"] = float(np.degrees(np.arctan2(-dy, dx)) % 360)
        out["distance_mm"] = float(np.hypot(dx, dy) * pixel_mm)
    if shuffled is not None:
        ref = 0.5 * (shuffled["strength_fwd"] + shuffled["strength_bwd"])
        out["ratio"] = float(0.5 * (sf + sb) / (ref + 1e-12))
    # verdict
    if out["ratio"] is not None and out["ratio"] < 1.5:
        out["verdict"] = "no organised flow (same as shuffled)"
    elif s is None and e is None:
        out["verdict"] = "no organised flow (nothing above FTLE floor)"
    elif coherence is not None and coherence >= 0.5 and out["direction_deg"] is not None:
        out["verdict"] = f"planar sweep toward {out['direction_deg']:.0f} deg"
    else:
        out["verdict"] = "radial / complex flow: read the source and sink locations"
    return out


def portrait_with_control(u, v, w, us, vs, mask, t_start, t_end, T_frames=40,
                          stride=5, percentile=90, fs=FS):
    """The version to use on REAL data.

    FTLE values are only comparable within one trial: the arrows are scaled
    by |dI/dt| relative to the trial's own 99th percentile, so a strong
    burst anywhere in the trial shrinks every other arrow.  A fixed FTLE
    floor therefore doesn't transfer between trials.  Instead the floor is
    set by the pixel-shuffled control of the SAME trial and window: a
    ridge must beat the top 5% of what scrambled data produces.

    u, v, w : flow of the real data (compute_flow)
    us, vs  : flow of the shuffled data (compute_flow on shuffle_pixels(...))
    Returns (portrait, summary, shuffled_summary)."""
    ps = flow_portrait(us, vs, mask, t_start, t_end, T_frames, stride,
                       percentile, min_ftle=0.0, fs=fs)
    ref = portrait_summary(ps, mask)
    floor = 0.5 * (ref["strength_fwd"] + ref["strength_bwd"])
    p = flow_portrait(u, v, mask, t_start, t_end, T_frames, stride,
                      percentile, min_ftle=floor, fs=fs)
    tl = flow_timeline(u, v, mask, w)
    coh = float(np.mean(tl["coherence"][p["window"][0]:p["window"][1] + 1]))
    summ = portrait_summary(p, mask, shuffled=ref, coherence=coh)
    if summ["ratio"] is not None and summ["ratio"] < 1.5:
        # not distinguishable from scrambled data: don't draw ridges
        p["ridge_fwd"] = np.zeros_like(p["ridge_fwd"])
        p["ridge_bwd"] = np.zeros_like(p["ridge_bwd"])
        summ["start_yx"] = summ["end_yx"] = summ["direction_deg"] = None
        summ["ridge_count_fwd"] = summ["ridge_count_bwd"] = 0
    p["min_ftle"] = floor
    return p, summ, ref


# =====================================================================
# Extra: per-frame flow timeline (front-to-back sweeps)
# =====================================================================

def flow_timeline(u, v, mask, w=None, fs=FS, pixel_mm=PIXEL_MM, anterior="up"):
    """Per frame: speed (mm/s), flow coherence (0..1), mean direction
    (deg, 0 = right, 90 = up on screen), anterior-posterior component
    (mm/s, + = toward anterior) and activity (mean |dI/dt| weight).

    Speed: the FLOW arrows are scaled by |dI/dt|, which shrinks them, so the
    speed divides that weighting back out (sum|arrow| / sum weight).  On
    synthetic waves this gives ~8.5 mm/s for a true 10.4 mm/s (Horn-Schunck
    smoothing biases it ~20% low).

    anterior: which side of the image is the front of the brain
    ('up', 'down', 'left', 'right').  CONFIRM WITH DIETER."""
    um, vm = u[mask], v[mask]                          # (n_pix, T-1)
    mag = np.hypot(um, vm)
    mean_u, mean_v = um.mean(axis=0), vm.mean(axis=0)
    coherence = np.hypot(mean_u, mean_v) / (mag.mean(axis=0) + 1e-12)
    direction = np.degrees(np.arctan2(-mean_v, mean_u)) % 360
    to_mm_s = pixel_mm * fs
    if w is not None:
        wm = w[mask]
        speed = mag.sum(axis=0) / (wm.sum(axis=0) + 1e-12)
        activity = wm.mean(axis=0)
        ap_px = (um.sum(axis=0), vm.sum(axis=0))
        ap_px = (ap_px[0] / (wm.sum(axis=0) + 1e-12), ap_px[1] / (wm.sum(axis=0) + 1e-12))
    else:
        speed = mag.mean(axis=0)
        activity = np.ones_like(speed)
        ap_px = (mean_u, mean_v)
    axis = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[anterior]
    ap = (ap_px[0] * axis[0] + ap_px[1] * axis[1]) * to_mm_s
    return {"speed_mm_s": speed * to_mm_s, "coherence": coherence,
            "direction_deg": direction, "ap_mm_s": ap, "activity": activity}


def find_sweeps(timeline, min_coherence=0.5, min_frames=5):
    """Contiguous runs where the flow is coherent; each sweep gets a start,
    end, mean direction and whether it is front->back or back->front."""
    c = timeline["coherence"] >= min_coherence
    sweeps = []
    i, n = 0, len(c)
    while i < n:
        if c[i]:
            j = i
            while j < n and c[j]:
                j += 1
            if j - i >= min_frames:
                ap = float(np.mean(timeline["ap_mm_s"][i:j]))
                ang = np.radians(timeline["direction_deg"][i:j])
                mdir = float(np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
                sweeps.append({"start": i, "end": j, "direction_deg": mdir,
                               "ap_mm_s": ap,
                               "kind": "back->front" if ap > 0 else "front->back"})
            i = j
        else:
            i += 1
    return sweeps


# =====================================================================
# Synthetic test movies (known answer)
# =====================================================================

def make_synthetic(kind="planar", n_frames=300, size=50, snr=5.0, seed=0,
                   event=(100, 200), mask=None):
    """Synthetic events with a known start and end:
      planar      : a band appears at x=12, travels right, vanishes at x=38
      ring        : a ring expands from the centre (25,25) to radius 18
      synchronous : the whole field brightens and fades together (no travel)
      converging  : a ring shrinks from radius 18 onto the centre
    Returns data (Y, X, T), mask, truth dict."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    if mask is None:
        mask = ((yy - 24.5) / 23) ** 2 + ((xx - 24.5) / 23) ** 2 <= 1
    data = np.zeros((size, size, n_frames))
    t0, t1 = event
    width = 2.5
    for t in range(t0, t1):
        s = (t - t0) / (t1 - t0)                       # 0 -> 1
        env = np.sin(np.pi * s) ** 0.5                 # fades in and out
        if kind == "planar":
            pos = 12 + 26 * s
            frame = np.exp(-((xx - pos) ** 2) / (2 * width ** 2))
        elif kind == "ring":
            r = 1 + 17 * s
            frame = np.exp(-((np.hypot(yy - 24.5, xx - 24.5) - r) ** 2) / (2 * width ** 2))
        elif kind == "converging":
            r = 18 - 17 * s
            frame = np.exp(-((np.hypot(yy - 24.5, xx - 24.5) - r) ** 2) / (2 * width ** 2))
        elif kind == "synchronous":
            frame = np.ones_like(xx)
            env = np.sin(np.pi * s)
        else:
            raise ValueError(kind)
        data[:, :, t] = env * frame
    data += rng.normal(0, 1.0 / snr, data.shape)
    data[~mask] = 0
    truths = {
        "planar": {"start_yx": (24.5, 12), "end_yx": (24.5, 38), "direction_deg": 0.0},
        "ring": {"start_yx": (24.5, 24.5), "end_yx": "outer ring", "direction_deg": None},
        "converging": {"start_yx": "outer ring", "end_yx": (24.5, 24.5), "direction_deg": None},
        "synchronous": {"start_yx": None, "end_yx": None, "direction_deg": None},
    }
    truth = dict(truths[kind], kind=kind, event=event)
    return data, mask, truth


# =====================================================================
# One-call convenience used by the GUI and scripts
# =====================================================================

def run_flow(data, mask, band=None, alpha=1.0, remove_global=False,
             spatial_sigma=1.0, time_sigma=2.0, fs=FS):
    d = preprocess(data, mask, fs, band, spatial_sigma, remove_global)
    u, v, w = compute_flow(d, mask, alpha=alpha, time_sigma=time_sigma)
    return d, u, v, w


if __name__ == "__main__":
    import time
    print("FLOW self-test on synthetic movies with a known answer")
    print("(positions are (row, col) in the 50x50 grid; event = frames 100-200)")
    for kind in ("planar", "ring", "converging", "synchronous"):
        data, mask, truth = make_synthetic(kind, snr=5)
        t = time.time()
        d, u, v, w = run_flow(data, mask)
        tl = flow_timeline(u, v, mask, w)
        ev = slice(115, 185)
        _, us, vs, ws = run_flow(shuffle_pixels(data, mask), mask)   # control
        p, s, ps = portrait_with_control(u, v, w, us, vs, mask, 95, 195)
        r = lambda c: None if c is None else tuple(round(x, 1) for x in c)
        print(f"\n--- {kind} ({time.time() - t:.1f} s) ---")
        print("  truth : start", truth["start_yx"], "| end", truth["end_yx"],
              "| direction", truth["direction_deg"])
        print("  FLOW  : start", r(s["start_yx"]), "| end", r(s["end_yx"]),
              "| direction", None if s["direction_deg"] is None else round(s["direction_deg"], 1))
        print(f"  strength fwd/bwd {s['strength_fwd']:.2f}/{s['strength_bwd']:.2f} /s"
              f"   shuffled {ps['strength_fwd']:.2f}/{ps['strength_bwd']:.2f} /s"
              f"   ratio {s['ratio']:.2f}")
        print("  verdict:", s["verdict"])
        print(f"  ridges fwd/bwd {s['ridge_count_fwd']}/{s['ridge_count_bwd']}"
              f"   speed {np.median(tl['speed_mm_s'][ev]):.1f} mm/s"
              f"   coherence {tl['coherence'][ev].mean():.2f}")
