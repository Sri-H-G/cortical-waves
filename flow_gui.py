"""
flow_gui.py

Interactive viewer for the FLOW method (flow.py) on synthetic waves and on
real EMXJ21 trials.  All the analysis lives in flow.py; this file only draws.

Panels
  top-left    : the movie (preprocessed) with optical-flow arrows
  top-middle  : FLOW portrait for the window: orange = where activity starts
                (forward-FTLE ridges), purple = where it ends (backward-FTLE
                ridges), stars = their centres, arrow = start -> end when the
                flow is a planar sweep
  top-right   : the two FTLE maps the ridges come from
  bottom      : timeline over the whole trial: activity, flow coherence
                (0..1, flow version of PGD), and the front/back component of
                the flow; green/red spans = detected sweeps; grey = portrait
                window.  Click anywhere on the timeline to jump there.

Controls
  Source      : synthetic planar / ring / converging / synchronous, or a real trial
  Band        : broadband (as in the paper) or a band-pass
  Frame       : current frame (movie + arrows)
  Window      : portrait window length (frames, centred on the frame)
  T           : FTLE integration time (frames).  Long T smears fast events.
  Percentile  : ridge threshold
  Compute portrait / Next sweep / Real-Shuffled / < Trial > / Play
  Remove shared signal : subtract the cortex-wide mean each frame

Run:  python flow_gui.py
"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

import flow as F

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------
# Settings to check with Dieter
# ---------------------------------------------------------------------
DATA_DIR = "data/EMXJ21_0406"
MASK_PNG = "data/masks/EMXJ21_20220406_true_mask.png"
ANTERIOR = "up"          # which side of the image is the front of the brain
TASK_START_S = 2.0       # end of 2 s baseline = air-puff onset (Wang et al. Fig 1b)
HS_ITERS = 50            # 50 gives the same answers as 100, twice as fast

BANDS = {
    "Broadband": None,
    "Delta 1-4 Hz": (1, 4),
    "2-6 Hz": (2, 6),
    "8 Hz (6-10)": (6, 10),
    "Gamma 37 Hz": (33, 41),
}
SOURCES = ["Synthetic planar", "Synthetic ring", "Synthetic converging",
           "Synthetic synchronous", "Real trial"]

# colours
C_FWD = "#f28e2b"        # orange: sources
C_BWD = "#7b3fa0"        # purple: sinks
C_B2F = "#59a14f"        # back->front sweep
C_F2B = "#e15759"        # front->back sweep

S = {"source": SOURCES[0], "band": "Broadband", "remove_global": False,
     "shuffled": False, "trial_idx": 0, "files": F.list_trials(DATA_DIR),
     "cache": {}, "cur": None, "portrait": None, "summary": None,
     "playing": False, "busy": False, "msg": ""}


# ---------------------------------------------------------------------
# Data + analysis (all via flow.py)
# ---------------------------------------------------------------------

def load_source():
    src = S["source"]
    if src.startswith("Synthetic"):
        kind = src.split()[1]
        data, mask, truth = F.make_synthetic(kind, n_frames=400, event=(150, 250), snr=5)
        return data, mask, f"synthetic {kind} (event 0.75-1.25 s)", truth
    files = S["files"]
    if not files:
        raise FileNotFoundError(f"no .mat files in {DATA_DIR}/")
    path = files[S["trial_idx"] % len(files)]
    data = F.load_trial(path)
    mask = F.get_mask(data, MASK_PNG)
    return data, mask, os.path.basename(path), None


def analyse():
    key = (S["source"], S["trial_idx"] if S["source"] == "Real trial" else 0,
           S["band"], S["remove_global"], S["shuffled"])
    if key in S["cache"]:
        return S["cache"][key]
    data, mask, name, truth = load_source()
    if S["shuffled"]:
        data = F.shuffle_pixels(data, mask, seed=0)
    d = F.preprocess(data, mask, F.FS, BANDS[S["band"]], 1.0, S["remove_global"])
    u, v, w = F.compute_flow(d, mask, n_iter=HS_ITERS)
    tl = F.flow_timeline(u, v, mask, w, anterior=ANTERIOR)
    sweeps = F.find_sweeps(tl, min_coherence=0.5, min_frames=10)
    # arrows shown on screen: undo the |dI/dt| shrinkage where there is activity
    res = {"data": data, "d": d, "mask": mask, "u": u, "v": v, "w": w, "tl": tl,
           "sweeps": sweeps, "name": name, "truth": truth}
    if len(S["cache"]) > 6:
        S["cache"].clear()
    S["cache"][key] = res
    return res


# ---------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9.5))
try:
    fig.canvas.manager.set_window_title("FLOW viewer")
except Exception:
    pass
title = fig.text(0.5, 0.975, "", ha="center", va="top", fontsize=13, weight="bold")
status = fig.text(0.5, 0.945, "", ha="center", va="top", fontsize=10.5)

ax_mov = fig.add_axes([0.03, 0.47, 0.25, 0.44])
ax_por = fig.add_axes([0.30, 0.47, 0.25, 0.44])
ax_ff = fig.add_axes([0.57, 0.70, 0.15, 0.21])
ax_fb = fig.add_axes([0.57, 0.47, 0.15, 0.21])
cax_ff = fig.add_axes([0.725, 0.71, 0.008, 0.19])
cax_fb = fig.add_axes([0.725, 0.48, 0.008, 0.19])
ax_act = fig.add_axes([0.05, 0.335, 0.69, 0.085])
ax_coh = fig.add_axes([0.05, 0.245, 0.69, 0.085], sharex=ax_act)
ax_ap = fig.add_axes([0.05, 0.155, 0.69, 0.085], sharex=ax_act)

for a in (ax_mov, ax_por, ax_ff, ax_fb):
    a.set_xticks([]); a.set_yticks([])

# controls (right column)
ax_src = fig.add_axes([0.77, 0.72, 0.21, 0.19]); ax_src.set_title("Source", fontsize=10, loc="left")
ax_band = fig.add_axes([0.77, 0.52, 0.21, 0.17]); ax_band.set_title("Band", fontsize=10, loc="left")
ax_chk = fig.add_axes([0.77, 0.455, 0.21, 0.05])
r_src = RadioButtons(ax_src, SOURCES, active=0)
r_band = RadioButtons(ax_band, list(BANDS), active=0)
c_glob = CheckButtons(ax_chk, ["Remove shared signal"], [False])

b_comp = Button(fig.add_axes([0.77, 0.385, 0.10, 0.05]), "Compute portrait")
b_sweep = Button(fig.add_axes([0.88, 0.385, 0.10, 0.05]), "Next sweep")
b_shuf = Button(fig.add_axes([0.77, 0.325, 0.10, 0.05]), "Data: REAL")
b_play = Button(fig.add_axes([0.88, 0.325, 0.10, 0.05]), "Play")
b_prev = Button(fig.add_axes([0.77, 0.265, 0.10, 0.05]), "< Trial")
b_next = Button(fig.add_axes([0.88, 0.265, 0.10, 0.05]), "Trial >")

s_frame = Slider(fig.add_axes([0.08, 0.095, 0.62, 0.022]), "Frame", 0, 399, valinit=200, valstep=1)
s_win = Slider(fig.add_axes([0.08, 0.06, 0.25, 0.022]), "Window", 20, 400, valinit=100, valstep=10)
s_T = Slider(fig.add_axes([0.45, 0.06, 0.25, 0.022]), "T (frames)", 10, 100, valinit=40, valstep=5)
s_pct = Slider(fig.add_axes([0.08, 0.025, 0.25, 0.022]), "Percentile", 80, 98, valinit=90, valstep=1)
fig.text(0.45, 0.03, "T = 40 frames (0.2 s) recovered the synthetic start/end best. "
         "Long T low-pass filters fast events.", fontsize=8.5, color="0.35")
fig.text(0.77, 0.215, "How to read the portrait:\n"
         "  activity flows from ORANGE (start)\n  to PURPLE (end).\n"
         "Strength ratio = real / shuffled FTLE;\n  >= 1.5 means organised flow.\nRidges must beat the shuffled control.\n"
         f"Front of brain assumed: {ANTERIOR} (check).",
         fontsize=8.5, va="top", color="0.25")


# ---------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------
_artists = {}


def setup_trial():
    """Called when the data changes: set up images, timeline, slider range."""
    R = S["cur"]
    d, mask, tl = R["d"], R["mask"], R["tl"]
    nT = d.shape[2]
    s_frame.valmax = nT - 2
    s_frame.ax.set_xlim(0, nT - 2)
    t = np.arange(nT - 1) / F.FS

    vlim = np.percentile(np.abs(d[mask]), 99) if mask.any() else 1
    ax_mov.clear(); ax_mov.set_xticks([]); ax_mov.set_yticks([])
    img = np.where(mask, d[:, :, 0], np.nan)
    _artists["mov"] = ax_mov.imshow(img, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax_mov.contour(mask, [0.5], colors="k", linewidths=0.8)
    yy, xx = np.mgrid[0:mask.shape[0]:3, 0:mask.shape[1]:3]
    _artists["qyx"] = (yy, xx)
    mags = np.hypot(R["u"], R["v"])[mask]
    _artists["qscale"] = max(np.percentile(mags, 99.5), 1e-6) * 12
    _artists["quiv"] = ax_mov.quiver(xx, yy, np.zeros(xx.shape), np.zeros(xx.shape),
                                     color="k", scale=_artists["qscale"], width=0.004,
                                     angles="xy", scale_units="width")
    _artists["movtitle"] = ax_mov.set_title("", fontsize=10)

    # timeline
    for a in (ax_act, ax_coh, ax_ap):
        a.clear()
    act = np.abs(d[mask]).mean(axis=0)[:-1]
    ax_act.plot(t, act, color="0.3", lw=0.8)
    ax_act.set_ylabel("activity\n|z|", fontsize=8)
    ax_coh.plot(t, tl["coherence"], color="k", lw=0.8)
    ax_coh.axhline(0.5, color="0.5", ls=":", lw=0.8)
    ax_coh.set_ylim(0, 1); ax_coh.set_ylabel("flow\ncoherence", fontsize=8)
    ap = np.clip(tl["ap_mm_s"], -np.percentile(np.abs(tl["ap_mm_s"]), 99) - 1e-9,
                 np.percentile(np.abs(tl["ap_mm_s"]), 99) + 1e-9)
    ax_ap.fill_between(t, 0, ap, where=ap >= 0, color=C_B2F, lw=0, alpha=0.8)
    ax_ap.fill_between(t, 0, ap, where=ap < 0, color=C_F2B, lw=0, alpha=0.8)
    ax_ap.axhline(0, color="k", lw=0.5)
    ax_ap.set_ylabel("to front\nmm/s", fontsize=8)
    ax_ap.set_xlabel("time (s)", fontsize=9)
    for sw in R["sweeps"]:
        col = C_B2F if sw["ap_mm_s"] > 0 else C_F2B
        for a in (ax_act, ax_coh, ax_ap):
            a.axvspan(sw["start"] / F.FS, sw["end"] / F.FS, color=col, alpha=0.15, lw=0)
    if S["source"] == "Real trial":
        for a in (ax_act, ax_coh, ax_ap):
            a.axvline(TASK_START_S, color="b", ls="--", lw=0.8)
        ax_act.text(TASK_START_S, ax_act.get_ylim()[1], " air puff?", color="b",
                    fontsize=8, va="top")
    for a in (ax_act, ax_coh):
        plt.setp(a.get_xticklabels(), visible=False)
    for a in (ax_act, ax_coh, ax_ap):
        a.tick_params(labelsize=8)
        a.set_xlim(0, t[-1])
    _artists["vlines"] = [a.axvline(0, color="k", lw=1) for a in (ax_act, ax_coh, ax_ap)]
    _artists["spans"] = []

    n_b2f = sum(sw["ap_mm_s"] > 0 for sw in R["sweeps"])
    n_f2b = len(R["sweeps"]) - n_b2f
    ax_act.set_title(f"Timeline: {len(R['sweeps'])} coherent sweeps "
                     f"(green {n_b2f} back->front, red {n_f2b} front->back). "
                     "Click to jump.", fontsize=9, loc="left")


def draw_frame(fr):
    R = S["cur"]
    if R is None:
        return
    fr = int(np.clip(fr, 0, R["u"].shape[2] - 1))
    mask = R["mask"]
    _artists["mov"].set_data(np.where(mask, R["d"][:, :, fr], np.nan))
    yy, xx = _artists["qyx"]
    u = R["u"][yy, xx, fr]
    v = R["v"][yy, xx, fr]
    _artists["quiv"].set_UVC(u, -v)        # screen y is flipped vs image rows
    tl = R["tl"]
    _artists["movtitle"].set_text(
        f"t = {fr / F.FS:.3f} s   speed {tl['speed_mm_s'][fr]:.1f} mm/s   "
        f"coherence {tl['coherence'][fr]:.2f}")
    for ln in _artists["vlines"]:
        ln.set_xdata([fr / F.FS, fr / F.FS])
    fig.canvas.draw_idle()


def window_bounds():
    fr = int(s_frame.val)
    half = int(s_win.val) // 2
    nT = S["cur"]["u"].shape[2]
    return max(0, fr - half), min(nT - 1, fr + half)


def shuffled_flow():
    """Flow of the pixel-shuffled control for the current trial/band (cached)."""
    key = (S["source"], S["trial_idx"] if S["source"] == "Real trial" else 0,
           S["band"], S["remove_global"], True)
    if key not in S["cache"]:
        was = S["shuffled"]; S["shuffled"] = True
        try:
            analyse()
        finally:
            S["shuffled"] = was
    return S["cache"][key]


def compute_portrait(_=None):
    if S["busy"] or S["cur"] is None:
        return
    S["busy"] = True
    status.set_text("computing portrait + shuffled control ..."); status.set_color("0.4")
    fig.canvas.draw(); fig.canvas.flush_events()
    try:
        R = S["cur"]
        t0, t1 = window_bounds()
        T, pct = int(s_T.val), int(s_pct.val)
        if S["shuffled"]:
            # viewing the control itself: no reference to compare against
            p = F.flow_portrait(R["u"], R["v"], R["mask"], t0, t1, T, 5, pct, min_ftle=0.0)
            coh = float(np.mean(R["tl"]["coherence"][t0:t1 + 1]))
            summ = F.portrait_summary(p, R["mask"], coherence=coh)
            summ["verdict"] = ("SHUFFLED control: expect low FTLE, scattered ridges, "
                               "no coherent sweeps. Compare FTLE with REAL")
        else:
            Rs = shuffled_flow()
            p, summ, _ = F.portrait_with_control(R["u"], R["v"], R["w"], Rs["u"], Rs["v"],
                                                 R["mask"], t0, t1, T, 5, pct)
        S["portrait"], S["summary"] = p, summ
        draw_portrait()
    except Exception as e:
        status.set_text(f"ERROR: {e}"); status.set_color("red")
    finally:
        S["busy"] = False
        fig.canvas.draw_idle()


def draw_portrait():
    R, p, s = S["cur"], S["portrait"], S["summary"]
    mask = R["mask"]
    t0, t1 = p["window"]
    for a in (ax_por, ax_ff, ax_fb):
        a.clear(); a.set_xticks([]); a.set_yticks([])
    bg = np.where(mask, np.abs(R["d"][:, :, t0:t1 + 1]).mean(axis=2), np.nan)
    ax_por.imshow(bg, cmap="Greys_r")
    ax_por.contour(mask, [0.5], colors="k", linewidths=0.8)
    ry, rx = np.nonzero(p["ridge_fwd"])
    ax_por.scatter(rx, ry, s=14, marker="s", color=C_FWD, label="start (fwd FTLE)")
    ry, rx = np.nonzero(p["ridge_bwd"])
    ax_por.scatter(rx, ry, s=14, marker="s", color=C_BWD, label="end (bwd FTLE)")
    if s["verdict"].startswith("planar"):
        # stars/arrow only for planar sweeps; for a ring the centre of mass
        # of the sink is the middle of the ring, which would mislead
        ax_por.plot(s["start_yx"][1], s["start_yx"][0], "*", ms=16, mec="k", color=C_FWD)
        ax_por.plot(s["end_yx"][1], s["end_yx"][0], "*", ms=16, mec="k", color=C_BWD)
        ax_por.annotate("", xy=(s["end_yx"][1], s["end_yx"][0]),
                        xytext=(s["start_yx"][1], s["start_yx"][0]),
                        arrowprops=dict(arrowstyle="-|>", color="lime", lw=2.5))
    ax_por.legend(loc="lower center", fontsize=7.5, ncol=2, framealpha=0.85,
                  bbox_to_anchor=(0.5, -0.12))
    ax_por.set_title(f"FLOW portrait  {t0 / F.FS:.2f}-{t1 / F.FS:.2f} s  (T = {p['T_frames']} fr)",
                     fontsize=10)
    for a, cax, fld, cm, lab in ((ax_ff, cax_ff, p["ftle_fwd"], "Oranges", "forward FTLE (sources)"),
                                 (ax_fb, cax_fb, p["ftle_bwd"], "Purples", "backward FTLE (sinks)")):
        im = a.imshow(np.where(mask, fld, np.nan), cmap=cm, vmin=0,
                      vmax=max(np.nanpercentile(fld[mask], 99), p["min_ftle"], 1e-6))
        a.set_title(lab, fontsize=8.5)
        cax.clear()
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("1/s", fontsize=7)
    # window on the timeline
    for sp in _artists.get("spans", []):
        try:
            sp.remove()
        except Exception:
            pass
    _artists["spans"] = [a.axvspan(t0 / F.FS, t1 / F.FS, color="0.5", alpha=0.18, lw=0)
                         for a in (ax_act, ax_coh, ax_ap)]
    ratio = "" if s["ratio"] is None else f"   strength vs shuffled: {s['ratio']:.2f}x"
    status.set_text(f"{s['verdict']}{ratio}   (FTLE fwd {s['strength_fwd']:.2f}, "
                    f"bwd {s['strength_bwd']:.2f} /s)")
    good = s["verdict"].startswith(("planar", "radial"))
    status.set_color("darkgreen" if good else "firebrick")


def reload(_=None):
    if S["busy"]:
        return
    S["busy"] = True
    data_lbl = "SHUFFLED" if S["shuffled"] else "REAL"
    title.set_text("loading + optical flow ...")
    status.set_text("")
    fig.canvas.draw(); fig.canvas.flush_events()
    try:
        S["cur"] = analyse()
        R = S["cur"]
        setup_trial()
        title.set_text(f"FLOW  |  {R['name']}  |  {S['band']}"
                       f"{'  |  shared signal removed' if S['remove_global'] else ''}"
                       f"  |  data: {data_lbl}")
        title.set_color("firebrick" if S["shuffled"] else "black")
        b_shuf.label.set_text(f"Data: {data_lbl}")
        # start on the strongest sweep if there is one, else the middle
        if R["sweeps"]:
            sw = max(R["sweeps"], key=lambda z: z["end"] - z["start"])
            fr = (sw["start"] + sw["end"]) // 2
        else:
            fr = R["u"].shape[2] // 2
        s_frame.eventson = False; s_frame.set_val(fr); s_frame.eventson = True
        draw_frame(fr)
    except Exception as e:
        S["cur"] = None
        title.set_text("could not load data"); title.set_color("firebrick")
        status.set_text(f"{type(e).__name__}: {e}"); status.set_color("red")
        fig.canvas.draw_idle()
        S["busy"] = False
        return
    S["busy"] = False
    compute_portrait()


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------

def on_source(label):
    S["source"] = label
    S["shuffled"] = False
    reload()


def on_band(label):
    S["band"] = label
    reload()


def on_check(_label):
    S["remove_global"] = not S["remove_global"]
    reload()


def on_shuffle(_):
    S["shuffled"] = not S["shuffled"]
    fr = int(s_frame.val)
    reload()
    s_frame.set_val(fr)          # same frame, so you compare like with like
    compute_portrait()


def change_trial(step):
    if S["source"] != "Real trial":
        status.set_text("Trial buttons work on the Real trial source."); status.set_color("0.4")
        fig.canvas.draw_idle()
        return
    if S["files"]:
        S["trial_idx"] = (S["trial_idx"] + step) % len(S["files"])
    reload()


def next_sweep(_):
    R = S["cur"]
    if R is None or not R["sweeps"]:
        status.set_text("no coherent sweeps in this trial"); status.set_color("0.4")
        fig.canvas.draw_idle()
        return
    fr = int(s_frame.val)
    later = [sw for sw in R["sweeps"] if (sw["start"] + sw["end"]) // 2 > fr]
    sw = later[0] if later else R["sweeps"][0]
    centre = (sw["start"] + sw["end"]) // 2
    length = sw["end"] - sw["start"] + 30
    s_win.eventson = False; s_win.set_val(int(np.clip(round(length, -1), 20, 400))); s_win.eventson = True
    s_frame.eventson = False; s_frame.set_val(centre); s_frame.eventson = True
    draw_frame(centre)
    compute_portrait()


def on_click(ev):
    if ev.inaxes in (ax_act, ax_coh, ax_ap) and ev.xdata is not None and S["cur"] is not None:
        fr = int(round(ev.xdata * F.FS))
        s_frame.set_val(fr)
        if ev.dblclick:
            compute_portrait()


def on_pct(_val):
    compute_portrait()


def on_play(_):
    S["playing"] = not S["playing"]
    b_play.label.set_text("Pause" if S["playing"] else "Play")
    fig.canvas.draw_idle()


def tick():
    if not S["playing"] or S["busy"] or S["cur"] is None:
        return
    nxt = int(s_frame.val) + 2
    if nxt > s_frame.valmax:
        nxt = 0
    s_frame.set_val(nxt)


r_src.on_clicked(on_source)
r_band.on_clicked(on_band)
c_glob.on_clicked(on_check)
b_comp.on_clicked(compute_portrait)
b_sweep.on_clicked(next_sweep)
b_shuf.on_clicked(on_shuffle)
b_play.on_clicked(on_play)
b_prev.on_clicked(lambda _e: change_trial(-1))
b_next.on_clicked(lambda _e: change_trial(+1))
s_frame.on_changed(lambda v: draw_frame(v))
s_pct.on_changed(on_pct)
fig.canvas.mpl_connect("button_press_event", on_click)

timer = fig.canvas.new_timer(interval=60)
timer.add_callback(tick)
timer.start()
fig._timer = timer


if __name__ == "__main__":
    reload()
    plt.show()
