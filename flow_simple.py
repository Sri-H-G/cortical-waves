"""
flow_simple.py  -  a simple FLOW viewer

Two pictures:
  LEFT : the brain movie, with arrows showing which way activity is moving
  RIGHT: the FLOW summary.  ORANGE = where activity STARTS,
                             PURPLE = where activity ENDS.

A sentence at the top says what happened, in plain English.

Buttons:
  Example (left side) : pick a fake wave (we know the right answer) or a real trial
  Play                : play the movie
  Summarize this moment : redo the summary picture for 0.5 s around the current frame

Run:  python flow_simple.py     (needs flow.py in the same folder)
"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy import ndimage

import flow as F

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = "data/EMXJ21_0406"
MASK_PNG = "data/masks/EMXJ21_20220406_true_mask.png"

EXAMPLES = {
    "Fake: wave left to right": "planar",
    "Fake: ring spreading out": "ring",
    "Fake: ring shrinking in": "converging",
    "Fake: everything at once": "synchronous",
    "Real trial": None,
}
ORANGE, PURPLE = "#f28e2b", "#7b3fa0"
WINDOW = 100           # frames summarized (0.5 s)

S = {"example": list(EXAMPLES)[0], "R": None, "playing": False, "busy": False,
     "files": F.list_trials(DATA_DIR), "trial": 0}


# ------------------------------------------------------------------
# analysis (all from flow.py)
# ------------------------------------------------------------------
def load():
    kind = EXAMPLES[S["example"]]
    if kind is not None:
        data, mask, _ = F.make_synthetic(kind, n_frames=300, event=(100, 200), snr=5)
        name = S["example"]
    else:
        if not S["files"]:
            raise FileNotFoundError(f"No trial files found in {DATA_DIR}/")
        path = S["files"][S["trial"] % len(S["files"])]
        data = F.load_trial(path)
        mask = F.get_mask(data, MASK_PNG)
        name = "Real trial " + os.path.basename(path)
    d, u, v, w = F.run_flow(data, mask)
    _, us, vs, _ = F.run_flow(F.shuffle_pixels(data, mask), mask)   # scrambled control
    return {"name": name, "d": d, "mask": mask, "u": u, "v": v, "w": w,
            "us": us, "vs": vs, "fake": kind is not None}


def spread(field, mask, floor):
    """How spread out the high-FTLE region is (pixels). Small = one spot."""
    m = ndimage.binary_erosion(mask, iterations=2)
    f = np.nan_to_num(field)
    sel = m & (f >= max(np.percentile(f[m], 90), floor))
    if sel.sum() < 3:
        return np.inf
    yy, xx = np.nonzero(sel)
    return float(np.hypot(yy - yy.mean(), xx - xx.mean()).mean())


def direction_words(deg):
    names = ["left to right", "bottom to top", "right to left", "top to bottom"]
    return names[int(((deg + 45) % 360) // 90)]


def plain_sentence(p, s, mask):
    """Turn the numbers into one sentence a person can read."""
    v = s["verdict"]
    if v.startswith("no organised"):
        return ("No wave here: the activity does not move from place to place "
                "(it looks the same as scrambled data).", "firebrick")
    if v.startswith("planar"):
        return (f"A wave: activity starts at the ORANGE line and moves "
                f"{direction_words(s['direction_deg'])} to the PURPLE line.", "darkgreen")
    sf = spread(p["ftle_fwd"], mask, p["min_ftle"])
    sb = spread(p["ftle_bwd"], mask, p["min_ftle"])
    if sf < 0.6 * sb:
        return ("A wave: activity starts at ONE SPOT (orange) and spreads "
                "outward (purple).", "darkgreen")
    if sb < 0.6 * sf:
        return ("A wave: activity comes in from all around (orange) and "
                "ends at ONE SPOT (purple).", "darkgreen")
    return ("Activity is moving, but in a complicated pattern: look at where "
            "the orange and purple lines are.", "darkorange")


# ------------------------------------------------------------------
# figure
# ------------------------------------------------------------------
fig = plt.figure(figsize=(13, 7.5))
try:
    fig.canvas.manager.set_window_title("FLOW - simple viewer")
except Exception:
    pass
headline = fig.text(0.5, 0.95, "", ha="center", fontsize=15, weight="bold", wrap=True)
subline = fig.text(0.5, 0.905, "", ha="center", fontsize=10.5, color="0.35")

ax_mov = fig.add_axes([0.24, 0.22, 0.36, 0.62])
ax_sum = fig.add_axes([0.62, 0.22, 0.36, 0.62])
ax_mov.set_title("1. The movie (arrows = which way activity moves)", fontsize=11)
ax_sum.set_title("2. Summary: ORANGE = start,  PURPLE = end", fontsize=11)

ax_ex = fig.add_axes([0.01, 0.45, 0.21, 0.39])
ax_ex.set_title("Pick an example", fontsize=11, loc="left")
r_ex = RadioButtons(ax_ex, list(EXAMPLES), active=0)

b_play = Button(fig.add_axes([0.24, 0.07, 0.09, 0.06]), "Play")
b_sum = Button(fig.add_axes([0.35, 0.07, 0.16, 0.06]), "Summarize this moment")
b_next = Button(fig.add_axes([0.53, 0.07, 0.10, 0.06]), "Next trial")
s_fr = Slider(fig.add_axes([0.30, 0.155, 0.62, 0.025]), "Time", 0, 298, valinit=150, valstep=1)

fig.text(0.01, 0.40,
         "How to read it\n\n"
         "LEFT: red = active brain.\n"
         "Arrows show where the\nactivity is moving.\n\n"
         "RIGHT: one picture for\n0.5 s of the movie.\n"
         "Activity flows from\nORANGE to PURPLE.\n\n"
         "The fake examples have\nknown answers, so they\n"
         "show the method works.",
         fontsize=9.5, va="top", color="0.25")

A = {}


def show_movie():
    R = S["R"]
    d, mask = R["d"], R["mask"]
    ax_mov.clear()
    ax_mov.set_title("1. The movie (arrows = which way activity moves)", fontsize=11)
    ax_mov.set_xticks([]); ax_mov.set_yticks([])
    lim = np.percentile(np.abs(d[mask]), 99)
    A["img"] = ax_mov.imshow(np.where(mask, d[:, :, 0], np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax_mov.contour(mask, [0.5], colors="k", linewidths=1)
    yy, xx = np.mgrid[0:mask.shape[0]:3, 0:mask.shape[1]:3]
    A["yx"] = (yy, xx)
    scale = max(np.percentile(np.hypot(R["u"], R["v"])[mask], 99.5), 1e-6) * 12
    A["q"] = ax_mov.quiver(xx, yy, np.zeros(xx.shape), np.zeros(xx.shape), color="k",
                           scale=scale, scale_units="width", angles="xy", width=0.005)
    n = R["u"].shape[2] - 1
    s_fr.valmax = n
    s_fr.ax.set_xlim(0, n)


def show_frame(fr):
    R = S["R"]
    if R is None:
        return
    fr = int(np.clip(fr, 0, R["u"].shape[2] - 1))
    A["img"].set_data(np.where(R["mask"], R["d"][:, :, fr], np.nan))
    yy, xx = A["yx"]
    A["q"].set_UVC(R["u"][yy, xx, fr], -R["v"][yy, xx, fr])
    s_fr.valtext.set_text(f"{fr / F.FS:.2f} s")
    fig.canvas.draw_idle()


def summarize(_=None):
    R = S["R"]
    if R is None or S["busy"]:
        return
    S["busy"] = True
    subline.set_text("working ..."); fig.canvas.draw(); fig.canvas.flush_events()
    fr = int(s_fr.val)
    t0, t1 = fr - WINDOW // 2, fr + WINDOW // 2
    p, s, _ = F.portrait_with_control(R["u"], R["v"], R["w"], R["us"], R["vs"],
                                      R["mask"], t0, t1)
    t0, t1 = p["window"]
    ax_sum.clear(); ax_sum.set_xticks([]); ax_sum.set_yticks([])
    ax_sum.set_title("2. Summary: ORANGE = start,  PURPLE = end", fontsize=11)
    bg = np.where(R["mask"], np.abs(R["d"][:, :, t0:t1 + 1]).mean(axis=2), np.nan)
    ax_sum.imshow(bg, cmap="Greys_r")
    ax_sum.contour(R["mask"], [0.5], colors="k", linewidths=1)
    ry, rx = np.nonzero(p["ridge_fwd"])
    ax_sum.scatter(rx, ry, s=28, marker="s", color=ORANGE)
    ry, rx = np.nonzero(p["ridge_bwd"])
    ax_sum.scatter(rx, ry, s=28, marker="s", color=PURPLE)
    if s["verdict"].startswith("planar"):
        ax_sum.annotate("", xy=(s["end_yx"][1], s["end_yx"][0]),
                        xytext=(s["start_yx"][1], s["start_yx"][0]),
                        arrowprops=dict(arrowstyle="-|>", color="lime", lw=3))
    text, col = plain_sentence(p, s, R["mask"])
    headline.set_text(text); headline.set_color(col)
    subline.set_text(f"{R['name']}   |   summary of {t0 / F.FS:.2f} to {t1 / F.FS:.2f} s"
                     f"   |   {s['ratio']:.1f}x stronger than scrambled data")
    S["busy"] = False
    fig.canvas.draw_idle()


def reload():
    S["busy"] = True
    headline.set_text("Loading ..."); headline.set_color("black"); subline.set_text("")
    fig.canvas.draw(); fig.canvas.flush_events()
    try:
        S["R"] = load()
    except Exception as e:
        S["R"] = None; S["busy"] = False
        headline.set_text("Could not load the data"); headline.set_color("firebrick")
        subline.set_text(str(e)); fig.canvas.draw_idle()
        return
    S["busy"] = False
    show_movie()
    if S["R"]["fake"]:
        fr = 150                     # middle of the fake wave
    else:                            # real data: jump to the most organised moment
        tl = F.flow_timeline(S["R"]["u"], S["R"]["v"], S["R"]["mask"], S["R"]["w"])
        c = np.convolve(tl["coherence"], np.ones(WINDOW) / WINDOW, mode="same")
        fr = int(np.argmax(c))
    s_fr.eventson = False; s_fr.set_val(fr); s_fr.eventson = True
    show_frame(fr)
    summarize()


def on_example(label):
    S["example"] = label
    reload()


def on_next(_):
    if EXAMPLES[S["example"]] is not None:
        subline.set_text("'Next trial' is for real trials. Pick 'Real trial' first.")
        fig.canvas.draw_idle(); return
    S["trial"] += 1
    reload()


def on_play(_):
    S["playing"] = not S["playing"]
    b_play.label.set_text("Pause" if S["playing"] else "Play")
    fig.canvas.draw_idle()


def tick():
    if S["playing"] and not S["busy"] and S["R"] is not None:
        nxt = int(s_fr.val) + 2
        s_fr.set_val(0 if nxt > s_fr.valmax else nxt)


r_ex.on_clicked(on_example)
b_sum.on_clicked(summarize)
b_next.on_clicked(on_next)
b_play.on_clicked(on_play)
s_fr.on_changed(show_frame)
timer = fig.canvas.new_timer(interval=60)
timer.add_callback(tick)
timer.start()
fig._timer = timer

if __name__ == "__main__":
    reload()
    plt.show()
