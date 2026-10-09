"""
wave_overlay_viewer.py
Overlay of detection.py's own output on the simulated movie.
All heavy computation happens ONCE on Regenerate. Frame changes only index
precomputed arrays, so Play stays smooth.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

from wave_sim import make_block, make_mask, FS, PIXEL_MM
from detection import (get_phase, phase_gradient_detect, erode_mask,
                       estimate_radial_source, estimate_frequency)

FREQ = 8.0
PLAY_STEP = 2          # frames advanced per timer tick
PLAY_MS = 50           # timer interval


def run_viewer():
    S = dict(data=None, phase=None, pgd=None, dirs=None, speeds=None,
             truth=None, src=None, freq=None, contour=None, arrow=None,
             playing=False, busy=False, drawing=False)
    mask = make_mask()
    emask = erode_mask(mask, 1)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([0.30, 0.18, 0.66, 0.76])
    im = ax.imshow(np.zeros((50, 50)), cmap="RdBu", vmin=-2, vmax=2)
    ax.axis("off")
    title = ax.set_title("")
    true_pt, = ax.plot([], [], "r+", ms=14, mew=2)
    est_pt, = ax.plot([], [], "gx", ms=12, mew=2)

    ax_radio = fig.add_axes([0.03, 0.72, 0.20, 0.15])
    radio = RadioButtons(ax_radio, ("Planar", "Radial", "Synchronous"))
    ax_snr = fig.add_axes([0.06, 0.62, 0.17, 0.03])
    snr_sl = Slider(ax_snr, "SNR", 0.3, 5.0, valinit=2.0)
    ax_reg = fig.add_axes([0.03, 0.53, 0.20, 0.05])
    btn_reg = Button(ax_reg, "Regenerate")
    ax_play = fig.add_axes([0.03, 0.45, 0.20, 0.05])
    btn_play = Button(ax_play, "Play")
    ax_stat = fig.add_axes([0.03, 0.20, 0.23, 0.2]); ax_stat.axis("off")
    stat = ax_stat.text(0, 1, "", va="top", fontsize=9)
    ax_fr = fig.add_axes([0.30, 0.08, 0.66, 0.03])
    fr_sl = Slider(ax_fr, "Frame", 0, 2047, valinit=600, valstep=1)

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
        if S["data"] is None or S["drawing"]:
            return
        S["drawing"] = True
        try:
            frame = int(frame)
            im.set_data(S["data"][:, :, frame])
            clear_overlay()
            ph = S["phase"][:, :, frame]
            if np.isfinite(ph).sum() > 10:
                S["contour"] = ax.contour(ph, levels=[-np.pi / 2, 0, np.pi / 2],
                                          colors="white", linewidths=1.2)
            pgd, d, sp = S["pgd"][frame], S["dirs"][frame], S["speeds"][frame]
            lines = [f"True: {S['truth']['wave_type']}",
                     f"Frame {frame}  ({frame / FS:.2f} s)",
                     f"Frequency: {S['freq']:.1f} Hz" if S["freq"] else "",
                     f"PGD: {pgd:.2f}"]
            if pgd >= 0.5 and np.isfinite(d):
                x0, y0 = 25, 25
                dx, dy = 10 * np.cos(np.deg2rad(d)), 10 * np.sin(np.deg2rad(d))
                S["arrow"] = ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                                         arrowprops=dict(arrowstyle="-|>", color="lime", lw=3))
                lines += [f"LIVE DETECTION", f"direction: {d:.1f} deg",
                          f"speed: {sp:.1f} mm/s"]
                stat.set_color("black")
            else:
                lines += ["no coherent wave"]
                stat.set_color("red")
            if S["src"] is not None:
                lines += [f"radial source: ({S['src'][0]:.1f}, {S['src'][1]:.1f})",
                          f"residual: {S['src'][2]:.2f} mm"]
            stat.set_text("\n".join(l for l in lines if l))
            fig.canvas.draw_idle()
        finally:
            S["drawing"] = False

    def regenerate(_=None):
        if S["busy"]:
            return
        S["busy"] = True
        S["playing"] = False
        btn_play.label.set_text("Play")
        stat.set_text("Computing...\n(a few seconds)")
        stat.set_color("black")
        fig.canvas.draw_idle(); plt.pause(0.001)
        try:
            wt = radio.value_selected.lower()
            data, truth = make_block(wave_type=wt, freq_hz=FREQ, speed_mm_s=20.0,
                                     direction_deg=45.0, source_xy=(2.0, 7.0),
                                     snr=snr_sl.val, mask=mask, seed=0)
            ph = get_phase(data, FREQ, FS).astype(float)
            ph[~emask, :] = np.nan
            pgd, dirs, speeds = phase_gradient_detect(data, FREQ, FS, PIXEL_MM, emask)
            a = int(truth["burst_start_s"] * FS) + 50
            b = int((truth["burst_start_s"] + truth["burst_dur_s"]) * FS) - 50
            src = None
            if wt == "radial":
                try:
                    src = estimate_radial_source(data, FREQ, FS, PIXEL_MM, emask,
                                                 frame_range=(a, b))
                except Exception:
                    src = None
            freq, _, _ = estimate_frequency(data, FS, mask=emask, frame_range=(a, b))
            S.update(data=data, phase=ph, pgd=np.asarray(pgd), dirs=np.asarray(dirs),
                     speeds=np.asarray(speeds), truth=truth, src=src, freq=freq)
            if wt == "radial":
                true_pt.set_data([truth["source_xy"][0] / PIXEL_MM],
                                 [truth["source_xy"][1] / PIXEL_MM])
                if src is not None:
                    est_pt.set_data([src[0] / PIXEL_MM], [src[1] / PIXEL_MM])
                else:
                    est_pt.set_data([], [])
            else:
                true_pt.set_data([], []); est_pt.set_data([], [])
            title.set_text(f"{wt} wave, SNR {snr_sl.val:.1f}")
            mid = (a + b) // 2
            fr_sl.eventson = False; fr_sl.set_val(mid); fr_sl.eventson = True
            draw(mid)
        except Exception as e:
            stat.set_text(f"ERROR: {e}"); stat.set_color("red")
            fig.canvas.draw_idle()
        finally:
            S["busy"] = False

    def on_play(_):
        S["playing"] = not S["playing"]
        btn_play.label.set_text("Pause" if S["playing"] else "Play")

    def tick():
        if not S["playing"] or S["data"] is None or S["busy"] or S["drawing"]:
            return
        nxt = int(fr_sl.val) + PLAY_STEP
        if nxt > 2047:
            nxt = 0
        fr_sl.eventson = False; fr_sl.set_val(nxt); fr_sl.eventson = True
        draw(nxt)

    fr_sl.on_changed(lambda v: draw(v))
    btn_reg.on_clicked(regenerate)
    radio.on_clicked(lambda _l: regenerate())
    btn_play.on_clicked(on_play)

    timer = fig.canvas.new_timer(interval=PLAY_MS)
    timer.add_callback(tick)
    timer.start()
    fig._timer = timer          # keep a reference or it gets garbage collected

    regenerate()
    return fig


if __name__ == "__main__":
    run_viewer()
    plt.show()