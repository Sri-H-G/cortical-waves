"""
shuffle_control.py

A standard validation control, requested by Dieter: scramble the
spatial arrangement of the data and rerun the exact same detector. If
detection still reports a confident wave on scrambled data, that's
proof the method is finding an artifact of its own pipeline, not real
structure. If detection collapses (as it should), that validates the
method genuinely depends on real spatial organization.
"""
import numpy as np
from wave_sim import make_block, make_mask, FS, PIXEL_MM
from detection import detect_in_burst_window


def shuffle_pixels(data, mask, seed=0):
    """
    Randomly reassigns which spatial location each pixel's time trace
    sits at, among valid (masked) pixels only.

    This destroys any real neighbor-to-neighbor spatial relationship
    (the thing a traveling wave depends on), while keeping every
    individual pixel's own time series completely intact -- so if
    detection still finds a "wave" here, the method itself is broken,
    not the data.
    """
    rng = np.random.default_rng(seed)
    shuffled = data.copy()
    ys, xs = np.where(mask)
    order = rng.permutation(len(ys))
    shuffled[ys, xs, :] = data[ys[order], xs[order], :]
    return shuffled


if __name__ == "__main__":
    mask = make_mask()

    print("Shuffle control test: does detection collapse on scrambled data?\n")
    data, truth = make_block("planar", freq_hz=8, speed_mm_s=20, direction_deg=45,
                              snr=3, seed=0, mask=mask)

    real_result = detect_in_burst_window(data, truth, 8, FS, PIXEL_MM, mask)
    print(f"Real data:     direction={real_result['direction_deg']}, "
          f"PGD={real_result['mean_pgd']:.3f}")

    shuffled_data = shuffle_pixels(data, mask, seed=1)
    shuffled_result = detect_in_burst_window(shuffled_data, truth, 8, FS, PIXEL_MM, mask)
    print(f"Shuffled data: direction={shuffled_result['direction_deg']}, "
          f"PGD={shuffled_result['mean_pgd']:.3f}")

    print("\nIf shuffled PGD is much lower than real PGD, the detector is")
    print("genuinely using spatial structure, not finding a pipeline artifact.")