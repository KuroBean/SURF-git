# velocity_vs_peak_accel_smooth.py
# Walk folders like "50V", "40 V", "30v", ...; compute pulse velocity and peak acceleration amplitude.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# ---------- Calibration / geometry ----------
# Accelerometers: 1.02 mV per (m/s^2)  ->  volts_to_mps2 = (V * 1000 mV/V) / 1.02
VOLTS_TO_MPS2 = 1000.0 / 1.02    # ≈ 980.392 m/s^2 per Volt

# ---------- Helpers ----------
def _best_odd_window(n_samples: int, target: int) -> int:
    """Valid odd window for Savitzky–Golay."""
    if n_samples < 3:
        return 1
    min_win = 5 if n_samples >= 5 else (3 if n_samples >= 3 else 1)
    win = min(target, n_samples if (n_samples % 2 == 1) else n_samples - 1)
    if win < min_win:
        win = min_win
    if win % 2 == 0:
        win -= 1
    return max(win, 1)


# ---------- Core: compute velocity & amplitude from a single CSV ----------
def compute_velocity_and_peak_accel(
    csv_path: str,
    distance: float,
    sep: str = ',',
    skiprows: int = 0,
    threshold_frac: float = 0.5,
    prominence=None,
    sample_dist=None,
    smooth: bool = True,
    smooth_window: int = 401,
    smooth_polyorder: int = 3,
    use_abs_peaks: bool = False,
    baseline_window_frac: float = 0.02,
    show: bool = False,
    overlay_raw: bool = True
):
    """
    Reads CSV (time, ch1, ch2, ...), smooths, finds 'main' peak in ch1 & ch2,
    computes velocity = distance / (t2 - t1) and the **peak acceleration amplitude**
    from the chosen peak of channel 1 (upstream). Returns dict.

    If your pulses are negative, set use_abs_peaks=True (peak finding on |signal|).
    """
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows, skipinitialspace=True)
    t_col = df.columns[0]
    ch_cols = df.columns[1:3]  # first two voltage channels

    df[t_col] = pd.to_numeric(df[t_col], errors='coerce')
    df[ch_cols] = df[ch_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[t_col, *ch_cols])

    time = df[t_col].to_numpy()
    v1_raw = df[ch_cols[0]].to_numpy()
    v2_raw = df[ch_cols[1]].to_numpy()

    # Smooth
    if smooth:
        w1 = _best_odd_window(len(v1_raw), int(smooth_window))
        w2 = _best_odd_window(len(v2_raw), int(smooth_window))
        p1 = min(smooth_polyorder, max(1, w1 - 2))
        p2 = min(smooth_polyorder, max(1, w2 - 2))
        v1 = savgol_filter(v1_raw, window_length=w1, polyorder=p1, mode='interp')
        v2 = savgol_filter(v2_raw, window_length=w2, polyorder=p2, mode='interp')
    else:
        v1, v2 = v1_raw, v2_raw

    # Optionally use absolute value for peak finding (helpful for negative pulses)
    v1_find = np.abs(v1) if use_abs_peaks else v1
    v2_find = np.abs(v2) if use_abs_peaks else v2

    th1 = threshold_frac * (np.nanmax(v1_find) if np.isfinite(np.nanmax(v1_find)) else 0.0)
    th2 = threshold_frac * (np.nanmax(v2_find) if np.isfinite(np.nanmax(v2_find)) else 0.0)

    peaks1, props1 = find_peaks(v1_find, height=th1, prominence=prominence, distance=sample_dist)
    peaks2, props2 = find_peaks(v2_find, height=th2, prominence=prominence, distance=sample_dist)

    if len(peaks1) == 0 or len(peaks2) == 0:
        raise ValueError(
            f"No peaks found above threshold in {os.path.basename(csv_path)}. "
            "Try lowering threshold_frac, reduce prominence, or set use_abs_peaks=True."
        )

    peak1 = int(peaks1[0])
    peak2 = int(peaks2[0])

    t_peak1 = float(time[peak1])
    t_peak2 = float(time[peak2])

    dt = t_peak2 - t_peak1
    if dt <= 0:
        raise ValueError(
            f"Non-positive dt={dt:.6g} for {os.path.basename(csv_path)}. "
            "Check channel ordering/wiring; ch1 should be upstream."
        )
    velocity = float(distance / dt)

    # Peak amplitude at ch1, relative to a local baseline (median of preceding window)
    preN = max(10, int(baseline_window_frac * len(v1)))
    start = max(0, peak1 - preN)
    baseline = 0#float(np.median(v1[start:1000])) if peak1 > 0 else 0.0
    amp_V = float(v1[peak1] - baseline)
    peak_accel_mps2 = float(abs(amp_V) * VOLTS_TO_MPS2)

    if show:
        plt.figure(figsize=(10, 5))
        if overlay_raw:
            plt.plot(time, v1_raw, alpha=0.3, linewidth=1, label=f"{ch_cols[0]} (raw)")
            plt.plot(time, v2_raw, alpha=0.3, linewidth=1, label=f"{ch_cols[1]} (raw)")
        plt.plot(time, v1, label=f"{ch_cols[0]} (smoothed)")
        plt.plot(time, v2, label=f"{ch_cols[1]} (smoothed)")
        plt.plot(time[peak1], v1[peak1], 'o', ms=7, label=f"ch1 peak @ {t_peak1:.6f}s")
        plt.plot(time[peak2], v2[peak2], 'o', ms=7, label=f"ch2 peak @ {t_peak2:.6f}s")
        ymin = float(min(np.nanmin(v1), np.nanmin(v2)))
        ymax = float(max(np.nanmax(v1), np.nanmax(v2)))
        plt.vlines([t_peak1, t_peak2], ymin, ymax, linestyles='dashed', alpha=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Pulse velocity: {velocity:.2f} m/s  |  ch1 peak accel: {peak_accel_mps2:.2f} m/s²')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        't_peak1': t_peak1,
        't_peak2': t_peak2,
        'delta_t': dt,
        'velocity_mps': velocity,
        'ch1_peak_amp_V': amp_V,
        'ch1_peak_accel_mps2': peak_accel_mps2,
        'peak1_idx': peak1,
        'peak2_idx': peak2
    }