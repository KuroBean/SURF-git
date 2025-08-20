
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


def _best_odd_window(n_samples: int, target: int) -> int:
    """
    Return a valid odd window length for Savitzky–Golay filtering.
    Ensures: 5 <= window <= n_samples - (n_samples+1)%2  (odd, <= n_samples)
    Falls back to the largest valid odd number if target is too large,
    or to 5 if the signal is very short (and increases to 3 if needed).
    """
    if n_samples < 3:
        # Too short to smooth meaningfully
        return 1

    # lower bound
    min_win = 5 if n_samples >= 5 else (3 if n_samples >= 3 else 1)

    # cap and make odd
    win = min(target, n_samples if n_samples % 2 == 1 else n_samples - 1)
    if win < min_win:
        win = min_win
    if win % 2 == 0:
        win -= 1
    win = max(win, 1)
    return win


def compute_pulse_velocity(
    csv_path,
    distance,
    sep=',',
    skiprows=0,
    threshold_frac=0.5,
    prominence=None,
    sample_dist=None,
    show=False,
    # --- new smoothing controls ---
    smooth=True,
    smooth_window=51,
    smooth_polyorder=3,
    overlay_raw=True
):
    """
    Reads a CSV (time, ch1, ch2, ...) and computes pulse velocity between the first
    two voltage channels. Now **smooths the data before peak detection** and plots
    the smoothed traces with the chosen peaks marked.

    Parameters
    ----------
    csv_path : str
        Path to CSV. First col = time (s), next cols = voltages.
    distance : float
        Separation between sensors (meters).
    sep : str
        CSV delimiter.
    skiprows : int
        Rows to skip at file start.
    threshold_frac : float in [0,1]
        Fraction of per-channel max used as absolute height threshold for peaks.
    prominence : float or dict, optional
        Forwarded to scipy.signal.find_peaks(prominence=...).
    sample_dist : int, optional
        Forwarded to scipy.signal.find_peaks(distance=...).
    show : bool
        If True, plot smoothed traces and mark chosen peaks.
    smooth : bool
        If True, apply Savitzky–Golay filter before peak-finding.
    smooth_window : int
        Target window length for smoothing (auto-adjusted to valid odd <= N).
    smooth_polyorder : int
        Polynomial order for Savitzky–Golay (must be < window length).
    overlay_raw : bool
        If True and `show`, overlay faint raw traces for reference.
    """
    # 1) Load
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows, skipinitialspace=True)
    t_col = df.columns[0]
    ch_cols = df.columns[1:3]  # use first two voltage columns
    df[t_col] = pd.to_numeric(df[t_col], errors='coerce')
    df[ch_cols] = df[ch_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[t_col, *ch_cols])

    time = df[t_col].to_numpy()
    v1_raw = df[ch_cols[0]].to_numpy()
    v2_raw = df[ch_cols[1]].to_numpy()

    # 2) Smooth before peak detection
    if smooth:
        print(f"2%: {len(v1_raw)*0.02}")
        w1 = _best_odd_window(len(v1_raw), int(smooth_window))
        w2 = _best_odd_window(len(v2_raw), int(smooth_window))
        # polyorder must be < window_length
        p1 = min(smooth_polyorder, max(1, w1 - 2))
        p2 = min(smooth_polyorder, max(1, w2 - 2))
        v1 = savgol_filter(v1_raw, window_length=w1, polyorder=p1, mode='interp')
        v2 = savgol_filter(v2_raw, window_length=w2, polyorder=p2, mode='interp')
    else:
        v1, v2 = v1_raw, v2_raw

    # 3) Peak detection on the (optionally) smoothed signals
    th1 = threshold_frac * np.nanmax(v1)
    th2 = threshold_frac * np.nanmax(v2)

    peaks1, props1 = find_peaks(v1, height=th1, prominence=prominence, distance=sample_dist)
    peaks2, props2 = find_peaks(v2, height=th2, prominence=prominence, distance=sample_dist)

    if len(peaks1) == 0 or len(peaks2) == 0:
        raise ValueError(
            "No peaks found above threshold. Try lowering threshold_frac, "
            "reducing prominence, or disabling smoothing."
        )

    # 4) Choose the *main* pulse peak per channel (highest amplitude)
    i1 = int(np.nanargmax(v1[peaks1]))
    i2 = int(np.nanargmax(v2[peaks2]))
    peak1 = int(peaks1[i1])
    peak2 = int(peaks2[i2])
    t_peak1 = float(time[peak1])
    t_peak2 = float(time[peak2])

    # 5) Velocity
    dt = t_peak2 - t_peak1
    if dt <= 0:
        raise ValueError(f"Non-positive time difference dt={dt:.6g}. Check channel ordering or sensor wiring.")
    velocity = float(distance / dt)

    # 6) Plot (smoothed + markers; optional raw overlay)
    if show:
        plt.figure(figsize=(10, 5))

        if overlay_raw:
            plt.plot(time, v1_raw, alpha=0.3, linewidth=1, label=f"{ch_cols[0]} (raw)")
            plt.plot(time, v2_raw, alpha=0.3, linewidth=1, label=f"{ch_cols[1]} (raw)")

        plt.plot(time, v1, label=f"{ch_cols[0]} (smoothed)")
        plt.plot(time, v2, label=f"{ch_cols[1]} (smoothed)")

        # mark *chosen* peak positions used for velocity
        plt.plot(time[peak1], v1[peak1], 'o', ms=7, label=f"Chosen peak {ch_cols[0]} @ {t_peak1:.6f}s")
        plt.plot(time[peak2], v2[peak2], 'o', ms=7, label=f"Chosen peak {ch_cols[1]} @ {t_peak2:.6f}s")

        # optional vertical lines to visualize dt
        ymin = np.nanmin([v1.min(), v2.min()])
        ymax = np.nanmax([v1.max(), v2.max()])
        plt.vlines([t_peak1, t_peak2], ymin, ymax, linestyles='dashed', alpha=0.5)

        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Pulse Detection (smoothed) and Velocity: {velocity:.2f} m/s')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # For quick visibility in logs
    print(f"[{csv_path}] peak1={t_peak1:.6f}s, peak2={t_peak2:.6f}s, dt={dt:.6g}s, v={velocity:.3f} m/s")

    return {
        't_peak1': t_peak1,
        't_peak2': t_peak2,
        'delta_t': dt,
        'velocity': velocity
    }


# --- Lightweight self-test (only runs if this file is executed directly) ---
if __name__ == '__main__':
    # Example (adjust to your local path)
    try:
        results = compute_pulse_velocity(
            r'.\1D chain tension changing exp\8_19 main data\10N\SCOPE_16.csv',
            distance=0.018*8,
            sep=',',
            skiprows=2,
            threshold_frac=0.5,
            prominence=0.1,
            show=True,
            smooth=True,
            smooth_window=201,
            smooth_polyorder=3,
            overlay_raw=True
        )
        print(results) 
    except Exception as e:
        print('Self-test failed:', e)
