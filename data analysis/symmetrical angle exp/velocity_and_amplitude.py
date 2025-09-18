import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

def compute_velocity_and_peak_accel(
    csv_path,
    distance,
    volts_per_mps2,
    sep=",",
    skiprows=0,
    time_col=0,
    s1_col=1,
    s2_col=2,
    smooth=True,
    smooth_window=401,
    smooth_polyorder=3,
    threshold_frac=0.5,
    prominence=0.05,
    show=False,
    overlay_raw=True,
    overlay_raw_alpha=0.25,
    baseline_us=300,
    gain=2
):
    # ---- load ----
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows, header=None)
    t = df.iloc[:, time_col].to_numpy().astype(float)
    s1 = df.iloc[:, s1_col].to_numpy().astype(float)
    s2 = df.iloc[:, s2_col].to_numpy().astype(float)
    #print(s1)
    # ---- time step & baseline window (300 µs) ----
    # Assume t is in seconds; if your files use microseconds, convert first.
    # dt is robust to minor jitter by using median diff.
    dt = float(np.mean(np.diff(t)))
    #print(f'dt: {dt}')
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Time column appears invalid (non-positive dt).")

    n_baseline = max(1, int((baseline_us * 1e-6) / dt))
    if n_baseline > len(t):
        n_baseline = len(t)
    #print(f'n_baseline: {n_baseline}')

    # ---- compute baseline on first 300 µs, subtract it ----
    base1 = float(np.nanmean(s1[:n_baseline]))
    base2 = float(np.nanmean(s2[:n_baseline]))
    #print(f'base1: {base1}, base2: {base2}')
    s1b = s1 - base1
    s2b = s2 - base2

    # ---- optional smoothing for peak picking/visualization ----
    if smooth and smooth_window > 3 and smooth_window % 2 == 1 and smooth_window <= len(s1b):
        s1v = savgol_filter(s1b, smooth_window, smooth_polyorder, mode="interp")
        s2v = savgol_filter(s2b, smooth_window, smooth_polyorder, mode="interp")
    else:
        s1v, s2v = s1b, s2b

    # ---- find first positive peaks (relative to baseline) ----
    # height=0 ensures positive-going; tweak `prominence` as needed
    pk1, _ = find_peaks(s1v, height=0, prominence=prominence)
    pk2, _ = find_peaks(s2v, height=0, prominence=prominence)
    if len(pk1) == 0 or len(pk2) == 0:
        raise RuntimeError("No positive peaks found on one or both sensors.")

    i1 = int(pk1[0])
    i2 = int(pk2[0])

    # ---- compute velocity from arrival time difference ----
    dt_samp = i2 - i1
    if dt_samp <= 0:
        # If sensor order in the CSV is reversed in space, swap them:
        dt_samp = i1 - i2
    delay = dt_samp * dt  # seconds
    if delay <= 0:
        raise RuntimeError("Non-positive delay; check sensor ordering and peaks.")
    velocity = distance / delay  # m/s

    # ---- peak acceleration amplitude (use the downstream sensor’s first peak) ----
    # If you prefer sensor 1, change s2v[i2] -> s1v[i1].
    peak_accel_mps2 = (s2v[i2]) / (volts_per_mps2*gain)

    # ---- optional plot ----
    if show:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

        # shaded baseline region (first 300 µs)
        ax[0].axvspan(t[0], t[min(n_baseline-1, len(t)-1)], alpha=0.15, color="gray", label=f"Baseline ({baseline_us} µs)")
        ax[1].axvspan(t[0], t[min(n_baseline-1, len(t)-1)], alpha=0.15, color="gray")

        # raw overlays (low opacity) – baseline-subtracted for visual comparability
        if overlay_raw:
            ax[0].plot(t, s1b, linewidth=1.0, alpha=overlay_raw_alpha, label="Sensor 1 (raw, baseline-sub)")
            ax[1].plot(t, s2b, linewidth=1.0, alpha=overlay_raw_alpha, label="Sensor 2 (raw, baseline-sub)")

        # smoothed/analysis traces
        ax[0].plot(t, s1v, linewidth=1.5, label="Sensor 1 (analysis)")
        ax[1].plot(t, s2v, linewidth=1.5, label="Sensor 2 (analysis)")

        # mark chosen peaks
        ax[0].plot(t[i1], s1v[i1], "o", ms=7, label="Peak (S1)")
        ax[1].plot(t[i2], s2v[i2], "o", ms=7, label="Peak (S2)")

        # cosmetics
        ax[0].set_ylabel("Voltage (V, baseline-sub)")
        ax[1].set_ylabel("Voltage (V, baseline-sub)")
        ax[1].set_xlabel("Time (s)")
        ax[0].legend(loc="best")
        ax[1].legend(loc="best")
        ttl = f"{csv_path}\nVelocity = {velocity:.2f} m/s, Peak accel ≈ {peak_accel_mps2:.2f} m/s²"
        fig.suptitle(ttl, fontsize=10)
        fig.tight_layout()
        plt.show()

    return velocity, peak_accel_mps2, {
        "t": t,
        "s1_raw": s1,
        "s2_raw": s2,
        "s1_blsub": s1b,
        "s2_blsub": s2b,
        "s1_used": s1v,
        "s2_used": s2v,
        "peak_idx_s1": i1,
        "peak_idx_s2": i2,
        "baseline_s1": base1,
        "baseline_s2": base2,
        "baseline_us": baseline_us,
        "dt": dt,
    }
# --- Lightweight self-test (only runs if this file is executed directly) ---
if __name__ == '__main__':
    # Example (adjust to your local path)
    try:
        results = compute_velocity_and_peak_accel(
            r'.\8_21 main data\50V\scope_4.csv',
            distance=0.018*8,
            sep=',',
            skiprows=13,
            threshold_frac=0.5,
            prominence=0.01,
            volts_per_mps2=0.00051,
            show=True,
            smooth=True,
            smooth_window=401,
            smooth_polyorder=3,
            overlay_raw=True,
            baseline_us=300
        )
        print(results) 
    except Exception as e:
        print('Self-test failed:', e)
