import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

def _coerce_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=cols)

def _smooth(x, window_samples=11, polyorder=3):
    # Ensure valid params
    window_samples = max(3, int(window_samples) | 1)  # odd
    polyorder = min(polyorder, window_samples - 2)
    try:
        return savgol_filter(x, window_samples, polyorder)
    except Exception:
        # Fallback: simple moving average if Savitzky-Golay fails
        k = max(3, window_samples)
        k += (k % 2 == 0)
        pad = k // 2
        xpad = np.pad(x, (pad, pad), mode='edge')
        ker = np.ones(k) / k
        return np.convolve(xpad, ker, mode='valid')

def _detect_peaks(time, sig, height_frac=0.25, prominence_frac=0.05,
                  polarity='auto'):
    """
    Returns indices of peaks and their properties.
    polarity: 'positive' | 'negative' | 'auto'
    """
    # Handle polarity (allow negative-going pulses)
    smax, smin = np.max(sig), np.min(sig)
    if polarity == 'auto':
        use_negative = abs(smin) > abs(smax)
    elif polarity == 'negative':
        use_negative = True
    else:
        use_negative = False

    sig_for_peaks = -sig if use_negative else sig

    # Relative thresholds
    amp_range = np.max(sig_for_peaks) - np.min(sig_for_peaks)
    height = height_frac * (np.max(sig_for_peaks) if np.max(sig_for_peaks) > 0 else 0)
    prominence = prominence_frac * (amp_range if amp_range > 0 else 0)

    peak_idx, props = find_peaks(
        sig_for_peaks,
        height=height if height > 0 else None,
        prominence=prominence if prominence > 0 else None
    )

    # Use original signal for amplitudes
    peak_times = time[peak_idx]
    peak_amps  = sig[peak_idx]
    return peak_idx, peak_times, peak_amps

def _group_peaks(peak_times, peak_amps, t_eps, a_eps, min_group_size=1):
    """
    Groups adjacent peaks if both time gap <= t_eps and amplitude gap <= a_eps.
    Returns list of dicts with group info.
    """
    if len(peak_times) == 0:
        return []

    order = np.argsort(peak_times)
    t = np.asarray(peak_times)[order]
    a = np.asarray(peak_amps)[order]

    groups = []
    g_idx = [order[0]]
    for i in range(1, len(t)):
        same_time_cluster = (t[i] - t[i-1]) <= t_eps
        similar_amp = abs(a[i] - a[i-1]) <= a_eps
        if same_time_cluster and similar_amp:
            g_idx.append(order[i])
        else:
            if len(g_idx) >= min_group_size:
                groups.append(g_idx.copy())
            g_idx = [order[i]]
    if len(g_idx) >= min_group_size:
        groups.append(g_idx)

    # Summaries
    out = []
    for g in groups:
        gt = peak_times[g]
        ga = peak_amps[g]
        out.append({
            "indices": g,
            "times": gt,
            "amps": ga,
            "t_mean": float(np.mean(gt)),
            "t_std":  float(np.std(gt, ddof=1)) if len(gt) > 1 else 0.0,
            "a_mean": float(np.mean(ga)),
            "a_std":  float(np.std(ga, ddof=1)) if len(ga) > 1 else 0.0,
            "a_max":  float(np.max(ga)),
            "a_abs_max": float(np.max(np.abs(ga))),
            "t_min":  float(np.min(gt))
        })
    # Sort by time
    out.sort(key=lambda d: d["t_mean"])
    return out

def _pick_main_group_by_max_amplitude(groups, global_max_amp, main_height_frac=0.4):
    """
    Pick the group with the largest absolute peak amplitude (a_abs_max).
    Require a_abs_max >= main_height_frac * abs(global_max_amp); if none pass,
    pick the absolute-largest group anyway.
    """
    if not groups:
        return None
    thresh = main_height_frac * abs(global_max_amp)

    # compute absolute max for each group if not present
    for g in groups:
        if "a_abs_max" not in g:
            g["a_abs_max"] = float(np.max(np.abs(g["amps"])))

    candidates = [g for g in groups if g["a_abs_max"] >= thresh]
    pool = candidates if candidates else groups
    return max(pool, key=lambda g: g["a_abs_max"])


def compute_pulse_velocity_with_uncertainty(
    
    csv_path,
    group_time_eps_samples=25,
    distance=0.018*8,      # meters
    u_distance=0.0001,        # meters (set if you have distance uncertainty)
    sep=",",
    skiprows=0,
    smooth=True,
    smooth_window_samples=15,
    smooth_polyorder=2,
    height_frac=0.05,      # peak must reach ≥ 25% of channel max (after smoothing)
    prominence_frac=0.005,  # peak must stick out by ≥ 5% of channel range
    polarity="auto",       # auto-detect negative vs positive pulses
    group_time_eps_s=5e-6,
    group_amp_eps_abs=2e4, # absolute amplitude proximity (if needed)
    main_height_frac=0.45,     # group must contain a peak ≥ 40% of channel max to count as "main"
    min_group_size=1,
    plot=True
):
    """
    1) Load CSV: time | ch1 | ch2 | (ch3 optional…)
    2) Smooth (optional)
    3) Detect peaks for ch1 & ch2 (with relative height/prominence)
    4) Group peaks by time & amplitude proximity
    5) Pick first 'main' group per channel
    6) Average times/amps + std devs (uncertainty)
    7) Compute Δt, velocity, and uncertainty propagation

    Returns a dict with detailed results.
    """
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows, skipinitialspace=True)
    time_col = df.columns[0]
    ch1_col  = df.columns[1]
    ch2_col  = df.columns[2]

    df = _coerce_numeric(df, [time_col, ch1_col, ch2_col])
    time = df[time_col].to_numpy()
    ch1  = df[ch1_col].to_numpy()
    ch2  = df[ch2_col].to_numpy()

    # Sort by time just in case
    order = np.argsort(time)
    time, ch1, ch2 = time[order], ch1[order], ch2[order]

    # Estimate dt and default grouping window in time
    if len(time) < 3:
        raise ValueError("Not enough samples.")
    dt = np.median(np.diff(time))
    # Time proximity window for grouping
    if group_time_eps_s is not None:
        t_eps = float(group_time_eps_s)              # use absolute seconds if given
    else:
        t_eps = float(group_time_eps_samples) * dt   # fallback to samples * dt

    # Smoothing (helps merge tiny wiggles)
    ch1_proc = _smooth(ch1, smooth_window_samples, smooth_polyorder) if smooth else ch1.copy()
    ch2_proc = _smooth(ch2, smooth_window_samples, smooth_polyorder) if smooth else ch2.copy()

    # Peak detection
    pk1_idx, pk1_t, pk1_a = _detect_peaks(time, ch1_proc, height_frac, prominence_frac, polarity=polarity)
    pk2_idx, pk2_t, pk2_a = _detect_peaks(time, ch2_proc, height_frac, prominence_frac, polarity=polarity)


    # Amplitude proximity window
    a_eps1 = float(group_amp_eps_abs)
    a_eps2 = float(group_amp_eps_abs)



    # Grouping
    groups1 = _group_peaks(pk1_t, pk1_a, t_eps=t_eps, a_eps=a_eps1, min_group_size=min_group_size)
    groups2 = _group_peaks(pk2_t, pk2_a, t_eps=t_eps, a_eps=a_eps2, min_group_size=min_group_size)

    if len(groups1) == 0 or len(groups2) == 0:
        raise ValueError("Could not form any peak groups on one or both channels. Try lowering height_frac/prominence_frac or widening group windows.")

    # Choose first main group for each channel
    g1 = _pick_main_group_by_max_amplitude(groups1, global_max_amp=np.max(ch1_proc), main_height_frac=main_height_frac)
    g2 = _pick_main_group_by_max_amplitude(groups2, global_max_amp=np.max(ch2_proc), main_height_frac=main_height_frac)

    if g1 is None or g2 is None:
        raise ValueError("Failed to select main peak group. Adjust main_height_frac or detection thresholds.")

    # Averages & uncertainties (std dev)
    t1_mean, t1_std = g1["t_mean"], g1["t_std"]
    t2_mean, t2_std = g2["t_mean"], g2["t_std"]

    delta_t = t2_mean - t1_mean
    if delta_t <= 0:
        raise ValueError(f"Non-positive Δt ({delta_t}). Check channel order or peak selection parameters.")

    u_dt = np.sqrt(t1_std**2 + t2_std**2)

    # Velocity and uncertainty propagation
    velocity = distance / delta_t
    # u_v = sqrt( (∂v/∂d * u_d)^2 + (∂v/∂Δt * u_Δt)^2 ) = sqrt( (1/Δt * u_d)^2 + (-d/Δt^2 * u_Δt)^2 )
    u_velocity = np.sqrt( (u_distance / delta_t)**2 + (distance * u_dt / (delta_t**2))**2 )

    results = {
        "channel_1": {
            "group_size": len(g1["indices"]),
            "t_mean": t1_mean,
            "t_std": t1_std,
            "a_mean": g1["a_mean"],
            "a_std": g1["a_std"],
            "t_values": g1["times"].tolist(),
            "a_values": g1["amps"].tolist(),
        },
        "channel_2": {
            "group_size": len(g2["indices"]),
            "t_mean": t2_mean,
            "t_std": t2_std,
            "a_mean": g2["a_mean"],
            "a_std": g2["a_std"],
            "t_values": g2["times"].tolist(),
            "a_values": g2["amps"].tolist(),
        },
        "delta_t": delta_t,
        "u_delta_t": u_dt,
        "distance": distance,
        "u_distance": u_distance,
        "velocity": velocity,
        "u_velocity": u_velocity,
        "dt_s": dt,
        "group_time_eps_s": t_eps,
        "group_amp_eps_ch1": a_eps1,
        "group_amp_eps_ch2": a_eps2
    }

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        # Channel 1
        ax[0].plot(time, ch1, alpha=0.5, label=f"{ch1_col} (raw)")
        ax[0].plot(time, ch1_proc, label=f"{ch1_col} (proc)")
        ax[0].scatter(pk1_t, pk1_a, s=25, label="peaks", zorder=3)
        # Highlight chosen group in red
        ax[0].scatter(g1["times"], g1["amps"], s=40, c="red", edgecolors="black", zorder=4, label="chosen group")
        ax[0].axvline(t1_mean, linestyle="--", color="red", label=f"grp mean t1={t1_mean:.6g} ± {t1_std:.2g}s")
        ax[0].set_ylabel("Voltage")
        ax[0].legend(loc="best")
        ax[0].set_title("Channel 1 peak grouping")

        # Channel 2
        ax[1].plot(time, ch2, alpha=0.5, label=f"{ch2_col} (raw)")
        ax[1].plot(time, ch2_proc, label=f"{ch2_col} (proc)")
        ax[1].scatter(pk2_t, pk2_a, s=25, label="peaks", zorder=3)
        # Highlight chosen group in red
        ax[1].scatter(g2["times"], g2["amps"], s=40, c="red", edgecolors="black", zorder=4, label="chosen group")
        ax[1].axvline(t2_mean, linestyle="--", color="red", label=f"grp mean t2={t2_mean:.6g} ± {t2_std:.2g}s")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Voltage")
        ax[1].legend(loc="best")
        ax[1].set_title(
            f"Δt = {delta_t:.6g} ± {u_dt:.2g} s | v = {velocity:.6g} ± {u_velocity:.2g} (units/s)"
        )

        plt.tight_layout()
        plt.show()

        return results

# --- Example usage ---
results = compute_pulse_velocity_with_uncertainty(
    r'.\1D chain tension changing exp\10N\scope_12.csv',
    distance=0.018*8,      # meters
    u_distance=0.0001,        # meters (set if you have distance uncertainty)
    sep=",",
    skiprows=0,
    smooth=True,
    smooth_window_samples=15,
    smooth_polyorder=2,
    height_frac=0.05,      # peak must reach ≥ 25% of channel max (after smoothing)
    prominence_frac=0.005,  # peak must stick out by ≥ 5% of channel range
    polarity="auto",       # auto-detect negative vs positive pulses
    group_time_eps_s=5e-6,
    group_amp_eps_abs=2e4, # absolute amplitude proximity (if needed)
    main_height_frac=0.45,     # group must contain a peak ≥ 40% of channel max to count as "main"
    min_group_size=1,
    plot=True
)
print(results)
