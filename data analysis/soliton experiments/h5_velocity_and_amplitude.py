import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

# =====================================================================
# H5 loading utilities (Keysight oscilloscope format)
# =====================================================================

def _parse_keysight_preamble(preamble_str):
    """
    Parse the 10-field Keysight/Agilent waveform preamble string.

    Fields (comma-separated):
        0  FORMAT        (0=BYTE, 1=WORD, 4=ASCII)
        1  TYPE          (0=NORMal, 1=PEAK, 2=AVERage, 3=HRESolution)
        2  POINTS        number of data points
        3  COUNT         averaging count
        4  XINCREMENT    time step (seconds per sample)
        5  XORIGIN       time of the first sample (seconds)
        6  XREFERENCE    reference point index (usually 0)
        7  YINCREMENT    voltage per LSB (volts per ADC count)
        8  YORIGIN       voltage offset (volts)
        9  YREFERENCE    ADC reference level (counts)

    Returns a dict with all ten fields, keyed by name.
    """
    fields = [f.strip() for f in preamble_str.split(",")]
    if len(fields) < 10:
        raise ValueError(
            f"Preamble has {len(fields)} fields (expected 10): {preamble_str!r}"
        )
    return {
        "format":       int(float(fields[0])),
        "type":         int(float(fields[1])),
        "points":       int(float(fields[2])),
        "count":        int(float(fields[3])),
        "x_increment":  float(fields[4]),
        "x_origin":     float(fields[5]),
        "x_reference":  int(float(fields[6])),
        "y_increment":  float(fields[7]),
        "y_origin":     float(fields[8]),
        "y_reference":  int(float(fields[9])),
    }


def _reconstruct_waveform(raw_data, preamble):
    """
    Convert raw ADC byte array + parsed preamble dict into physical
    (time, voltage) arrays.

    voltage[i] = (raw[i] - y_reference) * y_increment + y_origin
    time[i]    = x_origin + i * x_increment
    """
    raw = np.asarray(raw_data, dtype=float)
    voltage = (raw - preamble["y_reference"]) * preamble["y_increment"] + preamble["y_origin"]
    n = len(raw)
    time = preamble["x_origin"] + np.arange(n) * preamble["x_increment"]
    return time, voltage


def load_h5(file_path, s1_channel="CHANnel2", s2_channel="CHANnel3"):
    """
    Load a Keysight-format HDF5 waveform file and return (t, s1, s2).

    Parameters
    ----------
    file_path : str or Path
        Path to the .h5 file.
    s1_channel, s2_channel : str
        Dataset names inside the H5 file for sensor 1 and sensor 2.
        Defaults match the oscilloscope recording script.

    Returns
    -------
    t  : 1-D ndarray, time in seconds
    s1 : 1-D ndarray, sensor 1 voltage in volts
    s2 : 1-D ndarray, sensor 2 voltage in volts
    """
    import h5py  # deferred so CSV-only users don't need h5py installed

    with h5py.File(file_path, "r") as hf:
        available = list(hf.keys())

        for ch_name, label in [(s1_channel, "s1"), (s2_channel, "s2")]:
            if ch_name not in hf:
                raise KeyError(
                    f"Dataset '{ch_name}' not found for {label}. "
                    f"Available datasets: {available}"
                )

        # --- sensor 1 ---
        ds1 = hf[s1_channel]
        pre1 = _parse_keysight_preamble(ds1.attrs["preamble"])
        t, s1 = _reconstruct_waveform(ds1[:], pre1)

        # --- sensor 2 ---
        ds2 = hf[s2_channel]
        pre2 = _parse_keysight_preamble(ds2.attrs["preamble"])
        t2, s2 = _reconstruct_waveform(ds2[:], pre2)

    # Sanity: both channels should share the same timebase.
    # If x_increment differs, warn but still return sensor-1's time axis.
    if not np.isclose(pre1["x_increment"], pre2["x_increment"], rtol=1e-6):
        import warnings
        warnings.warn(
            f"Channel timebases differ: {pre1['x_increment']} vs "
            f"{pre2['x_increment']} s/sample. Using {s1_channel}'s timebase."
        )

    return t, s1, s2


def load_csv(file_path, sep=",", skiprows=0, time_col=0, s1_col=1, s2_col=2):
    """
    Original CSV loader — unchanged behaviour.
    """
    df = pd.read_csv(file_path, sep=sep, skiprows=skiprows, header=None)
    t  = df.iloc[:, time_col].to_numpy().astype(float)
    s1 = df.iloc[:, s1_col].to_numpy().astype(float)
    s2 = df.iloc[:, s2_col].to_numpy().astype(float)
    return t, s1, s2


# =====================================================================
# Main analysis function
# =====================================================================

def compute_velocity_and_peak_accel(
    csv_path,                       # kept name for backward compat; accepts .csv or .h5
    distance,
    volts_per_mps2,
    # --- CSV-specific (ignored for H5) ---
    sep=",",
    skiprows=0,
    time_col=0,
    s1_col=1,
    s2_col=2,
    # --- H5-specific (ignored for CSV) ---
    s1_channel="CHANnel2",
    s2_channel="CHANnel3",
    # --- analysis parameters ---
    smooth=True,
    smooth_window=401,
    smooth_polyorder=3,
    threshold_frac=0.5,
    prominence=0.05,
    show=False,
    overlay_raw=True,
    overlay_raw_alpha=0.25,
    baseline_us=300,
    gain=4,
    save_plot=False
):
    """
    Compute pulse velocity and peak acceleration from a two-sensor
    waveform file (.csv or .h5).

    File type is auto-detected from the extension:
        .h5 / .hdf5  →  Keysight HDF5 format (preamble-scaled)
        anything else →  CSV with columns [time, s1, s2, ...]

    All other parameters are unchanged from the original CSV-only version.
    """
    # ---- auto-detect and load ----
    file_path = str(csv_path)
    ext = Path(file_path).suffix.lower()

    if ext in (".h5", ".hdf5"):
        t, s1, s2 = load_h5(file_path, s1_channel=s1_channel, s2_channel=s2_channel)
    else:
        t, s1, s2 = load_csv(file_path, sep=sep, skiprows=skiprows,
                             time_col=time_col, s1_col=s1_col, s2_col=s2_col)

    # ---- time step & baseline window ----
    dt = float(np.mean(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Time column appears invalid (non-positive dt).")

    n_baseline = max(1, int((baseline_us * 1e-6) / dt))
    if n_baseline > len(t):
        n_baseline = len(t)

    # ---- baseline subtraction ----
    base1 = float(np.nanmean(s1[:n_baseline]))
    base2 = float(np.nanmean(s2[:n_baseline]))
    s1b = s1 - base1
    s2b = s2 - base2

    # ---- optional smoothing ----
    if smooth and smooth_window > 3 and smooth_window % 2 == 1 and smooth_window <= len(s1b):
        s1v = savgol_filter(s1b, smooth_window, smooth_polyorder, mode="interp")
        s2v = savgol_filter(s2b, smooth_window, smooth_polyorder, mode="interp")
    else:
        s1v, s2v = s1b, s2b

    # ---- find first positive peaks ----
    pk1, _ = find_peaks(s1v, height=0, prominence=prominence)
    pk2, _ = find_peaks(s2v, height=0, prominence=prominence)
    if len(pk1) == 0 or len(pk2) == 0:
        raise RuntimeError("No positive peaks found on one or both sensors.")

    i1 = int(pk1[0])
    i2 = int(pk2[0])

    # ---- velocity from arrival-time difference ----
    dt_samp = i2 - i1
    if dt_samp <= 0:
        dt_samp = i1 - i2
    delay = dt_samp * dt
    if delay <= 0:
        raise RuntimeError("Non-positive delay; check sensor ordering and peaks.")
    velocity = distance / delay

    # ---- peak acceleration amplitude ----
    peak_accel_mps2 = (s2v[i2]) / (volts_per_mps2 * gain)

    # ---- plotting ----
    if show or save_plot:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

        ax[0].axvspan(t[0], t[min(n_baseline - 1, len(t) - 1)],
                      alpha=0.15, color="gray", label=f"Baseline ({baseline_us} µs)")
        ax[1].axvspan(t[0], t[min(n_baseline - 1, len(t) - 1)],
                      alpha=0.15, color="gray")

        if overlay_raw:
            ax[0].plot(t, s1b, lw=1.0, alpha=overlay_raw_alpha, label="Sensor 1 (raw, baseline-sub)")
            ax[1].plot(t, s2b, lw=1.0, alpha=overlay_raw_alpha, label="Sensor 2 (raw, baseline-sub)")

        ax[0].plot(t, s1v, lw=1.5, label="Sensor 1 (analysis)")
        ax[1].plot(t, s2v, lw=1.5, label="Sensor 2 (analysis)")

        ax[0].plot(t[i1], s1v[i1], "o", ms=7, label="Peak (S1)")
        ax[1].plot(t[i2], s2v[i2], "o", ms=7, label="Peak (S2)")

        ax[0].set_ylabel("Voltage (V, baseline-sub)")
        ax[1].set_ylabel("Voltage (V, baseline-sub)")
        ax[1].set_xlabel("Time (s)")
        ax[0].legend(loc="best")
        ax[1].legend(loc="best")
        ttl = (f"{file_path}\n"
               f"Velocity = {velocity:.2f} m/s, Peak accel ≈ {peak_accel_mps2:.2f} m/s²")
        fig.suptitle(ttl, fontsize=10)
        fig.tight_layout()

        if save_plot:
            stem = Path(file_path).stem
            fig.savefig(f"{stem} peaks.png", dpi=200)

        if show:
            plt.show()
        plt.close(fig)

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


# --- Lightweight self-test ---
if __name__ == '__main__':
    try:
        length_per_unit = 0.098*2 #from inside of hook to inside of hook, in meters
        results = compute_velocity_and_peak_accel(
            r".\90 deg soliton data\60g_20260429-175718.h5",
            distance=length_per_unit * 13,
            sep=',',
            s1_channel="CHANnel2",
            s2_channel="CHANnel3",
            skiprows=13,
            prominence=0.4,
            volts_per_mps2=0.00051,
            show=True,
            save_plot=True,
            smooth=True,
            smooth_window=401,
            smooth_polyorder=3,
            overlay_raw=True,
            baseline_us=300
        )
        print(results)
    except Exception as e:
        print('CSV self-test failed:', e)
