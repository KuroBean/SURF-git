# velocity_vs_peak_accel_smooth.py
# Walk folders like "50V", "40 V", "30v", ...; compute pulse velocity and peak acceleration amplitude.

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from velocity_and_amplitude import compute_velocity_and_peak_accel

# ---------- Calibration / geometry ----------
# Accelerometers: 1.02 mV per (m/s^2)  ->  volts_to_mps2 = (V * 1000 mV/V) / 1.02
VOLTS_TO_MPS2 = 1000.0 / 1.02    # ≈ 980.392 m/s^2 per Volt
DISTANCE = 0.018 * 8             # meters, separation between sensors (adjust if needed)

# --- voltage parsing (robust: matches "5V", "5 V", "amp_5V_gain_6", "V5", "5 volts") ---
_V_PATTERNS = [
    re.compile(r'(?<!\d)(\d+(?:\.\d+)?)\s*[vV](?!\d)'),        # number ... V not followed by a digit  (handles underscores etc.)
    re.compile(r'[vV]\s*(\d+(?:\.\d+)?)'),                     # V first, then number: "V5", "v 12.5"
    re.compile(r'(?<!\d)(\d+(?:\.\d+)?)\s*volts?\b', re.I),    # "5 volt", "5 volts"
]

def _parse_voltage_from_folder(folder_name: str):
    for pat in _V_PATTERNS:
        m = pat.search(folder_name)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


# Accept patterns like "gain 6", "Gain=6", "g6"
_GAIN_PATTERNS = [
    re.compile(r'gain\s*[:=]?\s*(\d+(?:\.\d+)?)', re.IGNORECASE),
    re.compile(r'\bg\s*[:=]?\s*(\d+(?:\.\d+)?)\b', re.IGNORECASE),
]

def _parse_gain(folder_name: str, default=2.0):
    for pat in _GAIN_PATTERNS:
        m = pat.search(folder_name)
        if m:
            return float(m.group(1))
    return float(default)


def _find_csvs(folder_path: str):
    """Return CSV paths directly inside folder_path."""
    return sorted(
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith('.csv')
    )

# ---------- Batch over voltage folders ----------
def batch_velocity_vs_peak_accel(
    root_dir,
    distance,
    volts_per_mps2,
    compute_kwargs=None,
    csv_glob="*.csv",
    min_files_per_folder=1,
    max_files_per_folder=None,
    save_summary_name=None,
    save_detail_name=None,
    save_plot_name=None,
    verbose=True
):
    """
    Walk subfolders under root_dir whose names look like '50V', '40V', etc.
    For each CSV, compute (velocity, peak_accel) with compute_velocity_and_peak_accel,
    then aggregate per folder (mean, std) and make a velocity-vs-peak-accel plot.

    Returns:
        summary_df: one row per folder (voltage bin)
        detail_df: one row per CSV file
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    compute_kwargs = dict(compute_kwargs or {})
    # Ensure downstream plotting inside compute() is off during batching
    compute_kwargs.setdefault("show", False)
    # These are fine to pass through; your compute() already handles them
    # compute_kwargs.setdefault("overlay_raw", True)
    # compute_kwargs.setdefault("overlay_raw_alpha", 0.25)
    # compute_kwargs.setdefault("baseline_us", 300)

    rows_detail = []
    folders = [p for p in root.iterdir() if p.is_dir()]
    # Keep only folders that look like voltage bins
    folders = [p for p in folders if _parse_voltage_from_folder(p.name) is not None]
    if len(folders) == 0:
        raise RuntimeError("No voltage-named subfolders found (e.g., '50V', '40V').")

    # Sort by numeric voltage descending (e.g., 50V, 40V, 30V ...)
    folders = sorted(folders, key=lambda p: _parse_voltage_from_folder(p.name), reverse=True)

    for f in folders:
        V = _parse_voltage_from_folder(f.name)
        G = _parse_gain(f.name, default=2.0)
        csvs = sorted(f.glob(csv_glob))
        if max_files_per_folder is not None:
            csvs = csvs[:max_files_per_folder]
        if len(csvs) < min_files_per_folder:
            if verbose:
                print(f"[Skip] {f.name}: found {len(csvs)} CSV (< {min_files_per_folder})")
            continue

        if verbose:
            print(f"[Folder] {f.name}  (V={V})  files={len(csvs)}")

        for path in csvs:
            try:
                vel, peak_acc, meta = compute_velocity_and_peak_accel(
                    csv_path=str(path),
                    distance=distance,
                    volts_per_mps2=volts_per_mps2,
                    gain=G,
                    **compute_kwargs
                )
                rows_detail.append({
                    "folder": f.name,
                    "voltage": V,
                    "gain": G,
                    "csv": str(path),
                    "velocity_mps": vel,
                    "peak_accel_mps2": peak_acc,
                    "dt_s": meta.get("dt", np.nan)
                })
                if verbose:
                    print(f"  OK: {path.name}  vel={vel:.2f} m/s  peak={peak_acc:.3f} m/s²")
            except Exception as e:
                if verbose:
                    print(f"  FAIL: {path.name}  -> {e}")

    if len(rows_detail) == 0:
        raise RuntimeError("No successful computations; check inputs and parameters.")

    detail_df = pd.DataFrame(rows_detail)

    # Per-folder (voltage) aggregation
    grp = detail_df.groupby(["folder", "voltage", "gain"], dropna=False)  # <-- added gain
    summary_df = grp.agg(
        n=("velocity_mps", "count"),
        mean_velocity=("velocity_mps", "mean"),
        std_velocity=("velocity_mps", "std"),
        mean_peak_accel=("peak_accel_mps2", "mean"),
        std_peak_accel=("peak_accel_mps2", "std"),
    ).reset_index()

    # Compute standard errors (optional; safe if n>1)
    summary_df["sem_velocity"] = summary_df["std_velocity"] / np.sqrt(summary_df["n"])
    summary_df["sem_peak_accel"] = summary_df["std_peak_accel"] / np.sqrt(summary_df["n"])

    # Save tables if requested
    if save_detail_name:
        detail_path = root / save_detail_name
        detail_df.to_csv(detail_path, index=False)
        if verbose:
            print(f"[Saved] details -> {detail_path}")

    if save_summary_name:
        summary_path = root / save_summary_name
        summary_df.to_csv(summary_path, index=False)
        if verbose:
            print(f"[Saved] summary -> {summary_path}")

    # -------- Plot: velocity vs peak acceleration --------
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(detail_df["peak_accel_mps2"], detail_df["velocity_mps"], s=22, alpha=0.35, label="Per file")

    if len(summary_df):
        ax.errorbar(
            summary_df["mean_peak_accel"], summary_df["mean_velocity"],
            xerr=summary_df["std_peak_accel"], yerr=summary_df["std_velocity"],
            fmt="o", ms=7, capsize=3, elinewidth=1.2, label="Per folder mean ± SD"
        )
        for _, row in summary_df.iterrows():
            ax.annotate(str(row["folder"]), (row["mean_peak_accel"], row["mean_velocity"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax.set_xlabel("Peak acceleration (m/s²)")
    ax.set_ylabel("Pulse velocity (m/s)")
    ax.set_title("Pulse velocity vs. peak acceleration (labeled by input voltage)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    # saves
    if save_detail_name:
        (root / save_detail_name).write_text(detail_df.to_csv(index=False))
    if save_summary_name:
        (root / save_summary_name).write_text(summary_df.to_csv(index=False))
    if save_plot_name:
        fig.savefig(root / save_plot_name, dpi=200)

    plt.show()
    return summary_df, detail_df



# ---- Example run (adjust paths and options as needed) ----
if __name__ == "__main__":
    # Geometry
    DISTANCE = 0.018 * 8  # meters between the two sensors, adjust if yours differs

    # Accelerometer conversion: 1.02 mV per (m/s^2)  =>  0.00102 V per (m/s^2)
    ACCEL_MV_PER_MPS2 = 0.51
    VOLTS_PER_MPS2 = ACCEL_MV_PER_MPS2 / 1000.0  # 0.00102

    # Per-file detection settings (same style as your tension scripts)
    compute_kwargs = dict(
        sep=",",
        skiprows=13,
        threshold_frac=0.5,
        prominence=0.01,
        show=True,              # turn on if you want per-file plots
        # smoothing (if your script supports it via pulse_velocity_smooth)
        smooth=True,
        smooth_window=401,
        smooth_polyorder=3,
        overlay_raw=False
    )

    # Run the batch over folders like "50V", "40V", "30V", ...
    # (Change root_dir to your amplitude sweep data directory)
    summary_df, detail_df = batch_velocity_vs_peak_accel(
        root_dir=r".\1D amplitude changing exp\8_21 main data",
        distance=DISTANCE,
        volts_per_mps2=VOLTS_PER_MPS2,     # accelerometer scale
        compute_kwargs=compute_kwargs,
        min_files_per_folder=1,            # require at least 1 CSV in each V folder
        max_files_per_folder=None,         # use all CSVs found in each folder
        save_summary_name="velocity_vs_peak_accel_summary.csv",
        save_detail_name="velocity_vs_peak_accel_details.csv",
        save_plot_name="velocity_vs_peak_accel.png",
        verbose=True
    )

    # Quick peek
    print("\n=== Summary (per peak-accel bin or folder) ===")
    print(summary_df.head())
    print("\n=== Detail (per CSV) ===")
    print(detail_df.head())
