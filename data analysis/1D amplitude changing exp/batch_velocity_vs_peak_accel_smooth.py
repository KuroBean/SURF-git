# velocity_vs_peak_accel_smooth.py
# Walk folders like "50V", "40 V", "30v", ...; compute pulse velocity and peak acceleration amplitude.

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from velocity_and_amplitude import compute_velocity_and_peak_accel

# ---------- Calibration / geometry ----------
# Accelerometers: 1.02 mV per (m/s^2)  ->  volts_to_mps2 = (V * 1000 mV/V) / 1.02
VOLTS_TO_MPS2 = 1000.0 / 1.02    # ≈ 980.392 m/s^2 per Volt
DISTANCE = 0.018 * 8             # meters, separation between sensors (adjust if needed)


def _parse_voltage_from_folder(folder_name: str):
    """
    Extract numeric voltage from names like '50V', '40 v', '30 V '.
    Returns float or None if it doesn't match.
    """
    m = re.search(r'^\s*([-+]?\d*\.?\d+)\s*[vV]\s*$', folder_name)
    return float(m.group(1)) if m else None

def _find_csvs(folder_path: str):
    """Return CSV paths directly inside folder_path."""
    return sorted(
        os.path.join(folder_path, name)
        for name in os.listdir(folder_path)
        if name.lower().endswith('.csv')
    )

# ---------- Batch over voltage folders ----------
def batch_velocity_vs_peak_accel(
    root_dir: str,
    distance: float = DISTANCE,
    compute_kwargs: dict | None = None,
    min_files_per_folder: int = 1,
    max_files_per_folder: int | None = None,
    save_detail_name: str = 'velocity_vs_peak_accel_details.csv',
    save_summary_name: str = 'velocity_vs_peak_accel_summary.csv',
    save_plot_name: str = 'velocity_vs_peak_accel.png',
    verbose: bool = True
):
    """
    Scans `root_dir` for folders named like '50V', '40 V', ..., processes CSVs to compute
    pulse velocity and peak acceleration amplitude (from ch1). Saves detail & summary CSVs and a plot.
    """
    compute_kwargs = dict(compute_kwargs or {})

    detail_rows = []
    summary_rows = []
    folder_counts = []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        volts = _parse_voltage_from_folder(entry.name)
        if volts is None:
            continue

        csvs = _find_csvs(entry.path)
        folder_counts.append((entry.name, volts, len(csvs)))
        if len(csvs) < int(min_files_per_folder):
            continue
        if isinstance(max_files_per_folder, int) and max_files_per_folder > 0:
            csvs = csvs[:max_files_per_folder]

        vels, accs = [], []
        for i, csv_path in enumerate(csvs, start=1):
            try:
                res = compute_velocity_and_peak_accel(csv_path, distance=distance, **compute_kwargs)
            except Exception as e:
                if verbose:
                    print(f"[WARN] Skipping {csv_path}: {e}")
                continue

            vels.append(float(res['velocity_mps']))
            accs.append(float(res['ch1_peak_accel_mps2']))
            detail_rows.append({
                'folder_voltage_V': float(volts),
                'csv_index': int(i),
                'csv_path': csv_path,
                'velocity_mps': float(res['velocity_mps']),
                'peak_accel_mps2': float(res['ch1_peak_accel_mps2']),
                't_peak1_s': float(res['t_peak1']),
                't_peak2_s': float(res['t_peak2'])
            })

        if len(vels) == 0:
            continue

        v_arr = np.asarray(vels, dtype=float)
        a_arr = np.asarray(accs, dtype=float)
        summary_rows.append({
            'folder_voltage_V': float(volts),
            'n_files_used': int(len(v_arr)),
            'mean_velocity_mps': float(np.mean(v_arr)),
            'std_velocity_mps': float(np.std(v_arr, ddof=1)) if len(v_arr) > 1 else 0.0,
            'mean_peak_accel_mps2': float(np.mean(a_arr)),
            'std_peak_accel_mps2': float(np.std(a_arr, ddof=1)) if len(a_arr) > 1 else 0.0
        })

    if not summary_rows:
        if verbose:
            print("Diagnostic: CSV counts per V-folder under:", root_dir)
            for name, V, n in sorted(folder_counts, key=lambda x: x[1]):
                print(f"  {name:<8}  V={V:<6g}  csv_count={n}")
        raise RuntimeError("No suitable V-folders were processed. See diagnostics above.")

    df_detail = pd.DataFrame(detail_rows).sort_values(['folder_voltage_V', 'csv_index']).reset_index(drop=True)
    df_summary = pd.DataFrame(summary_rows).sort_values('folder_voltage_V').reset_index(drop=True)

    # Save CSVs
    out_detail = os.path.join(root_dir, save_detail_name)
    out_summary = os.path.join(root_dir, save_summary_name)
    df_detail.to_csv(out_detail, index=False)
    df_summary.to_csv(out_summary, index=False)
    if verbose:
        print(f"Saved details: {out_detail}")
        print(f"Saved summary: {out_summary}")

    # ---- Plot: pulse velocity (y) vs peak acceleration amplitude (x) ----
    plt.figure(figsize=(9, 6))

    # Individual trials
    x_pts = df_detail['peak_accel_mps2'].values.astype(float)
    y_pts = df_detail['velocity_mps'].values.astype(float)
    plt.scatter(x_pts, y_pts, s=22, alpha=0.55, label='individual CSVs', zorder=2)

    # Per-folder means with error bars (both x and y)
    x_mean = df_summary['mean_peak_accel_mps2'].values.astype(float)
    y_mean = df_summary['mean_velocity_mps'].values.astype(float)
    xerr   = df_summary['std_peak_accel_mps2'].values.astype(float)
    yerr   = df_summary['std_velocity_mps'].values.astype(float)
    plt.errorbar(x_mean, y_mean, xerr=xerr, yerr=yerr, fmt='o', capsize=4, linewidth=1.5,
                 color='k', label='per-folder mean ±1σ', zorder=3)

    plt.xlabel('Peak acceleration amplitude (m/s²)  [from ch1]')
    plt.ylabel('Pulse velocity (m/s)')
    plt.title('Pulse velocity vs peak acceleration amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    out_plot = os.path.join(root_dir, save_plot_name)
    plt.savefig(out_plot, dpi=200)
    plt.show()
    print(f"Saved plot: {out_plot}")

    return df_summary, df_detail


# ---- Example run (adjust paths and options as needed) ----
if __name__ == "__main__":
    # Geometry
    DISTANCE = 0.018 * 8  # meters between the two sensors, adjust if yours differs

    # Accelerometer conversion: 1.02 mV per (m/s^2)  =>  0.00102 V per (m/s^2)
    ACCEL_MV_PER_MPS2 = 1.02
    VOLTS_PER_MPS2 = ACCEL_MV_PER_MPS2 / 1000.0  # 0.00102

    # Per-file detection settings (same style as your tension scripts)
    compute_kwargs = dict(
        sep=",",
        skiprows=2,
        threshold_frac=0.5,
        prominence=0.02,
        sample_dist=None,
        show=False,              # turn on if you want per-file plots
        # smoothing (if your script supports it via pulse_velocity_smooth)
        smooth=True,
        smooth_window=401,
        smooth_polyorder=3,
        overlay_raw=False
    )

    # Run the batch over folders like "50V", "40V", "30V", ...
    # (Change root_dir to your amplitude sweep data directory)
    summary_df, detail_df = batch_velocity_vs_peak_accel(
        root_dir=r".\8_21 main data",
        distance=DISTANCE,
        #volts_per_mps2=VOLTS_PER_MPS2,     # accelerometer scale
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
