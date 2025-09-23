
# batch_velocity_vs_peak_accel_flat.py
# Scan every CSV directly in `root_dir`, compute (velocity, peak acceleration) per file,
# and plot per-file points only (no subfolder grouping or means).

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from velocity_and_amplitude import compute_velocity_and_peak_accel

def batch_velocity_vs_peak_accel_flat(
    root_dir,
    distance,
    volts_per_mps2,
    gain=2.0,
    compute_kwargs=None,
    csv_glob="*.csv",
    recursive=False,
    save_detail_name="velocity_vs_peak_accel_details_flat.csv",
    save_plot_name="velocity_vs_peak_accel_flat.png",
    annotate=False,
    verbose=True,
):
    '''
    Parameters
    ----------
    root_dir : str or Path
        Directory containing CSV files to process.
    distance : float
        Sensor separation in meters.
    volts_per_mps2 : float
        Accelerometer conversion, volts per (m/s^2).
    gain : float
        Amplifier gain applied during acquisition. Used to convert to m/s^2.
    compute_kwargs : dict
        Extra kwargs passed to compute_velocity_and_peak_accel (e.g., sep, skiprows, smoothing, etc.).
    csv_glob : str
        Pattern for CSV selection (default: '*.csv').
    recursive : bool
        If True, search recursively with rglob; otherwise only the top-level directory is scanned.
    save_detail_name : str or None
        If set, save a per-file table with results.
    save_plot_name : str or None
        If set, save the scatter plot here.
    annotate : bool
        If True, annotate each point with the CSV stem.
    verbose : bool
        Print progress and any failures.

    Returns
    -------
    detail_df : pandas.DataFrame
        Per-CSV results with columns: csv, velocity_mps, peak_accel_mps2, dt_s
    '''
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    compute_kwargs = dict(compute_kwargs or {})
    compute_kwargs.setdefault("show", False)  # don't spam plots in batch

    # Gather CSV files
    files = sorted(root.rglob(csv_glob) if recursive else root.glob(csv_glob))
    if len(files) == 0:
        raise RuntimeError(f"No CSV files matched in {root} with pattern {csv_glob!r} (recursive={recursive}).")

    rows = []
    for fp in files:
        try:
            vel, peak_acc, meta = compute_velocity_and_peak_accel(
                csv_path=str(fp),
                distance=distance,
                volts_per_mps2=volts_per_mps2,
                gain=gain,
                **compute_kwargs
            )
            rows.append({
                "csv": str(fp),
                "velocity_mps": float(vel),
                "peak_accel_mps2": float(peak_acc),
                "dt_s": float(meta.get("dt", np.nan)),
            })
            if verbose:
                print(f"[OK] {fp.name:30s}  vel={vel:8.3f} m/s   peak={peak_acc:8.3f} m/s^2")
        except Exception as e:
            if verbose:
                print(f"[FAIL] {fp.name:30s}  -> {e}")

    if len(rows) == 0:
        raise RuntimeError("All CSVs failed. Check compute parameters (skiprows, columns, prominence, etc.).")

    detail_df = pd.DataFrame(rows)

    # ---- Plot: per-CSV points only ----
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(detail_df["peak_accel_mps2"], detail_df["velocity_mps"], s=24, alpha=0.6, label="Per CSV")

    if annotate:
        for _, r in detail_df.iterrows():
            ax.annotate(Path(r["csv"]).stem, (r["peak_accel_mps2"], r["velocity_mps"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)

    ax.set_xlabel("Peak acceleration (m/s²)")
    ax.set_ylabel("Pulse velocity (m/s)")
    ax.set_title("Pulse velocity vs. peak acceleration (per CSV)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    # Save outputs
    if save_detail_name:
        out_csv = root / save_detail_name
        detail_df.to_csv(out_csv, index=False)
        if verbose:
            print(f"[Saved] details -> {out_csv}")

    if save_plot_name:
        out_png = root / save_plot_name
        fig.savefig(out_png, dpi=200)
        if verbose:
            print(f"[Saved] plot    -> {out_png}")

    plt.close(fig)
    return detail_df


# ---- Example usage (edit paths and parameters) ----
if __name__ == "__main__":
    # Geometry
    DISTANCE = 0.018 * 6  # meters between sensors (adjust if needed)

    # Accelerometer scale: 0.51 mV / (m/s^2) -> 0.00051 V / (m/s^2)
    ACCEL_MV_PER_MPS2 = 0.51
    VOLTS_PER_MPS2 = ACCEL_MV_PER_MPS2 / 1000.0

    compute_kwargs = dict(
        sep=",",
        skiprows=13,
        threshold_frac=0.5,
        prominence=0.05, #mV
        show=False,
        smooth=True,
        smooth_window=401,
        smooth_polyorder=3,
        overlay_raw=True,
        baseline_us=300,
        save_plot=False  # disable per-file plots to keep batch clean
    )

    # Run on CSVs directly under the folder (no subfolder parsing)
    detail = batch_velocity_vs_peak_accel_flat(
        root_dir=r"250922_compression_actuator",
        distance=DISTANCE,
        volts_per_mps2=VOLTS_PER_MPS2,
        gain=4.0,
        compute_kwargs=compute_kwargs,
        csv_glob="*.csv",
        recursive=False,
        save_detail_name="velocity_vs_peak_accel_details_flat.csv",
        save_plot_name="velocity_vs_peak_accel_flat.png",
        annotate=False,
        verbose=True,
    )

    print(detail.head())
