# batch_junction_angle_velocity_amplitude.py
# Sweep subfolders named like "45.6 deg" (or "45°", "45 DEG") and compute
# junction-split velocities and peak accelerations for left/right branches.
#
# CSV format (per file):
#   col0 = time (s)
#   col1 = sensor on incoming (actuated) chain
#   col2 = sensor on LEFT branch
#   col3 = sensor on RIGHT branch
#
# Requires: velocity_and_amplitude.py in your PYTHONPATH
# Uses compute_velocity_and_peak_accel() twice per CSV:
#   (1->2) for LEFT, (1->3) for RIGHT.

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# import sys
# import os
# # Get the absolute path to the directory containing the module
# module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '1D amplitude changing exp'))
# # Add the directory to sys.path
# sys.path.insert(0, module_dir) # Insert at the beginning for higher priority
from velocity_and_amplitude import compute_velocity_and_peak_accel  # ← your function

# --------- Parsing helpers ---------
def _parse_angle_from_folder(name: str):
    """
    Accepts: '45.6 deg', '45deg', '45°', '45 DEG', with/without spaces.
    Returns float angle in degrees, or None if not matched.
    """
    name = name.strip()
    m = re.match(r'^([-+]?\d*\.?\d+)\s*(?:deg|degree|degrees|°)?$', name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None

def _parse_gain_from_text(name: str):
    """
    Optional: override default gain if folder/file name contains 'gain X'.
    e.g., '45 deg gain 6' -> 6
    """
    m = re.search(r'gain\s*[:=]?\s*([-+]?\d*\.?\d+)', name, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None

def _find_csvs(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == '.csv')

# --------- Core batch ---------
def batch_junction_angle_velocity_amplitude(
    root_dir,
    distance_m,
    volts_per_mps2,               # sensor scale *without* gain, e.g. 0.00051 V/(m/s^2) if 0.51 mV/(m/s^2)
    compute_kwargs=None,          # forwarded into compute_velocity_and_peak_accel()
    min_files_per_folder=1,
    max_files_per_folder=None,
    save_detail_csv="junction_angle_details.csv",
    save_summary_csv="junction_angle_summary.csv",
    save_velocity_plot="velocity_vs_angle.png",
    save_amplitude_plot="amplitude_vs_angle.png",
    verbose=True
):
    """
    Returns (summary_df, detail_df).
    summary_df: one row per (angle, branch)
    detail_df:  one row per CSV × branch
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    compute_kwargs = dict(compute_kwargs or {})
    # ensure batch mode: no per-file popups unless you flip this on
    compute_kwargs.setdefault("show", False)
    # Defaults here mirror your existing amplitude batch style.  # :contentReference[oaicite:3]{index=3}

    rows = []
    folders = [p for p in root.iterdir() if p.is_dir()]
    # keep only folders that look like angles
    folders = [(p, _parse_angle_from_folder(p.name)) for p in folders]
    folders = [(p, ang) for (p, ang) in folders if ang is not None]
    if len(folders) == 0:
        raise RuntimeError("No angle-named subfolders found (e.g., '45.6 deg', '45°').")

    # sort by numeric angle ascending
    folders.sort(key=lambda x: x[1])

    for f, angle in folders:
        csvs = _find_csvs(f)
        if max_files_per_folder is not None:
            csvs = csvs[:max_files_per_folder]
        if len(csvs) < min_files_per_folder:
            if verbose:
                print(f"[Skip] {f.name}: only {len(csvs)} CSV (< {min_files_per_folder})")
            continue

        # Allow folder-level gain override (falls back to compute_kwargs.get('gain', 2))
        folder_gain = _parse_gain_from_text(f.name)
        if verbose:
            print(f"[Folder] {f.name}  angle={angle}°  files={len(csvs)}  gain={folder_gain or compute_kwargs.get('gain', 2)}")

        for path in csvs:
            # Also allow file-level gain override if present (rare but robust)
            file_gain = _parse_gain_from_text(path.name) or folder_gain
            local_kwargs = dict(compute_kwargs)
            if file_gain is not None:
                local_kwargs["gain"] = file_gain

            try:
                # LEFT: (incoming col1) -> (left col2)
                vL, aL, metaL = compute_velocity_and_peak_accel(
                    csv_path=str(path),
                    distance=distance_m,
                    volts_per_mps2=volts_per_mps2,
                    time_col=0, s1_col=1, s2_col=2,
                    **local_kwargs
                )

                rows.append({
                    "folder": f.name, "angle_deg": angle, "branch": "left",
                    "csv": str(path),
                    "velocity_mps": float(vL),
                    "peak_accel_mps2": float(aL),
                    "dt_s": float(metaL.get("dt", np.nan)),
                    "gain_used": float(local_kwargs.get("gain", compute_kwargs.get("gain", 2)))
                })

                # RIGHT: (incoming col1) -> (right col3)
                vR, aR, metaR = compute_velocity_and_peak_accel(
                    csv_path=str(path),
                    distance=distance_m,
                    volts_per_mps2=volts_per_mps2,
                    time_col=0, s1_col=1, s2_col=3,
                    **local_kwargs
                )

                rows.append({
                    "folder": f.name, "angle_deg": angle, "branch": "right",
                    "csv": str(path),
                    "velocity_mps": float(vR),
                    "peak_accel_mps2": float(aR),
                    "dt_s": float(metaR.get("dt", np.nan)),
                    "gain_used": float(local_kwargs.get("gain", compute_kwargs.get("gain", 2)))
                })

                if verbose:
                    print(f"  OK: {path.name}  vL={vL:.2f} m/s  vR={vR:.2f} m/s   aL={aL:.2f} m/s²  aR={aR:.2f} m/s²")

            except Exception as e:
                if verbose:
                    print(f"  FAIL: {path.name} -> {e}")

    if len(rows) == 0:
        raise RuntimeError("No successful computations; check inputs (skiprows, prominence) and CSV format.")

    detail_df = pd.DataFrame(rows)

    # Per-angle × branch aggregation
    grp = detail_df.groupby(["angle_deg", "branch"], dropna=False)
    summary_df = grp.agg(
        n=("velocity_mps", "count"),
        mean_velocity=("velocity_mps", "mean"),
        std_velocity=("velocity_mps", "std"),
        mean_peak_accel=("peak_accel_mps2", "mean"),
        std_peak_accel=("peak_accel_mps2", "std")
    ).reset_index()
    # standard errors (optional, safe if n>1)
    summary_df["sem_velocity"]   = summary_df["std_velocity"] / np.sqrt(summary_df["n"])
    summary_df["sem_peak_accel"] = summary_df["std_peak_accel"] / np.sqrt(summary_df["n"])

    # Save tables
    if save_detail_csv:
        Path(save_detail_csv).parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(Path(root_dir) / save_detail_csv, index=False)
        if verbose: print(f"[Saved] details -> {(Path(root_dir) / save_detail_csv)}")
    if save_summary_csv:
        Path(save_summary_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(Path(root_dir) / save_summary_csv, index=False)
        if verbose: print(f"[Saved] summary -> {(Path(root_dir) / save_summary_csv)}")

    # ---------- Plot 1: Velocity vs Angle (left & right) ----------
    fig1, ax1 = plt.subplots(figsize=(9, 6))

    # Individual CSV points (faint), colored by branch
    for br in ["left", "right"]:
        d = detail_df[detail_df["branch"] == br]
        ax1.scatter(d["angle_deg"].values, d["velocity_mps"].values, s=22, alpha=0.35, label=f"{br} (per file)")

    # Per-angle means ± SD
    for br in ["left", "right"]:
        s = summary_df[summary_df["branch"] == br].sort_values("angle_deg")
        if len(s) == 0: continue
        ax1.errorbar(
            s["angle_deg"].values,
            s["mean_velocity"].values,
            yerr=s["std_velocity"].values,
            fmt="o-", capsize=3, elinewidth=1.2, ms=6, label=f"{br} mean ± SD"
        )

    ax1.set_xlabel("Junction angle (deg)")
    ax1.set_ylabel("Pulse velocity (m/s)")
    ax1.set_title("Pulse velocity vs junction angle (left/right branches)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    fig1.tight_layout()
    out1 = Path(root_dir) / save_velocity_plot
    fig1.savefig(out1, dpi=200)
    if verbose: print(f"[Saved] plot -> {out1}")
    fig1.show()

    # ---------- Plot 2: Peak acceleration vs Angle ----------
    fig2, ax2 = plt.subplots(figsize=(9, 6))

    for br in ["left", "right"]:
        d = detail_df[detail_df["branch"] == br]
        ax2.scatter(d["angle_deg"].values, d["peak_accel_mps2"].values, s=22, alpha=0.35, label=f"{br} (per file)")

    for br in ["left", "right"]:
        s = summary_df[summary_df["branch"] == br].sort_values("angle_deg")
        if len(s) == 0: continue
        ax2.errorbar(
            s["angle_deg"].values,
            s["mean_peak_accel"].values,
            yerr=s["std_peak_accel"].values,
            fmt="o-", capsize=3, elinewidth=1.2, ms=6, label=f"{br} mean ± SD"
        )

    ax2.set_xlabel("Junction angle (deg)")
    ax2.set_ylabel("Peak acceleration (m/s²)")
    ax2.set_title("Peak acceleration vs junction angle (left/right branches)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")
    fig2.tight_layout()
    out2 = Path(root_dir) / save_amplitude_plot
    fig2.savefig(out2, dpi=200)
    if verbose: print(f"[Saved] plot -> {out2}")
    fig2.show()

    plt.close(fig1); plt.close(fig2)
    return summary_df, detail_df


# ---------------- Example run ----------------
if __name__ == "__main__":
    # Geometry: distance between the incoming-chain sensor and each branch sensor
    DISTANCE_M = 0.018 * 8        # adjust to your rig
    # Accelerometer scale (no gain): 0.51 mV/(m/s^2) -> 0.00051 V/(m/s^2)
    VOLTS_PER_MPS2 = 0.00051

    # Per-file detection settings — same style you used for amplitude batches  # :contentReference[oaicite:4]{index=4}
    compute_kwargs = dict(
        sep=",",
        skiprows=13,              # adjust to your CSV header length
        prominence=0.01,          # tweak for your SNR
        threshold_frac=0.5,       # not used directly in this helper but safe to pass
        smooth=True,
        smooth_window=401,
        smooth_polyorder=3,
        overlay_raw=False,
        baseline_us=300,
        gain=2,                    # default; folder/file can override via 'gain X'
        save_plot=False,
        show=False
    )

    ROOT = r"C:\Users\eiko\Desktop\college files\SURF git\data analysis\symmetrical angle exp\8_22 data"  # ← folder containing subfolders like "45.6 deg", each with CSVs

    summary_df, detail_df = batch_junction_angle_velocity_amplitude(
        root_dir=ROOT,
        distance_m=DISTANCE_M,
        volts_per_mps2=VOLTS_PER_MPS2,
        compute_kwargs=compute_kwargs,
        min_files_per_folder=1,
        max_files_per_folder=None,
        save_detail_csv="junction_angle_details.csv",
        save_summary_csv="junction_angle_summary.csv",
        save_velocity_plot="velocity_vs_angle.png",
        save_amplitude_plot="amplitude_vs_angle.png",
        verbose=True
    )

    print("\n=== Summary (per angle × branch) ===")
    print(summary_df.head())
    print("\n=== Detail (per CSV × branch) ===")
    print(detail_df.head())
