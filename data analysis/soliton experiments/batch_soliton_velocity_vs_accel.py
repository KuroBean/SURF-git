# batch_soliton_velocity_vs_accel.py
# Walk angle folders like "135 deg soliton data"; inside each, process H5 files
# named like "100g_20260429-173433.h5" to extract pulse velocity and peak acceleration.
# Final plot: velocity vs peak accel, colored by pretension, marker shape by angle.

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from h5_velocity_and_amplitude import compute_velocity_and_peak_accel

# =====================================================================
# Physical constants & geometry
# =====================================================================
G_ACCEL = 9.81                          # m/s^2
LENGTH_PER_UNIT = 0.0098 * 2             # m, hook-to-hook
N_UNITS_BETWEEN_SENSORS = 13
DISTANCE = LENGTH_PER_UNIT * N_UNITS_BETWEEN_SENSORS

ACCEL_MV_PER_MPS2 = 0.51               # accelerometer sensitivity
VOLTS_PER_MPS2 = ACCEL_MV_PER_MPS2 / 1000.0
GAIN = 4

# =====================================================================
# Parsers
# =====================================================================

# "135 deg soliton data" -> 135.0
_ANGLE_RE = re.compile(r'(\d+(?:\.\d+)?)\s*deg', re.IGNORECASE)

def _parse_angle_from_folder(name: str):
    m = _ANGLE_RE.search(name)
    return float(m.group(1)) if m else None


# "100g_20260429-173433.h5" -> 100.0  (grams)
_MASS_RE = re.compile(r'^(\d+(?:\.\d+)?)\s*g[_\s]', re.IGNORECASE)

def _parse_mass_grams_from_filename(name: str):
    m = _MASS_RE.search(name)
    return float(m.group(1)) if m else None


def mass_to_pretension(mass_g):
    """Convert hanging mass in grams to pretension force in Newtons."""
    return (mass_g / 1000.0) * G_ACCEL

# =====================================================================
# Batch processor
# =====================================================================

def batch_soliton(
    root_dir,
    distance=DISTANCE,
    volts_per_mps2=VOLTS_PER_MPS2,
    gain=GAIN,
    compute_kwargs=None,
    min_files_per_folder=1,
    max_files_per_folder=None,
    save_summary_name="soliton_summary.csv",
    save_detail_name="soliton_details.csv",
    save_plot_name="no_halfmax_soliton_velocity_vs_accel.png",
    verbose=True,
):
    """
    Walk angle-named subfolders under root_dir (e.g. '135 deg soliton data').
    Inside each, process .h5 files whose names encode the hanging mass.

    Returns
    -------
    summary_df : one row per (angle, pretension) group
    detail_df  : one row per H5 file
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    compute_kwargs = dict(compute_kwargs or {})
    compute_kwargs.setdefault("show", False)

    rows_detail = []

    # --- discover angle folders ---
    angle_folders = []
    for p in root.iterdir():
        if p.is_dir():
            angle = _parse_angle_from_folder(p.name)
            if angle is not None:
                angle_folders.append((angle, p))

    if not angle_folders:
        raise RuntimeError(
            "No angle-named subfolders found (expected e.g. '135 deg soliton data')."
        )

    angle_folders.sort(key=lambda x: x[0])
    if verbose:
        print(f"Found {len(angle_folders)} angle folder(s): "
              f"{[f'{a} deg' for a, _ in angle_folders]}")

    # --- iterate angle folders -> H5 files ---
    for angle, folder in angle_folders:
        h5_files = sorted(folder.glob("*.h5"))
        if max_files_per_folder is not None:
            h5_files = h5_files[:max_files_per_folder]

        if len(h5_files) < min_files_per_folder:
            if verbose:
                print(f"[Skip] {folder.name}: {len(h5_files)} H5 files "
                      f"(< {min_files_per_folder})")
            continue

        if verbose:
            print(f"\n[Angle {angle} deg]  {folder.name}  ({len(h5_files)} H5 files)")

        for h5_path in h5_files:
            mass_g = _parse_mass_grams_from_filename(h5_path.name)
            if mass_g is None:
                if verbose:
                    print(f"  SKIP: {h5_path.name}  (cannot parse mass)")
                continue

            pretension_N = mass_to_pretension(mass_g)

            try:
                vel, peak_acc, meta = compute_velocity_and_peak_accel(
                    csv_path=str(h5_path),
                    distance=distance,
                    volts_per_mps2=volts_per_mps2,
                    gain=gain,
                    **compute_kwargs,
                )
                rows_detail.append({
                    "angle_deg": angle,
                    "angle_folder": folder.name,
                    "mass_g": mass_g,
                    "pretension_N": pretension_N,
                    "file": h5_path.name,
                    "file_path": str(h5_path),
                    "velocity_mps": vel,
                    "peak_accel_mps2": peak_acc,
                    "dt_s": meta.get("dt", np.nan),
                })
                if verbose:
                    print(f"  OK: {h5_path.name}  "
                          f"({mass_g:.0f}g -> {pretension_N:.3f}N)  "
                          f"vel={vel:.2f} m/s  peak={peak_acc:.1f} m/s^2")
            except Exception as e:
                if verbose:
                    print(f"  FAIL: {h5_path.name}  -> {e}")

    if not rows_detail:
        raise RuntimeError("No successful computations; check data and parameters.")

    # --- build DataFrames ---
    detail_df = pd.DataFrame(rows_detail)

    grp = detail_df.groupby(["angle_deg", "pretension_N", "mass_g"], dropna=False)
    summary_df = grp.agg(
        n=("velocity_mps", "count"),
        mean_velocity=("velocity_mps", "mean"),
        std_velocity=("velocity_mps", "std"),
        mean_peak_accel=("peak_accel_mps2", "mean"),
        std_peak_accel=("peak_accel_mps2", "std"),
    ).reset_index()
    summary_df["sem_velocity"] = summary_df["std_velocity"] / np.sqrt(summary_df["n"])
    summary_df["sem_peak_accel"] = summary_df["std_peak_accel"] / np.sqrt(summary_df["n"])

    # --- save tables ---
    if save_detail_name:
        detail_df.to_csv(root / save_detail_name, index=False)
        if verbose:
            print(f"\n[Saved] details  -> {root / save_detail_name}")
    if save_summary_name:
        summary_df.to_csv(root / save_summary_name, index=False)
        if verbose:
            print(f"[Saved] summary  -> {root / save_summary_name}")

    # =================================================================
    # Plot: velocity vs peak accel
    #   color  = pretension (discrete legend)
    #   marker = angle
    # =================================================================
    MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    unique_angles = sorted(detail_df["angle_deg"].unique())
    unique_pretensions = sorted(detail_df["pretension_N"].unique())

    # One color per pretension level
    cmap = plt.cm.viridis
    pretension_colors = {
        pt: cmap(i / max(len(unique_pretensions) - 1, 1))
        for i, pt in enumerate(unique_pretensions)
    }
    angle_markers = {
        a: MARKERS[i % len(MARKERS)] for i, a in enumerate(unique_angles)
    }

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # --- individual data points ---
    for angle in unique_angles:
        for pt in unique_pretensions:
            mask = (detail_df["angle_deg"] == angle) & (detail_df["pretension_N"] == pt)
            sub = detail_df[mask]
            if sub.empty:
                continue
            ax.scatter(
                sub["peak_accel_mps2"], sub["velocity_mps"],
                c=[pretension_colors[pt]],
                marker=angle_markers[angle],
                s=40, alpha=0.55, edgecolors="none",
            )

    # --- legend: two sections (pretension color + angle shape) ---
    from matplotlib.lines import Line2D

    legend_handles = []

    # pretension colors
    legend_handles.append(Line2D([], [], color="none", label="Pretension"))
    for pt in unique_pretensions:
        mass_g = pt / G_ACCEL * 1000
        legend_handles.append(
            Line2D([], [], marker="o", color="none",
                   markerfacecolor=pretension_colors[pt],
                   markeredgecolor="black", markersize=8,
                   label=f"{mass_g:.0f}g ({pt:.2f} N)")
        )

    # angle markers
    legend_handles.append(Line2D([], [], color="none", label=""))  # spacer
    legend_handles.append(Line2D([], [], color="none", label="Hammer angle"))
    for angle in unique_angles:
        legend_handles.append(
            Line2D([], [], marker=angle_markers[angle], color="none",
                   markerfacecolor="gray", markeredgecolor="black",
                   markersize=8, label=f"{angle:.0f} deg")
        )

    ax.legend(handles=legend_handles, loc="best", fontsize=9, framealpha=0.9)
    ax.set_xlabel("Peak acceleration (m/s$^2$)")
    ax.set_ylabel("Pulse velocity (m/s)")
    ax.set_title("Soliton pulse velocity vs. peak acceleration")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_plot_name:
        fig.savefig(root / save_plot_name, dpi=200)
        if verbose:
            print(f"[Saved] plot     -> {root / save_plot_name}")

    plt.show()
    return summary_df, detail_df


# =====================================================================
# Run
# =====================================================================
if __name__ == "__main__":

    compute_kwargs = dict(
        # H5 channel mapping (matches oscilloscope recording script)
        s1_channel="CHANnel2",
        s2_channel="CHANnel3",

        # analysis
        threshold_frac=0.5,
        prominence=0.4,

        show=False,         # set True to see per-file plots
        save_plot=False,     # save per-file peak plots

        # smoothing
        smooth=True,
        smooth_window=401,
        smooth_polyorder=3,
        overlay_raw=True,
        baseline_us=300,
    )

    summary_df, detail_df = batch_soliton(
        root_dir=r".\soliton experiments\soliton exp 5_13",    # parent dir containing angle folders
        distance=DISTANCE,
        volts_per_mps2=VOLTS_PER_MPS2,
        gain=GAIN,
        compute_kwargs=compute_kwargs,
        min_files_per_folder=1,
        max_files_per_folder=None,  # set None to process all files
        save_summary_name="soliton_summary.csv",
        save_detail_name="soliton_details.csv",
        save_plot_name="soliton_velocity_vs_accel.png",
        verbose=True,
    )

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== Detail (first 10) ===")
    print(detail_df.head(10).to_string(index=False))
