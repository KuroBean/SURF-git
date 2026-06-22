# batch_soliton_velocity_vs_accel.py
# Walk angle folders like "135 deg soliton data"; inside each, process H5 files
# named like "100g_20260429-173433.h5" to extract pulse velocity and peak acceleration.
# Final plots: one velocity vs peak-accel graph PER pretension level,
#              marker shape encodes hammer angle.

import re
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from half_max_h5_velocity_and_amplitude import compute_velocity_and_peak_accel

# =====================================================================
# Physical constants & geometry
# =====================================================================
G_ACCEL = 9.81                          # m/s^2
LENGTH_PER_UNIT = 0.0098 * 2            # m, hook-to-hook
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
# Per-pretension plot helper
# =====================================================================

def _plot_one_pretension(
    detail_df: pd.DataFrame,
    pretension_N: float,
    unique_angles: list,
    angle_markers: dict,
    root: Path,
    save_plot_name_template: str,
    verbose: bool,
):
    """
    Create and save a velocity vs peak-acceleration figure for a single
    pretension level.  Marker shape encodes hammer angle.

    Parameters
    ----------
    detail_df              : full detail DataFrame (will be filtered inside)
    pretension_N           : the pretension level to plot (Newtons)
    unique_angles          : sorted list of all angle values (for consistent legend)
    angle_markers          : dict mapping angle -> matplotlib marker string
    root                   : Path to root_dir (for saving)
    save_plot_name_template: filename template; must contain '{mass_g:.0f}'
                             e.g. 'soliton_vel_vs_accel_{mass_g:.0f}g.png'
    verbose                : print save path
    """
    mass_g = pretension_N / G_ACCEL * 1000.0
    sub_pt = detail_df[detail_df["pretension_N"] == pretension_N]

    # Angles actually present for this pretension (subset used for data points)
    angles_present = sorted(sub_pt["angle_deg"].unique())

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for angle in angles_present:
        sub = sub_pt[sub_pt["angle_deg"] == angle]
        ax.scatter(
            sub["peak_accel_mps2"],
            sub["velocity_mps"],
            marker=angle_markers[angle],
            s=55,
            alpha=0.75,
            edgecolors="k",
            linewidths=0.5,
            label=f"{angle:.0f}°",
            zorder=3,
        )

    # --- axes labels & title ---
    ax.set_xlabel("Peak acceleration (m/s²)", fontsize=12)
    ax.set_ylabel("Pulse velocity (m/s)", fontsize=12)
    ax.set_title(
        f"Soliton: velocity vs. peak acceleration\n"
        f"Pretension = {mass_g:.0f} g  ({pretension_N:.3f} N)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25)

    # --- legend: marker shape = angle ---
    legend_handles = [
        Line2D(
            [], [],
            marker=angle_markers[a],
            color="none",
            markerfacecolor="steelblue",
            markeredgecolor="black",
            markersize=9,
            label=f"{a:.0f}°",
        )
        for a in angles_present
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Hammer angle",
            loc="best",
            fontsize=9,
            framealpha=0.9,
        )

    fig.tight_layout()

    # --- save ---
    fname = save_plot_name_template.format(mass_g=mass_g, pretension_N=pretension_N)
    out_path = root / fname
    fig.savefig(out_path, dpi=200)
    if verbose:
        print(f"[Saved] plot ({mass_g:.0f}g) -> {out_path}")

    plt.show()
    plt.close(fig)


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
    # Template for per-pretension plot filenames.
    # Available format keys: {mass_g:.0f}  {pretension_N:.3f}
    save_plot_name_template="soliton_vel_vs_accel_{mass_g:.0f}g.png",
    verbose=True,
):
    """
    Walk angle-named subfolders under root_dir (e.g. '135 deg soliton data').
    Inside each, process .h5 files whose names encode the hanging mass.

    Outputs one velocity-vs-peak-accel plot per unique pretension level.

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
                    "angle_deg":      angle,
                    "angle_folder":   folder.name,
                    "mass_g":         mass_g,
                    "pretension_N":   pretension_N,
                    "file":           h5_path.name,
                    "file_path":      str(h5_path),
                    "velocity_mps":   vel,
                    "peak_accel_mps2": peak_acc,
                    "fwhm_s1_s":      meta.get("fwhm_s1", np.nan),
                    "fwhm_s2_s":      meta.get("fwhm_s2", np.nan),
                    "dt_s":           meta.get("dt", np.nan),
                })
                if verbose:
                    fwhm1_us = meta.get("fwhm_s1", np.nan) * 1e6
                    fwhm2_us = meta.get("fwhm_s2", np.nan) * 1e6
                    print(f"  OK: {h5_path.name}  "
                          f"({mass_g:.0f}g -> {pretension_N:.3f}N)  "
                          f"vel={vel:.2f} m/s  peak={peak_acc:.1f} m/s^2  "
                          f"fwhm_s1={fwhm1_us:.1f} µs  fwhm_s2={fwhm2_us:.1f} µs")
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
        mean_fwhm_s1_s=("fwhm_s1_s", "mean"),
        std_fwhm_s1_s=("fwhm_s1_s", "std"),
        mean_fwhm_s2_s=("fwhm_s2_s", "mean"),
        std_fwhm_s2_s=("fwhm_s2_s", "std"),
    ).reset_index()
    summary_df["sem_velocity"]    = summary_df["std_velocity"]    / np.sqrt(summary_df["n"])
    summary_df["sem_peak_accel"]  = summary_df["std_peak_accel"]  / np.sqrt(summary_df["n"])
    summary_df["sem_fwhm_s1_s"]   = summary_df["std_fwhm_s1_s"]  / np.sqrt(summary_df["n"])
    summary_df["sem_fwhm_s2_s"]   = summary_df["std_fwhm_s2_s"]  / np.sqrt(summary_df["n"])

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
    # Plots: one figure per pretension level
    # =================================================================
    MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

    unique_angles      = sorted(detail_df["angle_deg"].unique())
    unique_pretensions = sorted(detail_df["pretension_N"].unique())

    # Fixed marker assignment (consistent across all pretension plots)
    angle_markers = {
        a: MARKERS[i % len(MARKERS)] for i, a in enumerate(unique_angles)
    }

    if verbose:
        print(f"\nGenerating {len(unique_pretensions)} per-pretension plot(s)...")

    for pt in unique_pretensions:
        _plot_one_pretension(
            detail_df=detail_df,
            pretension_N=pt,
            unique_angles=unique_angles,
            angle_markers=angle_markers,
            root=root,
            save_plot_name_template=save_plot_name_template,
            verbose=verbose,
        )

    return summary_df, detail_df


# =====================================================================
# Run
# =====================================================================
if __name__ == "__main__":

    compute_kwargs = dict(
        # H5 channel mapping (matches oscilloscope recording script)
        s1_channel="CHANnel2",
        s2_channel="CHANnel3",

        # peak finder settings
        threshold_frac=0.5,
        prominence=0.4,

        show=False,         # set True to see per-file plots
        save_plot=False,    # save per-file peak plots

        # smoothing
        smooth=True,
        smooth_window=401,
        smooth_polyorder=3,
        overlay_raw=True,
        baseline_us=300,
    )

    summary_df, detail_df = batch_soliton(
        root_dir=r".\soliton experiments\soliton exp 6_20 farther sensors",
        distance=DISTANCE,
        volts_per_mps2=VOLTS_PER_MPS2,
        gain=GAIN,
        compute_kwargs=compute_kwargs,
        min_files_per_folder=1,
        max_files_per_folder=None,
        save_summary_name="soliton_summary.csv",
        save_detail_name="soliton_details.csv",
        # Output filenames will be e.g.:
        #   soliton_vel_vs_accel_60g.png
        #   soliton_vel_vs_accel_100g.png
        #   soliton_vel_vs_accel_200g.png
        save_plot_name_template="soliton_vel_vs_accel_{mass_g:.0f}g.png",
        verbose=True,
    )

    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== Detail (first 10) ===")
    print(detail_df.head(10).to_string(index=False))
