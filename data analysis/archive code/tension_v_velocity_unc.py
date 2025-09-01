import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
from u_velocity_calc import *

# Assumes compute_pulse_velocity_with_uncertainty(...) is already defined/imported.
# It should accept at least (csv_path, distance, **kwargs) and return a dict
# that includes 'velocity' and 'u_velocity'.

DISTANCE = 0.018 * 8  # meters (given)

def _parse_pretension_from_folder(folder_name: str):
    """
    Extract numeric pretension in N from a folder like '3N', '10N', '6n'.
    """
    m = re.search(r'([-+]?\d*\.?\d+)\s*[nN]\b', folder_name)
    return float(m.group(1)) if m else None

def batch_pretension_velocity_with_uncertainty(
    root_dir=r'.\1D chain tension changing exp',
    distance=DISTANCE,
    compute_kwargs=None,
    save_table_name='pulse_velocity_vs_pretension_with_uncertainty.csv',
    save_plot_name='pulse_velocity_vs_pretension_with_uncertainty.png'
):
    """
    - Walk immediate subfolders of `root_dir` (e.g., 3N, 4N, 6N, ...)
    - Pick the FIRST .csv in each subfolder
    - Run compute_pulse_velocity_with_uncertainty()
    - Save a CSV summary and a PNG plot with error bars (±u_velocity)
    """
    compute_kwargs = dict(compute_kwargs or {})
    # Force no per-file plot to keep batch clean
    compute_kwargs.setdefault("plot", False)

    rows = []
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        pretension_N = _parse_pretension_from_folder(entry.name)
        if pretension_N is None:
            continue

        csvs = sorted(glob.glob(os.path.join(entry.path, "*.csv")))
        if not csvs:
            continue
        csv_path = csvs[0]

        try:
            res = compute_pulse_velocity_with_uncertainty(csv_path, distance=distance, **compute_kwargs)
        except TypeError:
            # Handle signature (csv_path, distance, ...) without keyword
            res = compute_pulse_velocity_with_uncertainty(csv_path, distance, **compute_kwargs)

        vel = res.get("velocity", None)
        uvel = res.get("u_velocity", None)

        rows.append({
            "pretension_N": pretension_N,
            "csv_path": csv_path,
            "velocity": vel,
            "u_velocity": uvel,
            "delta_t_s": res.get("delta_t"),
            "u_delta_t_s": res.get("u_delta_t"),
            "t1_mean_s": res.get("channel_1", {}).get("t_mean"),
            "t1_std_s":  res.get("channel_1", {}).get("t_std"),
            "t2_mean_s": res.get("channel_2", {}).get("t_mean"),
            "t2_std_s":  res.get("channel_2", {}).get("t_std"),
        })

    if not rows:
        raise RuntimeError("No valid pretension folders with CSV files were found (or all runs failed).")

    df = pd.DataFrame(rows).sort_values("pretension_N").reset_index(drop=True)

    # Save table
    out_csv = os.path.join(root_dir, save_table_name)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plot velocity vs pretension with error bars (±u_velocity)
    x = df["pretension_N"].values
    y = df["velocity"].values
    yerr = df["u_velocity"].values

    plt.figure(figsize=(8, 5))
    # Error bars with modest caps so they’re visible
    plt.errorbar(x, y, yerr=yerr, fmt='o-', capsize=4, linewidth=1.5)
    plt.xlabel("Pretension (N)")
    # If your distance is meters, this is m/s:
    plt.ylabel("Pulse velocity (m/s)")
    plt.title("Pulse velocity vs pretension (with ±1σ error bars)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(root_dir, save_plot_name)
    plt.savefig(out_png, dpi=200)
    plt.show()
    print(f"Saved: {out_png}")

    return df

# ==== Example call ====
# Put your preferred detector/grouping settings here.
# If you now specify time window in seconds and absolute amplitude window:
compute_kwargs = dict(
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

summary_df = batch_pretension_velocity_with_uncertainty(
    root_dir=r'.\1D chain tension changing exp',
    distance=DISTANCE,
    compute_kwargs=compute_kwargs
)
print(summary_df)
