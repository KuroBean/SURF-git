import os
import re
import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
from pulse_velocity import *

def _parse_pretension_from_folder(folder_name: str):
    """
    Extract a numeric pretension in Newtons from a folder name like '3N' or '10N'.
    Returns float or None if it can't parse.
    """
    m = re.search(r'([-+]?\d*\.?\d+)\s*[nN]\b', folder_name)
    return float(m.group(1)) if m else None

def batch_pretension_velocity(
    root_dir=r'.\1D chain tension changing exp',
    distance=0.018*8,
    compute_kwargs=None,
    save_table_name='pulse_velocity_vs_pretension.csv',
    save_plot_name='pulse_velocity_vs_pretension.png'
):
    """
    Walks `root_dir`, finds immediate subfolders named like '3N', '4N', '6N', ...
    Picks the FIRST .csv file found in each subfolder.
    Calls your `compute_pulse_velocity(csv_path, distance, **compute_kwargs)`.
    Saves a CSV and a PNG plot into `root_dir`.
    """
    compute_kwargs = compute_kwargs or {}

    # Collect (pretension, csv_path) pairs
    rows = []
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        pretension = _parse_pretension_from_folder(entry.name)
        if pretension is None:
            continue  # skip non-N folders
        # pick the first CSV in this folder (ignore the rest for now)
        csvs = sorted(glob.glob(os.path.join(entry.path, '*.csv')))
        if not csvs:
            continue
        csv_path = csvs[0]
        # Run the user's compute function
        try:
            res = compute_pulse_velocity(csv_path, distance=distance, **compute_kwargs)
        except TypeError:
            # If their signature is (csv_path, distance, sep, skiprows, ...)
            res = compute_pulse_velocity(csv_path, distance, **compute_kwargs)

        # Try to extract values robustly
        if isinstance(res, dict):
            velocity = res.get('velocity', None)
            t1 = res.get('t_peak1', res.get('t1', None))
            t2 = res.get('t_peak2', res.get('t2', None))
            dt = res.get('delta_t', res.get('dt', None))
        else:
            # If the function returned velocity directly
            velocity, t1, t2, dt = res, None, None, None

        rows.append({
            'pretension_N': pretension,
            'csv_path': csv_path,
            't1_s': t1,
            't2_s': t2,
            'delta_t_s': dt,
            'velocity_(units_per_s)': velocity
        })

    if not rows:
        raise RuntimeError("No valid pretension folders with CSV files were found or velocity computation failed.")

    df = pd.DataFrame(rows).sort_values('pretension_N').reset_index(drop=True)

    # Save table
    out_csv = os.path.join(root_dir, save_table_name)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plot velocity vs pretension
    x = df['pretension_N'].values
    y = df['velocity_(units_per_s)'].values

    plt.figure(figsize=(8,5))
    plt.plot(x, y, marker='o')
    plt.xlabel('Pretension (N)')
    plt.ylabel('Pulse velocity (units/s)')  # change to m/s if your distance is meters
    plt.title('Pulse velocity vs pretension')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(root_dir, save_plot_name)
    plt.savefig(out_png, dpi=200)
    plt.show()
    print(f"Saved: {out_png}")

    return df

# ---- Example call ----
# Tweak kwargs to match your loader/detector settings (delimiter, skiprows, thresholds, etc.)
# e.g., if your compute_pulse_velocity takes `sep`, `skiprows`, `threshold_frac`, `prominence`
# compute_kwargs = dict(sep=",", skiprows=0, threshold_frac=0.05, prominence=0.0)
compute_kwargs = {}  # fill with your usual parameters if needed

summary_df = batch_pretension_velocity(
    root_dir=r'.\1D chain tension changing exp',
    distance=0.018*8,
    compute_kwargs=compute_kwargs
)
print(summary_df)
