import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulse_velocity import *
        # --- add these imports near the top of your file ---
from scipy.optimize import curve_fit
import textwrap


DISTANCE = 0.018 * 8  # meters (given)

def _parse_pretension_from_folder(folder_name: str):
    """
    Extract numeric pretension in N from names like '3N', '10 N', '6n'.
    More permissive: allows spaces and trailing whitespace.
    """
    m = re.search(r'^\s*([-+]?\d*\.?\d+)\s*[nN]\s*$', folder_name)
    return float(m.group(1)) if m else None

def _find_csvs(folder_path: str):
    """
    Return CSV paths (case-insensitive) directly inside folder_path.
    """
    paths = []
    for name in os.listdir(folder_path):
        if name.lower().endswith('.csv'):
            paths.append(os.path.join(folder_path, name))
    return sorted(paths)

def batch_threefile_velocity_avg(
    root_dir=r'.\1D chain tension changing exp',
    distance=DISTANCE,
    show=False,
    compute_kwargs=None,
    require_exactly_three=False,     # set True to force exactly 3 CSVs per folder
    max_files_per_folder=3,          # we only use the first 3 files
    save_table_name='pulse_velocity_vs_pretension_3file_avg.csv',
    save_plot_name='pulse_velocity_vs_pretension_3file_avg.png',
    save_detail_name='pulse_velocity_vs_pretension_3file_details.csv',
    verbose=True
):
    """
    - Scans subfolders under `root_dir` named like '3N', '4N', ...
    - Uses either folders with >=3 CSVs (default) or exactly 3 (if require_exactly_three=True)
    - Runs compute_pulse_velocity on the first 3 CSVs in each folder
    - Averages the three velocities and uses sample std (ddof=1) as uncertainty
    - Saves a summary CSV, details CSV, and an error-bar plot into `root_dir`
    """
    compute_kwargs = dict(compute_kwargs or {})

    # Gather candidates and keep track for helpful diagnostics
    summary_rows = []
    detail_rows = []
    folder_counts = []  # (folder, pretension, n_csv)

    # Validate root_dir exists
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        pretension = _parse_pretension_from_folder(entry.name)
        if pretension is None:
            continue

        csvs = _find_csvs(entry.path)
        folder_counts.append((entry.name, pretension, len(csvs)))

        if require_exactly_three:
            if len(csvs) != 3:
                continue
        else:
            if len(csvs) < 3:
                continue

        # Take the first three CSVs deterministically
        csvs = csvs[:max_files_per_folder]
        velocities = []

        for i, csv_path in enumerate(csvs, start=1):
            try:
                res = compute_pulse_velocity(csv_path, distance=distance, **compute_kwargs)
            except TypeError:
                # handle positional distance signature
                res = compute_pulse_velocity(csv_path, distance, **compute_kwargs)

            # Extract velocity from result
            if isinstance(res, dict) and 'velocity' in res:
                v = float(res['velocity'])
            elif isinstance(res, (int, float)):
                v = float(res)
            else:
                raise RuntimeError(f"compute_pulse_velocity did not return a 'velocity' for: {csv_path}")

            velocities.append(v)
            detail_rows.append({
                'pretension_N': pretension,
                'csv_index': i,
                'csv_path': csv_path,
                'velocity': v
            })

        v_arr = np.asarray(velocities, dtype=float)
        v_mean = float(np.mean(v_arr))
        v_std  = float(np.std(v_arr, ddof=1)) if len(v_arr) > 1 else 0.0

        summary_rows.append({
            'pretension_N': pretension,
            'n_files_used': len(v_arr),
            'velocity_mean': v_mean,
            'velocity_std': v_std
        })

    if not summary_rows:
        # Helpful diagnostics so you can see what was found
        if verbose:
            print("Diagnostic: CSV counts per N-folder under:", root_dir)
            for name, N, n in sorted(folder_counts, key=lambda x: x[1]):
                print(f"  {name:<8}  N={N:<6g}  csv_count={n}")
            if require_exactly_three:
                print("\nNo folders had exactly 3 CSV files. "
                      "Set require_exactly_three=False to accept folders with ≥3 and use the first three.")
        raise RuntimeError("No suitable N-folders were processed. See diagnostics above.")

    # Build DataFrames
    df_summary = pd.DataFrame(summary_rows).sort_values('pretension_N').reset_index(drop=True)
    df_detail  = pd.DataFrame(detail_rows).sort_values(['pretension_N','csv_index']).reset_index(drop=True)

    # Save CSVs
    out_summary = os.path.join(root_dir, save_table_name)
    out_detail  = os.path.join(root_dir, save_detail_name)
    df_summary.to_csv(out_summary, index=False)
    df_detail.to_csv(out_detail, index=False)
    if verbose:
        print(f"Saved summary: {out_summary}")
        print(f"Saved details: {out_detail}")

    # Plot with error bars (±1σ)
    x = df_summary['pretension_N'].values
    y = df_summary['velocity_mean'].values
    yerr = df_summary['velocity_std'].values

    #FITTING AND PLOTTING


    # --- Prepare data from your summary dataframe (already computed above) ---
    x = df_summary['pretension_N'].values.astype(float)
    y = df_summary['velocity_mean'].values.astype(float)
    yerr = df_summary['velocity_std'].values.astype(float)

    plt.figure(figsize=(8,5))
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=4, linewidth=1.5, label='data (mean ± 1σ)')

    # ---- model: a*sqrt(x) + d ----
    def model(x, a, d):
        return a * np.sqrt(np.clip(x, 0.0, np.inf)) + d

    # Initial guesses
    a0 = (y[-1] - y[0]) / (np.sqrt(max(x[-1], 1e-12)) - np.sqrt(max(x[0], 1e-12)) + 1e-12)
    d0 = float(np.min(y))
    p0 = [a0 if np.isfinite(a0) else 1.0, d0]

    # Optional weighting by yerr (ignore nonpositive/NaN)
    sigma = None
    if np.any(np.isfinite(yerr) & (yerr > 0)):
        sigma = np.where(yerr > 0, yerr, np.nan)
        if not np.any(np.isfinite(sigma)):
            sigma = None

    # Fit
    bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
    popt, pcov = curve_fit(
        model, x, y, p0=p0, sigma=sigma, absolute_sigma=bool(sigma is not None),
        bounds=bounds, maxfev=50000
    )
    a_hat, d_hat = popt
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    a_se, d_se = perr

    # R^2
    y_fit = model(x, *popt)
    sse = float(np.sum((y - y_fit)**2))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - sse/sst if sst > 0 else np.nan

    # Overlay smooth fit
    xx = np.linspace(np.min(x), np.max(x), 400)
    yy = model(xx, *popt)
    plt.plot(xx, yy, '-', label='fit: a·√x + d')

    plt.xlabel('Pretension (N)')
    plt.ylabel('Pulse velocity (m/s)')
    plt.title('Pulse velocity vs pretension (3-file avg ± 1σ)')
    eqn = r"$v(x)=a\sqrt{x}+d$"
    txt = "\n".join([
        eqn,
        fr"$a={a_hat:.4g}\ \pm\ {a_se:.2g}$",
        fr"$d={d_hat:.4g}\ \pm\ {d_se:.2g}$",
        fr"$R^2={r2:.4f}$"
    ])
    plt.gca().text(
        0.02, 0.98, txt, transform=plt.gca().transAxes,
        va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    out_plot = os.path.join(root_dir, save_plot_name)  # reuse your existing filename
    plt.savefig(out_plot, dpi=200)
    plt.show()
    print(f"Saved plot with fit: {out_plot}")

    # Save fit parameters next to your summary
    fit_csv_path = os.path.join(root_dir, 'pulse_velocity_fit_params_simple_sqrt.csv')
    pd.DataFrame({
        'param': ['a','d'],
        'value': [a_hat, d_hat],
        'stderr': [a_se, d_se],
        'R2': [r2, r2]
    }).to_csv(fit_csv_path, index=False)
    print(f"Saved fit params: {fit_csv_path}")



    return df_summary, df_detail

# ---- Example call ----
# Include any parameters your compute_pulse_velocity needs (delimiter, skiprows, thresholds, etc.)
compute_kwargs = dict(
    sep=',',
    skiprows=2,
    show=False,
    threshold_frac=0.5,  # 50% of max amplitude
    prominence=0.01       # adjust as needed
)

# Accept folders with ≥3 CSVs and use the first three:
summary_df, detail_df = batch_threefile_velocity_avg(
    root_dir=r'.\1D chain tension changing exp',
    distance=DISTANCE,
    compute_kwargs=compute_kwargs,
    require_exactly_three=False,   # set True if you want strictly exactly 3
    verbose=True
)
print(summary_df)
