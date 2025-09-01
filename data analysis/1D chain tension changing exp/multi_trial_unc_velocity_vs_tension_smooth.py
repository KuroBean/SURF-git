# multi trial uncertainty v vs tension.py  — multi-file version

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulse_velocity_smooth import compute_pulse_velocity
from scipy.optimize import curve_fit

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

def batch_multifile_velocity_avg(
    root_dir=r'.\1D chain tension changing exp',
    distance=DISTANCE,
    compute_kwargs=None,
    min_files_per_folder=1,            # process folders with at least this many CSVs
    max_files_per_folder=None,         # None = use all CSVs in the folder
    save_table_name='smooth_pulse_velocity_vs_pretension_multi_avg.csv',
    save_plot_name='smooth_pulse_velocity_vs_pretension_multi_avg.png',
    save_detail_name='smooth_pulse_velocity_vs_pretension_multi_details.csv',
    verbose=True
):
    """
    - Scans subfolders under `root_dir` named like '3N', '4N', ...
    - Uses ALL CSVs found in each N folder (or up to max_files_per_folder, if set)
    - Runs compute_pulse_velocity on each CSV to get velocities
    - Aggregates by pretension: mean velocity and sample std (ddof=1)
    - Saves summary CSV, details CSV, and a plot with: individual points + mean ±1σ + sqrt-fit (to means)
    """
    compute_kwargs = dict(compute_kwargs or {})

    summary_rows, detail_rows = [], []
    folder_counts = []  # (folder, pretension, n_csv)

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    # pass 1: collect and compute per-file velocities
    for entry in os.scandir(root_dir):
        if not entry.is_dir():
            continue
        pretension = _parse_pretension_from_folder(entry.name)
        if pretension is None:
            continue

        csvs = _find_csvs(entry.path)
        folder_counts.append((entry.name, pretension, len(csvs)))
        if len(csvs) < int(min_files_per_folder):
            continue

        if isinstance(max_files_per_folder, int) and max_files_per_folder > 0:
            csvs = csvs[:max_files_per_folder]

        velocities = []
        for i, csv_path in enumerate(csvs, start=1):
            try:
                res = compute_pulse_velocity(csv_path, distance=distance, **compute_kwargs)
            except TypeError:
                # handle positional distance signature
                res = compute_pulse_velocity(csv_path, distance, **compute_kwargs)

            # Extract velocity from result (dict with 'velocity' or numeric)
            if isinstance(res, dict) and 'velocity' in res:
                v = float(res['velocity'])
            elif isinstance(res, (int, float)):
                v = float(res)
            else:
                raise RuntimeError(f"compute_pulse_velocity did not return a 'velocity' for: {csv_path}")

            velocities.append(v)
            detail_rows.append({
                'pretension_N': float(pretension),
                'csv_index': i,
                'csv_path': csv_path,
                'velocity': v
            })

        if len(velocities) == 0:
            continue

        v_arr = np.asarray(velocities, dtype=float)
        v_mean = float(np.mean(v_arr))
        v_std  = float(np.std(v_arr, ddof=1)) if len(v_arr) > 1 else 0.0

        summary_rows.append({
            'pretension_N': float(pretension),
            'n_files_used': int(len(v_arr)),
            'velocity_mean': v_mean,
            'velocity_std': v_std
        })

    if not summary_rows:
        if verbose:
            print("Diagnostic: CSV counts per N-folder under:", root_dir)
            for name, N, n in sorted(folder_counts, key=lambda x: x[1]):
                print(f"  {name:<8}  N={N:<6g}  csv_count={n}")
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

    # --- Plot: individuals + mean ±1σ; fit uses means only ---
    x_mean = df_summary['pretension_N'].values.astype(float)
    y_mean = df_summary['velocity_mean'].values.astype(float)
    yerr   = df_summary['velocity_std'].values.astype(float)

    plt.figure(figsize=(9,6))

    # scatter all individual CSV velocities
    x_pts = df_detail['pretension_N'].values.astype(float)
    y_pts = df_detail['velocity'].values.astype(float)
    plt.scatter(x_pts, y_pts, s=22, alpha=0.55, label='individual CSVs', zorder=2)

    # error bars on means
    plt.errorbar(x_mean, y_mean, yerr=yerr, fmt='o', capsize=4, linewidth=1.5,
                 color='k', label='mean ± 1σ', zorder=3)

    # ---- model: a*sqrt(x) + d (fit to means) ----
    def model(x, a, d):
        return a * np.sqrt(np.clip(x, 0.0, np.inf)) + d

    # Initial guess from endpoints
    a0 = (y_mean[-1] - y_mean[0]) / (np.sqrt(max(x_mean[-1], 1e-12)) - np.sqrt(max(x_mean[0], 1e-12)) + 1e-12)
    d0 = float(np.min(y_mean))
    p0 = [a0 if np.isfinite(a0) else 1.0, d0]

    # Weighted fit if we have usable stds
    sigma = None
    if np.any(np.isfinite(yerr) & (yerr > 0)):
        sigma = np.where(yerr > 0, yerr, np.nan)
        if not np.any(np.isfinite(sigma)):
            sigma = None

    popt, pcov = curve_fit(
        model, x_mean, y_mean, p0=p0, sigma=sigma, absolute_sigma=bool(sigma is not None),
        bounds=([-np.inf, -np.inf], [np.inf, np.inf]), maxfev=50000
    )
    a_hat, d_hat = map(float, popt)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, np.inf))
    a_se, d_se = (float(perr[0]), float(perr[1])) if perr.size >= 2 else (np.nan, np.nan)

    y_fit = model(x_mean, *popt)
    sse = float(np.sum((y_mean - y_fit)**2))
    sst = float(np.sum((y_mean - np.mean(y_mean))**2))
    r2  = 1.0 - sse/sst if sst > 0 else np.nan

    # smooth overlay
    xx = np.linspace(np.min(x_mean), np.max(x_mean), 400)
    yy = model(xx, *popt)
    

    plt.xlabel('Pretension (N)')
    plt.ylabel('Pulse velocity (m/s)')
    plt.title('Pulse velocity vs pretension (multi-file avg ± 1σ)')
    
    #plt.plot(xx, yy, '-', label='fit: a·√x + d', zorder=4)
    # eqn = r"$v(x)=a\sqrt{x}+d$"
    # box = "\n".join([
    #     eqn,
    #     fr"$a={a_hat:.4g}\ \pm\ {a_se:.2g}$",
    #     fr"$d={d_hat:.4g}\ \pm\ {d_se:.2g}$",
    #     fr"$R^2={r2:.4f}$"
    # ])
    # plt.gca().text(
    #     0.02, 0.98, box, transform=plt.gca().transAxes,
    #     va='top', ha='left',
    #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    # )

    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()

    out_plot = os.path.join(root_dir, save_plot_name)
    plt.savefig(out_plot, dpi=200)
    plt.show()
    print(f"Saved plot with fit: {out_plot}")

    # Save fit parameters
    fit_csv_path = os.path.join(root_dir, 'smooth_pulse_velocity_fit_params_simple_sqrt.csv')
    pd.DataFrame({
        'param': ['a','d'],
        'value': [a_hat, d_hat],
        'stderr': [a_se, d_se],
        'R2': [r2, r2]
    }).to_csv(fit_csv_path, index=False)
    print(f"Saved fit params: {fit_csv_path}")

    return df_summary, df_detail


# ---- Example run (adjust root_dir / compute_kwargs as needed) ----
compute_kwargs = dict(
    sep=',',
    skiprows=2,
    show=True,                 # per-file plots off during batch run
    threshold_frac=0.5,         # 50% of per-channel max
    prominence=0.05,            # tweak if needed
    # --- smoothing (AUTO) ---
    smooth=True,
    smooth_window=401,         # None/'auto' => choose from data length
    smooth_polyorder=3,
    overlay_raw=True           # set True only if you also set show=True
)


summary_df, detail_df = batch_multifile_velocity_avg(
    root_dir=r'.\8_19 main data',
    distance=DISTANCE,
    compute_kwargs=compute_kwargs,
    min_files_per_folder=1,
    max_files_per_folder=None,  # use all CSVs found
    verbose=True
)
print(summary_df)
