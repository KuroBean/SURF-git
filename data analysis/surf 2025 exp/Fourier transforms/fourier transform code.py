import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_and_plot_fft(csv_path, output_csv_path=None, sep=',', skiprows=0):
    """
    Reads a CSV file (time, voltage1, voltage2, voltage3), coerces to numeric,
    computes the FFT for each channel, returns a DataFrame of frequency & magnitude,
    and plots all three spectra.
    
    Parameters:
    - csv_path: str
    - output_csv_path: str or None
    - sep: delimiter for your CSV (',' or ';' or '\t')
    - skiprows: number of top rows to skip if you have e.g. a unit/header line
    """
    # 1) Load, with flexible delimiter and optional rows skipped
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows, skipinitialspace=True)
    
    # 2) Identify columns
    time_col = df.columns[0]
    volt_cols = df.columns[1:5]
    
    # 3) Force numeric conversion
    df[time_col]  = pd.to_numeric(df[time_col],  errors='coerce')
    df[volt_cols] = df[volt_cols].apply(pd.to_numeric, errors='coerce')
    
    # 4) Drop any rows that failed to parse
    n_bad = df.isna().any(axis=1).sum()
    if n_bad > 0:
        print(f"Warning: Dropping {n_bad} rows with non‑numeric entries.")
        df = df.dropna()
    
   # 5) Compute sampling interval and FFT
    dt = df[time_col].diff().iloc[1]
    n  = len(df)
    desired_bin_width = 5  # Hz
    n_fft = int(1 / (dt * desired_bin_width))
    n_fft = max(n_fft, n)  # Ensure n_fft is at least as large as your data
    freqs = np.fft.rfftfreq(n_fft, d=dt)

    fft_data = {'frequency': freqs}
    for col in volt_cols:
        y  = df[col].values
        yf = np.fft.rfft(y, n=n_fft)  # Zero-padded FFT
        fft_data[f'{col}_magnitude'] = np.abs(yf)

    fft_df = pd.DataFrame(fft_data)
    
    # 6) Save if requested
    if output_csv_path:
        fft_df.to_csv(output_csv_path, index=False)
        print(f"FFT results saved to {output_csv_path}")
    
    # 7) Plot with frequency range filter
    freq_min = 1      # Hz
    freq_max = 3000  # Hz
    mask = (freqs >= freq_min) & (freqs <= freq_max)

    # Custom legend labels and colors
    legend_labels = ["sensor 1", "sensor 2", "boundary sensor", "Actuator"]
    colors = ["orange", "green", "blue", "red"]

    plt.figure(figsize=(10,6)) 
    for col, label, color in zip(volt_cols, legend_labels, colors):
        plt.plot(freqs[mask], fft_df[f'{col}_magnitude'][mask], label=label, color=color)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'FFT of Voltage Signals ({freq_min} Hz to {freq_max} Hz)')
    plt.legend()
    plt.tight_layout()


    # Save plot as PNG with same name as CSV
    import os
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    plot_filename = f"Fourier {base_name}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as {plot_filename}")

    plt.show()
    
    return fft_df

# Example: if your file uses semicolons and has a units line:
# fft_df = compute_and_plot_fft(
#     'trial3.csv',
#     sep=';',
#     skiprows=1,
#     output_csv_path='fft_results.csv'
# )


# Example usage
fft_results = compute_and_plot_fft(r".\7_22 pulse data\Aluminum\interesting waveforms\triangular wave 1 pulse matched freq to boundary ringing.csv", "fft_results.csv",skiprows=2)
# Display first few rows
print(fft_results.head())

