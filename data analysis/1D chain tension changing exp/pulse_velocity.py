import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def compute_pulse_velocity(csv_path, distance, sep=',', skiprows=0, threshold_frac=0.5, prominence=None,sample_dist=None,show=False):
    """
    Reads a CSV file (time, ch1, ch2, ...), identifies the first main pulse peak
    in channels 1 and 2, and computes pulse velocity = distance / (t2 - t1).

    Parameters:
    - csv_path: str, path to CSV (first col time, next cols voltage)
    - distance: float, known separation between sensors (same units as time distance/time)
    - sep: delimiter, default ','
    - skiprows: rows to skip at top (e.g. units row)
    - threshold_frac: float [0,1], fraction of each channel's max amplitude to set as a threshold
    - prominence: dict or float, passed to scipy.signal.find_peaks for peak prominence
    """
    # 1) Load and coerce numeric
    df = pd.read_csv(csv_path, sep=sep, skiprows=skiprows, skipinitialspace=True)
    t_col = df.columns[0]
    ch_cols = df.columns[1:3]  # first two voltage channels
    df[t_col]  = pd.to_numeric(df[t_col], errors='coerce')
    df[ch_cols] = df[ch_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=[t_col, *ch_cols])
    
    time = df[t_col].values
    v1 = df[ch_cols[0]].values
    v2 = df[ch_cols[1]].values
    
    # 2) Detect peaks
    # Calculate individual thresholds based on max amplitude
    th1 = threshold_frac * np.max(v1)
    th2 = threshold_frac * np.max(v2)
    
    print("Finding peaks with thresholds:", th1, th2)
    peaks1, props1 = find_peaks(v1, height=th1, prominence=prominence,distance=sample_dist)
    peaks2, props2 = find_peaks(v2, height=th2, prominence=prominence,distance=sample_dist)
    
    if len(peaks1) == 0 or len(peaks2) == 0:
        raise ValueError("No peaks found above threshold. Try lowering threshold_frac or adjusting prominence.")
    
    # 3) Select first main pulse: earliest peak above threshold
    max_idx1 = np.argmax(v1[peaks1])  # index in peaks1 array
    max_idx2 = np.argmax(v2[peaks2])
    peak1 = peaks1[max_idx1]
    peak2 = peaks2[max_idx2]
    
    t_peak1 = time[peak1]
    t_peak2 = time[peak2]
    
    # 4) Compute velocity
    dt = t_peak2 - t_peak1
    if dt <= 0:
        raise ValueError(f"Non-positive time difference: {dt}. Check your data ordering.")
    velocity = distance / dt
    
    # 5) Plot signals and peaks
    if show:
        plt.figure(figsize=(10, 5))
        plt.plot(time, v1, label=ch_cols[0])
        plt.plot(time[peaks1], v1[peaks1], 'x', label=f'Peak {ch_cols[0]} at {t_peak1:.4f}s')
        plt.plot(time, v2, label=ch_cols[1])
        plt.plot(time[peaks2], v2[peaks2], 'x', label=f'Peak {ch_cols[1]} at {t_peak2:.4f}s')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Pulse Detection and Velocity: {velocity:.2f} m/s')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(time, v1, label=ch_cols[0])
        plt.plot(time[peaks1], v1[peaks1], 'x', label=f'Peak {ch_cols[0]} at {t_peak1:.4f}s')
        plt.plot(time, v2, label=ch_cols[1])
        plt.plot(time[peaks2], v2[peaks2], 'x', label=f'Peak {ch_cols[1]} at {t_peak2:.4f}s')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'Pulse Detection and Velocity: {velocity:.2f} m/s')
        plt.legend()
        plt.tight_layout()
  
    print(f"for {csv_path}:\n")
    print(time[peaks1])
    print(time[peaks2])

    return {
        't_peak1': t_peak1,
        't_peak2': t_peak2,
        'delta_t': dt,
        'velocity': velocity
    }

# Example usage:
if __name__ == "__main__":
    results = compute_pulse_velocity(
        r'.\1D chain tension changing exp\8_19 main data\14N\scope_14.csv', 
        distance=0.018*8,        # meters
        sep=',',
        skiprows=2,
        threshold_frac=0.5,  # 50% of max amplitude
        prominence=0.1,       # adjust as needed
        show=True
    )
    print(results)