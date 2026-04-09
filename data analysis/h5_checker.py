import h5py
import matplotlib.pyplot as plt
import numpy as np

# Change this to the path of the file you just saved
file_path = "Experiment_Data_Run1/waveform_20260408-174734.h5"

with h5py.File(file_path, 'r') as f:
    # Load the data
    raw_data = np.array(f['waveform'])
    preamble_str = f.attrs['preamble']
    
    # The preamble is a string of 10 values separated by commas:
    # Format, Type, Points, Count, X-Increment, X-Origin, X-Reference, 
    # Y-Increment, Y-Origin, Y-Reference
    pre = [float(x) for x in preamble_str.split(',')]
    
    x_inc, x_org, x_ref = pre[4], pre[5], pre[6]
    y_inc, y_org, y_ref = pre[7], pre[8], pre[9]

    # Convert raw bytes to Volts and Seconds
    time_axis = ((np.arange(len(raw_data)) - x_ref) * x_inc) + x_org
    voltage_axis = ((raw_data - y_ref) * y_inc) + y_org

    # Plot it
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, voltage_axis)
    plt.title(f"Verified Waveform: {file_path}")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.grid(True)
    plt.show()