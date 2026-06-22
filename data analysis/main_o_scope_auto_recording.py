import pyvisa
import os
import time
import numpy as np
import h5py

# --- CONFIGURATION ---
DIRECTORY_NAME = "./soliton_experiments/test_exp/135_deg"
SCOPE_USB_ADDRESS = 'USB0::0x0957::0x17A8::MY51360495::0::INSTR' 
filename_prefix = "200g"
# ---------------------

def setup_oscilloscope():
    rm = pyvisa.ResourceManager('C:/Windows/System32/visa64.dll')
    scope = rm.open_resource(SCOPE_USB_ADDRESS)
    # Increase timeout to 60 seconds for large data transfers
    scope.timeout = 7000 
    scope.clear()
    print(f"Connected to: {scope.query('*IDN?').strip()}")
    return scope
def capture_data(scope, folder):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    h5_path = os.path.join(folder, f"{filename_prefix}_{timestamp}.h5")
    
    # Define what we want to capture
    # 'MATH1' or 'MATH' depending on your model
    sources = ["CHANnel2", "CHANnel3"] 
    
    # 1. Capture Screen (Once per trigger)
    scope.write(":DISPlay:DATA? PNG, COLor")
    raw_screen_data = scope.read_raw()
    with open(os.path.join(folder, f"{filename_prefix}_{timestamp}.png"), 'wb') as f:
        f.write(raw_screen_data[10:])

    # 2. Capture Waveforms into one H5 file
    with h5py.File(h5_path, 'w') as hf:
        for src in sources:
            try:
                print(f"Pulling data from {src}...")
                scope.write(f":WAVeform:SOURce {src}")
                print(f"Configuring waveform for {src}...")
                scope.write(":WAVeform:FORMat BYTE")
                scope.write(":WAVeform:POINts:MODE RAW")
                scope.write(":WAVeform:POINts 100000")

                # Get scaling for THIS specific channel
                preamble = scope.query(":WAVeform:PREamble?")
                data = scope.query_binary_values(":WAVeform:DATA?", datatype='B', container=np.array)
                
                # Create a group or unique dataset name in the H5
                print(f"Saving {src} data to H5...")
                ds = hf.create_dataset(src, data=data, compression="gzip")
                ds.attrs["preamble"] = preamble
                
            except Exception as e:
                print(f"Could not capture {src}: {e}")

    print(f"Saved all channels to: {h5_path}")

def main():
    if not os.path.exists(DIRECTORY_NAME):
        os.makedirs(DIRECTORY_NAME)
        
    scope = setup_oscilloscope()
    
    # Put scope in 'Single' mode to wait for a specific event
    scope.write("*CLS")  # Clear all status/event registers before arming
    scope.write(":SINGle")
    time.sleep(1) # Give it a moment to settle
    scope.query(":TER?")  # Flush any stale trigger event that fired during connection/setup
    try:
        while True:
            # Query the 'Operation Complete' register or Trigger status
            # :TER? checks if a trigger event occurred since the last query
            triggered = scope.query(":TER?").strip()
            #print(f"Trigger status: {triggered}")
            if triggered == "+1":
                print("Trigger detected!")
                # Give the scope a tiny bit of time to finish drawing
                time.sleep(0.2) 
                capture_data(scope, DIRECTORY_NAME)
                
                # Rearm the trigger for the next one
                print("Rearming for next trigger...")
                scope.write(":SINGle")
            
            time.sleep(0.1) # Don't spam the USB bus
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        scope.close()

if __name__ == "__main__":
    main()