import pyvisa
import os
import time
import numpy as np
import h5py

# --- CONFIGURATION ---
DIRECTORY_NAME = "Experiment_Data_Run1"
SCOPE_USB_ADDRESS = 'USB0::0x0957::0x17A8::MY51360495::0::INSTR' 
# ---------------------

def setup_oscilloscope():
    rm = pyvisa.ResourceManager('C:/Windows/System32/visa64.dll')
    scope = rm.open_resource(SCOPE_USB_ADDRESS)
    # Increase timeout to 60 seconds for large data transfers
    scope.timeout = 60000 
    scope.clear()
    print(f"Connected to: {scope.query('*IDN?').strip()}")
    return scope

def capture_data(scope, folder):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    try:
        # 1. Capture Screen (PNG)
        print("Capturing screen...")
        scope.write(":DISPlay:DATA? PNG, COLor")
        raw_screen_data = scope.read_raw()
        
        png_path = os.path.join(folder, f"screenshot_{timestamp}.png")
        with open(png_path, 'wb') as f:
            # Keysight/Agilent usually has a 10-byte TMC header (#800xxxxxx)
            f.write(raw_screen_data[10:]) 

        # 2. Capture Waveform
        print("Capturing waveform data...")
        scope.write(":WAVeform:SOURce CHANnel1")
        scope.write(":WAVeform:FORMat BYTE") 
        
        # Change 'MAXimum' to 'NORMal' or set a specific limit
        # 'NORMal' usually captures the points currently visible on the screen (~1000 pts)
        # 'RAW' allows you to specify a count
        scope.write(":WAVeform:POINts:MODE RAW")
        scope.write(":WAVeform:POINts 100000") # Start with 100k points
        
        # Check how many points the scope ACTUALLY decided to provide
        actual_pts = scope.query(":WAVeform:POINts?")
        print(f"Requesting {actual_pts.strip()} points...")
        
        preamble = scope.query(":WAVeform:PREamble?")
        
        # Using query_binary_values with a chunk size can help with timeouts
        raw_data = scope.query_binary_values(":WAVeform:DATA?", datatype='b', container=np.array)
        
        print(f"Data received. Array size: {len(raw_data)}")
        
        h5_path = os.path.join(folder, f"waveform_{timestamp}.h5")
        with h5py.File(h5_path, 'w') as hf:
            dset = hf.create_dataset("waveform", data=raw_data, compression="gzip")
            hf.attrs["preamble"] = preamble
            print(f"H5 file saved successfully: {h5_path}")

        print(f"Success! Saved to {folder}")
        
    except Exception as e:
        print(f"Error during capture: {e}")

def main():
    if not os.path.exists(DIRECTORY_NAME):
        os.makedirs(DIRECTORY_NAME)
        
    scope = setup_oscilloscope()
    
    # Put scope in 'Single' mode to wait for a specific event
    scope.write(":SINGle")

    try:
        while True:
            # Query the 'Operation Complete' register or Trigger status
            # :TER? checks if a trigger event occurred since the last query
            triggered = scope.query(":TER?").strip()
            print(f"Trigger status: {triggered}")
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