import pyvisa
import time

# Use the absolute path to the Keysight DLL
visa_dll = 'C:/Windows/System32/visa64.dll'
# PASTE YOUR ADDRESS FROM CONNECTION EXPERT HERE
my_scope_address = 'USB0::0x0957::0x17A8::MY51360495::0::INSTR' 

try:
    # 1. Initialize with the explicit DLL
    rm = pyvisa.ResourceManager(visa_dll)
    
    # 2. Force open the resource even if list_resources() is empty
    print(f"Attempting direct connection to {my_scope_address}...")
    scope = rm.open_resource(my_scope_address)
    
    # 3. Set a long timeout and clear the buffer
    scope.timeout = 5000
    scope.clear()
    
    # 4. Ask for ID
    idn = scope.query("*IDN?")
    print(f"SUCCESS! Connected to: {idn}")
    
    # If this works, we can proceed with the saving logic.
    scope.close()

except Exception as e:
    print(f"\n--- Connection Failed ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Details: {e}")