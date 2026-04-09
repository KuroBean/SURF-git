import pyvisa

# The 'py' argument tells PyVISA to use the python-native backend 
# instead of the Keysight/NI .dll files.
try:
    rm = pyvisa.ResourceManager('@py') 
    resources = rm.list_resources()
    print(f"Resources: {resources}")
except Exception as e:
    print(f"PyVISA-py failed too: {e}")