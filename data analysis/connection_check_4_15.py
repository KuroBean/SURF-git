import pyvisa
import sys

print(sys.version)
print("64-bit Python:", sys.maxsize > 2**32)

rm = pyvisa.ResourceManager(r"C:\Windows\System32\visa64.dll")
print("Resources:", rm.list_resources())
print("Resource info:", rm.list_resources_info())

addr = r"USB0::0x0957::0x17A8::MY51360495::0::INSTR"
# Better: replace this with the VISA alias from Connection Expert if available

scope = rm.open_resource(addr)
scope.timeout = 5000
scope.write_termination = '\n'
scope.read_termination = '\n'

print(scope.query("*IDN?"))

scope.close()
rm.close()