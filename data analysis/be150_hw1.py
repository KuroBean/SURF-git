import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
# def dmdt(m,p,k,n):
#     return 1/(1+(p/k)**n) - m
def dmdt(m,p,k,n):
    return 1 - m
def dpdt(m,p,g):
    return g*(m-p)

m0=0
p0=0
dt=0.1
t=np.arange(0, 30, dt)
#k=176.4*2.77e-3*200/(1*144)
k=1
n=3
#g=176.4/2.77e-3
g=1
m=np.zeros(len(t))
p=np.zeros(len(t))
m[0]=m0
p[0]=p0
#euler's method
for i in range(1, len(t)):
    m[i]=m[i-1]+dmdt(m[i-1],p[i-1],k,n)*dt
    p[i]=p[i-1]+dpdt(m[i-1],p[i-1],g)*dt

plt.plot(t,m,label='mRNA')
plt.plot(t,p,label='protein')
plt.xlabel('time (s)')
plt.ylabel('concentration (#/cell)')
plt.legend()
plt.show()