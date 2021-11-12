import math
import numpy as np
import matplotlib.pyplot as plt

LUT=[
     1.0000, 0.9848,  0.9396, 0.8660,
     0.7660, 0.6427,  0.5000, 0.3420,
     0.1736, 0.0000, -0.1736,-0.3420,
    -0.5000,-0.6427, -0.7660,-0.8660,
    -0.9396,-0.9848, -1.0000,-0.9848,
    -0.9396,-0.8660, -0.7660,-0.6427,
    -0.5000,-0.3420, -0.1736,-0.0000,
     0.1736, 0.3420,  0.5000, 0.6427,
     0.7660, 0.8660,  0.9396, 0.9848, 1.0]

def lut_cos_deg(x):
    x%=360
    u=x//10
    v=x%10
    a,b=LUT[u],LUT[u+1]
    return a+(b-a)*float(v)/10.0

x =np.array([np.cos(math.radians(d)) for d in range(360)])
x0=np.array([lut_cos_deg(d)          for d in range(360)])

plt.clf()
plt.plot(x,'--k')
plt.plot(x0,'k')
plt.grid(True)
plt.show()
    
plt.plot(np.array(x)-np.array(x0),'k')
plt.grid(True)
plt.show()

