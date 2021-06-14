import sys
sys.path.append(".")

from seawave import config
import xarray as xr
import numpy as np
from seawave.radar import radar
from seawave.surface.core import stat
import matplotlib.pyplot as plt

r0 = [0, 0, 30]

config['Radar']['Position'] = r0
config['Radar']['GainWidth'] = 30


ds = xr.open_dataset('test-dataset.nc')


kernels = [stat, radar]

for kernel in kernels:
    kernel(ds)



# Radius from radar to surface point
r = np.array([ds['X'], ds['Y'], ds['Z']])
# Normal vector
n = np.array([*ds['slopes']])


"""
Tild modulation
"""
# Vector product 

fac = np.sum(r*n, axis=0)
# see [6] in Research of X-Band Radar Sea Clutter Image Simulation Method 
fac[fac <= 0]  = 0
elev = ds['elevations']
tilt_modulation = elev * fac

"""
Shadow effect
"""
# see [7] in Research of X-Band Radar Sea Clutter Image Simulation Method 
rho = np.sqrt(ds['X']**2  + ds['Y']**2)
alpha = np.arctan(rho/(r0[2] - elev))
# That is R0?
# Thai is Omega? 
# I don't seem to know how to quickly build the shadow effect right now. I leave it to you
# R0 = np.zeros_like(rho)

X = ds['X'][0]
Y = ds['Y'][0]

fig, ax = plt.subplots()
img = ax.contourf(X, Y, tilt_modulation[0], levels=100)
bar = plt.colorbar(img)
ax.set_xlabel("x")
ax.set_ylabel("y")
bar.set_label("tilt modulation")
fig.savefig("examples/radar-img.png")

ds.to_netcdf('test-dataset-radar.nc')
