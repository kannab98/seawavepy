import sys
sys.path.append(".")

from seawave import config
import xarray as xr
import numpy as np
from seawave.radar import radar
from seawave.surface.core import stat
import matplotlib.pyplot as plt

r0 = [0, 0, 15]

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
n = n / np.linalg.norm(n, axis=0)


"""
Tilt modulation
"""
# Vector product 

fac = np.sum(r*n, axis=0)
# see [6] in Research of X-Band Radar Sea Clutter Image Simulation Method 
fac[fac <= 0]  = 0
# elevations of surface
elev = ds['elevations']
# angle of departure
aod = ds['AoD'][0]
tilt_modulation = fac

"""
Shadow effect
"""
N, M = elev.shape[1:]

rho = np.sqrt(r[0]**2 + r[1]**2)
rho = rho[0].flatten()
chi = np.zeros(elev.size)
for i in range(elev.size):
    r0 = rho[i]
    r = rho[rho < r0]
    aodi = aod.values.flat[rho < r0]
    aod0 = aod.values.flat[i]
    try:
        chi[i] = 1 + np.sign(
            np.min(
                np.heaviside(r0-r, 1) *
                np.heaviside(r, 1) *
                (aodi - aod0)
            )
        )
    except:
        chi[i] = 0
chi = chi.reshape(elev.shape[1:])
shadow_effect = chi
# see [7] in Research of X-Band Radar Sea Clutter Image Simulation Method 
# rho = np.sqrt(ds['X']**2  + ds['Y']**2)
# alpha = np.arctan(rho/(r0[2] - elev))
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



fig, ax = plt.subplots()
img = ax.contourf(X, Y, shadow_effect, levels=100)
bar = plt.colorbar(img)
ax.set_xlabel("x")
ax.set_ylabel("y")
bar.set_label("shadow effectr")
fig.savefig("examples/shadow-img.png")

ds.to_netcdf('test-dataset-radar.nc')

