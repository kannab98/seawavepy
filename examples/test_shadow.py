
import sys
sys.path.append(".")

from seawave import config
import xarray as xr
import numpy as np
from seawave.radar import radar
from seawave.surface.core import stat
import matplotlib.pyplot as plt

r0 = [0, 0, 10]

config['Radar']['Position'] = r0
config['Radar']['GainWidth'] = 30


ds = xr.open_dataset('test-dataset.nc')


kernels = [stat, radar]

for kernel in kernels:
    kernel(ds)



# Radius from radar to surface point
r = np.array([ds['X'], ds['Y'], ds['Z']])
# elevations of surface
elev = ds['elevations'].values[0]
# Distance in XY projection
rho = np.sqrt(r[0]**2 + r[1]**2)
rho = rho[0]

"""
Shadow effect
"""
def radar_ray(r, rho, z):
    # Equation of a straight line from a surface point to a radar
    return z + (r - rho) * (r0[2] - z)/rho + r0[2]

# Azimuthal angle
phi = np.rad2deg(np.arctan(r[1]/r[0]))
phi = phi[0]

shadow_effect = elev
N = elev.shape[0]
# Loop over all surface points
for i in range(elev.shape[0]):
    for j in range(elev.shape[1]):
        phi0 = phi[i,j]
        # Select points lying on a straight line 
        # connecting the point of the surface and the radar
        # The selection is carried out by azimuth angle and distance.
        idx = (np.abs(phi0 - phi) <= 0.1) & (rho <= rho[i,j])
        ray = radar_ray(rho[idx], rho[i,j], elev[i,j])
        # print(ray)
        # If any point of the radar line intersects the sea surface, a shadow effect appears.
        if (ray < elev[idx]).any():
            shadow_effect[i,j] = None

X = ds['X'][0]
Y = ds['Y'][0]



fig, ax = plt.subplots()
img = ax.contourf(X, Y, shadow_effect, levels=100)
bar = plt.colorbar(img)
ax.set_xlabel("x")
ax.set_ylabel("y")
bar.set_label("shadow effect")
fig.savefig("examples/shadow-img.png")