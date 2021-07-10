import sys
sys.path.append(".")

from seawave import config
import xarray as xr
import numpy as np
from seawave.radar import radar
from seawave.surface.core import stat
import matplotlib.pyplot as plt

r0 = [0, 0, 1e6]

config['Radar']['Position'] = r0
config['Radar']['GainWidth'] = 30
config['Radar']['Direction'] = np.deg2rad([0, 0])


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
elev = ds['elevations'].values[0]
# angle of departure
aod = ds['AoD'][0]
tilt_modulation = fac * elev

"""
Shadow effect
"""
rho = np.sqrt(r[0]**2 + r[1]**2)
rho = rho[0]
chi = np.zeros(elev.shape)
# def shadow():
#     1 + np.sign( np.min(aodi - aod0))
#     return

def radar_ray(r, rho, z):
    # Equation of a straight line from a surface point to a radar
    return z + (r - rho) * (r0[2] - z)/rho



# Azimuthal angle
phi = np.rad2deg(np.arctan(r[1]/r[0]))
phi = phi[0]

shadow_effect = elev
# Loop over all surface points
for i in range(elev.shape[0]):
    for j in range(elev.shape[1]):
        phi0 = phi[i,j]
        # Select points lying on a straight line 
        # connecting the point of the surface and the radar
        # The selection is carried out by azimuth angle and distance.
        idx = (np.abs(phi0 - phi) < 1e-32) & (rho < rho[i,j])
        ray = radar_ray(rho[idx], rho[i,j], elev[i,j])
        # If any point of the radar line intersects the sea surface, a shadow effect appears.
        if (ray < elev[idx]).any():
            shadow_effect[i,j] = None

X = ds['X'][0]
Y = ds['Y'][0]

# # fig, ax = plt.subplots()
# # img = ax.contourf(X, Y, tilt_modulation[0], levels=100)
# # bar = plt.colorbar(img)
# # ax.set_xlabel("x")
# # ax.set_ylabel("y")
# # bar.set_label("tilt modulation")
# # fig.savefig("examples/radar-img.png")

# # ds.to_netcdf('test-dataset-radar.nc')



fig, ax = plt.subplots()
img = ax.contourf(X, Y, shadow_effect, levels=100)
bar = plt.colorbar(img)
ax.set_xlabel("x")
ax.set_ylabel("y")
bar.set_label("shadow effectr")
fig.savefig("examples/shadow-img.png")

# # ds.to_netcdf('test-dataset-radar.nc')

