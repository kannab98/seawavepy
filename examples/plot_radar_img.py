import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

srf = xr.open_dataset("test-dataset-radar.nc")

d = srf["AoA"].values
X = srf["X"].values
Y = srf["Y"].values


fig, ax = plt.subplots()
img = ax.contourf(X[0], Y[0], d[0], levels=100)
bar = plt.colorbar(img)
ax.set_xlabel("x")
ax.set_ylabel("y")
bar.set_label("elevation tilt modulation")
fig.savefig("examples/radar-img.png")