import xarray as xr
import matplotlib.pyplot as plt

srf = xr.open_dataset("test-dataset.nc")

elev = srf["elevations"].values
X = srf["X"].values
Y = srf["Y"].values


fig, ax = plt.subplots()
ax.contourf(X, Y, elev[0], levels=100)
fig.savefig("elev-img.png")