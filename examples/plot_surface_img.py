import xarray as xr
import matplotlib.pyplot as plt

srf = xr.open_dataset("test-dataset.nc")

elev = srf["elevations"].values
X = srf["X"].values
Y = srf["Y"].values
t = srf["time"].values

for i in range(t.size):
	fig, ax = plt.subplots()
	img = ax.contourf(X, Y, elev[i], levels=100)
	bar = plt.colorbar(img)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	bar.set_label("elevations")
	fig.savefig("examples/elev-img-%d.png" % i)