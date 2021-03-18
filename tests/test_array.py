import xarray as xr
import numpy as np





t = np.zeros(4)
x = np.zeros((256, 256))
y = np.zeros((256, 256))
srf = np.zeros((4, 256, 256))


# ds = xr.DataArray(srf, coords=[("x", x), ("y", y)])
ds = xr.Dataset(

    {
        "elevations": (["time", "x", "y"], srf),
        "slopes x": (["time", "x", "y"], srf),
        "slopes y": (["time", "x", "y"], srf),
        "velocity x": (["time", "x", "y"], srf),
        "velocity y": (["time", "x", "y"], srf),
        "velocity z": (["time", "x", "y"], srf),
    },
    coords={
        "lon": (["x", "y"], x),
        "lat": (["x", "y"], y),
        "time": t,
    },

)

print(ds)