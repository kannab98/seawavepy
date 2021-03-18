
import sys
sys.path.append(".")
import seawave
import xarray as xr 


ds = seawave.Dataset()
surf = ds.empty_surface_dataset()
print(ds._ds)


ds0 = xr.merge([ds._ds, surf])
print(ds0)


