import sys
sys.path.append(".")


import numpy as np
import xarray as xr
from seawave import config
from seawave.spectrum.module import dispersion

ds = xr.open_dataset('database_1800.nc')
k = ds['k'].values

w = dispersion.omega(k)
jac = dispersion.det(k)

omega = xr.DataArray(
    data=w,
    dims=['k'],
    coords=dict(k=k),
)

Jac = xr.DataArray(
    data=w,
    dims=['k'],
    coords=dict(k=k)
)

ds['omega'] = omega 
ds['jacobian'] = Jac 

ds.to_netcdf('database_new.nc')


