import sys
sys.path.append(".")

from seawave import config
import xarray as xr
from seawave.radar import radar
from seawave.surface.core import stat

config['Radar']['Position'] = [0, 0, 10]
config['Radar']['GainWidth'] = 30


ds = xr.open_dataset('test-dataset.nc')


kernels = [stat, radar]

for kernel in kernels:
    kernel(ds)

ds.to_netcdf('test-dataset-radar.nc')
