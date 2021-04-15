import sys
sys.path.append(".")

import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
from seawave.spectrum import spectrum
from seawave.retracking import retracking
from seawave.radar import radar
from seawave import config
from seawave.surface.core  import wind, stat, surface
import pandas as pd



config["Surface"]["LimitsOfModeling"] = [[-15, 15], [-15, 15]]
config['Surface']['GridSize'] = [128, 128]
config['Surface']['ThreadPerBlock'] = [16, 16, 1]
config['Wind']['Speed']=10.2
config['Dataset']['File'] = "dataset_impulses.nc"

df = pd.read_excel("impulses/example-impulse.xlsx")

t0 = df['t'].values
P0 = df['P'].values
popt = retracking.pulse(t0, P0)
h = retracking.height(popt[2])

swh = retracking.swh(popt[3])



fs = 1
x = np.linspace(*config["Surface"]["LimitsOfModeling"][0], config['Surface']['GridSize'][0], endpoint=False)
y = np.linspace(*config["Surface"]["LimitsOfModeling"][1], config['Surface']['GridSize'][1], endpoint=False)
T = np.arange(0, 1, 1/fs, dtype=np.float32)

ds = surface((x,y,T), config=config)

kernels = [wind, stat, radar]
for kernel in kernels:
    kernel(ds)

P = radar.power(ds, t=t0)
print(ds)

plt.figure()
plt.plot(t0, P0)
plt.plot(t0, retracking.ice(t0, *popt))
plt.savefig('kek.png')
