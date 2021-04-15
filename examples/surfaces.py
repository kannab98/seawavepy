import sys
sys.path.append(".")


import numpy as np
from seawave import config
from seawave.radar import radar
from seawave.surface.core  import wind, stat, surface
from seawave.surface import dataset
import matplotlib.pyplot as plt


config["Surface"]["LimitsOfModeling"] = [[-15, 15], [-15, 15]]
config['Surface']['GridSize'] = [512, 512]
config['Surface']['ThreadPerBlock'] = [16, 16, 1]
config['Radar']['Position'] = [0, 0, 28.3688]
config['Dataset']['File'] = "dataset_impulses.nc"
config['Wind']['Direction'] = 0
config['Wind']['Speed'] = 10.2



fs = 1
x = np.linspace(*config["Surface"]["LimitsOfModeling"][0], config['Surface']['GridSize'][0], endpoint=False)
y = np.linspace(*config["Surface"]["LimitsOfModeling"][1], config['Surface']['GridSize'][1], endpoint=False)
t = np.arange(0, 1, 1/fs, dtype=np.float32)

ds = surface((x,y,t), config=config)

kernels = [stat, radar, radar.power]
for kernel in kernels:
    kernel(ds)


