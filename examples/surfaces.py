import sys
sys.path.append(".")


import numpy as np
from seawave import config
from seawave.radar import radar
from seawave.surface.core  import wind, stat, surface
from seawave.surface import dataset
import matplotlib.pyplot as plt


config["Surface"]["LimitsOfModeling"] = [[-15, 15], [-15, 15]]
config['Surface']['GridSize'] = [128, 128]
config['Surface']['ThreadPerBlock'] = [16, 16, 1]
config['Surface']["Kernel"] = ["default"]
config['Radar']['Position'] = [0, 0, 30]
config['Dataset']['File'] = "dataset.nc"
config['Wind']['Direction'] = 0
config['Wind']['Speed'] = 5



fs = 1
x = np.linspace(*config["Surface"]["LimitsOfModeling"][0], config['Surface']['GridSize'][0], endpoint=True)
y = np.linspace(*config["Surface"]["LimitsOfModeling"][1], config['Surface']['GridSize'][1], endpoint=True)
t = np.arange(0, 1, 1/fs, dtype=np.float32)

ds = surface((x,y,t), config=config)

kernels = [wind]
for kernel in kernels:
    kernel(ds)

print(ds)