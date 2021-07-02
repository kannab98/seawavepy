import numpy as np
from seawave import config
from seawave.surface.core  import wind, surface



## Config lines
config['Radar']['WaveLength'] = "X"
config["Surface"]["LimitsOfModeling"] = [[-50, 50], [-50, 50]]
config['Surface']['GridSize'] = [128, 128]
config['Surface']['ThreadPerBlock'] = [16, 16, 1]
config['Surface']["Kernel"] = "default"


config['Wind']['Direction'] = 30
config['Wind']['Speed'] = 5


config['Dataset']['File'] = "test-dataset.nc"

fs = 1
x = np.linspace(*config["Surface"]["LimitsOfModeling"][0], config['Surface']['GridSize'][0], endpoint=True)
y = np.linspace(*config["Surface"]["LimitsOfModeling"][1], config['Surface']['GridSize'][1], endpoint=True)
t = np.arange(0, 1, 1/fs, dtype=np.float32)




## Main code
ds = surface((x,y,t), config=config)

wind(ds)
# print(ds)
