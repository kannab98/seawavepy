import numpy as np
from seawave import rc,kernel,cuda
from seawave.radar import radar
from seawave.retracking import retracking
from seawave.surface import surface

rc.antenna.z = 30
rc.constants.lightSpeed = 1500 
rc.antenna.impulseDuration = 40e-6
rc.wind.speed = 5


N = 100
P, t = radar.create_multiple_pulses(N, dump=True)
popt = retracking.pulse(t, P)