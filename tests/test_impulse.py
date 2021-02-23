import sys

sys.path.append(".")
import unittest

import numpy as np
from seawave import rc
from seawave.radar import radar
from seawave.retracking import retracking
from seawave.surface import surface

rc.antenna.z = 30
rc.constants.lightSpeed = 1500 
rc.antenna.impulseDuration = 40e-6

class TestPulseModeling(unittest.TestCase):
    def setUp(self):
        self.retracking = retracking
        self.radar = radar
        self.t0 = rc.antenna.z/rc.constants.lightSpeed
        self.tau = rc.antenna.impulseDuration
        self.t = np.linspace(0.019, 0.0235, 512)
        self.P = np.zeros_like(self.t) 

    def test_modeling(self):
        
        N = 1
        t = self.t
        P = radar.create_multiple_pulses(t, N, dump=True)
        popt = retracking.pulse(2*t, P)

if __name__=="__main__":
    unittest.main()

