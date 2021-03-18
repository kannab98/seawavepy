import sys
sys.path.append(".")
import unittest
from seawave.spectrum import spectrum
from seawave import rc

rc.wind.speed = 5
k = spectrum.k
spectrum(k)
# seawave.rc = rcParams('/home/ponur/modeling-git/rc.json')
# class TestrcParams(unittest.TestCase):
#     def setUp(self):
#         self.rc = rc


# if __name__=="__main__":
#     unittest.main()