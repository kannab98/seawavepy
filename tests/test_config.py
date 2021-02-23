import sys
sys.path.append(".")
import unittest

import seawave
from seawave import rcParams

seawave.rc = rcParams('/home/ponur/modeling-git/rc.json')
# class TestrcParams(unittest.TestCase):
#     def setUp(self):
#         self.rc = rc


# if __name__=="__main__":
#     unittest.main()