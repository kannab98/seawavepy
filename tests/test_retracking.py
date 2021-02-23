import sys
sys.path.append(".")
import unittest

from seawave.retracking import retracking

class TestRetracking(unittest.TestCase):
    def setUp(self):
        self.retracking = retracking


if __name__=="__main__":
    unittest.main()