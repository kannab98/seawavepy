import sys, os
sys.path.append(".")
import unittest

from seawave.retracking import retracking


df0, df = retracking.from_file('impulses/.*.txt')
print(df)