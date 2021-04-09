import sys, os
sys.path.append(".")
import unittest

from seawave.retracking import retracking
from seawave import config 




config['Dataset']['RetrackingFileName'] = 'kek.xlsx'
df0, df = retracking.from_file('impulses/.*.txt', config)
print(df)