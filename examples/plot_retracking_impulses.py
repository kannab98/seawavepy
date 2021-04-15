
import sys
sys.path.append(".")
import pandas as pd
import matplotlib.pyplot as plt
from seawave.retracking import retracking

df = pd.read_excel("impulses/example-impulse.xlsx")

t0 = df['t'].values
P0 = df['P'].values
popt = retracking.pulse(t0, P0)
P = retracking.ice(t0, *popt)

plt.figure()
plt.plot(t0, P0)

plt.plot(t0, P)

df = pd.DataFrame({'t': t0, 'P_raw': P0, 'P_brown': P})
df.to_excel("impulse_brown.xlsx")