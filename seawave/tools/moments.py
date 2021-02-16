import os
from os import path
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

df = np.zeros(2, dtype=object)

with open(path.join("data","TheoreticalMoments.xlsx"), "rb") as f:
    df[0] = pd.read_excel(f, header=[0,1,2])

with open(path.join("data","ModelMoments.xlsx"), "rb") as f:
    df[1] = pd.read_excel(f, header=[0,1,2])


fig, ax = plt.subplots()

ax.plot(df[0]["U"], df[0]["theory", "Ku", "var"])
ax.plot(df[1]["U"], df[1]["model", "Ku", "var"])

fig, ax = plt.subplots()

ax.plot(df[0]["U"], df[0]["theory", "Ku", "xx"])
ax.plot(df[1]["U"], df[1]["model", "Ku", "xx"])

fig, ax = plt.subplots()
ax.plot(df[0]["U"], df[0]["theory", "Ku", "yy"])
ax.plot(df[1]["U"], df[1]["model", "Ku", "yy"])

fig, ax = plt.subplots()
ax.plot(df[0]["U"], df[0]["theory", "Ku", "xx+yy"])
ax.plot(df[1]["U"], df[1]["model", "Ku", "xx+yy"])


plt.show()


