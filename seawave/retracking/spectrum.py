import numpy as np
import re
import os
import pandas as pd

from scipy.optimize import curve_fit
from scipy.special import erf


from  .. import config

import logging
logger = logging.getLogger(__name__)



def to_xlsx(files):
    df = []
    for i in range(len(files)):

        df.append(pd.read_csv(files[i], sep="\s+", comment="#"))
    


    df0 = pd.DataFrame({
                            'file': files, "varelev": np.zeros(len(files)), 
                            "varslopes": np.zeros(len(files)), 
    })
    for i, f in enumerate(files):
        f = df[i].iloc[:, 0].values
        s = df[i].iloc[:, 1].values
        df0.iloc[i, 1] = np.trapz(s, f)
        df0.iloc[i, 2] = np.trapz(s*f**2, f)


    excel_name = "%s_%s.xlsx" % (config["Dataset"]["RetrackingFileName"], 'spectrum')
    df0.to_excel(excel_name)







    
    

