import numpy as np
import xarray as xr
import atexit
from .. import rc

datasets = []

def radar(srf: xr.Dataset):
    # angle of arrival
    # angle of departure
    labels = ['distance', 'AoD', 'AoA', 'mask']
    srf.coords["X"] += rc.antenna.x
    srf.coords["Y"] += rc.antenna.y
    srf.coords["Z"] = srf['elevations'] + rc.antenna.z


    r = np.array([srf['X'], srf['Y'], srf['Z']])

    distance = srf.elevations.copy()
    distance.values = np.linalg.norm(r, axis=0)
    distance.attrs = dict(
            description="Array with surface fields data.",
    )

    AoD = srf.elevations.copy()
    AoD.values = np.arccos(srf['Z']/distance.values)
    AoD.attrs=dict( description = "Angle of departure")


    d = distance.values
    n = srf.slopes.values/np.linalg.norm(srf.slopes.values, axis=0)
    AoA = srf.elevations.copy()
    AoA.values = np.arccos(np.einsum('ijkm, ijkm -> jkm', r/d, n/np.linalg.norm(n, axis=0)))
    AoA.attrs=dict(
            description="Angle of arrival",
        )


    # gain = srf.elevations.copy()
    # gain.attrs=dict( description="Gain", )

    mask = srf.elevations.copy()
    mask.values = np.abs( np.cos(AoA) ) > np.cos( np.deg2rad(1))  
    mask.attrs=dict( description="Mirror points flag", )


    srf['distance'] = distance
    srf['AoD'] = AoD
    srf['AoA'] = AoA
    srf['mask'] = mask


    return srf