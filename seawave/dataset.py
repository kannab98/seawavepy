from . import rc
import numpy as np
import xarray as xr




def float_surface(x: np.ndarray, y: np.ndarray, t: np.ndarray, ) -> xr.Dataset:
    z = elevations(x, y, t)
    n = slopes(x, y, z.values, t)
    v = velocities(x, y, z.values, t)

    ds = xr.Dataset( {'elevations': z, 'slopes': n, 'velocities':v} )
    return ds

def slopes(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray, ):

    arr = np.zeros((3, x.size, y.size, t.size))
    arr[-1] = 1
    X, Y, T = np.meshgrid(x, y, t)

    da = xr.DataArray(
        data=arr,
        dims=["proj", "x", "y", "time"],
        coords=dict(
            proj=["x", "y", "z"],
            X=(["x", "y", "time"], X),
            Y=(["x", "y", "time"], Y),
            Z=(["x", "y", "time"], z),
            time=t,
        ),
        attrs=dict(
            description="Array with surface slopes data.",
            units="non-dimensional, (dz/dx, dz/dy, 1)"
        ),
    )
    return da

def velocities(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray, ):

    arr = np.zeros((3, x.size, y.size, t.size))
    X, Y, T = np.meshgrid(x, y, t)

    da = xr.DataArray(
        data=arr,
        dims=["proj", "x", "y", "time"],
        coords=dict(
            proj=["x", "y", "z"],
            X=(["x", "y", "time"], X),
            Y=(["x", "y", "time"], Y),
            Z=(["x", "y", "time"], z),
            time=t,
        ),
        attrs=dict(
            description="Array with surface slopes data.",
            units="m/s"

        ),
    )
    return da

def elevations(x: np.ndarray, y: np.ndarray, t: np.ndarray):
    arr = np.zeros((x.size, y.size, t.size))
    X, Y, T = np.meshgrid(x, y, t)

    da = xr.DataArray(
        data=arr,
        dims=["x", "y", "time"],
        coords=dict(
            X=(["x", "y", "time"], X),
            Y=(["x", "y", "time"], Y),
            time=t,
        ),
        attrs=dict(
            description="Array with surface elevations data.",
            units="m"

        ),
    )
    return da

def amplitudes(k: np.ndarray, A: np.ndarray):
    da = xr.DataArray(
        data=A,
        dims=["k", "phi"],
        coords=dict(
            waveNumber = (["k", "phi"], np.abs(k)),
            azimuth = (["k", "phi"], np.angle(k))
        ),
        attrs=dict( 
            description=""" Комплексный волновой вектор -- форма записи x и y компонент. 
            В комплексной амплитуде A учитываются текущие случайные фазы 
            """
        )
    )
    return da





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
    AoD.values = srf['Z']/distance.values
    AoD.attrs=dict( description = "Angle of departure")


    d = distance.values
    n = srf.slopes.values/np.linalg.norm(srf.slopes.values, axis=0)
    n.attrs = dict(
        description="normal to surface"
    )

    AoA = srf.elevations.copy()
    AoA.values = np.arccos(np.einsum('ijkm, ijkm -> jkm', r/d, n/np.linalg.norm(n, axis=0)))
    AoA.attrs=dict(
            description="Angle of arrival",
        )


    # gain = srf.elevations.copy()
    # gain.attrs=dict( description="Gain", )

    mask = srf.elevations.copy()
    mask.values = AoA < np.deg2rad(1)
    mask.attrs=dict( description="Mirror points flag", )


    srf['distance'] = distance
    srf['normal'] = n
    srf['AoD'] = AoD
    srf['AoA'] = AoD
    srf['mask'] = mask


    return srf
