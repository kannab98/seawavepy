import numpy as np
import xarray as xr

from .. import config

datasets = []


def slopes(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray, ):

    arr = np.zeros((3, t.size, x.size, y.size))
    arr[-1] = 1

    X, Y, _ = np.meshgrid(x, y, t)
    X = np.transpose(X, axes=(2,1,0))
    Y = np.transpose(Y, axes=(2,1,0))

    da = xr.DataArray(
        data=arr,
        dims=["proj", "time", "x", "y"],
        coords=dict(
            proj=["x", "y", "z"],
            X=(["time", "x", "y"], X),
            Y=(["time", "x", "y"], Y),
            Z=(["time", "x", "y"], z),
            time=t,
        ),
        attrs=dict(
            description="Array with surface slopes data.",
            units="non-dimensional, (dz/dx, dz/dy, 1)"
        ),
    )
    return da

def velocities(x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray, ):

    arr = np.zeros((3, t.size, x.size, y.size))
    X, Y, _ = np.meshgrid(x, y, t)
    X = np.transpose(X, axes=(2,1,0))
    Y = np.transpose(Y, axes=(2,1,0))

    da = xr.DataArray(
        data=arr,
        dims=["proj", "time", "x", "y"],
        coords=dict(
            proj=["x", "y", "z"],
            X=(["time", "x", "y"], X),
            Y=(["time", "x", "y"], Y),
            Z=(["time", "x", "y"], z),
            time=t,
        ),
        attrs=dict(
            description="Array with surface slopes data.",
            units="m/s"

        ),
    )
    return da

def harmonics(N: int, M: int):
    da = xr.DataArray(data=np.zeros((N, M), dtype=np.float32), 
                        dims=["k", "phi"], 
                        coords=dict(
                        k = np.zeros(N),
                        phi = np.zeros(M),
        ))
    return da

def phases(N: int, M: int):
    da = xr.DataArray(data=np.zeros((N, M), dtype=np.float32), 
                        dims=["k", "phi"], 
                        coords=dict(
                        k = np.zeros(N),
                        phi = np.zeros(M),
        ))
    return da


def elevations(x: np.ndarray, y: np.ndarray, t: np.ndarray):

    X, Y, _ = np.meshgrid(x, y, t)
    k = np.array([[]],dtype=object)
    X = np.transpose(X, axes=(2,1,0))
    Y = np.transpose(Y, axes=(2,1,0))


    arr = np.zeros(X.shape)

    da = xr.DataArray(
        data=arr,
        dims=["time", "x", "y"],
        coords=dict(
            X=(["time", "x", "y"], X),
            Y=(["time", "x", "y"], Y),
            x = x, 
            y = y,
            time=t,

        ),
        attrs=dict(
            description="Array with surface elevations data.",
            units="m"

        ),
    )
    return da


def spectrum(k: np.ndarray, A: np.ndarray):
    da = xr.DataArray(
        data=[np.abs(A), np.angle(A)],
        dims=["type", "k", "phi"],
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

def float_surface(x: np.ndarray, y: np.ndarray, t: np.ndarray, ):
    z = elevations(x, y, t)
    n = slopes(x, y, z.values, t)
    v = velocities(x, y, z.values, t)

    ds = xr.Dataset( {'elevations': z, 'slopes': n, 'velocities':v, 'spectrum': xr.DataArray()} )
    return ds

def restore_z(srf):
    P = srf['pulse'].values
    t = srf['time_relative'].values
    timp = config['Radar']['ImpulseDuration']
    c = config['Constants']['WaveSpeed']

    imax = np.argmax(P, axis=0)
    z = (t[imax] - timp/4) * c
    srf['z_restored'] = xr.DataArray(
        data=z,
        dims = ["time"],
        coords = dict(time = srf['time'].values),
    )
    return z

def rough_surface(host_constants, *args):
    ds = float_surface(*args)
    ds['spectrum'] = spectrum(*host_constants)
    return ds

def radar(srf: xr.Dataset):
    srf.coords["X"] += config['Radar']['Position'][0]
    srf.coords["Y"] += config['Radar']['Position'][1]
    srf.coords["Z"] = srf['elevations'] + config['Radar']['Position'][2]


    for coord in ['X', 'Y']:
        if len(srf[coord].shape) != len(srf['Z'].shape):
            srf[coord] = srf[coord].expand_dims("time")
            srf[coord] = np.repeat(srf[coord], srf['Z'].shape[0], axis=0)

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

def pulse(P: np.ndarray, t: np.ndarray, time_relative: np.ndarray):
    da = xr.DataArray(
        data = P,
        dims = ["time_relative", "time"],
        coords = dict(
                time_relative = time_relative,
                time = t,
        ),
        attrs=dict(
            description="Мощность отраженного импульса"
        )
    )
    return da

def statistics(srf: xr.Dataset) -> xr.Dataset:

    varelev = xr.DataArray(
        data   = np.var(srf['elevations'].values, axis=(-1, -2)),
        dims   = ["time"],
        coords = dict(
            time = srf['time'].values
        ),
        attrs=dict(
            description="Variance of elevations",
            units = "m^2",
        )
    )

    meanelev = xr.DataArray(
        data   =  np.mean(srf['elevations'].values, axis=(-1, -2)),
        dims   = ["time"],
        coords = dict(
            time = srf['time'].values
        ),
        attrs=dict(
            description="Mean of elevations",
            units = "m",)
    )


    print(srf.slopes.values.shape)
    varslopes = xr.DataArray(
        data   = np.var(srf['slopes'].values, axis=(-1, -2)),
        dims   = ["proj", "time"],
        coords = dict(
            time = srf['time'].values
        ),
        attrs=dict(
            description="Variance of slopes",
            units = "a.u.",
        )
    )

    srf["VoE"] = varelev
    srf["MoE"] = meanelev
    srf["VoS"] = varslopes
