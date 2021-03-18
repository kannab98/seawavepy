import numpy as np
import os.path
import json
import pandas as pd
import xarray as xr
import logging
from . import cuda
from . import rc
from .spectrum import spectrum
from .surface import surface

import datetime
import math

logger = logging.getLogger(__name__)



try:
    from numba import cuda, float32
    from multiprocessing import Process, Array
    GPU = True
except:
    print("CUDA not installed")
    GPU = False 

band = np.array(["C", "X", "Ku", "Ka"])

TPB=16


    #         logger.debug("Use regular meshgrid")

    #     else:
    #         logger.warning("Can't complete regularization. " + 
    #                         "Use irregular meshgrid with epsabs=%.6f" % epsabs)
    #     # print(X)
    #     # print(X0)

    # X0 = cuda.to_device(X0)
    # Y0 = cuda.to_device(Y0)
    # kernel[blockspergrid, threadsperblock](arr, X0, Y0, t, *cuda_constants)




    # return arr, X, Y

def simple_launch(kernel):
    logger.debug("Use %s kernel" % kernel.__dict__['py_func'].__name__)

    model_coeffs = surface.export()

    X, Y = surface.meshgrid
    X = X.flatten()
    Y = Y.flatten()
    arr = np.zeros((6, X.size))

    t = surface.time
    arr, X0, Y0 = init(kernel, arr, X, Y, t, model_coeffs)
    X0 = np.array([X0])
    Y0 = np.array([Y0])

    size = rc.surface.gridSize

    # var0 = spectrum.quad(0, 0)
    # var0_s = spectrum.quad(2, 0)

    # var = np.var(arr[0])
    # var_s = np.var(arr[1]+arr[2])

    # logger.info('Practice variance of heights sigma^2_h=%.6f' % var)
    # logger.info('Practice variance of heights sigma^2=%.6f' % var_s)

    # if 0.5*var0 <= var <= 1.5*var0:
    #     logger.info("Practice variance of heights sigma^2_h matches with theory")
    # else:
    #     logger.warning("Practice variance of heights sigma^2_h not matches with theory")

    # if 0.5*var0_s <= var_s <= 1.5*var0_s:
    #     logger.info("Practice variance of full slopes sigma^2 matches with theory")
    # else:
    #     logger.warning("Practice variance of full slopes sigma^2 not matches with theory")


    # labels = ['z', 'sx', 'sy', 'vz', 'vx', 'vy']
    labels = ['elevations', 'slopes x', 'slopes y', 'velocity z', 'velocity x', 'velocity y']

    ds = xr.Dataset(
        {label: ( ["time", "x", "y"], arr[i]) for i, label in enumerate(labels)}, 
        coords={
            "lon": (["x", "y"], x),
            "lat": (["x", "y"], y),
            "time": t,
        },
    )

    return df


# def cache_file(kernel):
#     srf = rc.surface
#     wind = rc.wind
#     return "%s_%s_%s_%s_%s_%s_%s" % (kernel.__name__, srf.band, srf.gridSize, wind.speed, wind.direction)





def __kernel_per_band__(kernel, X, Y, model_coeffs):
    logger.info("Use %s kernel" % kernel.__dict__['py_func'].__name__)

    k = np.abs(model_coeffs[0][:, 0])
    N = spectrum.KT.size - 1

    process = [ None for i in range(N)]
    arr = [ None for i in range(N)]


    edge = spectrum.KT
    edge = [ np.max(np.where(k <= edge[i] )) for i in range(1, edge.size)]
    edge = [0] + edge

    def __multiple_kernels_(kernel, X, Y, model_coeffs):

        for j in range(N):
            # Срез массивов коэффициентов по оси K, условие !=1 для phi (он не зависит от band)
            host_constants = tuple([ model_coeffs[i][edge[j]:edge[j+1]]  for i in range(len(model_coeffs))])
            # Create shared array
            arr_share = Array('d', 6*X.size )
            # arr_share and arr share the same memory
            arr[j] = np.frombuffer( arr_share.get_obj() ).reshape((6, X.size)) 


            X0 = X.flatten()
            Y0 = Y.flatten()
            process[j] = Process(target = init, args = (kernel, arr[j], X0, Y0, host_constants) )
            process[j].start()

        for j in range(N):
            process[j].join()

        return arr
            
    arr = __multiple_kernels_(kernel, X, Y, model_coeffs)
        
    return arr


def launch(kernel):


    model_coeffs = surface.export()

    X, Y = surface.meshgrid
    t = surface.time
    arr = __kernel_per_band__(kernel, X, Y, model_coeffs)

    size = (rc.surface.gridSize, rc.surface.gridSize)

    return arr


def convert_to_band(arr, band):
    band0 = ["C", "X", "Ku", "Ka"]

    if isinstance(band, str):
        for i, item in enumerate(band0): 
            if band in item:
                ind = i
                break

    elif isinstance(band, int):
        ind = band

    srf = np.sum(arr[:ind], axis=0)
    labels = ['z', 'sx', 'sy', 'vz', 'vx', 'vy']

    df = pd.DataFrame({label: srf[i] for i, label in enumerate(labels)})

    return df



def reshape_meshgrid(X0, Y0):
    X0 = np.array(X0)
    Y0 = np.array(Y0)
    X0 = X0.reshape((3, rc.surface.gridSize, rc.surface.gridSize))
    Y0 = Y0.reshape((3, rc.surface.gridSize, rc.surface.gridSize))
    return X0, Y0
    
def dump(arr0, band0):

    arr = convert_to_band(arr0, band0) 

    X, Y = surface.meshgrid
    X = X.flatten()
    Y = Y.flatten()
    data = np.array([arr, X, Y], dtype=object)

    name = ["surface", "rc"]
    ext = [".pkl", ".json"]


    field = ["heights", "sigmaxx", "sigmayy"]

    iterables = [band, field]

    columns = pd.MultiIndex.from_arrays([["x"],[""],[""]])
    df = pd.DataFrame({"x": X, "y": Y} )
    columns = pd.MultiIndex.from_product(iterables) 
    df0 = pd.DataFrame(columns=columns, index=df.index)
    df = pd.concat([df, df0], axis=1)

    # label = ["$Z$, м", "$arctan(\\frac {\\partial Z}{\\partial X}), град$", "$arctan(\\frac {\\partial Z}{\\partial Y}), град$"]

    # suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    # file = [''.join(name[i]+suffix+ext[i]) for i in range(len(name)) ]

    for i, b in enumerate(band):
        for j, m in enumerate(field):
            df[(b, m)] = arr[i][j].flatten()
    
    # df.to_pickle(file[0])

    # rc.surface.varPreCalc = np.sum((surface.ampld(surface.k, surface.phi)**2/2))
    # rc.surface.varReal = np.var(arr[-1][0])
    # rc.surface.varTheory = spectrum.dblquad(0,0,0)

    # rc.surface.covReal = np.cov(arr[-1][1],arr[-1][2]).tolist()
    # rc.surface.covTheory = spectrum.cov().tolist()
    # rc.surface.kEdge = spectrum.KT.tolist()

    # with open(file[1], 'w') as f:
        # json.dump(dt, f, indent=4)
    #     dt = {}
    #     for Key, Value in vars(rc).items():
    #         if type(Value) ==  type(rc.surface):
    #             dt.update({Key: {}})
    #             for key, value in Value.__dict__.items():
    #                 if key[0] != "_":
    #                     dt[Key].update({key: value})
    return df




