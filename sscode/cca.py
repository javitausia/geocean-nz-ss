# time
from datetime import datetime

# arrays and math
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

# custom
from .config import default_region_reduced
from .plotting.cca import plot_ccs


def CCA_Analysis(pres, pres_vars: tuple = ('msl','longitude','latitude'),
                 ss, ss_vars: tuple = ('dac','longitude','latitude'),
                 time_resample: str = '1D',
                 region: tuple = (True,default_region_reduced,default_region_reduced),
                 cca_plot: bool = True):

    # TODO: add docstring

    # perform cca analysis
    print('\n lets calculate the CCs... \n')
    # crop slp to the region selected
    if region[0]:
        pres = pres.sel({
            pres_vars[1]:slice(region[1][0],region[1][1]),
            pres_vars[2]:slice(region[1][2],region[1][3])
        })
        ss = ss.sel({
            pres_vars[1]:slice(region[2][0],region[2][1]),
            pres_vars[2]:slice(region[2][3],region[2][2])
        })
        # TODO: check order in lat/lon coords!!

    # check if data is resampled and dropna
    pres = pres.resample(time=time_resample).mean().dropna(dim='time')
    ss = ss.resample(time=time_resample).mean().dropna(dim='time')

    # input same time frames
    common_times = np.intersect1d(
        pd.to_datetime(pres.time.values).round('H'),
        pd.to_datetime(ss.time.values).round('H'),
        return_indices=True
    )
    pres = pres.isel(time=common_times[1])
    ss = ss.isel(time=common_times[2])

    # get data ready to CCA
    pres_data = pres.values.reshape(len(pres.time.values),-1)
    stan_scal = StandardScaler() # standarize SLP data
    pres_stan = stan_scal.fit_transform(pres_data)
    pres_stan[np.isnan(pres_stan)] = 0.0 # check additional nans
    ss_data = pres.values.reshape(len(ss.time.values),-1)

    # calculate the CCAs
    cca = CCA(n_components=min(pres_stan.shape[0],pres_stan.shape[1]))
    cca.fit(pres_stan, ss_stan) # fit the object
    pres_c, ss_c = cca.transform(pres_stan, ss_stan)

    # lons/lats for pres and ss
    pres_lons, pres_lats = pres[pres_vars[1]].values, pres[pres_vars[2]].values
    ss_lons, ss_lats = ss[ss_vars[1]].values, ss[ss_vars[2]].values

    # return data
    CCA_return = xr.Dataset(
        data_vars = {
            'x_loadings': (('pres_longitude','pres_latitude','n_components'),
                cca.x_loadings_.reshape(len(pres_lons),len(pres_lats),-1)),
            'y_loadings': (('ss_longitude','ss_latitude','n_components'), 
                cca.y_loadings_.reshape(len(ss_lons),len(ss_lats)),-1),
            'x_rotations': (('pres_longitude','pres_latitude','n_components'), 
                cca.x_rotations_.reshape(len(pres_lons),len(pres_lats)),-1),
            'y_rotations': (('ss_longitude','ss_latitude','n_components'), 
                cca.y_rotations_.reshape(len(ss_lons),len(ss_lats)),-1),
            'coefs': (('n_features','n_targets'), cca.coef_)
        },
        coords = {
            'pres_longitude': pres_lons, 'pres_latitude': pres_lats,
            'ss_longitude': ss_lons, 'pres_latitude': ss_lats
        }
    )

    # TODO: plot the data

    return cca, CCA_return

