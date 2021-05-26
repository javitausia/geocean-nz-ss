# time
from datetime import datetime

# arrays and math
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA

# custom
from .config import default_region_reduced, default_region
from .pca import PCA_DynamicPred
from .plotting.cca import plot_ccs


def CCA_Analysis(pres, ss, # pressure and ss datasets
                 pres_vars: tuple = ('msl','longitude','latitude'),
                 ss_vars: tuple = ('ss','lon','lat'),
                 time_resample: str = '1D',
                 region: tuple = (True,default_region,default_region_reduced),
                 percentage_PCs: float = 0.96,
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
            ss_vars[1]:slice(region[2][0],region[2][1]),
            ss_vars[2]:slice(region[2][2],region[2][3])
        })
        # TODO: check order in lat/lon coords!!

    # check if data is resampled and dropna
    pres = pres.resample(time=time_resample).mean().dropna(dim='time')
    if False: # check time loss
        ss = ss.resample(time=time_resample).quantile(0.95).dropna(dim='time',how='all')
    else:
        ss = ss.dropna(dim='time',how='all')

    # extract PCs from SLP and SS
    pcs_pres, pres_scaler = PCA_DynamicPred(
        pres,time_resample=time_resample,
        time_lapse=1,region=(False,None),
        pca_plot=(True,False),
        pca_ttls=['SLP in t','SLP in t-1']
    )
    pcs_ss, ss_scaler = PCA_DynamicPred(
        ss,pres_vars=ss_vars,time_resample=time_resample,
        time_lapse=1,region=(False,None),
        pca_plot=(True,False),
        pca_ttls=['SS in t','SS in t-1']
    )
    
    # num of pcs to use
    num_pcs_pres = int(
        np.where(np.cumsum(pcs_pres.variance.values/\
            np.sum(pcs_pres.variance.values))>percentage_PCs
            )[0][0]
    ) + 4
    num_pcs_ss = int(
        np.where(np.cumsum(pcs_ss.variance.values/\
            np.sum(pcs_ss.variance.values))>percentage_PCs
            )[0][0]
    ) + 4
    print('\n we will use {} PCs from the pressure and {} from the ss \n'.format(
        num_pcs_pres, num_pcs_ss
    ))

    # input same time frames
    common_times = np.intersect1d(
        pd.to_datetime(pcs_pres.time.values).round('H'),
        pd.to_datetime(pcs_ss.time.values).round('H'),
        return_indices=True
    )
    pres_data = pcs_pres.isel(time=common_times[1]).PCs.values[:,:num_pcs_pres]
    ss_data = pcs_ss.isel(time=common_times[2]).PCs.values[:,:num_pcs_ss]

    # calculate the CCAs
    cca = CCA(n_components=min(
        pres_data.shape[0],pres_data.shape[1],ss_data.shape[1]
        )
    )
    cca.fit(pres_data, ss_data) # fit the object
    pres_cca, ss_cca = cca.transform(pres_data, ss_data)

    # check model results    
    print('\n R score: {} -- in TEST data'.format(
        round(cca.score(pres_data,ss_data),2)
    ))

    # lons/lats for pres and ss
    pres_lons, pres_lats = pres[pres_vars[1]].values, pres[pres_vars[2]].values
    ss_lons, ss_lats = ss[ss_vars[1]].values, ss[ss_vars[2]].values

    # return data
    CCA_return = xr.Dataset(
        data_vars = {
            'x_scores': (('time','n_components'), pres_cca),
            'y_scores': (('time','n_components'), ss_cca),
            'x_loadings': (('n_features','n_components'), cca.x_loadings_),
            'y_loadings': (('n_targets','n_components'), cca.y_loadings_),
            'x_rotations': (('n_features','n_components'), cca.x_rotations_),
            'y_rotations': (('n_targets','n_components'), cca.y_rotations_),
            'coefs': (('n_features','n_targets'), cca.coef_)
        },
        coords={
            'time': common_times[0]
        }
    )

    # if plot: plot
    region_plot = region[1] if region[0] else default_region
    if cca_plot:
        plot_ccs(CCA_return,(pcs_pres,pcs_ss),(pres_scaler,ss_scaler),
                 n_plot=3,region=region_plot)

    return CCA_return, (pcs_pres,pcs_ss), (pres_scaler,ss_scaler)

