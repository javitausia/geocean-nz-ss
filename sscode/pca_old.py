# time
from datetime import datetime

# arrays and math
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# distributed pca calculations
from datetime import datetime
import dask.array as dask_array
from dask_ml.decomposition import PCA as dask_ml_PCA

# custom
from .config import default_region_reduced, default_region
from .utils import spatial_gradient
from .plotting.pca import plot_pcs


def PCA_DynamicPred(pres, pres_vars: tuple = ('SLP','longitude','latitude'),
                    calculate_gradient: bool = False,
                    winds: tuple = (False,None),
                    wind_vars: tuple = ('wind_proj_mask','lon','lat'),
                    time_lapse: int = 1, # 1 equals to NO time delay 
                    time_resample: str = '1D',
                    region: tuple = (True,default_region),
                    pca_plot: tuple = (True,False,2),
                    verbose: bool = True,
                    pca_ttls = None, pca_borders = None):
    """
    Perform PCA decomposition given the pressure, winds...

    Args:
        pres (xarray.Dataarray/Dataset): These are the sea-level-pressure fields,
            that could be loaded  both with load_era5() or load_cfsr(), and
            obviously with the Loader object!!
                ** this parameter can be used to calculate PCA 
                   with different data 
                **
        pres_vars (tuple, optional): These are the slp xarray coordinates. 
            - Defaults to ('SLP','longitude','latitude').
        calculate_gradient (bool, optional): weather to calculate or not the slp gradient.
            - Defaults to False.
        winds (tuple, optional): This is the wind data, if available...
            - Defaults to (False,None), data must be xarray!!
        wind_vars (tuple, optional): These are the wind coordinates. 
            - Defaults to ('wind_proj','longitude','latitude').
        time_lapse (int, optional): Time delay for the dynamic predictor. 
            - Defaults to 2.
        time_resample (str, optional): This is the time resample.
            - Defaults to '1D'.
        region (tuple, optional): Region to crop the slp to. 
            - Defaults to (True,default_region).
        pca_plot (tuple, bool): Wether to plot or not the final results.
            - Defaults to (True,False,1), as the second boolean indicator refers
            to the de-standarization of the EOFs, which is not totally validated
            ** pca_plot[2] == n_plot
        verbose (bool, optional): Wether or not to debug the actions.
            - Defaults to True.
        pca_ttls: These are the titles for the pca plots if wanted.
        pca_borders: Limits to use in the pca plots.

    Returns:
        [xarray.Dataset]: PCA decomposition final results
    """

    # perform pca analysis
    print('\n lets calculate the PCs... \n') if verbose else None
    # crop slp and winds to the region selected
    if region[0]:
        pres = pres.sel({
            pres_vars[1]:slice(region[1][0],region[1][1]),
            pres_vars[2]:slice(region[1][2],region[1][3])
        })
        if winds[0]:
            print('\n adding the wind to the predictor... \n') if verbose else None
            wind = winds[1].sel({
                wind_vars[1]:slice(region[1][0],region[1][1]),
                wind_vars[2]:slice(region[1][2],region[1][3])
            }) # TODO: check lat order when cropping

    # check if data is resampled and dropna
    if pres_vars[0]=='wind_proj' or pres_vars[0]=='wind_proj_mask': # when just winds are loaded
        pres = pres.resample(time=time_resample).mean().fillna(0.0)
    else:
        pres = pres.resample(time=time_resample).mean().dropna(dim='time',how='all')
    if winds[0]:
        print(wind)
        wind = wind[wind_vars[0]].resample(time=time_resample).mean().fillna(0.0)\
            .interp(coords={wind_vars[1]:pres[pres_vars[1]],
                            wind_vars[2]:pres[pres_vars[2]]}
                    )\
            .sel(time=pres.time) # interp to pressure coords
        print('\n winds predictor with shape: \n {} \n'.format(wind.shape)) if verbose else None
        wind_add = 1 # for the pcs matrix
    else:
        wind_add = 0

    # calculate the gradient
    if calculate_gradient:
        print('\n calculating the gradient of the sea-level-pressure fields... \n') if verbose else None
        pres = spatial_gradient(pres,pres_vars[0]) # from utils.py
        print('\n pressure/gradient predictor both with shape: \n {} \n'\
            .format(pres[pres_vars[0]].shape)) if verbose else None
        grad_add = 2
    else:
        grad_add = 1

    # lets now create the PCs matrix
    x_shape = len(pres.time.values)-time_lapse
    y_shape = len(pres[pres_vars[1]].values)*len(pres[pres_vars[2]].values)
    pcs_matrix = np.zeros((x_shape,(time_lapse*(grad_add+wind_add))*y_shape))
    # fill the pcs_matrix array with data
    for t in range(x_shape):
        for tl in range(0,time_lapse*(grad_add+wind_add),grad_add+wind_add):
            try:
                pcs_matrix[t,y_shape*tl:y_shape*(tl+1)] = \
                    pres.isel(time=t-tl)[pres_vars[0]].values.reshape(-1)
            except:
                pcs_matrix[t,y_shape*tl:y_shape*(tl+1)] = \
                    pres.isel(time=t-tl).values.reshape(-1)
            if calculate_gradient:
                pcs_matrix[t,y_shape*(tl+1):y_shape*(tl+2)] = \
                    pres.isel(time=t-tl)[pres_vars[0]+'_gradient'].values.reshape(-1)
            if wind_add:
                pcs_matrix[t,y_shape*(tl+grad_add):y_shape*(tl+grad_add+1)] = \
                    wind.isel(time=t-tl).values.reshape(-1)

    # standarize the features
    pcs_scaler = StandardScaler()
    pcs_stan = pcs_scaler.fit_transform(pcs_matrix)
    pcs_stan[np.isnan(pcs_stan)] = 0.0 # check additional nans

    # --------------------------------------------------------------------------------- #
    # THIS IS JUST TO ALLOW MY 32GB-RAM COMPUTER TO RUN
    # if pcs_stan.shape[1]>18000:
    #     pcs_stan = pcs_stan[:,::12]
    # elif pcs_stan.shape[1]>10000:
    #     pcs_stan = pcs_stan[:,::8] # if pcs_stan.shape[0]<20000 else pcs_stan[::2,::8]
    # elif pcs_stan.shape[1]>6000:
    #     pcs_stan = pcs_stan[:,::6] # if pcs_stan.shape[0]<20000 else pcs_stan[::2,::6]
    # elif pcs_stan.shape[1]>2000:
    #     pcs_stan = pcs_stan[:,::4] # if pcs_stan.shape[0]<20000 else pcs_stan[::2,::4]
    # --------------------------------------------------------------------------------- #

    # calculate de PCAs
    print('\n calculating PCs matrix with shape: \n {} \n'.format(pcs_stan.shape)) \
        if verbose else None
    if True:
        pca_fit = PCA(n_components=min(pcs_stan.shape[0],pcs_stan.shape[1]))
        PCs = pca_fit.fit_transform(pcs_stan)
    elif False:
        # Incremental version
        from sklearn.decomposition import IncrementalPCA

        # calculate de PCAs (this is non-distributed but uses all CPU's on the machine)
        print("Using incremental PCA algo")
        # batch_size is set to the number of features which is the minimum allowed
        pca_fit = IncrementalPCA(n_components=min(pcs_stan.shape[0],pcs_stan.shape[1]),
                                 batch_size=pcs_stan.shape[0])
        PCs = pca_fit.fit_transform(pcs_stan)

    else: # Distributed version
        # Turn numpy array into dask array
        dask_pcs_stan = dask_array.from_array(pcs_stan,
                                              chunks=(pcs_stan.shape[0],
                                                      pcs_stan.shape[1]//32))

        pca_fit = dask_ml_PCA(n_components=min(pcs_stan.shape[0],pcs_stan.shape[1]))
        pca_fit.fit_transform(dask_pcs_stan)
    
    # return data
    PCA_return = xr.Dataset(
        data_vars = {
            'PCs': (('time','n_components'), PCs),
            'EOFs': (('n_components','n_features'), pca_fit.components_),
            'variance': (('n_components'), pca_fit.explained_variance_),
            'pcs_lon': (('n_lon'), pres[pres_vars[1]].values),
            'pcs_lat': (('n_lat'), pres[pres_vars[2]].values)
        },
        coords = {
            'time': pres.time.values[:-time_lapse]
        }
    )

    # if plot, plot
    region_plot = region[1] if region[0] else default_region
    if pca_plot[0]:
        pca_plot_scale = pcs_scaler if pca_plot[1] else None
        plot_pcs(PCA_return,pcs_scaler=pca_plot_scale,
                 n_plot=pca_plot[2],region=region_plot,
                 pca_ttls=pca_ttls,pca_borders=pca_borders,
                 verbose=verbose)

    # return the PCA xarray.Dataset and the scaler
    return PCA_return, pcs_scaler


# --------------------------------------------------------------------------------- #
# More PCA analysis / tools can be seen below, but are not used in the project yet **
# --------------------------------------------------------------------------------- #


def running_mean(x, N, mode_str='mean'):
    '''
    computes a running mean (also known as moving average)
    on the elements of the vector X. It uses a window of 2*M+1 datapoints

    As always with filtering, the values of Y can be inaccurate at the
    edges. RUNMEAN(..., MODESTR) determines how the edges are treated. MODESTR can be
    one of the following strings:
      'edge'    : X is padded with first and last values along dimension
                  DIM (default)
      'zeros'   : X is padded with zeros
      'ones'    : X is padded with ones
      'mean'    : X is padded with the mean along dimension DIM

    X should not contains NaNs, yielding an all NaN result.
    '''

    # if nan in data, return nan array
    if np.isnan(x).any():
        return np.full(x.shape, np.nan)

    nn = 2*N+1

    if mode_str == 'zeros':
        x = np.insert(x, 0, np.zeros(N))
        x = np.append(x, np.zeros(N))

    elif mode_str == 'ones':
        x = np.insert(x, 0, np.ones(N))
        x = np.append(x, np.ones(N))

    elif mode_str == 'edge':
        x = np.insert(x, 0, np.ones(N)*x[0])
        x = np.append(x, np.ones(N)*x[-1])

    elif mode_str == 'mean':
        x = np.insert(x, 0, np.ones(N)*np.mean(x))
        x = np.append(x, np.ones(N)*np.mean(x))


    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[nn:] - cumsum[:-nn]) / float(nn)


def RunnningMean_Monthly(xds, var_name, window=5):
    '''
    Calculate running average grouped by months

    xds:
        (longitude, latitude, time) variables: var_name

    returns xds with new variable "var_name_runavg"
    '''

    tempdata_runavg = np.empty(xds[var_name].shape)

    for lon in xds.longitude.values:
        for lat in xds.latitude.values:
            for mn in range(1, 13):
                
                # indexes
                ix_lon = np.where(xds.longitude == lon)[0]
                ix_lat = np.where(xds.latitude == lat)[0]
                ix_mnt = np.where(xds['time.month'] == mn)[0]

                # point running average
                time_mnt = xds.time[ix_mnt]
                data_pnt = xds[var_name].isel(
                    time=ix_mnt,longitude=ix_lon,latitude=ix_lat
                )
                
                tempdata_runavg[ix_mnt,ix_lat[0],ix_lon[0]] = running_mean(
                    data_pnt.values, window)

    # store running average
    xds['{0}_runavg'.format(var_name)]= (
        ('longitude', 'latitude', 'time'),
        tempdata_runavg)

    return xds


def PCA_LatitudeAverage(xds, var_name, y1, y2, m1, m2):
    '''
    Principal component analysis
    method: remove monthly running mean and latitude average

    xds:
        (longitude, latitude, time), pred_name | pred_name_runavg

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # calculate monthly running mean
    xds = RunnningMean_Monthly(xds, var_name)

    # predictor variable and variable_runnavg from dataset
    var_val = xds[var_name]
    var_val_ra = xds['{0}_runavg'.format(var_name)]

    # use datetime for indexing
    dt1 = datetime(y1, m1, 1)
    dt2 = datetime(y2+1, m2, 28)
    time_PCA = [datetime(y, m1, 1) for y in range(y1, y2+1)]

    # use data inside timeframe
    data_ss = var_val.loc[:,:,dt1:dt2]
    data_ss_ra = var_val_ra.loc[:,:,dt1:dt2]

    # anomalies: remove the monthly running mean
    data_anom = data_ss - data_ss_ra

    # average across all latitudes
    data_avg_lat = data_anom.mean(dim='latitude')

    # collapse 12 months of data to a single vector
    nlon = data_avg_lat.longitude.shape[0]
    ntime = data_avg_lat.time.shape[0]
    hovmoller = xr.DataArray(
        np.reshape(data_avg_lat.values, (12*nlon, ntime//12), order='F')
    )
    hovmoller = hovmoller.transpose()

    # mean and standard deviation
    var_anom_mean = hovmoller.mean(axis=0)
    var_anom_std = hovmoller.std(axis=0)

    # remove means and normalize by the standard deviation at anomaly
    # rows = time, columns = longitude
    nk_m = np.kron(np.ones((y2-y1+1,1)), var_anom_mean)
    nk_s = np.kron(np.ones((y2-y1+1,1)), var_anom_std)
    var_anom_demean = (hovmoller - nk_m) / nk_s

    # sklearn principal components analysis
    ipca = PCA(n_components=var_anom_demean.shape[0])
    PCs = ipca.fit_transform(var_anom_demean)

    pred_lon = xds.longitude.values[:]

    return xr.Dataset(
        {
            'PCs': (('n_components', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),

            'var_anom_std': (('n_features',), var_anom_std),
            'var_anom_mean': (('n_features',), var_anom_mean),

            'time': (('n_components'), time_PCA),
            'pred_lon': (('n_lon',), pred_lon),
        },

        # store PCA algorithm metadata
        attrs = {
            'method': 'anomalies, latitude averaged',
        }
    )


def PCA_EstelaPred(xds, pred_name):
    '''
    Principal component analysis
    method: custom for estela predictor

    xds:
        (time, latitude, longitude), pred_name_comp | pred_name_gradient_comp

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # estela predictor and estela gradient predictor
    pred_est_var = xds['{0}_comp'.format(pred_name)]
    pred_est_grad = xds['{0}_gradient_comp'.format(pred_name)]

    # use data inside timeframe
    dp_var = pred_est_var.values
    dp_grd = pred_est_grad.values
    shape_grid = dp_var[0].shape  # needed to handle data after PCs

    # unravel and join var and grad data 
    dp_ur = np.nan * np.ones(
        (dp_var.shape[0], 2*dp_var.shape[1]*dp_var.shape[2])
    )

    # we use .T to equal matlab
    for ti in range(dp_ur.shape[0]):
        dp_ur[ti,:] = np.concatenate(
            [np.ravel(dp_var[ti].T) , np.ravel(dp_grd[ti].T)]
        )

    # remove nans from predictor    
    data_pos = ~np.isnan(dp_ur[0,:])
    clean_row = dp_ur[0, data_pos]
    dp_ur_nonan = np.nan * np.ones(
        (dp_ur.shape[0], len(clean_row))
    )
    for ti in range(dp_ur.shape[0]):
        dp_ur_nonan[ti,:] = dp_ur[ti, data_pos]

    # standarize predictor
    pred_mean = np.mean(dp_ur_nonan, axis=0)
    pred_std = np.std(dp_ur_nonan, axis=0)
    pred_norm = (dp_ur_nonan[:,:] - pred_mean) / pred_std
    pred_norm[np.isnan(pred_norm)] = 0

    # principal components analysis
    ipca = PCA(n_components=min(pred_norm.shape[0], pred_norm.shape[1]))
    PCs = ipca.fit_transform(pred_norm)

    # return dataset
    return xr.Dataset(
        {
            'PCs': (('time', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),

            'pred_mean': (('n_features',), pred_mean),
            'pred_std': (('n_features',), pred_std),

            'pred_lon': (('n_lon',), xds.longitude.values[:]),
            'pred_lat': (('n_lat',), xds.latitude.values[:]),
            'pred_time': (('time',), xds.time.values[:]),
            'pred_data_pos':(('n_points',), data_pos)
        },

        attrs = {
            'method': 'gradient + estela',
            'pred_name': pred_name,
        }
    )

