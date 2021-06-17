# time
from datetime import datetime

# arrays
import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
import xarray as xr

# maths
from scipy import stats
from scipy.spatial import distance_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# custom
from .config import data_path
from .statistical import gev_matrix
from .plotting.kma import Plot_DWTs_Mean_Anom, Plot_DWTs_Probs


def KMA_simple(slp_data, ss_data, pca_data, 
               n_clusters=64, plot: bool = True,
               calculate_gev_stats: bool = True,
               plot_gev: bool = (False,None,None)):
    """
    This function performs the commonly known weather typing
    analysis (daily), where given the PCs of the sea-level-pressure
    fields, this slp and the ss are classified

    Args:
        slp_data (xarray.Dataset): These are the sea-level-pressure fields
            data
        ss_data (xarray.Dataset): This is the storm surge in the region where
            the daily-wt will be applied
        pca_data (xarray.Dataset): These are the PCs that were previously calculated
            with the PCA_DynamicPred
        n_clusters (int, optional): Number of clusters to get. Defaults to 64.
        plot (bool, optional): Wheter to plot or not the results. Defaults to True.
        calculate_gev_stats (bool, optional): Wheter to adjust or not the
            nodes in each cluster to a gev with pyextremes. Defaults to True.
        plot_gev (tuple, optional): This is a tuple with the validation data,
            where this validation here refers to the data thats wants also to
            be fitted to a GEV. Defaults to (False,None,None). An example of this:
                - (True,[lons,lats],dataset) (see analysis_kma.ipynb)

    Returns:
        [xarray.Datasets]: These are the xarray datasets of the analysis
    """

    # check time coherence
    common_times_pslp = np.intersect1d(
        pd.to_datetime(slp_data.time.values).round('H'), # add D??
        pd.to_datetime(pca_data.time.values).round('H'),
        return_indices=True
    )
    common_times = np.intersect1d(
        common_times_pslp[0],
        pd.to_datetime(ss_data.time.values).round('H'),
        return_indices=True
    )

    # number of PCs to use
    num_pcs = int(
        np.where(np.cumsum(pca_data.variance.values/\
            np.sum(pca_data.variance.values))>0.9
        )[0][0]
    ) +1 # number of Pcs to use
        
    # split predictors into train and test
    X_train, X_test, t_train, t_test = \
    train_test_split(
        pca_data.PCs[
            common_times_pslp[2][common_times[1]], :num_pcs
        ].values,
        common_times_pslp[0][common_times[1]], 
        train_size=0.95, 
        shuffle=False
    )

    # calculate the kma
    kma = KMeans(n_clusters=n_clusters).fit(X_train)

    # predict
    prediction = kma.predict(X_test)

    # order the clusters (km, x and var_centers)
    centers = kma.cluster_centers_
    bmus = kma.labels_
    kma_order = np.argsort(np.mean(-centers,axis=1))
    sorted_bmus = np.zeros((len(bmus),),)*np.nan
    sorted_pred = np.zeros((len(prediction),),)*np.nan
    for i in range(n_clusters):
        posc = np.where(bmus==kma_order[i])
        sorted_bmus[posc] = i
        posc = np.where(prediction==kma_order[i])
        sorted_pred[posc] = i

    # save slp and ss clusters + gev stats
    slp_clusters_list = []
    ss_clusters_list_mean, ss_clusters_list_max = [], []
    if calculate_gev_stats:
        gev_stats_list = [] # save gev stats by cluster
        print('\n lets calculate the GEV parameters for each cluster... \n')
    for clus in np.unique(sorted_bmus):
        slp_clusters_list.append(
            slp_data.sel(time=t_train).isel(
                time=np.where(sorted_bmus==clus)[0]
            ).mean(dim='time').expand_dims(
                {'n_clusters':[clus]}
            )
        )
        ss_cluster = ss_data.sel(time=t_train).isel(
            time=np.where(sorted_bmus==clus)[0]
        )
        ss_clusters_list_mean.append(
            ss_cluster.mean(dim='time').expand_dims(
                {'n_clusters':[clus]}
            )
        )
        ss_clusters_list_max.append(
            ss_cluster.quantile(0.95,dim='time').expand_dims(
                {'n_clusters':[clus]}
            )
        )
        if calculate_gev_stats:
            gev_stats_list.append(
                gev_matrix(
                    (ss_cluster * xr.open_dataarray(
                        data_path+'/bathymetry/nz_300m_elev_mask.nc'
                    )).dropna(dim='lon',how='all').dropna(dim='lat',how='all'), 
                    # .interp(
                    #     lon=np.arange(160,185,0.7),
                    #     lat=np.arange(-52,-30,0.7)
                    # ),
                    'lon','lat',plot=False,
                    cluster_number=clus # just to verbose
                )[['ss','mu','phi','xi']].expand_dims(
                    {'n_clusters':[clus]}
                )
            ) # check this GEV analysis

    # concat all cluster lists
    slp_clusters = xr.concat(
        slp_clusters_list,dim='n_clusters'
    ).sortby('n_clusters')
    ss_clusters_mean = xr.concat(
        ss_clusters_list_mean,dim='n_clusters'
    ).sortby('n_clusters')
    ss_clusters_max = xr.concat(
        ss_clusters_list_max,dim='n_clusters'
    ).sortby('n_clusters')
    if calculate_gev_stats:
        ss_clusters_gev = xr.concat(
            gev_stats_list,dim='n_clusters'
        ).sortby('n_clusters')

    # save and plot results
    KMA_data = xr.Dataset(
        data_vars={
            'bmus': (('train_time'), kma.labels_.astype(int)),
            'sorted_bmus': (('train_time'), sorted_bmus.astype(int)),
            'bmus_pred': (('test_time'), sorted_pred.astype(int)),
            'cluster_centers': (('n_clusters','n_components'),
                kma.cluster_centers_),
            'slp_clusters': slp_clusters,
            'ss_clusters_mean': ss_clusters_mean,
            'ss_clusters_max': ss_clusters_max
        },
        coords={
            'train_time': t_train, 'test_time': t_test
        }
    )

    # plot if specified
    if plot:
        # plot all the wheather types
        ss_clusters_gev = ss_clusters_gev if calculate_gev_stats else None
        plottting_data = Plot_DWTs_Mean_Anom(
            KMA_data,cmap=['yellow','lime','green','lightblue','blue','purple','pink','grey','black']
        )
        Plot_DWTs_Probs(KMA_data.bmus,n_clusters)

    # return calculated data
    return_data = [KMA_data,ss_clusters_gev] if calculate_gev_stats else [KMA_data]

    return return_data


def persistences(series):
    '''
    locates persistences for each element in a series

    series - cluster series  (numpy.array)

    returns list of persistences for each cluster in a dictionary
    '''

    # locate dates where series changes
    s_diff = np.diff(series)
    ix_ch = np.where((s_diff != 0))[0]+1
    ix_ch = np.insert(ix_ch, 0, 0)

    wt_ch = series[ix_ch][:-1]  # bmus where WT changes
    wt_dr = np.diff(ix_ch)

    # output dict
    d_pers = {}
    for e in set(series):
        d_pers[e] = wt_dr[wt_ch == e]

    return d_pers


def cluster_probabilities(series, set_values):
    'return series probabilities for each item at set_values'

    us, cs = np.unique(series, return_counts=True)
    d_count = dict(zip(us, cs))

    # cluster probabilities
    cprobs = np.zeros((len(set_values)))
    for i, c in enumerate(set_values):
        cprobs[i] = 1.0*d_count[c]/len(series) if c in d_count.keys() else 0.0

    return cprobs


def change_probabilities(series, set_values):
    'return series transition count and probabilities'

    # count matrix
    count = np.zeros((len(set_values), len(set_values)))
    for ix, c1 in enumerate(set_values):
        for iy, c2 in enumerate(set_values):

            # count cluster-next_cluster ocurrences
            us, cs = np.unique((series[:-1] == c1) &
                               (series[1:] == c2), return_counts=True)
            d_count = dict(zip(us, cs))
            count[ix, iy] = d_count[True] if True in d_count.keys() else 0

    # probabilities
    probs = np.zeros((len(set_values), len(set_values)))
    for ix, _ in enumerate(set_values):

        # calculate each row probability
        probs[ix, :] = count[ix, :] / np.sum(count[ix, :])

    return count, probs


def sort_cluster_gen_corr_end(centers, dimdim):
    '''
    SOMs alternative
    '''

    # get dimx, dimy
    dimy = np.floor(np.sqrt(dimdim)).astype(int)
    dimx = np.ceil(np.sqrt(dimdim)).astype(int)

    if not np.equal(dimx*dimy, dimdim):
        # TODO: RAISE ERROR
        pass

    dd = distance_matrix(centers, centers)
    qx = 0
    sc = np.random.permutation(dimdim).reshape(dimy, dimx)

    # get qx
    for i in range(dimy):
        for j in range(dimx):

            # row F-1
            if not i == 0:
                qx += dd[sc[i-1, j], sc[i, j]]

                if not j == 0:
                    qx += dd[sc[i-1, j-1], sc[i, j]]

                if not j+1 == dimx:
                    qx += dd[sc[i-1, j+1], sc[i, j]]

            # row F
            if not j == 0:
                qx += dd[sc[i, j-1], sc[i, j]]

            if not j+1 == dimx:
                qx += dd[sc[i, j+1], sc[i, j]]

            # row F+1
            if not i+1 == dimy:
                qx += dd[sc[i+1, j], sc[i, j]]

                if not j == 0:
                    qx += dd[sc[i+1, j-1], sc[i, j]]

                if not j+1 == dimx:
                    qx += dd[sc[i+1, j+1], sc[i, j]]

    # test permutations
    q = np.inf
    go_out = False
    for i in range(dimdim):
        if go_out:
            break

        go_out = True

        for j in range(dimdim):
            for k in range(dimdim):
                if len(np.unique([i, j, k])) == 3:

                    u = sc.flatten('F')
                    u[i] = sc.flatten('F')[j]
                    u[j] = sc.flatten('F')[k]
                    u[k] = sc.flatten('F')[i]
                    u = u.reshape(dimy, dimx, order='F')

                    f = 0
                    for ix in range(dimy):
                        for jx in range(dimx):

                            # row F-1
                            if not ix == 0:
                                f += dd[u[ix-1, jx], u[ix, jx]]

                                if not jx == 0:
                                    f += dd[u[ix-1, jx-1], u[ix, jx]]

                                if not jx+1 == dimx:
                                    f += dd[u[ix-1, jx+1], u[ix, jx]]

                            # row F
                            if not jx == 0:
                                f += dd[u[ix, jx-1], u[ix, jx]]

                            if not jx+1 == dimx:
                                f += dd[u[ix, jx+1], u[ix, jx]]

                            # row F+1
                            if not ix+1 == dimy:
                                f += dd[u[ix+1, jx], u[ix, jx]]

                                if not jx == 0:
                                    f += dd[u[ix+1, jx-1], u[ix, jx]]

                                if not jx+1 == dimx:
                                    f += dd[u[ix+1, jx+1], u[ix, jx]]

                    if f <= q:
                        q = f
                        sc = u

                        if q <= qx:
                            qx = q
                            go_out = False

    return sc.flatten('F')


def kma_simple(xds_PCA, num_clusters, repres=0.95):
    '''
    KMeans Classification for PCA data

    xds_PCA       - Principal Component Analysis  (xarray.Dataset)
                    (n_components, n_components) PCs
                    (n_components, n_features) EOFs
                    (n_components, ) variance
    num_clusters  - number of clusters desired from classification 
    repres        -

    returns a xarray.Dataset containing KMA data
    '''

    # PCA data
    variance = xds_PCA.variance.values[:]
    EOFs = xds_PCA.EOFs.values[:]
    PCs = xds_PCA.PCs.values[:]

    var_anom_std = xds_PCA.var_anom_std.values[:]
    var_anom_mean = xds_PCA.var_anom_mean.values[:]
    time = xds_PCA.time.values[:]

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance) / np.sum(variance)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    PCsub = PCs[:, :nterm+1]
    EOFsub = EOFs[:nterm+1, :]

    # KMEANS
    kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(PCsub)

    # groupsize
    _, group_size = np.unique(kma.labels_, return_counts=True)

    # groups
    d_groups = {}
    for k in range(num_clusters):
        d_groups['{0}'.format(k)] = np.where(kma.labels_ == k)
    # TODO: STORE GROUPS WITHIN OUTPUT DATASET

    # centroids
    centroids = np.dot(kma.cluster_centers_, EOFsub)

    # km, x and var_centers
    km = np.multiply(
        centroids,
        np.tile(var_anom_std, (num_clusters, 1))
    ) + np.tile(var_anom_mean, (num_clusters, 1))

    # sort kmeans
    kma_order = np.argsort(np.mean(-km, axis=1))

    # reorder clusters: bmus, km, cenEOFs, centroids, group_size
    sorted_bmus = np.zeros((len(kma.labels_),),)*np.nan
    for i in range(num_clusters):
        posc = np.where(kma.labels_ == kma_order[i])
        sorted_bmus[posc] = i
    sorted_km = km[kma_order]
    sorted_cenEOFs = kma.cluster_centers_[kma_order]
    sorted_centroids = centroids[kma_order]
    sorted_group_size = group_size[kma_order]

    return xr.Dataset(
        {
            'bmus': (('n_pcacomp'), sorted_bmus.astype(int)),
            'cenEOFs': (('n_clusters', 'n_features'), sorted_cenEOFs),
            'centroids': (('n_clusters', 'n_pcafeat'), sorted_centroids),
            'Km': (('n_clusters', 'n_pcafeat'), sorted_km),
            'group_size': (('n_clusters'), sorted_group_size),

            # PCA data
            'PCs': (('n_pcacomp', 'n_features'), PCsub),
            'variance': (('n_pcacomp',), variance),
            'time': (('n_pcacomp',), time),
        }
    )


def kma_regression_guided(
        xds_PCA, xds_Yregres, num_clusters,
        repres=0.95, alpha=0.5, min_group_size=None):
    '''
    KMeans Classification for PCA data: regression guided

    xds_PCA         - Principal Component Analysis  (xarray.Dataset)
                        (n_components, n_components) PCs
                        (n_components, n_features) EOFs
                        (n_components, ) variance
    xds_Yregres     - Simple multivariate regression  (xarray.Dataset)
                        (time, vars) Ym
    num_clusters    - number of clusters desired from classification 
    repres          -
    alpha           -
    min_group_size  - minimun number of samples to accept classification  
    '''

    # PCA data
    variance = xds_PCA['variance'].values[:]
    EOFs = xds_PCA['EOFs'].values[:]
    PCs = xds_PCA['PCs'].values[:]

    # Yregres data
    Y = xds_Yregres['Ym'].values[:]

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance) / np.sum(variance)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    nterm = nterm+1
    PCsub = PCs[:, :nterm]

    # append Yregres data to PCs
    data = np.concatenate((PCsub, Y), axis=1)
    data_std = np.std(data, axis=0)
    data_mean = np.mean(data, axis=0)

    # normalize but keep PCs weigth
    data_norm = np.ones(data.shape)*np.nan
    for i in range(PCsub.shape[1]):
        data_norm[:, i] = np.divide(data[:, i]-data_mean[i], data_std[0])
    for i in range(PCsub.shape[1], data.shape[1]):
        data_norm[:, i] = np.divide(data[:, i]-data_mean[i], data_std[i])

    # apply alpha (PCs - Yregress weight)
    data_a = np.concatenate(
        ((1-alpha)*data_norm[:, :nterm],
         alpha*data_norm[:, nterm:]),
        axis=1
    )

    # KMeans
    keep_iter = True
    count_iter = 0
    while keep_iter:
        # n_init: number of times KMeans runs with different centroids seeds
        # kma = KMeans(
        #    n_clusters = num_clusters,
        #    init='random', n_init=30, max_iter=500,
        #    n_jobs=-1
        # ).fit(data_a)

        # much faster KMeans algorithm
        kma = MiniBatchKMeans(
            n_clusters=num_clusters,
            n_init=10,
            max_iter=500
        ).fit(data_a)

        # check minimun group_size
        group_keys, group_size = np.unique(kma.labels_, return_counts=True)

        # sort output
        group_k_s = np.column_stack([group_keys, group_size])
        group_k_s = group_k_s[group_k_s[:, 0].argsort()]  # sort by cluster num

        if not min_group_size:
            keep_iter = False

        else:
            # keep iterating?
            keep_iter1 = np.where(group_k_s[:, 1] < min_group_size)[0].any()
            keep_iter2 = len(group_keys) != num_clusters
            keep_iter = keep_iter1 or keep_iter2
            count_iter += 1

            # log kma iteration
            for rr in group_k_s:
                if rr[1] < min_group_size:
                    print('  c: {0} - s: {1}'.format(rr[0], rr[1]))
            print('total attemps: ', count_iter)
            print()

    # groups
    d_groups = {}
    for k in range(num_clusters):
        d_groups['{0}'.format(k)] = np.where(kma.labels_ == k)
    # TODO: STORE GROUPS WITHIN OUTPUT DATASET

    # centroids
    centroids = np.zeros((num_clusters, data.shape[1]))
    for k in range(num_clusters):
        centroids[k, :] = np.mean(data[d_groups['{0}'.format(k)], :], axis=1)

    # sort kmeans
    kma_order = sort_cluster_gen_corr_end(kma.cluster_centers_, num_clusters)

    bmus_corrected = np.zeros((len(kma.labels_),),)*np.nan
    for i in range(num_clusters):
        posc = np.where(kma.labels_ == kma_order[i])
        bmus_corrected[posc] = i

    # reorder centroids
    sorted_cenEOFs = kma.cluster_centers_[kma_order, :]
    sorted_centroids = centroids[kma_order, :]

    return xr.Dataset(
        {
            # KMA data
            'bmus': (('n_components',), kma.labels_),
            'cenEOFs': (('n_clusters', 'n_features'), kma.cluster_centers_),
            'centroids': (('n_clusters', 'n_features'), centroids),
            'group_size': (('n_clusters'), group_k_s[:, 1]),

            # sorted KMA data
            'sorted_order': (('n_clusters'), kma_order),
            'sorted_bmus': (('n_components'), bmus_corrected.astype(int)),
            'sorted_cenEOFs': (('n_clusters', 'n_features'), sorted_cenEOFs),
            'sorted_centroids': (('n_clusters', 'n_features'), sorted_centroids),

        },
        attrs={
            'method': 'regression guided',
            'alpha': alpha,
        }
    )


def simple_multivariate_regression_model(xds_PCA, xds_VARS, name_vars):
    '''
    Regression model between daily predictor and predictand
    PCA and VARS input data have to share time dimension values.

    xds_PCA    - PREDICTOR Principal Component Analysis  (xarray.Dataset)
                 (n_components, n_components) PCs
                 (n_components, n_features) EOFs
                 (n_components, ) variance

    xds_VARS   - predictand data variables  (xarray.Dataset)
                 dataset dimension: time
                 dataset variables: name_vars

    name_vars  - will be used as predictand  (ex: ['hs','t02'])

    returns regression for each variable indicated in name_vars
    '''

    # 95% repres
    repres = 0.951

    # PREDICTOR: PCA data
    variance = xds_PCA['variance'].values[:]
    EOFs = xds_PCA['EOFs'].values[:]
    PCs = xds_PCA['PCs'].values[:]

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance) / np.sum(variance)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    PCsub = PCs[:, :nterm-1]
    EOFsub = EOFs[:nterm-1, :]

    PCsub_std = np.std(PCsub, axis=0)
    PCsub_norm = np.divide(PCsub, PCsub_std)

    X = PCsub_norm  #  predictor

    # PREDICTAND: variables data
    wd = np.array([xds_VARS[vn].values[:] for vn in name_vars]).T
    wd_std = np.nanstd(wd, axis=0)
    wd_norm = np.divide(wd, wd_std)

    Y = wd_norm  # predictand

    # Adjust
    [n, d] = Y.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1)

    clf = LinearRegression(fit_intercept=True)
    Ymod = np.zeros((n, d))*np.nan
    for i in range(d):
        clf.fit(X, Y[:, i])
        beta = clf.coef_
        intercept = clf.intercept_
        Ymod[:, i] = np.ones((n,))*intercept
        for j in range(len(beta)):
            Ymod[:, i] = Ymod[:, i] + beta[j]*X[:, j]

    # de-scale
    Ym = np.multiply(Ymod, wd_std)

    # TODO: calculate errors

    return xr.Dataset(
        {
            'Ym': (('time', 'vars'), Ym),
        },
        {
            'time': xds_VARS.time,
            'vars': [vn for vn in name_vars],
        }
    )

