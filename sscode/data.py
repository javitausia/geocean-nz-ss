def data_loading(station, isite, num_pcs):
    '''
    Use this function to load the storm surge data and the slp

    Parameters
    ----------
    station : str
        This is the name of the station that wants to be loaded from the
        already specified file
    num_pcs : int
        This is the number of PCs that will be used to estimate the density
        probability function based on the likelihood

    Returns
    -------
    data : pandas dataframe
        It must return a dataframe with a column ss (data to be analyzed),
        a day column (day of the year) and a time one (time)
            time = day/365.25 + year

    '''
    
    # data loading and formatting
    time_group = '1D'
    # ss
    print('\n Loading the ss data... \n')
    print('\n loading the tidal gauge... \n')
    ss_tg = xr.open_dataset('data/TG_NZ.nc').sel(name=station)
    tg_location = (ss_tg.latitude.values, ss_tg.longitude.values) # loc to filter
    ss_tg = ss_tg.resample(time=time_group).max().dropna(dim='time')
    print('\n loading the ss hindcast data... \n')
    ss_hind = xr.open_dataset('data/SS_NZ_CoDEC.nc').sel(name=station)
    ss_hind = ss_hind.isel(time=slice(100,len(ss_hind.time.values))).ss
    ss_hind = ss_hind.resample(time=time_group).max()
    data = ss_hind.to_dataframe().drop(columns=['name', 
                                                'tg_location_lon', 'tg_location_lat', 
                                                'codec_coords_lon', 'codec_coords_lat'])
    data = data.rename(columns={'ss':'ss_CoDEC'})
    data.index = data.index.round('D')
    ss_hind = xr.open_dataset('data/SS_NZ_NunMod_MOANA.nc').isel(site=8).ss
    ss_hind = ss_hind.resample(time=time_group).max()
    data_join = ss_hind.to_dataframe().drop(columns=['site'])
    data_join = data_join.rename(columns={'ss':'ss_Moana'})
    data_join.index = data_join.index.round('D')
    data = data.join(data_join)
    # plot moana and codec series
    data.plot(figsize=(13,5),alpha=0.8) # TODO: add colors
    ss_num = data[['ss_CoDEC','ss_Moana']].loc[ss_tg.time.values].dropna(how='any',axis=0)
    # and we maintain tg in times existent
    ss_tg = ss_tg.sel(time=ss_num.index.values)
    fig, ax = plt.subplots(figsize=(13,13))
    ax.plot(ss_tg.ss.values, ss_num.ss_CoDEC, '.', c='darkgreen', label='CoDEC', alpha=0.6)
    ax.plot(ss_tg.ss.values, ss_num.ss_Moana, '.', c='darkblue', label='Moana', alpha=0.6)
    ax.set_xlabel('TG Measurements'), ax.set_ylabel('MOANA / CoDEC numerical predictions')
    ax.legend(loc='upper left',fontsize=20)
    ax.plot([-0.3,0.7], [-0.3,0.7], c='orange', lw=3)
    ax.set_xlim(-0.3, 0.7), ax.set_ylim(-0.3,0.7)
    ax.set_aspect('equal')
    # choose hindcast
    bias_codec = np.mean(ss_tg.ss.values-ss_num['ss_CoDEC'])
    label = '\n'.join((r'RMSE = %.2f' % (np.sqrt(np.mean((ss_tg.ss.values-ss_num['ss_CoDEC'])**2)), ),
                       r'BIAS = %.2f' % (bias_codec,  ),
                       r'SI = %.2f' % (si(ss_tg.ss.values, ss_num['ss_CoDEC']), )))
    ax.text(0.75, 0.25, label, transform=ax.transAxes, size=20, color='darkgreen')
    bias_moana = np.mean(ss_tg.ss.values-ss_num['ss_Moana'])
    label = '\n'.join((r'RMSE = %.2f' % (np.sqrt(np.mean((ss_tg.ss.values-ss_num['ss_Moana'])**2)), ),
                       r'BIAS = %.2f' % (bias_moana,  ),
                       r'SI = %.2f' % (si(ss_tg.ss.values, ss_num['ss_Moana']), )))
    ax.text(0.75, 0.10, label, transform=ax.transAxes, size=20, color='darkblue')
    fig.suptitle('TG / Numerical model visual comparisons', fontsize=20)
    # keep preprocessing data
    data = ss_num.copy()
    data['ss_CoDEC'] = data['ss_CoDEC'] + bias_codec # delete bias
    data['ss_Moana'] = data['ss_Moana'] + bias_moana # delete bias
    data['day_of_year'] = data.index.dayofyear
    data['time'] = data.day_of_year.values/365.25 \
        + (data.index.year.values - data.index.year.min())
    # plot ss data
    # fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(14,14))
    # data.ss.plot.hist(ax=axes[0][0], color='k')
    # axes[0][0].axvline(x=data.ss.mean(), color='red', lw=4)
    # data.ss.plot.kde(ax=axes[0][1], c='k', label='Empiric')
    # ss_array = np.arange(data.ss.min(),data.ss.max(),0.01)
    # norm = stats.norm.fit(data.ss.values)
    # genextreme = stats.genextreme.fit(data.ss.values) 
    # axes[0][1].plot(ss_array, stats.norm.pdf(
    #     ss_array,*norm),
    #     c='royalblue', label='Gaussian')
    # axes[0][1].plot(ss_array, stats.genextreme.pdf(
    #     ss_array,*genextreme),
    #     c='seagreen', label='GEV')
    # axes[0][1].legend()
    # stats.probplot(data.ss.values, dist=stats.norm, 
    #                plot=axes[1][0])
    # axes[1][0].set_title('Gaussian distribution', fontsize=12)
    # stats.probplot(data.ss.values, dist=stats.genextreme(0.1), 
    #                plot=axes[1][1])
    # axes[1][1].set_title('GEV distribution', fontsize=12)
    # fig.suptitle('Storm Surge (numerical model) statistics', fontsize=16, y=0.94)
    # data.plot.scatter(x='day_of_year', y='ss', s=10, xlim=(0,366),
    #                   c='royalblue', alpha=0.7, figsize=(14,5),
    #                   title='Storm Surge for each day in a year')
    # data.plot.scatter(x='time', y='ss', s=10, xlim=(1,40),
    #                   c='royalblue', alpha=0.7, figsize=(14,5),
    #                   title='Storm Surge for time in history')
    # calculate the loglikelihood
    # lnorm, lgev = loglikelihood(data.ss, norm, genextreme)
    # print('\n The loglikelihoods (Gaussian / GEV) of ss are: \n {} / {} \n'\
    #       .format(lnorm, lgev))
    # load the slp (predictor)
    print('\n Loading the slp dataset... \n')
    slp = xr.open_dataset('data/ERA5_SLP_6H_1979_2021.nc').sel(latitude=slice(tg_location[0]+2.5,
                                                                              tg_location[0]-2.5),
                                                               longitude=slice(tg_location[1]-2.5,
                                                                               tg_location[1]+2.5))
    slp = slp.resample(time=time_group).mean().sel(time=data.index.values)
    slp = slp.assign({'msl':(('time','longitude','latitude'),slp.msl/100)})
    print(' \n and calculating the gradient... \n')
    slp = spatial_gradient(slp,'msl')
    # lets now create the PCs matrix
    x_shape = len(slp.time.values)-1
    y_shape = len(slp.latitude.values)*len(slp.longitude.values)
    pcs_matrix = np.zeros((x_shape,4*y_shape))
    for t in range(1,x_shape):
        pcs_matrix[t-1,:y_shape] = slp.isel(time=t).msl.values.reshape(-1)
        pcs_matrix[t-1,y_shape:y_shape*2] = slp.isel(time=t).msl_gradient.values.reshape(-1)
        pcs_matrix[t-1,y_shape*2:y_shape*3] = slp.isel(time=t-1).msl.values.reshape(-1)
        pcs_matrix[t-1,y_shape*3:] = slp.isel(time=t-1).msl_gradient.values.reshape(-1)
    pcs_matrix = pcs_matrix[:-2]
    # standardizing the features
    pcs_stan = StandardScaler().fit_transform(pcs_matrix)
    pcs_stan[np.isnan(pcs_stan)] = 0.0
    # calculate de PCAs
    pca_fit = PCA(n_components=pcs_stan.shape[1])
    PCs = pca_fit.fit_transform(pcs_stan)
    # PCs dataset
    PCs_data = xr.Dataset(
        {
            'PCs': (('time', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), pca_fit.components_),
            'variance': (('n_components',), pca_fit.explained_variance_),
        }
    )
    # check PC1
    fig, axes = plt.subplots(ncols=2, figsize=(13,6))
    axes[0].pcolormesh(np.flipud(PCs_data.EOFs.isel(n_components=0).values[:y_shape]\
                                 .reshape(len(slp.latitude.values),len(slp.longitude.values))), 
                       cmap='RdBu', shading='gouraud')
    axes[0].set_title('EOF1 (SLP)')
    axes[1].pcolormesh(np.flipud(PCs_data.EOFs.isel(n_components=0).values[y_shape:y_shape*2]\
                                 .reshape(len(slp.latitude.values),len(slp.longitude.values))), 
                       cmap='RdBu', shading='gouraud')
    axes[1].set_title('EOF1 (Gradient)')
    fig.suptitle('Component 1 (EOF and PC): \n variance={} \n % explained_variance={}'\
                 .format(PCs_data.variance.values[0], 
                         (PCs_data.variance.values/np.sum(PCs_data.variance.values))[0]), 
                 y=1.15, fontsize=16)
    fig, ax = plt.subplots(figsize=(13,4))
    ax.plot(PCs_data.PCs.isel(n_components=0).values, alpha=0.7)
    ax.set_xlim(0,len(PCs_data.PCs.isel(n_components=0).values))
    ax.set_title('PC1 (time evolution)')
    # total data
    data = data.iloc[1:-2].copy()
    for ipc in range(num_pcs):
        data['pc{}'.format(ipc+1)] = PCs_data.PCs.values[:,ipc]
    # standarize the PCs
    data.iloc[:,3:] = StandardScaler().fit_transform(data.iloc[:,3:].values)
    print('\n lets plot the preprocessed data... \n')
    
    return data, PCs_data, ss_tg


def si(predictions, targets):
    S = predictions.mean()
    O = targets.mean()
    return np.sqrt(sum(((predictions-S)-(targets-O))**2)/((sum(targets**2))))
