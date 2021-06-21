# arrays
import numpy as np
from numpy.core.fromnumeric import size
from numpy.core.numeric import False_
from numpy.lib.index_tricks import ix_
import pandas as pd
# genextreme
from pyextremes import EVA
from scipy.stats import genextreme

# time
from datetime import date, datetime
from cftime._cftime import DatetimeGregorian

# plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

# custom
from ..data import join_load_uhslc_tgs
from .config import _figsize_width, _figsize_height, _fontsize_title, \
    _fontsize_label, _mbar_diff, _fontsize_legend
from ..config import default_location
from ..utils import GetBestRowsCols
from .utils import colors_dwt, custom_cmap, get_n_colors
# from ..kma import cluster_probabilities


def Plot_DWTs_Mean_Anom(xds_KMA, kind='anom', 
                        scale_: bool = True,
                        press_diff = _mbar_diff,
                        cmap = 'jet', gev_data = None,
                        plot_gev: tuple = (False,None,None)):
    '''
    Plot Daily Weather Types (bmus mean)
    kind - mean/anom
    '''

    # bmus = xds_KMA['bmus'].values # add sorted
    n_clusters = len(xds_KMA.n_clusters.values)

    # var_max = np.nanmax(xds_var.values)
    # var_min = np.nanmin(xds_var.values)
    scale = 100.0 if scale_ else 1.0 # scale from Pa to mbar

    # get number of rows and cols for gridplot 
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # get cluster colors
    cmap, cs_dwt = get_n_colors(
        cmap, # ['lime','yellow','green','blue','purple','pink','grey','black']
        # list of colors can be eddited
        n_clusters
    )

    # plotting the SLP clusters
    fig, axes = plt.subplots(
        ncols=n_cols,nrows=n_rows,
        figsize=(n_cols*(_figsize_width/1.6),
                 n_rows*(_figsize_height/1.8)),
        subplot_kw={
            'projection': ccrs.PlateCarree(
                central_longitude=default_location[0]
            )
        }
    )
    fig.subplots_adjust(hspace=0.02,wspace=0.02)
    for i,ax in enumerate(axes.flatten()):
        if kind=='anom':
            # p = ((xds_var.isel(time=np.where(xds_KMA.bmus.values==i)[0])\
            #     .mean(dim='time') - (
            #     xds_var.mean(dim='time')
            # )) / scale).plot(
            #         cmap='RdBu_r',vmin=-press_diff,vmax=press_diff,
            #         ax=ax,transform=ccrs.PlateCarree(),
            #         add_colorbar=False
            #     )
            pslp = ((xds_KMA.slp_clusters.isel(n_clusters=i) - \
                xds_KMA.slp_clusters.mean(dim='n_clusters')) / scale).plot(
                    cmap='RdBu_r',vmin=-press_diff,vmax=press_diff,
                    ax=ax,transform=ccrs.PlateCarree(),
                    add_colorbar=False
                )
        else:
            # p = ((xds_var.isel(time=np.where(xds_KMA.bmus.values==i)[0])\
            #     .mean(dim='time')) / scale).plot(
            #         cmap='RdBu_r',vmin=1013-press_diff,vmax=1013+press_diff,
            #         ax=ax,transform=ccrs.PlateCarree(),
            #         add_colorbar=False
            #     )
            pslp = (xds_KMA.slp_clusters.isel(n_clusters=i) / scale).plot(
                    cmap='RdBu_r',vmin=1013-press_diff,vmax=1013+press_diff,
                    ax=ax,transform=ccrs.PlateCarree(),
                    add_colorbar=False
                )
        # axis customization
        ax.coastlines(linewidth=2)
        ax.set_title('')
        ax.text(
            184.5,-24,str(i), # add better text?
            transform=ccrs.PlateCarree(),
            zorder=100,size=12,
            bbox=dict(boxstyle='round',
                ec=(1.,0.5,0.5),
                fc=(1.,0.8,0.8),
            )
        )
        [ax.spines[loc_ax].set_color(cs_dwt[i]) \
            for loc_ax in ax.spines] # axis color
        [ax.spines[loc_ax].set_linewidth(3) \
            for loc_ax in ax.spines] # axis linewidth
        
    # add the colorbar
    cbar_ax = fig.add_axes([0.135,0.10,0.75,0.02])
    cb = fig.colorbar(pslp,cax=cbar_ax,orientation='horizontal')
    if kind=='mean':
        cb.set_label('Pressure [mbar]',fontsize=_fontsize_label)
    elif kind=='anom':
        cb.set_label('Pressure anomalies [mbar]',fontsize=_fontsize_label)

    # plotting the ss mean clusters
    fig, axes = plt.subplots(
        ncols=n_cols,nrows=n_rows,
        figsize=(n_cols*(_figsize_width/1.6),
                 n_rows*(_figsize_height/1.9)),
        subplot_kw={
            'projection': ccrs.PlateCarree(
                central_longitude=default_location[0]
            )
        }
    )
    fig.subplots_adjust(hspace=0.02,wspace=0.02)
    for i,ax in enumerate(axes.flatten()):
        pss = xds_KMA.ss_clusters_mean.isel(n_clusters=i).plot(
            cmap=custom_cmap(15,'YlOrRd',0.15,0.9,'YlGnBu_r',0,0.85),
            vmin=-0.2,vmax=0.3,add_colorbar=False,
            ax=ax,transform=ccrs.PlateCarree(),
        )
        # axis customization
        ax.coastlines(linewidth=2)
        ax.set_title('')
        ax.text(
            182,-33.3,str(i), # add better text?
            transform=ccrs.PlateCarree(),
            zorder=100,size=12,
            bbox=dict(boxstyle='round',
                ec=(1.,0.5,0.5),
                fc=(1.,0.8,0.8),
            )
        )
        [ax.spines[loc_ax].set_color(cs_dwt[i]) \
            for loc_ax in ax.spines] # axis color
        [ax.spines[loc_ax].set_linewidth(3) \
            for loc_ax in ax.spines] # axis linewidth
    # add the colorbar
    cbar_ax = fig.add_axes([0.135,0.10,0.75,0.02])
    cb = fig.colorbar(pss,cax=cbar_ax,orientation='horizontal')
    cb.set_label('Storm surge mean [m]',fontsize=_fontsize_label)

    # plot gev stats
    if plot_gev[0]:
        # TODO: plot the different lons/lats to analyze
        clusters_to_plot = np.random.randint(0,n_clusters,5)
        tgs_names = join_load_uhslc_tgs().name.values
        moana_models = []
        for clus in clusters_to_plot:
            fig_hists, axes_hists = plt.subplots(
                ncols=len(plot_gev[1][0]),
                figsize=(_figsize_width*len(plot_gev[1][0])*1.2,_figsize_height/1.6)
            )
            # plot histogram stats for every site specified
            site_counter=0
            for lon_site,lat_site in zip(plot_gev[1][0],plot_gev[1][1]):
                try:
                    gev_data_clus_site = plot_gev[2].sel(
                        time=xds_KMA.train_time.where(
                            xds_KMA.sorted_bmus==clus,drop=True
                        ).values
                    ).isel(
                        site=site_counter
                    ).dropna(dim='time')
                except: # not working for memory problems
                    gev_data_clus_site = gev_data.sel(
                        n_clusters=clus,lon=lon_site,lat=lat_site
                    ).dropna(dim='time')
                # continue loop if no data is available
                if len(gev_data_clus_site.ss.values)==0:
                    site_counter+=1
                    continue
                gev_data_clus_site.ss.plot.hist(
                    ax=axes_hists[site_counter],alpha=0.7,density=True,
                    label='SS fit in ({},{})'.format(lon_site,lat_site) 
                    # TODO: add color
                )
                print('fitting the site {} in cluster {} to GEV...'.format(site_counter,clus),end='\r')
                model = EVA(data=gev_data_clus_site.ss.to_dataframe()['ss'])
                model.get_extremes(method='POT',threshold=min(gev_data_clus_site.ss.values))
                model.fit_model(
                    distribution=genextreme,model='Emcee',n_samples=50,progress=False
                )
                moana_models.append(model)
                # TODO: add extra plotting
                shape, loc, scale = tuple(
                    [mle_value for mle_value in model.distribution.mle_parameters.values()]
                )
                # shape, loc, scale = genextreme.fit(gev_data_clus_site.ss.values)
                print('predicting values for site {} in cluster {}...'.format(site_counter,clus),end='\r')
                gev_pdf = genextreme.rvs(shape,loc=loc,scale=scale,size=10000)
                pd.Series(gev_pdf).plot.kde(
                    ax=axes_hists[site_counter],lw=3,c='k'
                )
                axes_hists[site_counter].axvline(x=0,c='k',lw=1.5,ls='--')
                # axis customization
                # axes_hists[site_counter].legend(fontsize=_fontsize_legend)
                axes_hists[site_counter].set_title(
                    'GEV parameters in ({},{}) \n - {} - with {} points: \n min={}, max={} \n mu={}, phi={}, xi={}'.format(
                        str(lon_site)[:5],str(lat_site)[:5],
                        str(tgs_names[site_counter])[2:],
                        len(gev_data_clus_site.ss.values),
                        str(gev_data_clus_site.ss.min().values)[:5],
                        str(gev_data_clus_site.ss.max().values)[:5],
                        str(loc)[:5],str(scale)[:5],str(-shape)[:5]
                    )
                )
                axes_hists[site_counter].set_xlim(-0.4,0.8)
                [axes_hists[site_counter].spines[loc_ax].set_color(cs_dwt[clus]) \
                    for loc_ax in axes_hists[site_counter].spines] # axis color
                [axes_hists[site_counter].spines[loc_ax].set_linewidth(3) \
                    for loc_ax in axes_hists[site_counter].spines] # axis linewidth
                fig_hists.suptitle(
                    'GEV analysis for cluster  {}'.format(clus),
                    fontsize=_fontsize_title, y=1.35
                )
                site_counter+=1

    # show results
    plt.show()

    return_data = [cmap,moana_models] if plot_gev[0] else [cmap]

    return return_data


def Plot_DWTs_Probs(bmus, n_clusters, show_cbar=True):
    '''
    Plot Daily Weather Types bmus probabilities
    '''

    wt_set = np.arange(n_clusters) + 1

    # best rows cols combination
    n_rows, n_cols = GetBestRowsCols(n_clusters)

    # figure
    fig = plt.figure(figsize=(18,14))

    # layout
    gs = gridspec.GridSpec(4, 7, wspace=0.10, hspace=0.25)

    # list all plots params
    l_months = [
        (1, 'January',   gs[1,3]),
        (2, 'February',  gs[2,3]),
        (3, 'March',     gs[0,4]),
        (4, 'April',     gs[1,4]),
        (5, 'May',       gs[2,4]),
        (6, 'June',      gs[0,5]),
        (7, 'July',      gs[1,5]),
        (8, 'August',    gs[2,5]),
        (9, 'September', gs[0,6]),
        (10, 'October',  gs[1,6]),
        (11, 'November', gs[2,6]),
        (12, 'December', gs[0,3]),
    ]

    l_3months = [
        ([12, 1, 2],  'DJF', gs[3,3]),
        ([3, 4, 5],   'MAM', gs[3,4]),
        ([6, 7, 8],   'JJA', gs[3,5]),
        ([9, 10, 11], 'SON', gs[3,6]),
    ]

    # plot total probabilities
    c_T = cluster_probabilities(bmus, wt_set)
    C_T = np.reshape(c_T, (n_rows, n_cols))

    ax_probs_T = plt.subplot(gs[:2, :3])
    pc = axplot_WT_Probs(ax_probs_T, C_T, ttl = 'DWT Probabilities')

    # plot counts histogram
    ax_hist = plt.subplot(gs[2:, :3])
    axplot_WT_Hist(ax_hist, bmus, n_clusters, ttl = 'DWT Counts')

    # plot probabilities by month
    vmax = 0.15
    for m_ix, m_name, m_gs in l_months:

        # get probs matrix
        c_M = ClusterProbs_Month(bmus, bmus.train_time.values, wt_set, m_ix)
        C_M = np.reshape(c_M, (n_rows, n_cols))

        # plot axes
        ax_M = plt.subplot(m_gs)
        axplot_WT_Probs(ax_M, C_M, ttl = m_name, vmax=vmax)

    # TODO: add second colorbar?

    # plot probabilities by 3 month sets
    vmax = 0.15
    for m_ix, m_name, m_gs in l_3months:

        # get probs matrix
        c_M = ClusterProbs_Month(bmus, bmus.train_time.values, wt_set, m_ix)
        C_M = np.reshape(c_M, (n_rows, n_cols))

        # plot axes
        ax_M = plt.subplot(m_gs)
        axplot_WT_Probs(ax_M, C_M, ttl = m_name, vmax=vmax, cmap='Greens')

    # add custom colorbar
    if show_cbar:
        pp = ax_probs_T.get_position()
        # cbar_ax = fig.add_axes([pp.x1+0.02, pp.y0, 0.02, pp.y1 - pp.y0])
        cbar_ax = fig.add_axes([pp.x0+0.02, pp.y0+0.03, pp.x1-0.04 - pp.x0, 0.02])
        cb = fig.colorbar(pc, cax=cbar_ax, cmap='Blues', orientation='horizontal')
        cb.ax.tick_params(labelsize=8)

    # show probability results
    plt.show()


def cluster_probabilities(series, set_values):
    'return series probabilities for each item at set_values'

    us, cs = np.unique(series, return_counts=True)
    d_count = dict(zip(us, cs))

    # cluster probabilities
    cprobs = np.zeros((len(set_values)))
    for i, c in enumerate(set_values):
        cprobs[i] = 1.0*d_count[c]/len(series) if c in d_count.keys() else 0.0

    return cprobs


def ClusterProbs_Month(bmus, time, wt_set, month_ix):
    'Returns Cluster probs by month_ix'

    # get months
    _, months, _ = get_years_months_days(time)

    if isinstance(month_ix, list):

        # get each month indexes
        l_ix = []
        for m_ix in month_ix:
            ixs = np.where(months == m_ix)[0]
            l_ix.append(ixs)

        # get all indexes     
        ix = np.unique(np.concatenate(tuple(l_ix)))

    else:
        ix = np.where(months == month_ix)[0]

    bmus_sel = bmus[ix]

    return cluster_probabilities(bmus_sel, wt_set)


def axplot_WT_Probs(ax, wt_probs,
                     ttl = '', vmin = 0, vmax = 0.1,
                     cmap = 'Blues', caxis='black'):
    'axes plot WT cluster probabilities'

    # clsuter transition plot
    pc = ax.pcolor(
        np.flipud(wt_probs),
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors='k',
    )

    # customize axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(ttl, {'fontsize':_fontsize_title, 'fontweight':'bold'})

    # axis color
    plt.setp(ax.spines.values(), color=caxis)
    plt.setp(
        [ax.get_xticklines(), ax.get_yticklines()],
        color=caxis,
    )

    # axis linewidth
    if caxis != 'black':
        plt.setp(ax.spines.values(), linewidth=3)

    return pc


def axplot_WT_Hist(ax, bmus, n_clusters, ttl=''):
    'axes plot WT cluster count histogram'

    # cluster transition plot
    ax.hist(
        bmus,
        bins = np.arange(1, n_clusters+2),
        edgecolor='k'
    )

    # customize axes
    # ax.grid('y')

    ax.set_xticks(np.arange(1,n_clusters+1)+0.5)
    ax.set_xticklabels(np.arange(1,n_clusters+1))
    ax.set_xlim([1, n_clusters+1])
    ax.tick_params(axis='both', which='major', labelsize=6)

    ax.set_title(ttl, {'fontsize':_fontsize_title, 'fontweight':'bold'})


def get_years_months_days(time):
    '''
    Returns years, months, days of time in separete lists

    (Used to avoid problems with dates type)
    '''

    t0 = time[0]
    if isinstance(t0, (date, datetime, DatetimeGregorian)):
        ys = np.asarray([x.year for x in time])
        ms = np.asarray([x.month for x in time])
        ds = np.asarray([x.day for x in time])

    else:
        tpd = pd.DatetimeIndex(time)
        ys = tpd.year
        ms = tpd.month
        ds = tpd.day

    return ys, ms, ds

