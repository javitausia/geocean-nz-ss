"""
This python file will host all the plotting functions for the results.
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import seaborn as sns

from ..data import load_moana_hindcast
from .config import locations_colors, locations_colors_letters, \
    delta_lons, delta_lats, metrics_cmaps
from .utils import plot_ccrs_nz, get_n_colors


# these are the common dimensions in the models
variables_dictionary = {
    'grad': ['GrF','GrT'],
    'winds': ['WF','WT'],
    'tlapse': ['TL1','TL2','TL3'],
    'tresample': ['1D'],
    'region': ['local1','local2','NZ']
}
def generate_xlabels(variables_dictionary):
    """
    Generates the xlabels for the plot.
    """
    x_labels = []
    for cg in variables_dictionary['grad']:
        for ws in variables_dictionary['winds']:
            for tl in variables_dictionary['tapse']:
                for tr in variables_dictionary['tesample']:
                    for r in variables_dictionary['region']:
                        x_labels.append(
                            cg+' - '+ws+' - '+tl+' - '+tr+' - '+r
                        )
    return x_labels


class ResultsPlotter:

    def __init__(self, 
                 linear_stats,
                 knn_stats,
                 xgb_stats,
                 linear_stats_red,
                 knn_stats_red,
                 xgb_stats_red,
                 mda_rbf_stats = None):

        self.linear_stats = linear_stats.copy()
        self.knn_stats = knn_stats.copy()
        self.xgb_stats = xgb_stats.copy()
        self.linear_stats_red = linear_stats_red.copy()
        self.knn_stats_red = knn_stats_red.copy()
        self.xgb_stats_red = xgb_stats_red.copy()
        self.mda_rbf_stats = mda_rbf_stats

    
    def plot_histogram(self, metric='kge'):

        fig, ax = plt.subplots(figsize=(12,5))
        self.linear_stats.where(self.linear_stats[metric]>0,np.nan)[metric].plot(
            alpha=0.6,label='linear',density=True,bins=50,color='crimson',ax=ax)
        self.knn_stats.where(self.knn_stats[metric]>0,np.nan)[metric].plot(
            alpha=0.6,label='knn',density=True,bins=50,color='navy',ax=ax)
        self.xgb_stats.where(self.xgb_stats[metric]>0,np.nan)[metric].plot(
            alpha=0.6,label='xgb',density=True,bins=50,color='dimgray',ax=ax)
        ax.axes.get_yaxis().set_visible(False)
        ax.grid(ls='--')
        ax.set_xlim(0.5,0.9)
        fig.legend(loc='upper left')
        plt.show()

    
    def plot_studied_locations(self, metric='kge'):

        # and now we load the longitudes and latitudes for the selected sites,
        # these sites should be equal in all statistical methods
        locations = load_moana_hindcast(plot=False)[['lon','lat']].sel(
            site=self.linear_stats.site.values)
        # create the figure you want to plot on
        fig, ax = plt.subplots(
            figsize=(15,15), # this size correctly displays everything
            subplot_kw={
                'projection':ccrs.PlateCarree(central_longitude=180)
            }
        )
        # plot best performing models for each site in scatter color
        ax.scatter(
            x = locations.lon.values, y = locations.lat.values, # lon and lat coordinates
            c = [np.argsort(
                    np.concatenate([
                        self.linear_stats_red[metric].values.reshape(len(self.linear_stats_red.site.values),-1,1),
                        self.knn_stats_red[metric].values.reshape(len(self.knn_stats_red.site.values),-1,1),
                        self.xgb_stats_red[metric].values.reshape(len(self.xgb_stats_red.site.values),-1,1)
                    ], axis=2), axis=2
                )[:,:,-1][i_site,model] for i_site,model in enumerate(
                    np.argsort(np.max(
                        np.concatenate([
                            self.linear_stats_red[metric].values.reshape(len(self.linear_stats_red.site.values),-1,1),
                            self.knn_stats_red[metric].values.reshape(len(self.knn_stats_red.site.values),-1,1),
                            self.xgb_stats_red[metric].values.reshape(len(self.xgb_stats_red.site.values),-1,1)
                        ], axis=2), axis=2
                    ), axis=1)[:,-1]
                )
                ], # these are the best performing models by site
            transform=ccrs.PlateCarree(),zorder=20,s=200, 
            cmap=ListedColormap(['crimson','dimgray','navy'])
        )
        for isite,color in zip(range(len(locations.site)),locations_colors):
            ax.text(
                x=locations.lon.values[isite]+delta_lons[isite],
                y=locations.lat.values[isite]+delta_lats[isite],
                s=locations.site.values.astype(str)[isite],
                transform=ccrs.PlateCarree(),size=14,zorder=40,
                backgroundcolor=color,color=locations_colors_letters[isite],
                bbox=dict(facecolor=color,edgecolor='black',boxstyle='round')
            )
        ax.set_facecolor('lightblue')
        plot_ccrs_nz([ax],plot_labels=(False,5,5))
        plt.show()


    def plot_allmodels_stats(self, 
                             stats_plot='linear', 
                             metrics_to_plot=['kge']):

        # plot stuff!!
        stats_plot = self.linear_stats_red if stats_plot=='linear' else \
            self.knn_stats_red if stats_plot=='knn' else \
            self.xgb_stats_red if stats_plot=='xgb' else None

        # extract len sites
        num_sites = len(stats_plot.site.values)
            
        for metric_to_plot in metrics_to_plot:
            fig, ax = plt.subplots(figsize=(15,10))
            # extract metric values from xarray.Dataset
            metric_values = stats_plot[metric_to_plot].values.reshape(num_sites,-1) \
                if metric_to_plot!='bias' else np.abs(
                    stats_plot[metric_to_plot].values.reshape(num_sites,-1)
                )
            # plot pcolors 
            if metric_to_plot=='kgeprime':
                pc = ax.pcolor(
                    metric_values, cmap=metrics_cmaps[metric_to_plot],
                    vmin=0.45, vmax=0.85 # to maintain consistency
                )
            elif metric_to_plot=='kge':
                pc = ax.pcolor(
                    metric_values, cmap=metrics_cmaps[metric_to_plot],
                    vmin=0.45, vmax=0.85 # to maintain consistency
                )
            elif metric_to_plot=='pearson':
                pc = ax.pcolor(
                    metric_values, cmap=metrics_cmaps[metric_to_plot],
                    vmin=0.45, vmax=0.85 # to maintain consistency
                )
            elif metric_to_plot=='kgeprime_gamma':
                pc = ax.pcolor(
                    metric_values, cmap=metrics_cmaps[metric_to_plot],
                    vmin=0.5, vmax=1.0 # to maintain consistency
                )
            elif metric_to_plot=='kgeprime_beta':
                pc = ax.pcolor(
                    metric_values, cmap=metrics_cmaps[metric_to_plot],
                    vmin=0.7, vmax=1.3 # to maintain consistency
                )
            else:
                pc = ax.pcolor(
                    metric_values, cmap=metrics_cmaps[metric_to_plot],
                    vmin=np.nanmin(metric_values), vmax=np.nanmax(metric_values)
                )
            #ax.scatter(
            #    np.argsort(metric_values)[:,:1].reshape(-1) + 0.5, # first-worst values in array
            #    np.repeat(np.arange(num_sites),1).reshape(num_sites,1) + 0.5, # y_positons
            #    marker='v',facecolors='pink',edgecolors='black',linewidth=1,s=100,zorder=10
            #)
            ax.scatter(
                np.argsort(metric_values)[:,-2:].reshape(-1) + 0.5, # last-best values in array
                np.repeat(np.arange(num_sites),2).reshape(num_sites,2) + 0.5, # y_positons
                marker='^',facecolors='pink',edgecolors='black',linewidth=1,s=150,zorder=10
            )
            divider = make_axes_locatable(ax)
            ax_cbar = divider.append_axes("right",size="4%",pad=0.15) 
            cbar = fig.colorbar(pc,cax=ax_cbar) # colorbar definition
            cbar.set_label(metric_to_plot,size=28)
            cbar.set_ticks([0.5,0.6,0.7,0.8]) if metric_to_plot=='kgeprime' \
                or metric_to_plot=='kge' or metric_to_plot=='pearson' else None
            cbar.ax.set_yticklabels([' < 0.5',' 0.6',' 0.7',' > 0.8'],fontsize=24,fontweight='bold') \
                if metric_to_plot=='kgeprime' or metric_to_plot=='kge' or metric_to_plot=='pearson' else None
            ax.set_yticks(np.arange(num_sites)+0.5) # these are the positions, and below the labels = sites
            ax.set_yticklabels(stats_plot.site.values[:],fontweight='bold',fontsize=18)
            for ytick,color in zip(ax.get_yticklabels(),locations_colors):
                ytick.set_color(color)
            ax.set_xticks([]) # leave x labels empty
            # ax.set_xticks(np.arange(0,metric_values.shape[1],3)+0.5)
            # ax.set_xticklabels(x_labels[::3],fontsize=14)
            # plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
            # ax.set_xticks([]) if metric_to_plot=='kgeprime' else None
            # ax.set_title(metric_to_plot.upper(),fontsize=20) if metric_to_plot!='kgeprime' else None
            for xline,vs,lws in zip( # add lines to better visualization
                [11.5,14.5,17.5,11.5+18,14.5+18,17.5+18],[29]*6,[24]*6
            ):
                ax.vlines(xline,lw=lws,colors='lightgrey',ymax=vs,ymin=0)
            for xline,vs,lws in zip( # add lines to better visualization
                [np.arange(0,36,18)[1:],np.arange(0,36,9)[1:],np.arange(0,36,3)[1:]],
                [29,20,10],[8,6,4]
            ):
                ax.vlines(xline,lw=lws,colors='k',ymax=vs,ymin=0)
            plt.show()


    def plot_best_stat(self, stat='kge'):

        # extract len sites
        num_sites = len(self.linear_stats.site.values)
        # get better models for stat
        all_models_stat = np.concatenate([
            self.linear_stats_red[stat].values.reshape(num_sites,-1,1),
            self.knn_stats_red[stat].values.reshape(num_sites,-1,1),
            self.xgb_stats_red[stat].values.reshape(num_sites,-1,1)
        ], axis=2)

        # NOW, WE ADD MORE CUSTOM PLOTS!! hahahahahahahaha
        fig, ax = plt.subplots(figsize=(17,10))
        best = ax.pcolor(
            np.argmax(
                all_models_stat, axis=2
            ), # this is the best model for each site
            cmap=ListedColormap(['crimson','dimgray','navy']),
            vmin=0,vmax=3
        )
        ax.scatter(
            np.argsort(
                np.max(
                    all_models_stat, axis=2
                ), axis=1
            )[:,-1].reshape(-1) + 0.5, # last-best value in array
            np.repeat(np.arange(num_sites),1).reshape(num_sites,1) + 0.5, # y_positons
            marker='^',facecolors='pink',edgecolors='black',linewidth=1,s=200,zorder=10
        )
        ax.scatter(
            np.arange(36)+0.5, # x positions
            np.argsort(
                np.max(
                    all_models_stat, axis=2
                ), axis=0
            )[-1,:].reshape(-1) + 0.5, # last-best value in array
            marker='*',facecolors='pink',edgecolors='black',linewidth=1,s=300,zorder=1
        )
        ax.set_yticks(np.arange(num_sites)+0.5) # these are the positions, and below the labels = sites
        ax.set_yticklabels([
            str(self.linear_stats.site.values[i])+' - '+str(np.argsort(np.max(
                all_models_stat, axis=2
            ), axis=1)[:,-1][i]+1) for i in range(num_sites)],fontweight='bold',fontsize=18)
        for ytick,color in zip(ax.get_yticklabels(),locations_colors):
            ytick.set_color(color)
        ax.set_xticks([]) # leave x labels empty
        divider = make_axes_locatable(ax)
        ax_cbar = divider.append_axes("right",size="4%",pad=0.15) 
        cbar = fig.colorbar(best,cax=ax_cbar) # colorbar definition
        cbar.set_ticks([0.5,1.5,2.5])
        cbar.ax.set_yticklabels([' linear',' KNN',' XGBoost'],fontsize=24,fontweight='bold')
        for xline,vs,lws in zip( # add lines to better visualization
            [np.arange(0,36,18)[1:],np.arange(0,36,9)[1:],np.arange(0,36,3)[1:]],
            [29,20,10],[8,6,4]
        ):
            ax.vlines(xline,lw=lws,colors='white',ymax=vs,ymin=0)
        plt.show()
        fig, ax = plt.subplots(figsize=(16,10))
        best = ax.pcolor(
            np.max(
                all_models_stat, axis=2
            ), cmap='Spectral', vmin=0.45, vmax=0.85
        ) if stat=='kgeprime' or stat=='kge' or stat=='pearson' else ax.pcolor(
            np.max(
                all_models_stat, axis=2
            ), cmap='Spectral', vmin=np.nanmin(all_models_stat),
            vmax=np.nanmax(all_models_stat)
        )
        ax.scatter(
            np.argsort(
                np.max(
                    all_models_stat, axis=2
                ), axis=1
            )[:,-1].reshape(-1) + 0.5, # last-best value in array
            np.repeat(np.arange(num_sites),1).reshape(num_sites,1) + 0.5, # y_positons
            marker='^',facecolors='pink',edgecolors='black',linewidth=1,s=200,zorder=10
        )
        ax.scatter(
            np.arange(36)+0.5, # x positions
            np.argsort(
                np.max(
                    all_models_stat, axis=2
                ), axis=0
            )[-1,:].reshape(-1) + 0.5, # last-best value in array
            marker='*',facecolors='pink',edgecolors='black',linewidth=1,s=300,zorder=1
        )
        ax.set_yticks(np.arange(num_sites)+0.5) # these are the positions, and below the labels = sites
        ax.set_yticklabels([
            str(self.linear_stats.site.values[i])+' - '+str(np.argsort(np.max(
                all_models_stat, axis=2
            ), axis=1)[:,-1][i]+1) for i in range(num_sites)],fontweight='bold',fontsize=18)
        for ytick,color in zip(ax.get_yticklabels(),locations_colors):
            ytick.set_color(color)
        divider = make_axes_locatable(ax)
        ax_cbar = divider.append_axes("right",size="4%",pad=0.15) 
        cbar = fig.colorbar(best,cax=ax_cbar) # colorbar definition
        cbar.set_label(stat,size=28)
        cbar.set_ticks([0.5,0.6,0.7,0.8]) if stat=='kgeprime' \
            or stat=='kge' or stat=='pearson' else None
        cbar.ax.set_yticklabels([' < 0.5',' 0.6',' 0.7',' > 0.8'],fontsize=24,fontweight='bold') \
            if stat=='kgeprime' or stat=='kge' or stat=='pearson' else None
        ax.set_yticks(np.arange(num_sites)+0.5) # these are the positions, and below the labels = sites
        ax.set_yticklabels(self.linear_stats.site.values[:],fontweight='bold',fontsize=18)
        for ytick,color in zip(ax.get_yticklabels(),locations_colors):
            ytick.set_color(color)
        ax.set_xticks([]) # leave x labels empty
        # ax.set_xticks(np.arange(0,metric_values.shape[1],3)+0.5)
        # ax.set_xticklabels(x_labels[::3],fontsize=14)
        # plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
        # ax.set_xticks([]) if metric_to_plot=='kgeprime' else None
        # ax.set_title(metric_to_plot.upper(),fontsize=20) if metric_to_plot!='kgeprime' else None
        for xline,vs,lws in zip( # add lines to better visualization
            [np.arange(0,36,18)[1:],np.arange(0,36,9)[1:],np.arange(0,36,3)[1:]],
            [29,20,10],[8,6,4]
        ):
            ax.vlines(xline,lw=lws,colors='k',ymax=vs,ymin=0)
        plt.show()


    def plot_knn_stats(self):

        # plot knn stuf!!
        fig, ax = plt.subplots(figsize=(16,10))
        best = ax.pcolor(np.argsort(
            np.concatenate([
            self.knn_stats.isel(k_neighbors=nn).kgeprime.values.reshape(
                len(self.knn_stats_red.site.values),-1,1) for nn in range(11)
            ], axis=2), axis=2
        )[:,:,-1], cmap=ListedColormap(get_n_colors('viridis',5)[1]))
        ax.set_yticks(np.arange(len(self.knn_stats_red.site.values))+0.5) # these are the positions, and below the labels = sites
        ax.set_yticklabels(self.knn_stats_red.site.values,fontweight='bold',fontsize=18)
        for ytick,color in zip(ax.get_yticklabels(),locations_colors):
            ytick.set_color(color)
        divider = make_axes_locatable(ax)
        ax_cbar = divider.append_axes("right",size="4%",pad=0.15) 
        cbar = fig.colorbar(best,cax=ax_cbar) # colorbar definition
        cbar.set_ticks([1,3,5,7,9])
        cbar.ax.set_yticklabels(
                [' < 5',' < 10',' < 15',' < 20',' < 25'],fontsize=34,fontweight='bold')
        ax.set_xticks([])
        for xline,vs,lws in zip( # add lines to better visualization
            [np.arange(0,36,18)[1:],np.arange(0,36,9)[1:],np.arange(0,36,3)[1:]],
            [29,20,10],[8,6,4]
        ):
            ax.vlines(xline,lw=lws,colors='k',ymax=vs,ymin=0)
        ax.scatter(
            np.argsort(self.knn_stats_red.kgeprime.values.reshape(
                len(self.knn_stats_red.site.values),-1))[:,-3:].reshape(-1) + 0.5, # last values in array
            np.repeat(np.arange(len(self.knn_stats_red.site.values)),3).reshape(
                len(self.knn_stats_red.site.values),3) + 0.5, # y_positons
            marker='^',facecolors='pink',edgecolors='black',linewidth=1,s=150,
            zorder=10
        )
        plt.show()
        fig, axes = plt.subplots(figsize=(16,5),ncols=3)
        knn_values = np.argsort(np.concatenate([
            self.knn_stats.isel(k_neighbors=nn).kgeprime.values.reshape(
                len(self.knn_stats_red.site.values),-1,1) for nn in range(11)
            ], axis=2), axis=2
        )[:,:,-1:]
        sns.kdeplot(np.argsort(np.concatenate([
            self.knn_stats.isel(k_neighbors=nn).kgeprime_r.values.reshape(
                len(self.knn_stats_red.site.values),-1,1) for nn in range(11)
            ], axis=2), axis=2
        )[:,:,-1:].reshape(-1),
            color='blue',alpha=0.5,label="KGE' - Pearson",shade=True,clip=(0,11),ax=axes[0]
        )
        sns.kdeplot(np.argsort(np.concatenate([
            np.where(
                self.knn_stats.isel(k_neighbors=nn).kgeprime_gamma.values.reshape(
                    len(self.knn_stats_red.site.values),-1,1)<1,
                self.knn_stats.isel(k_neighbors=nn).kgeprime_gamma.values.reshape(
                    len(self.knn_stats_red.site.values),-1,1),
                1-(self.knn_stats.isel(k_neighbors=nn).kgeprime_gamma.values.reshape(
                    len(self.knn_stats_red.site.values),-1,1)-1)
            ) for nn in range(11)
            ], axis=2), axis=2
        )[:,:,-1:].reshape(-1),
            color='green',alpha=0.5,label="KGE' - Gamma",shade=True,clip=(0,11),ax=axes[0]
        )
        sns.kdeplot(np.argsort(np.concatenate([
            np.where(
                self.knn_stats.isel(k_neighbors=nn).kgeprime_beta.values.reshape(
                    len(self.knn_stats_red.site.values),-1,1)<1,
                self.knn_stats.isel(k_neighbors=nn).kgeprime_beta.values.reshape(
                    len(self.knn_stats_red.site.values),-1,1),
                1-(self.knn_stats.isel(k_neighbors=nn).kgeprime_beta.values.reshape(
                    len(self.knn_stats_red.site.values),-1,1)-1)
            ) for nn in range(11)
            ], axis=2), axis=2
        )[:,:,-1:].reshape(-1),
            color='red',alpha=0.9,label="KGE' - Beta",shade=True,clip=(0,11),ax=axes[0]
        )
        sns.kdeplot(np.argsort(np.concatenate([
            self.knn_stats.isel(k_neighbors=nn).kgeprime.values.reshape(
                len(self.knn_stats_red.site.values),-1,1) for nn in range(11)
            ], axis=2), axis=2
        )[:,:,-1:].reshape(-1),
            color='k',alpha=1,label="KGE'",shade=True,clip=(0,11),ax=axes[0]
        )
        axes[0].legend(fontsize=16,loc='upper center')
        models = []
        models_1 = list([models+[
            model+0,model+1,model+2#,model+3,model+4,model+5,model+6,model+7,model+8
        ] for model in np.arange(0,36,9)])
        models_2 = list([models+[
            model+0,model+1,model+2#,model+3,model+4,model+5,model+6,model+7,model+8
        ] for model in np.arange(3,36,9)])
        models_3 = list([models+[
            model+0,model+1,model+2#,model+3,model+4,model+5,model+6,model+7,model+8
        ] for model in np.arange(6,36,9)])
        sns.kdeplot(knn_values[:,np.array(models_1).reshape(-1),:].reshape(-1), 
            color="red", shade=True, clip=(0,11), label='TL 1', ax=axes[1])
        sns.kdeplot(knn_values[:,np.array(models_2).reshape(-1),:].reshape(-1), 
            color="green", shade=True, clip=(0,11), label='TL 2', ax=axes[1])
        sns.kdeplot(knn_values[:,np.array(models_3).reshape(-1),:].reshape(-1), 
            color="blue", shade=True, clip=(0,11), label='TL 3', ax=axes[1])
        axes[1].legend(fontsize=16)
        #models = []
        #models_1 = list([models+[model+0,model+1,model+2] for model in np.arange(0,36,3)])
        #models_2 = list([models+[model+0,model+1,model+2] for model in np.arange(1,36,3)])
        #models_3 = list([models+[model+0,model+1,model+2] for model in np.arange(2,36,3)])
        #sns.kdeplot(knn_values[:,np.array(models_1).reshape(-1),:].reshape(-1), 
        #    color="red", shade=True, clip=(0,11), label='6 H', ax=axes[2])
        #sns.kdeplot(knn_values[:,np.array(models_2).reshape(-1),:].reshape(-1), 
        #    color="green", shade=True, clip=(0,11), label='12 H', ax=axes[2])
        #sns.kdeplot(knn_values[:,np.array(models_3).reshape(-1),:].reshape(-1), 
        #    color="blue", shade=True, clip=(0,11), label='1 D', ax=axes[2])
        #axes[2].legend(fontsize=16)
        sns.kdeplot(knn_values[:,np.arange(0,knn_values.shape[1],3),:].reshape(-1), 
            color="red", shade=True, clip=(0,11), label='local - 1.5$\degree$', ax=axes[2])
        sns.kdeplot(knn_values[:,np.arange(1,knn_values.shape[1],3),:].reshape(-1), 
            color="green", shade=True, clip=(0,11), label='local - 2.5$\degree$', ax=axes[2])
        sns.kdeplot(knn_values[:,np.arange(2,knn_values.shape[1],3),:].reshape(-1), 
            color="blue", shade=True, clip=(0,11), label='regional - NZ', ax=axes[2])
        axes[2].legend(fontsize=16)
        plt.show()


# c'est fini!!