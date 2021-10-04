# basics
import os, sys
import progressbar

# arrays
import numpy as np
import pandas as pd
import xarray as xr

import pickle

# plotting
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs

# append sscode to path
sys.path.insert(0, '/home/metocean/geocean-nz-ss')
data_path = '/data' #'/data/storm_surge_data/'
os.environ["SSURGE_DATA_PATH"] = data_path
#default_region=(140,190,-70,-20)
#default_region_reduced = (160,185,-52,-30)

#sys.path.insert(0, os.path.join(os.path.abspath(''), '..'))

# custom
from .config import data_path, default_location, \
    default_region, default_region_reduced
from .data import Loader
from .utils import calculate_relative_winds
from .data import datasets_attrs

# warnings
import warnings
warnings.filterwarnings('ignore')


import gc

# time                                                                                  
from datetime import datetime

# arrays and math                                                                       
import numpy as np
import xarray as xr

from sklearn.preprocessing import StandardScaler

# custom                                                                                
from .config import default_region_reduced, default_region
from .utils import spatial_gradient, calculate_relative_winds
from .plotting.pca import plot_pcs


class PCA_DynamicPred(object):
    
    
    def __init__(self,
                 pres, pres_vars: tuple = ('SLP','longitude','latitude'),
                 calculate_gradient: bool = False,
                 wind=None,
                 wind_vars: tuple = ('wind_proj_mask','longitude','latitude','ugrd10m','vgrd10m'),
                 time_lapse: int = 1, # 1 equals to NO time delay                    
                 time_resample: str = '1D',
                 region: tuple = (True,default_region),
                 #ss_site: tuple	= (False, None),
                 site_location=None,
                 pca_plot: tuple = (True,False,2),
                 verbose: bool = True,
                 pca_ttls = None,
                 pca_borders = None,
                 pcs_folder = None,
                 site_id = None,
                 pca_percent=0.99,
                 pca_method='cpu'):
        
        self.pres = pres
        self.pres_vars = pres_vars
        self.calculate_gradient = calculate_gradient
        self.wind = wind
        self.wind_vars = wind_vars
        self.time_lapse = time_lapse
        self.time_resample = time_resample
        self.region = region
   #     self.ss_site = ss_site
        self.site_location = site_location
        self.pca_plot = pca_plot
        self.verbose = verbose
        self.pca_ttls = pca_ttls
        self.pca_borders = pca_borders
        self.pcs_folder = pcs_folder
        self.site_id = site_id
        self.pca_percent = 0.99
        self.pca_method = pca_method
        
        self.args = locals()
        
        print("time resample", time_resample)
    

    def pca_assemble_matrix(self):
        
        
        pres = self.pres
        pres_vars = self.pres_vars
        calculate_gradient = self.calculate_gradient
        wind_vars = self.wind_vars
        time_lapse = self.time_lapse
        time_resample = self.time_resample
        region = self.region 

        # if winds then calculate projected winds
        if self.wind:
            
            winds = (True,
                     calculate_relative_winds(
                                    location=self.site_location, # this is the location of site
                                    uw=self.wind[self.wind_vars[3]],
                                    vw=self.wind[self.wind_vars[4]],
                                    lat_name=self.wind_vars[2],
                                    lon_name=self.wind_vars[1]
                                )
                     )
        else:
            winds = (False, None)
        
        
        print('Assembling matrix')
        print('Start', os.system('free -h'))
        # crop slp and winds to the region selected                                         
        if region[0]:
            pres = pres.sel({
                pres_vars[1]:slice(region[1][0],region[1][1]),
                pres_vars[2]:slice(region[1][2],region[1][3])
            })
            if winds[0]:
                print('\n adding the wind to the predictor... \n') if self.verbose else None
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
            print('\n winds predictor with shape: \n {} \n'.format(wind.shape))\
                  if self.verbose else None
            wind_add = 1 # for the pcs matrix                                               
        else:
            wind_add = 0

        # calculate the gradient                                                            
        if calculate_gradient:
            print('\n calculating the gradient of the sea-level-pressure fields... \n')\
                  if self.verbose else None
            pres = spatial_gradient(pres,pres_vars[0]) # from utils.py                      
            print('\n pressure/gradient predictor both with shape: \n {} \n'\
                .format(pres[pres_vars[0]].shape)) if self.verbose else None
            grad_add = 2
        else:
            grad_add = 1
            
        #self.pres = pres
            
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
                    
        # Might not be necessary but in some cases keeping memory usage
        # to a minimum is critical
        del winds
        gc.collect()
                    
        return pcs_matrix, pres.time.values[:-self.time_lapse]
        

    def pcs_plot(self, pcs_data, pcs_scaler):

        if self.pca_plot[0]:
            region_plot = self.region[1] if self.region[0] else default_region
            pca_plot_scale = pcs_scaler if self.pca_plot[1] else None
            plot_pcs(PCA_return,
                     pcs_scaler=pca_plot_scale,
                     n_plot=self.pca_plot[2],
                     region=region_plot,
                     pca_ttls=self.pca_ttls,
                     pca_borders=self.pca_borders,
                     verbose=self.verbose)    


    def pcs_get(self):
        
        if self.pcs_folder:
            # Attempt at loading the PCs
            pca_data, pca_scaler = self.pcs_load()

            if pca_data and pca_scaler:
                print("PCs loaded from file")
                self.pcs_plot(pcs_data, pcs_scaler)
                return pca_data, pca_scaler
            
        
        # PCs need calculating
        pcs_matrix, pcs_time = self.pca_assemble_matrix()
        gc.collect()
            
        pca_data, pca_scaler = self.pca_calculate(pcs_matrix,
                                                  pcs_time,
                                                  method='cpu',
                                                  percent=None)

        # If folder we save the result for later use
        if self.pcs_folder:
            self.pcs_save(pca_data=pca_data,
                          pca_scaler=pca_scaler,
                          pca_percent=self.pca_percent)

        # Plot if required
        self.pcs_plot(pcs_data, pcs_scaler)
                    
        return pca_data, pca_scaler

    
    def pcs_compute_and_save(self):
        
        pcs_matrix, pcs_time = self.pca_assemble_matrix()
            
        pca_data, pca_scaler = self.pca_calculate(pcs_matrix,
                                                  pcs_time,
                                                  method='cpu',
                                                  percent=None)
        
        self.pcs_save(pca_data=pca_data,
                      pca_scaler=pca_scaler,
                      pca_percent=self.pca_percent)

        
    def _pca_file_name(self):
        
        name_attrs = []

        if self.region[1] == default_region:
            name_attrs.append('default_region')
        elif self.region[1] == default_region_reduced:
            name_attrs.append('default_region_reduced')
        else:
            name_attrs.append('local_'+str((self.region[1][1]-self.region[1][0])/2.)+'_'+str((self.region[1][3]-self.region[1][2])/2.))

        if (self.region[1] != default_region and self.region[1] != default_region_reduced)\
            or self.wind: # PCA is site dependent
            name_attrs.append(str(self.site_id))

        if self.wind:
            name_attrs.append('winds')
        else:
            name_attrs.append('no_winds')

        if self.calculate_gradient:
            name_attrs.append('gradients')
        else:
            name_attrs.append('no_gradients')

        name_attrs.append(self.time_resample)

        name_attrs.append("tl"+str(self.time_lapse))

        print(self.pcs_folder, name_attrs)
        
        return os.path.join(self.pcs_folder, "_".join(name_attrs) + ".nc"),\
               os.path.join(self.pcs_folder, "_".join(name_attrs) + ".pickle")
    
    
    def _pca_calculate_cpu(self, pcs_stan):
        from sklearn.decomposition import PCA
        
        print("Computing using CPU")
        
        pca_fit = PCA(n_components=min(pcs_stan.shape[0],
                                       pcs_stan.shape[1]))
        
        PCs = pca_fit.fit_transform(pcs_stan)
    
        return PCs, pca_fit.components_, pca_fit.explained_variance_
    
    
    def _pca_calculate_cpu_distributed(self, pcs_stan):
        # Distributed version
        import dask.array as dask_array
        from dask_ml.decomposition import PCA
        
        # Turn numpy array into dask array                                              
        dask_pcs_stan = dask_array.from_array(pcs_stan,
                                              chunks=(pcs_stan.shape[0],
                                                      pcs_stan.shape[1]//32))

        pca_fit = PCA(n_components=min(pcs_stan.shape[0],
                                       pcs_stan.shape[1]))
        
        PCs = pca_fit.fit_transform(dask_pcs_stan)
    
        return PCs, pca_fit.components_, pca_fit.explained_variance_
    
    
    def _pca_calculate_gpu(self, pcs_stan):
        # GPU version
        #import pandas as pd
        #import cudf
        import cuml
        from cuml.decomposition import PCA
        
        cuml.common.memory_utils.set_global_output_type('numpy')

        pca_fit = PCA(n_components = min(pcs_stan.shape[0],
                                         pcs_stan.shape[1]))

        # First we convert the matrix into a dataframe on the GPU
        import pandas as pd
        import cudf
        #def np2cudf(df):
        #    # convert numpy array to cuDF dataframe
        #    df = pd.DataFrame({'fea%d'%i:df[:,i] for i in range(df.shape[1])})
        #    pdf = cudf.DataFrame()
        #    for c,column in enumerate(df):
        #        pdf[str(c)] = df[column]
        #    return pdf
        #pcs_stan_df = np2cudf(pcs_stran)

        #return pca_fit.fit_transform(pcs_stan_df)
        
        PCs = pca_fit.fit_transform(pcs_stan)
        
        return PCs, pca_fit.components_, pca_fit.explained_variance_
    
    
    def _pca_calculate_sklearn(self,
                                pcs_stan):
        from sklearn.decomposition import PCA
        
        pca_fit = PCA(n_components=min(pcs_stan.shape[0],
                                       pcs_stan.shape[1]))
        
        PCs = pca_fit.fit_transform(pcs_stan)
        
        return PCS, pca_fit.components_, pca_fit.explained_variance_
    
    
    def pca_calculate(self,
                      pcs_matrix,
                      pcs_time,
                      method='cpu',
                      percent=None):
        
        # standarize the features                                                           
        pcs_scaler = StandardScaler()
        pcs_stan = pcs_scaler.fit_transform(pcs_matrix)
        pcs_stan[np.isnan(pcs_stan)] = 0.0 # check additional nans                          

        # ---------------------------------------------------------------------------------\
        #                                                                                      
        # THIS IS JUST TO ALLOW MY 32GB-RAM COMPUTER TO RUN                                 
        # if pcs_stan.shape[1]>18000:                                                       
        #     pcs_stan = pcs_stan[:,::12]                                                   
        # elif pcs_stan.shape[1]>10000:                                                     
        #     pcs_stan = pcs_stan[:,::8] # if pcs_stan.shape[0]<20000 else pcs_stan[::2,::8]                                                                                       
        # elif pcs_stan.shape[1]>6000:                                                      
        #     pcs_stan = pcs_stan[:,::6] # if pcs_stan.shape[0]<20000 else pcs_stan[::2,::6]                                                                                       
        # elif pcs_stan.shape[1]>2000:                                                      
        #     pcs_stan = pcs_stan[:,::4] # if pcs_stan.shape[0]<20000 else pcs_stan[::2,::4]                                                                                       
        # ---------------------------------------------------------------------------------\
 
        # calculate de pca                                                                 
        print('\n calculating PCs matrix with shape: \n {} \n'.format(pcs_stan.shape)) \
            if self.verbose else None
        if self.pca_method == 'cpu':
            PCs, components_,  explained_variance_ = self._pca_calculate_cpu(pcs_stan)
        elif self.pca_method == 'distributed':
            PCs, components_,  explained_variance_ = self._pca_calculate_cpu_distributed(pcs_stan)
        elif self.pca_method == 'gpu':
            PCs, components_,  explained_variance_ = self._pca_calculate_gpu(pcs_stan)
        else: # Distributed version                                                         
            print("Uknown method {}".format(method))

        print("matrix", pcs_matrix.shape)
        print("PCs", PCs.shape)
        print("comp", components_.shape)
        print("var", explained_variance_.shape)
        print("lon", self.pres[self.pres_vars[1]])
        print("lat", self.pres[self.pres_vars[2]])
        print("time", self.pres.time.values[:-self.time_lapse].shape )
                                                                           
        return xr.Dataset(
            data_vars = {
                'PCs': (('time','n_components'), PCs),
                'EOFs': (('n_components','n_features'), components_),
                'variance': (('n_components'), explained_variance_),
                'pcs_lon': (('n_lon'), self.pres[self.pres_vars[1]].values),
                'pcs_lat': (('n_lat'), self.pres[self.pres_vars[2]].values)
            },
            coords = {
                'time': pcs_time
                #'time': self.pres.time.values[:-self.time_lapse]
            }
        ), pcs_scaler
    
    
    def pcs_load(self):
        
        pca_data_file, pca_scaler_file = self._pca_file_name()
        print("FILE", pca_data_file)
        
        if not os.path.isfile(pca_data_file) or\
            not os.path.isfile(pca_scaler_file):
            return None, None
        
        # open a file, where you stored the pickled data
        with open(pca_scaler_file, 'rb') as fin:
            pca_scaler = pickle.load(fin)
            
        pca_data = xr.open_dataset(pca_data_file)

        return pca_data, pca_scaler
    
    
    def pcs_save(self,
                  pca_data,
                  pca_scaler,
                  pca_percent=None):
        """
        This function saves the results of the PCA calculations and the scaler in files.
        As the results can be very large, the function offers the option of only keeping
        a subset of them that accounts for a given percentage of the variance.
        
        Args:
            pca_data (xarray dataset): The xarray dataset that contains the pca
            pca_scaler (Scikitlearn scaler): The scaler used to normalise the data prior to
                                             PCA analysis
            pca_percent (double): A double between 0-1 corresponding the the proportion
                                   of explained variance the pca need saving for.
        """
        
        pca_data_file, pca_scaler_file = self._pca_file_name()
        
        if not pca_percent is None:
            pca_cutoff = np.where((pca_data.variance/pca_data.variance.sum()).cumsum().values > pca_percent)[0][0]+2
            pca_data = pca_data.isel(n_components=slice(0,pca_cutoff))
        
        # Store data to drive
        pca_data.to_netcdf(pca_data_file)
                
        # Store pca_scaler
        file_to_store = open(pca_scaler_file, "wb")
        pickle.dump(pca_scaler, file_to_store)
        file_to_store.close()
