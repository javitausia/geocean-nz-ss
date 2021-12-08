import os
"""
    The user can save here all the useful variables that might be
    used in the project
"""

# useful variables
data_path = os.getenv('SSURGE_DATA_PATH',
                      '/home/javitausia/Documentos/geocean-nz-ss/data')

default_location = (173.9,-40.5) # location at NZ

default_region = (140,190,-70,-20) # region of NZ where pressure, winds... are downloaded
default_region_reduced = (160,185,-52,-30) # reduced region of NZ (similar to moana region)

# to model evaluation
default_evaluation_metrics = [
    'bias','si', # these are just the bias and the scatter index
    'rmse','rel_rmse','ext_rmse','ext_rel_rmse', # all metrics regarding the RMSE
    'pearson','rscore','spearman', # correlation coefficients (they may differ)
    'ext_pearson','ext_rscore','ext_spearman', # exterior corr coefficients
    'pocid','tu_test','expl_var', # some more metrics are computed
    'nse','kge','kgeprime','ext_nse','ext_kge','ext_kgeprime' # hydrologic metrics
] # add more metrics if required
default_ext_quantile = ([0.9,0.95,0.99],0) # quantiles for extreme analysis

