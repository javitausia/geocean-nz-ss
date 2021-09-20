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
    'bias','si','rmse','rel_rmse','pearson','rscore',
    'ext_rmse','ext_rel_rmse','ext_pearson',
    'pocid','tu_test','expl_var'
] # add more metrics if required
