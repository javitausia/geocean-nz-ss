# common fig parameters
_figaspect = 2
_figsize = (18,5)
_figdpi = 128
_figsize_width = 4
_figsize_height = 5

# common font parameters
_fontsize_label = 12
_fontsize_legend = 14
_fontsize_title = 18

# pressure difference to plot
_mbar_diff = 20
_mbar100_diff = 2000

import cmocean
# scatter and qqplot colors
scatter_cmap =  'rainbow' # cmocean.cm.haline
qqplot_points_col = 'red'
qqplot_edges_col = 'orange'

# time plots colors
real_obs_col = 'k'
pred_val_col = 'green'

# DWTs colors list
dwts_colors = [
    'white','cyan','cornflowerblue','darkseagreen','olivedrab','gold','darkorange',
    'orangered','red','deeppink','violet','darkorchid','purple','midnightblue'
]

# select colors for each location depending on the shore
locations_colors = [
    'orange','orange','red','red','gold','blue','darkred','darkred','darkred','blue','pink',
    'purple','purple','pink','purple','pink','lightgreen','pink','lightgreen','orchid','orchid','orchid',
    'lightgreen','orchid','green','lightgreen','green','green','darkgreen'
]
locations_colors_letters = [
    'black','black','black','black','black','white','white','white','white','white','black',
    'white','white','black','white','black','black','black','black','black','black','black',
    'black','black','white','black','white','white','white'
]
# [ 116,  200,  224,  328,  378,  393,  480,  488,  578,  613,  689,  708,
#   744,  780,  803,  949,  999, 1025, 1064, 1124, 1146, 1174, 1177, 1214,
#   1217, 1260, 1296, 1327, 1442
# ]

# these deltas are used to correctly plot the text in the locations map
delta_lons = [ 0.35,  # 116
              -1.2,   # 200
               0.45,  # 224
               0.4,   # 328
              -1.2,   # 378
              -1.25,  # 393
               0.55,  # 480
               0.65,  # 488
               0.5,   # 578
              -0.65,  # 613 
               0.5,   # 689
              -0.8,   # 708
               0.35,  # 744
               0.45,  # 780
              -0.9,   # 803
               0.45,  # 949
              -1.3,   # 999
               0.5,   # 1025
              -1.5,   # 1064
              -0.9,   # 1124
               0.4,   # 1146
               0.9,   # 1174
              -1.55,  # 1177
               1.1,   # 1214
               0.3,   # 1217
              -1.55,  # 1260
               0.3,   # 1296
              -0.9,   # 1327
              -0.8    # 1442
]
delta_lats = [-0.65,  # 116
              -0.6,   # 200
              -0.4,   # 224
              -0.5,   # 328
              -0.8,   # 378
               0.3,   # 393
              -0.85,  # 480
              -0.05,  # 488
              -0.25,  # 578
               0.6,   # 613 
              -0.4,   # 689
              -0.8,   # 708
               0.7,   # 744
              -0.2,   # 780
               0.7,   # 803
              -0.5,   # 949
               0.1,   # 999
              -0.3,   # 1025
              -0.1,   # 1064
              -0.8,   # 1124
               0.3,   # 1146
               0.1,   # 1174
              -0.3,   # 1177
               0.6,   # 1214
               1.1,   # 1217
              -0.15,  # 1260
               1.3,   # 1296
               0.6,   # 1327
               0.5    # 1442
]

metrics_cmaps = { # ascendig = True
    'expl_var': 'Spectral', 'me': 'plasma_r', 'mae': 'hot', 'mse': 'hot', 'ext_rmse': 'plasma_r',
    'medae': 'hot', 'tweedie': 'jet', 'bias': 'pink_r', 'si': 'Spectral_r',
    'rmse': 'plasma_r', 'pearson': 'Spectral', 'spearman': 'jet', 'rscore': 'Spectral',
    'ext_pearson': 'Spectral', 'rel_rmse': 'hot', 'ext_rel_rmse': 'hot',
    'tu_test': 'jet', 'pocid': 'jet', 'final_metric': 'Spectral',
    'pearson_99':'Spectral','ext_kgeprime_99':'Spectral','kgeprime':'Spectral',
    'kge':'Spectral','kgeprime_r':'Spectral','kgeprime_beta':'bwr','kgeprime_gamma':'Spectral',
}