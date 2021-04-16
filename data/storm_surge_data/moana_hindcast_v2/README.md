This folder contains processed time series of water level extracted along the coastline in the Moana model (hindcast v2).
Information on the numerical model are the following:
- Domain: New Zealand
- Model: ROMS
- Atmospheric forcing: CFSR
- Tidal components used in forcing: M4, K2, S2, M2, N2, K1, P1, O1, Q1, Mf, Mm
- Spacial resolution: ~ 5km
- Output time resolution: hourly
- Main contact for the data: Joao De Souza (j.souza@metocean.co.nz)

File description:
- **moana_coast.tar.gz**: The file contains processed sea level data for all points of the model grid adjacent to the coastline. Data points are store in the file as "sites" in an unordered manner. The tarball contains a dataset in zarr format which can be opened using the python library xarray (function open_zarr). The following variables are available in the file:
   * *lon*: The longitude of the nodes at which the water level data were extracted.
   * *lat*: The latitude of the nodes at which the water level data were extracted.
   * *elev*: The raw sea surface height from the model in meters.
   * *trend*: The linear trend of the water elevation timeseries in meters.
   * *tide* is the astronomical tides in meter extracted by fitting the water elevation time series using the python software toto which uses utide. Only the modes used to force the model were used to fit the signal.
   * *msea* The monthly mean sea level variation in meters extracted using a second order cosine-Laczos filter with cut-off period of 30 days.
   * *ss* is the storm surge in meters extracted using a second order cosine-Laczos filter of cut-off period of 40 hours.
   * *res* is the residual in meters left after filetring out all the other components.

