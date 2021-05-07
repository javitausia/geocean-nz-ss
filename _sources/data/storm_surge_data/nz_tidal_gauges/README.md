This folder contains raw and processed data originating from various tidal gauges around the country.

The data is stored in 4 subfolders that are representative of the source of the data:
- **uhslc** contains hourly research grade data downloaded from the University of Hwai Sea Level Center (https://uhslc.soest.hawaii.edu/datainfo/)
- **linz** contains high resolution data downloaded from the Land Information New Zealand Website (LINZ) (https://sealevel-data.linz.govt.nz/index.html)
- **other** contains high resolution data that were sourced by approaching various regional councils and port authorities around New Zealand (private)
- **geocean** contains the processed tidal gauges used in previous works (private)

Each subfolder has the following structure:
- a **raw** folder that contains the data in a shape as close as possible to the original data (i.e. if the data was downloaded as NetCDF it is left in its original shape, if it was obtained in either other shape it was packed in NetCDF 4 format before storing in the folder.
- a **processed** folder that contains the data in processed form. This means that datum was identified, spikes removed, gaps filled when appropriate, data were resampled to the appropriate frequency. Data were then processed to extract different signals. Each file should contain the following variables:

   * *lon*: The longitude of the nodes at which the water level data were extracted.
   * *lat*: The latitude of the nodes at which the water level data were extracted.
   * *elev*: The raw sea surface height from the model in meters.
   * *trend*: The linear trend of the water elevation timeseries in meters.
   * *tide* is the astronomical tides in meter extracted by fitting the water elevation time series using the python software toto which uses utide. Only the modes used to force the model were used to fit the signal.
   * *msea*: The monthly mean sea level variation in meters extracted using a second order cosine-Laczos filter with cut-off period of 30 days.
   * *ss* is the storm surge in meters extracted using a second order cosine-Laczos filter of cut-off period of 40 hours.
   * *res* is the residual in meters left after filetring out all the other components.
