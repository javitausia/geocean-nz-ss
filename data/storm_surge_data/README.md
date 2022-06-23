This repository contains the data used in the MBIE funded smart ideas research project.

---------------------------------------------------

The content of the different folders is the following:
- **moana_hindcast_v2**: Numerical data from version 2 of the Moana hindcast generated at MetOcean
- **nz_tidal_gauges**:   Observation data collected for various tidal gauges around New Zealand

Below, the daily validation metrics for different locations in the hindcast where UHSLC tidal gauges are located can be seen:

| **null**                      | Bias   | Si    | Rmse  | Rmse_99 | Pearson | Spearman | Kge   |
|:-----------------------------:|:------:|:-----:|:-----:|:-------:|:-------:|:--------:|:-----:|
| **H076a_Taranaki - 999**      | 0.001  | 0.338 | 0.034 | 0.093   | 0.935   | 0.93     | 0.912 |
| **H072a_Bluff - 116**         | 0.002  | 0.357 | 0.046 | 0.126   | 0.925   | 0.925    | 0.904 |
| **H071a_Wellington - 689**    | 0.003  | 0.379 | 0.038 | 0.094   | 0.916   | 0.912    | 0.828 |
| **H668a_Napier - 949**        | 0.002  | 0.394 | 0.033 | 0.06    | 0.911   | 0.907    | 0.86  |
| **H667a_Lyttelton - 480**     | 0.005  | 0.382 | 0.043 | 0.114   | 0.917   | 0.913    | 0.8   |
| **H077a_Nelson - 708**        | -0.001 | 0.44  | 0.036 | 0.071   | 0.889   | 0.884    | 0.876 |
| **H403a_JacksonBay - 393**    | -0.005 | 0.426 | 0.043 | 0.093   | 0.909   | 0.908    | 0.789 |
| **H398a_MarsdenPoint - 1327** | 0.001  | 0.415 | 0.029 | 0.084   | 0.902   | 0.88     | 0.857 |
| **H073a_Tauranga - 1124**     | 0.0    | 0.424 | 0.03  | 0.063   | 0.896   | 0.888    | 0.879 |
| **H403b_JacksonBay - 393**    | -0.007 | 0.442 | 0.043 | 0.062   | 0.91    | 0.908    | 0.741 |
| **H665a_Timaru - 328**        | 0.001  | 0.397 | 0.047 | 0.145   | 0.909   | 0.911    | 0.875 |
| **H669a_PortChalmers - 224**  | 0.004  | 0.381 | 0.043 | 0.111   | 0.918   | 0.916    | 0.813 |