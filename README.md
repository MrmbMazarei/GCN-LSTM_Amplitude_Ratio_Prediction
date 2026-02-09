# GCN-LSTM_Amplitude_Ratio_Prediction

For this model please reference to: 
Behbahani, M. R. M., Rey, D. M., Briggs, M. A., & Bagtzoglou, A. C. (2025). A spatiotemporal deep learning approach for predicting daily air-water temperature signal coupling and identification of key watershed physical parameters in a montane watershed. Journal of Hydrology, 134139.

This repository is related to the paper entitled "A Spatial-Temporal Deep Learning Approach for Predicting Daily Air-Water Temperature Signal Coupling and Identification of Key Watershed Physical Parameters in a Montane Watershed".

Abstract
Seasonal shifts from runoff to groundwater dominance influence daily headwater stream temperatures, especially where local groundwater input is strong. This input buffers temperature during hot periods, supporting cold-water habitats. Recent studies use air–water temperature signal metrics to identify zones of strong stream–groundwater connectivity. While Previous studies used air–water signal ratios as proxies for groundwater influence but were limited to specific sites and periods, without dynamic forecasting. This study is the first to model the daily amplitude ratio (Ar) as a continuous spatiotemporal signal using deep learning, enabling fine-scale tracking of groundwater–stream interactions. Graph convolutional network – long-short term memory (GCN-LSTM) was used in this study to predict Ar with high accuracy, achieving an R² (NSE, RMSE) of 0.86 (0.73, 0.0004) for one-day-ahead to 0.52 (0.50, 0.0009) for seven-days ahead forecasts. Prior studies often have not explicitly incorporated spatial hydrogeologic drivers (e.g., depth to bedrock, Topographic Wetness Index (TWI)), but this model explicitly incorporates them to assess their impact on Ar forecasting and stream-groundwater connectivity.   Feature analysis identified mean sand, elevation, slope, clay, and TWI as key predictors of Ar. Stronger groundwater signals appeared in hillslopes, high elevations, and small tributaries, highlighting the influence of watershed characteristics on streamflow. Unlike previous studies relying on measured in-situ stream and air temperature, this study forecasts Ar directly from climate and physiographic features, avoiding in-situ data requirements. Findings reveal key drivers of seasonal runoff–groundwater shifts, aiding predictions of stream ecosystem resilience.


This spatiotemporal model is used to forecast Amplitude Ratio, a proxy for groundwater-surface water interaction.
1. For Ar data plese cite to:
   Rey, D. M., Hare, D. K., Fair, J. H., & Briggs, M. A. (2024). Diel temperature signals track seasonal shifts in localized groundwater contributions to headwater streamflow generation at network scale. Journal of Hydrology, 639, 131528.
   
3. For AT and hydrometeorological data please cite to the following website:
   https://waterdata.usgs.gov/monitoring-location/01434498/

4. For stream temperature (WT) data please cite to:
   Terry, N., Briggs, M.A, Kushner, D., Dickerson, H., Baldwin, A., Trottier, B., Haynes, A., Besteder, C., Glas, R., Doctor, D., Gazoorian, C., Odom, W., Benton, J., and Fleming, B., 2022, Stream temperature, dissolved radon, and stable water isotope data collected along headwater streams in the upper Neversink River watershed, NY, USA (ver. 3.0, June 2025): U.S. Geological Survey data release, https://doi.org/10.5066/P9R3TYOZ.
   
6. For "DRB_1000m_wedge_vars_Neversink" please cite to the following references:
   Benton, J. R., and D. H. Doctor. 2025. “Dynamic Baseflow Storage Estimates and the Role of Topography, Geology and Evapotranspiration on Streamflow Recession Characteristics in the Neversink Reservoir Watershed, New York.” Hydrol. Process., 39 (3): e70106. https://doi.org/10.1002/hyp.70106.
   Doctor, D. H., W. E. Odom, and J. R. Benton. n.d. “Hydrogeomorphic Map of the Neversink Reservoir Watershed, New York | U.S. Geological Survey.” Accessed February 24, 2025. https://www.usgs.gov/data/hydrogeomorphic-map-neversink-reservoir-watershed-new-york.
   Glas, R. L., M. Briggs, N. C. Terry, C. L. Gazoorian, and D. H. Doctor. 2021b. “Depth to bedrock determined from passive seismic measurements, Neversink River watershed, NY (USA).” U.S. Geological Survey.
   “NHDPlus High Resolution | U.S. Geological Survey.” n.d. Accessed July 10, 2025. https://www.usgs.gov/national-hydrography/nhdplus-high-resolution.

7. For the GCN_LSTM model, please cite to this paper (Citation will be provided after aceptance).

