# S2S - Probabilistic Sea Ice Guidance

## Project Summary

This repo is an archive of code, data, models, and notebooks for the [S2S - Proababilistic Sea Ice Guidance Project](http://stage-s2s.srv.axiomptk).

The goal of the project was to develop a method to provide week 3 to seasonal (S2S) sea ice guidance for predefined locations based on historical output from the operational NCEP Climate Forecast System Version 2 (CFSv2) coupled with an experimental sea ice model (CPC).

### Training data

Historical model results for sea ice concentration and air temperature were ingested from the model for the years 2012 to 2020 to provide the basis of the sea ice guidance model development. Time series of model results at 315 points of interest identified by NWS were extracted from the model and saved to netCDF files.

### Model development

The extracted station based time series signals were then used to train a regression based model to provide daily sea ice concentration forecasts for the next year.

### Model assessment

The sea ice concentration RMSE over every station derived from a cross-validation horizon of 365 days based on 8 years of training data had a mean value of 0.138, a minimum of 0.008, and a maximum of 0.434. The station with the highest average RMSE (N73W145) had an average RMSE of 0.26. Generally, the highest error correspond to a difference in phase between ice growth and retreat and that predicted by the model.  Skill metrics for individual stations are saved in `station-models` in text files on a per station file and in a compiled text file `data/station-skill.txt`.

## Repo Content

### Data 

`ak-ice-locs.csv` conatiners the locations of stations from which training data was extracted from CFS and models were trianed.

The directory `station-data` includes the data extracted from CFS at the locations defined in `data/ak-ice-locs.csv`.

A summary table of model RMSE per station is saved in `data/station-skill.txt`.

A browesable Bokeh map of the stations and model skill is available in `data/station-map-rmse-mean.html`.

### Notebooks

`evalute-prophet.ipynb` contains code from the evaulation and evolution of `prophet` based models.

`make-prophet-station-models.ipynb` contains code from the creation of `prophet` based models for each station.

### Code (`src`)

`src/make-nc.py` extracts the data at the locations defined in `ak-ice-locs.csv` from the model output and saves as station netCDF files.
`src/experiements` contains code for the evaluation of various model configurations for the `prophet` include integration with an internal `MLflow` instance.

### Models

`station-models` contains the `prophet` models trained for each station which can be loaded and used for predictions.  Skill scores for each station are saved in text files on a per station basis and a summary table of model RMSE is available in `data/station-skill.txt`.
