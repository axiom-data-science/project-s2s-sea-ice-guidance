# S2S - Probabilisitc Sea Ice Guidance

## Summary

The goal of the project is to develop models that can predict sea ice at a given set of locations from
3 weeks to 6 months in advance trained on CFSv2 with the CPC experimental coupled ice model.

## Project links

[Jira](https://axds.atlassian.net/jira/software/projects/S2S/boards/69)
[Confluence](https://axds.atlassian.net/wiki/spaces/LEM/pages/846200880/S2S+Probabilistic+Sea+Ice+Guidance)

## Data 

### Raw

1. CFSv2 with experimental CPC ice model
  - Contact: Wanqiu Wang (NOAA/NWS/NCEP), wanqiu.wang@noaa.gov
  - Delivery: Uploaded via ftp to backup1
  - Path: /mnt/store/data/assets/s2s/data/45day
  - Format: grib
  - Description:
    - `ocn` - Ocean fields from 2019 - 2020
    - `sic` - Sea ice from 2012 - 2020
    - `sst` - Sea surface temperature from 2012 - 2020
    - Daily files extracted from 45 day forecasts

2. CFSv2 with experimental CPC ice model
  - Contact: Wanqiu Wang (NOAA/NWS/NCEP), wanqiu.wang@noaa.gov
  - Delivery: Uploaded via ftp to backup1
  - Path: /mnt/store/data/assets/s2s/data/seasonal
  - Format: grib
  - Description:
    - Daily files extracted from seasonal forecasts
    - Data not used in the project

3. Stations of interest
  - Contact: Eugene Petrescu (NOAA/NWS), eugene.m.petrescu@noaa.gov
  - Delivery: Emailed
  - Path: /mnt/store/data/assets/s2s/data/ak_ice_locs.csv
  - Format: csv
  - Desription:
    - List of stations where model predictions would be useful

### Processed

1. Extracted CFS data at station locations from
  - Contact: Jesse
  - Process:
    - Extract stations:
      - Run `make-nc`
        - Path: /mnt/store/data/assets/s2s/src/data-preprocessing-scripts/make-nc.py
        - Environment: Used S2S from `dask-cluster` repo as environment and for Dask
  - Path: /mnt/store/data/assets/s2s/data/station-data
  - dvc: true 

## Models
