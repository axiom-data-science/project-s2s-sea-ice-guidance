# Notes
- 2021-04-20 - Started extration job with two nodes on dask-cluster staging hosts
- 2021-04-23/24 - Wrong locations being extracted? lon using different numbers in nc files and csv file (-180,180 vs. 0,360)
- 2021-04-24 - Started job with correct locations on dask-cluster prod hosts, but some won't launch due to Python/Ansible issue on host (told Shane)
- 2021-04-26 - Dask cluster stopped responding at STLISW10 station, relaunched using remainder-ak-ice-locs.csv on prod hosts


# run script 
- create tmux session
- run docker container with same image as cluster: `sudo docker run -v /mnt/store/data/assets/s2s/data-exploration:/srv -v /mnt/store/data/:/data -it registry.axiom/dask-cluster:s2s bash`
- run command: `/env/bin/python make-nc.py` 

# Notebooks:
- convert-365-to-leap-calendar.ipynb: Develop method to convert calendars
- sea-ice-analysis.ipynb: Initial exploration of data

## ARIMA like models:
- evaluate-arima-models.ipynb: Statsmodels based models
- make-s2s-model-01.ipynb
- make-s2s-model-02.ipynb
- s2s-model-grid-search-dask.ipynb
- s2s-model-grid-search.ipynb

## Prophet models:
- evaluate-prophet.ipynb: Preliminary + secondary development of models
- make-prophet-station-models.ipynb: Make models for every station using best model developed in evaluate-prophet.ipynb
