#!python
"""Create extraction station time-series from CFSV2-exp"""
from pathlib import Path

import geopandas as gdp
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from dask import delayed
from dask.distributed import Client, progress
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.seasonal import STL

register_matplotlib_converters()



locs = pd.read_csv("/data/assets/s2s/data/ak_ice_locs.csv")
# convert from -180, 180 -> 0, 360
locs.lon = (locs.lon + 360) % 360

var_map = {
    'ocn': {
        'temp': 'TMP_GDS0_SFC_ave6h',
        'salt': 'SALTY_GDS0_DBSL_ave6h',
        'ice_thk': 'ICETK_GDS0_SFC_ave6h',
    },
    'sic': {
        'ice_conc': 'ICE_C_GDS0_SFC_ave6h'
    },
    'sst': {
        'sst': 'POT_GDS0_DBSL_ave6h'
    }
}

ocn_files = list(Path('/data/assets/s2s/data/45day/ocn/').glob('**/*ocn*'))
ocn_files.sort()
ocn_files = [str(f) + '.grb' for f in ocn_files]

sic_files = list(Path('/data/assets/s2s/data/45day/sic/').glob('**/*sic*'))
sic_files.sort()
sic_files = [str(f) + '.grb' for f in sic_files]

sst_files = list(Path('/data/assets/s2s/data/45day/sst/').glob('**/*sst*'))
sst_files.sort()
sst_files = [str(f) + '.grb' for f in sst_files]


files = {
    'ocn': ocn_files,
    'sic': sic_files,
    'sst': sst_files
}

client = Client('estuaries01.ib.axiomptk:7030')

# Anywhere from 100 to 250 ms to extract a single time step from a file.
def get_station(fname, var='temp', coords=(-160.05500, 70.6870), var_map=var_map):
    if 'ocn' in fname:
        var = var_map['ocn'][var]
    elif 'sic' in fname:
        var = var_map['sic'][var]
    elif 'sst' in fname:
        var = var_map['sst'][var]
    else:
        raise ValueError(f'Cannot identify file type. fname={fname}')
    
    lon, lat = coords
    with xr.open_dataset(fname, engine='pynio') as ds:
        return ds[var].isel(initial_time0_hours=0).sel(dict(g0_lat_1=lat, g0_lon_2=lon), method='nearest')


# Try map-reduce the xr.concat
@delayed
def agg(x, y):
    return xr.concat([x, y], dim='initial_time0_hours')


def bfs_merge(seq):
    if len(seq) < 2:
        return seq
    middle = len(seq) // 2
    left = bfs_merge(seq[:middle])
    right = bfs_merge(seq[middle:])
    if not right:
        return left
    return [agg(left[0], right[0])]


for row in locs.itertuples():
    # row.name, row.lon, row.lat

    # (files, var)
    files_vars = [
        (sic_files, 'ice_conc'),
        (ocn_files, 'temp'),
        (ocn_files, 'salt'),
        (ocn_files, 'ice_thk'),
        (sst_files, 'sst')
    ]
    for fv in files_vars:
        files, var = fv
        print(f'Extracting {row.name} ({row.lon}, {row.lat}):{var}')
        try:
            futures = client.map(get_station, files, var=var, coords=(row.lon, row.lat))
            results = client.gather(iter(futures))
            results = list(results)
            merge = bfs_merge(results)[0]
            merge = merge.compute()

            fname = f'{row.name}-{var}.nc'
            merge.to_netcdf(fname)
        except Exception as e:
            print(f"ERROR: {e}")

