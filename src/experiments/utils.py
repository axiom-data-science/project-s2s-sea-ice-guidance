import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from prophet import serialize

STATIONS = [
    'PAWIC'
    'PAWI10'
    'PAWI20'
    'PABRC'
    'PTBRC'
    'PTBR10'
    'PTBR20'
    'LNLAC'
    'LNLA10'
    'LNLA20'
    'NUQTC'
    'NUQT10'
    'NUQT20'
    'PABAC'
    'PABA10'
    'PABA20'
    'PAHPC'
    'PAHP10'
    'PAHP20'
    'PAOMC'
    'PAOM10'
    'PAOM20'
    'PAOT'
    'DEERC'
    'DEER10'
    'DEER20'
    'KTZSM'
    'PAOT'
    'PAOT10'
    'PAPO10'
    'PAPO20'
    'PAPOC'
    'ICYPC'
    'ICYPC10'
    'ICYPC20'
    'KIVAC'
    'KIVA10'
    'KIVAC20'
    'PPIZC'
    'PPIZ10'
    'PPIZ20'
    'PASCC'
    'PASC10'
    'PASC20'
    'PTTPC'
    'PTTP10'
    'PTTP20'
    'CMPTC'
    'CMPT10'
    'CMPT20'
    'DMPTC'
    'DMPT10'
    'DMPT20'
    'PASHC'
    'PASH10'
    'PASH20'
    'BRSTC'
    'BRST10'
    'DIOM2'
    'PATCC'
    'PASNC'
    'PASN10'
    'PASN20'
    'PASGC'
    'PASG10'
    'PASG20'
    'PASAC'
    'PASA10'
    'PASA20'
    'PAGMC'
    'PAGM10'
    'PAGM20'
    'PATGC'
    'PATG10'
    'PATG20'
    'PATG40'
    'PAUNC'
    'PAUN10'
    'PAUN20'
    'PTCRC'
    'PTCR10'
    'PTCR20'
    'NRTBC'
    'YKRDC'
    'YKRD10'
    'YKRD20'
    'PAMYC'
    'PAMY10'
    'PAMY20'
    'NUNVC'
    'NUNV10'
    'NUNV20'
    'MRSTC'
    'NECPC'
    'NECP10'
    'NECP20'
    'PADLC'
    'PADL20'
    'PAKNC'
    'PAENC'
    'PAEN10'
    'PAEN20'
    'PLPTC'
    'PLPT10'
    'PLPT20'
    'PAPHC'
    'PAPH10'
    'PAPH20'
    'NSLGC'
    'NSLG10'
    'NSLG20'
    'FLPSC'
    'FLPS10'
    'FLPS20'
    'STMNE'
    'STMSW'
    'STMSE'
    'STMNW'
    'STLISE'
    'STLISE10'
    'STLISE20'
    'STLISW'
    'STLISW10'
    'STLISW20'
    'NRTS1'
    'NRTS2'
    'NRTS3'
    'NRTS4'
    'NRTS6'
    'NRTS7'
    'NRTS8'
    'N77W140'
    'N77W145'
    'N77W150'
    'N77W155'
    'N77W160'
    'N77W165'
    'N77W170'
    'N76W140'
    'N76W145'
    'N76W150'
    'N76W155'
    'N76W160'
    'N76W165'
    'N76W170'
    'N75W1375'
    'N75W140'
    'N75W1425'
    'N75W145'
    'N75W1475'
    'N75W150'
    'N75W1525'
    'N75W155'
    'N75W1575'
    'N75W160'
    'N75W1625'
    'N75W165'
    'N75W1675'
    'N75W170'
    'N75W1725'
    'N74W1375'
    'N74W140'
    'N74W1425'
    'N74W145'
    'N74W1475'
    'N74W150'
    'N74W1525'
    'N74W155'
    'N74W1575'
    'N74W160'
    'N74W1625'
    'N74W165'
    'N74W1675'
    'N74W170'
    'N74W1725'
    'N73W1375'
    'N73W140'
    'N73W1425'
    'N73W145'
    'N73W1475'
    'N73W150'
    'N73W1525'
    'N73W155'
    'N73W1575'
    'N73W160'
    'N73W1625'
    'N73W165'
    'N73W1675'
    'N73W170'
    'N73W1725'
    'N72W1375'
    'N72W140'
    'N72W1425'
    'N72W145'
    'N72W1475'
    'N72W150'
    'N72W1525'
    'N72W155'
    'N72W1575'
    'N72W160'
    'N72W1625'
    'N72W165'
    'N72W1675'
    'N72W170'
    'N72W1725'
    'N71W1375'
    'N71W140'
    'N71W1425'
    'N71W145'
    'N711W1475'
    'N712W150'
    'N714W1525'
    'N715W155'
    'N714W1575'
    'N713W160'
    'N712W1625'
    'N71W165'
    'N71W1675'
    'N71W170'
    'N71W1725'
    'N70W1375'
    'N7025W141'
    'N705W1635'
    'N70W165'
    'N70W1675'
    'N70W170'
    'N70W1725'
    'N6925W16475'
    'N69W1675'
    'N69W170'
    'N69W1725'
    'N68W1675'
    'N68W170'
    'N68W1725'
    'N6710W165'
    'N67W1675'
    'N67W170'
    'N663W1683'
    'N647W16775'
    'N65W169'
    'N653W1704'
    'N654W1715'
    'N64W1675'
    'N638W1689'
    'N644W170'
    'N628W1663'
    'N628W1683'
    'N63W171'
    'N63W173'
    'N63W175'
    'N634W1734'
    'N62W167'
    'N62W169'
    'N62W171'
    'N62W173'
    'N62W175'
    'N61W167'
    'N61W169'
    'N61W167'
    'N61W169'
    'N61W171'
    'N612W173'
    'N61W175'
    'N61W177'
    'N597W165'
    'N596W167'
    'N60W169'
    'N60W171'
    'N5975W173'
    'N60W175'
    'N60W177'
    'N60W179'
    'N593W163'
    'N59W165'
    'N59W167'
    'N59W169'
    'N59W171'
    'N59W175'
    'N59W177'
    'N59W179'
    'N58W163'
    'N58W165'
    'N58W167'
    'N58W169'
    'N58W171'
    'N59W173'
    'N58W159'
    'N58W161'
    'N58W163'
    'N58W165'
    'N58W173'
    'N58W175'
    'N58W177'
    'N58W179'
    'N57W161'
    'N57W163'
    'N57W165'
    'N57W167'
    'N57W1687'
    'N57W1717'
    'N57W173'
    'N57W175'
    'N57W177'
    'N56W163'
    'N56W165'
    'N56W167'
    'N56W169'
    'N56W171'
    'N56W173'
    'N56W175'
    'N55W165'
    'N55W167'
    'N55W169'
    'N55W171'
]

def read_config():
    try:
        config_file = os.environ['MLFLOW_CONFIG']
    except KeyError:
        config_file = 'config.yml'

    try:
        with open(config_file) as f:
            config = yaml.load(f, yaml.CLoader)
    except IOError:
        raise Exception(f'Unable to location experiment config.  Set env var MLFLOW_CONFIG')

    return config

def load_station(station_dir, station, resample='W'):
    variables = {
        'ice': 'ICE_C_GDS0_SFC_ave6h',
        'sst': 'POT_GDS0_DBSL_ave6h', 
    }
    files = {
        'ice': station_dir / f'{station}-ice_conc.nc',
        'sst': station_dir / f'{station}-sst.nc'
    }
    df = load_station_var(files['ice'], variables['ice'], resample)
    sst_df = load_station_var(files['sst'], variables['sst'], resample)
    df['sst'] = sst_df.y

    return df


def load_station_var(fname, var, resample='W'):
    with xr.open_dataset(fname) as ds:
        df = ds.to_dataframe()

    df = _remove_nonmonotonic_times(df)

    new_times = pd.date_range(start=df.index[0], end=df.index[-1], freq='D')
    df = (df
        .reindex(index=new_times, columns=[var])
        .fillna(method='ffill')
        .resample(resample).mean()
    )
    return _prep_df_for_prophet(df, var)


def _remove_nonmonotonic_times(df):
    # Remove non-monotonic time-stamps
    # - find time diff > 2 days
    # - max in original dataset without problems of 2 days (leap year)
    # - data uses non-leap calendar
    ns_per_day = 86400000000000
    ndays = 2
    limit = ndays * ns_per_day
    if np.max(np.diff(df.index.values)) > limit:
        tmp_df = df.copy()
        while np.max(np.diff(tmp_df.index.values)) > limit:
            drop_ix = np.argmax(np.diff(tmp_df.index.values)) + 1
            drop_time = tmp_df.iloc[drop_ix, :]
            new_df = tmp_df.drop(index=[drop_time.name])
            tmp_df = new_df
        df = tmp_df

    return df


def _prep_df_for_prophet(df, var):
    return (df
        .rename(columns={var: 'y'})
        .reset_index()
        .rename(columns={'index': 'ds'})
    )


def extract_model_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}
