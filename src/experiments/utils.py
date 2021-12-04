import numpy as np
import pandas as pd
import xarray as xr
from prophet import serialize

variables = {
    'ice': 'ICE_C_GDS0_SFC_ave6h',

}

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


def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}
