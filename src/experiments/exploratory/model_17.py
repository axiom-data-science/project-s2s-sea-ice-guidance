from prophet import Prophet


def model():
    model = Prophet(
        growth='logistic',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=None,
        seasonality_mode='multiplicative',
        changepoint_range=0.9,
    )
    model.add_regressor('sst', standardize=False)

    return model
