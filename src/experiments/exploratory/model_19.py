from prophet import Prophet


def model():
    model = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=None,
        seasonality_mode='additive'
    )
    model.add_regressor('sst', standardize=False)

    return model
