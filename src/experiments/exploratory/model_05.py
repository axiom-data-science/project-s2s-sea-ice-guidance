from prophet import Prophet


def model():
    model = Prophet(
        growth='logistic',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=None,
        seasonality_mode='multiplicative'
    ).add_seasonality(
        name='biannual',
        period=365.25*4,
        fourier_order=20,  # total guess
        prior_scale=5
    )

    return model
