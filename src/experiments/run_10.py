"""Model Fd from `evaluate-prophet.ipynb`"""
from pathlib import Path

import mlflow
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

import utils

# Load some data
STATION_FILE_DIR = '/mnt/store/data/assets/s2s/data/station-data'
station_files = list(Path(STATION_FILE_DIR).glob('*ice_conc.nc'))
# <station>-<var>.nc
station_names =[station_file.name.split('.')[0].split('-')[0] for station_file in station_files]

fname = Path(STATION_FILE_DIR) / 'PABRC-ice_conc.nc' 
station_ice = utils.load_station_var(fname, utils.variables['ice'], resample='W')
station_name = 'PABRC'

# Prep MLflow
TRACKING_URI = 'http://mlflow.srv.axiomptk:80'
mlflow.set_tracking_uri(TRACKING_URI)
EXPERIMENT = 'S2S_prophet'
mlflow.set_experiment(EXPERIMENT)

# Save model?
SAVE_MODEL = False
ARTIFACT_DIR = '/mnt/store/data/assets/s2s/models/'
artifact_dir = Path(ARTIFACT_DIR)
artifact_dir.mkdir(exist_ok=True)

# Git tag (tag used to ease specific verison of data in  dvc)
GIT_TAG = 'v1.0'

# Name of this file is good for tracking the experiment
script_name = Path(__file__).name.split('.py')[0]

with mlflow.start_run(run_name=script_name) as active_run:
    mlflow.set_tags(
        {
            'git-tag': GIT_TAG,
            'station': station_name
        }
    )

    model = Prophet(
        growth='logistic',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=None,
        seasonality_mode='multiplicative',
        changepoint_range=0.9,
        changepoint_prior_scale=0.5
    )
    # logistic requires 'cap' column in data
    station_ice['cap'] = 1
    model.fit(station_ice)
    params = utils.extract_model_params(model)

    metric_keys = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]
    metrics_raw = cross_validation(
        model=model,
        horizon="365 days",
        parallel="threads",
        disable_tqdm=True,
    )
    cv_metrics = performance_metrics(metrics_raw)
    # if some metrics are close to 0 they are not included, so need to check that
    metrics = {k: cv_metrics[k].mean() for k in metric_keys if k in cv_metrics.keys()} 
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    if SAVE_MODEL:
        run_name = mlflow.get_run()
        artifact_path = artifact_dir / f'{mflow.active_run().info.run_id}'
        mlflow.prophet.log_model(model, artifact_path=artifact_path)
        model_uri = mlflow.get_artifact_uri(artifact_path)
        print(f"Model artifact logged to: {model_uri}")