import json
from pathlib import Path

import mlflow
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

import utils

# Load some data
STATION_FILE_DIR = '/mnt/store/data/assets/s2s/data/station-data'
station_files = Path(STATION_FILE_DIR).glob('*ice_conc.nc')
fname = next(station_files)
station_ice = utils.load_station_var(fname, utils.variables['ice'])
# - get station from file name
fname = Path(fname)
# <station>-<var>.nc
station_name = fname.name.split('.')[0].split('-')[0]

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

# Run experiment
with mlflow.start_run(run_name=script_name):
    mlflow.set_tags(
        {
            'git-tag': GIT_TAG,
            'station': station_name
        }
    )

    model = Prophet().fit(station_ice)
    params = utils.extract_params(model)

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
