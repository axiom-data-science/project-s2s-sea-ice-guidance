"""Runs training and evaluation of Prophet models."""
import importlib
import json
import os
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import utils
from dask import distributed
from prophet import plot
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.serialize import model_to_json

import mlflow

config = utils.read_config()
# Client needs to this set (along with S3 creds if perms required)
# Also need to set envvar
# - AWS_ACCESS_KEY_ID to minio-user (see mlflow minio vault in ansible)
# - AWS_SECRET_ACCESS_KEY to minio-user-password

os.environ['MLFLOW_S3_ENDPOINT_URL'] = config['mlflow_s3_endpoint_url']


def __load_model(model):
    """Given path to model, return loaded Prophet model."""
    # boilerplate: https://docs.python.org/3/library/importlib.html
    model_name = model.name.split('.')[0]
    spec = importlib.util.spec_from_file_location(model_name, model) 
    model_module = importlib.util.module_from_spec(spec)
    sys.modules['model_module'] = model_module
    spec.loader.exec_module(model_module)

    return model_module.model()


@click.command()
@click.option('--data_path', default=config['station_file_path'], type=Path, help='Path to station data directory')
@click.option('--tracking_uri', default=config['tracking_uri'], type=str, help='URI to MLFlow tracking')
@click.option('--artifact_path', default=config['artifact_path'], type=Path, help='Path to directory where artifacts will be saved')
@click.option('--git_tag', default=config['git_tag'], type=str, help='DVC git tag (version of data)')
@click.option('--save_model', is_flag=True, help='Save model')
@click.option('--dask', default=None, type=str, help='URL to connect to Dask to parallelize cross validation')
@click.argument('model')
@click.argument('station')
@click.argument('experiment')
def run_model(model, station, experiment, data_path, tracking_uri, artifact_path, git_tag, save_model, dask):
    script_name = Path(__file__).name.split('.py')[0]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    artifact_path = Path(artifact_path)
    artifact_path.mkdir(exist_ok=True)

    # Load data
    # - set cap and floor as physical bounds and bounds respected by logistic growth models 
    station_df = utils.load_station(data_path, station)
    station_df['cap'] = 1
    station_df['floor'] = 0

    model = Path(model)

    if station == 'all':
        stations = utils.STATIONS
    else:
        stations = [station]

    if dask:
        client = distributed.Client(dask)
        parallel = "dask"
    else:
        parallel = "threads"

    for station in stations:
        with mlflow.start_run(run_name=f'{script_name}-{model}') as active_run:
            mlflow.set_tags(
                {
                    'git-tag': git_tag,
                    'station': station,
                    'model': model
                }
            )

            # Load model
            model = __load_model(model)
            # - fit model
            model.fit(station_df)

            # Calculate metrics from cross validation
            # - Start cross-validation every 30 days and forecast for next 180
            metric_keys = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]
            df_cv = cross_validation(
                model=model,
                period="30 days",
                horizon="180 days",
                parallel=parallel,
                disable_tqdm=True,
            )
            cv_metrics = performance_metrics(df_cv)
            # if some metrics are close to 0 they are not included, so need to check that
            metrics = {k: cv_metrics[k].mean() for k in metric_keys if k in cv_metrics.keys()}

            # Create forecast
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            # this is the fake SST because we don't have that information now
            # - for illustrative purposes
            forecast['sst'] = station_df['sst'].copy()
            guess_temp = station_df['sst'].iloc[-52::].values.copy()
            forecast['sst'].iloc[-52::] = guess_temp

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model params
            params = utils.extract_model_params(model)
            mlflow.log_params(params)

            # Save image
            fig, axes = plt.subplots(nrows=3, figsize=(10, 8))
            # - Station data
            axes[0].plot(station_df.ds, station_df.y)
            axes[0].set_ylabel('Ice coverage')
            ax2 = axes[0].twinx()
            ax2.plot(station_df.ds, station_df.sst, color='r')
            ax2.set_ylabel('SST')
            # - Cross-validation plot with error
            plot.plot_cross_validation_metric(df_cv, metric='rmse', ax=axes[1])
            # - Forecast
            model.plot(forecast, ax=axes[2], ylabel='Ice coverage')
            plot.add_changepoints_to_plot(axes[2], model, forecast)

            image_path = artifact_path / 'training'
            image_path.mkdir(exist_ok=True)
            fname = image_path / f'{station}.png'
            fig.savefig(fname)
            plt.close(fig)
            mlflow.log_artifact(str(fname))

            # Save forecast image (annoyingly doesn't take ax)
            fig = model.plot_components(forecast)
            fname = image_path / f'{station}-forecast-components.png'
            fig.savefig(fname)
            plt.close(fig)
            mlflow.log_artifact(str(fname))

            if save_model:
                model_path = artifact_path / 'station-models'
                model_path.mkdir(exist_ok=True)
                fname = model_path / f'{station}-model.json'
                with open(fname, 'w') as fout:
                    json.dump(model_to_json(model), fout)

    # Saves as a runnable artifact.  we'll start with just the json file
    #            mlflow.prophet.save_model(model, f'{station-model}.json')

if __name__ == '__main__':
    run_model()
