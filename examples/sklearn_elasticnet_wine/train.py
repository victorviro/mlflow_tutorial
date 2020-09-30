# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality


import warnings
import click

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

#mlflow.set_tracking_uri('file:/home/viro/mlrun_store')


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



@click.command()
@click.option("--alpha", type=click.FLOAT, default=0.1, help="Constant that multiplies the penalty terms.")
@click.option("--l1_ratio", type=click.FLOAT, default=0.1, help="The ElasticNet mixing parameter.")
@click.argument("training_data", default="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")

def run(training_data, alpha, l1_ratio):
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file
    data = pd.read_csv(training_data, sep=";")

    # The predicted column is "quality" which is a scalar from [3, 9]
    X = data.drop(["quality"], axis=1)
    y = data[["quality"]]

    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # Scale regressor variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    with mlflow.start_run(experiment_id=0): #experiment_id=1
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        
        fit_elasticnet = model.fit(X_train_scaled, y_train)

        predicted_qualities = fit_elasticnet.predict(X_test_scaled)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print(f'Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):')
        print(f'  RMSE: {rmse}\n  MAE: {mae}\n  R2: {r2}')

        # use the MLflow tracking APIs to log information about each training run
        # hyperparameters
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        # metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # serialize the model in a format that MLflow knows how to deploy it
        mlflow.sklearn.log_model(model, "model")



        dict_importance_features = {'Features':list(X_train), 'Importance': abs(fit_elasticnet.coef_)}
        df_importance_features = pd.DataFrame(dict_importance_features)
        df_importance_features_sorted = df_importance_features.sort_values(by=['Importance'], ascending=False)

        # plot importance of the features
        fig = plt.figure(figsize=(15, 10))
        sns.barplot(data=df_importance_features_sorted, x='Importance', y = 'Features')
        # Save figure
        fig.savefig("Importance_features.png")
        plt.close(fig)

        # Log artifacts (output files)
        mlflow.log_artifact("Importance_features.png")

if __name__ == "__main__":
    run()

        