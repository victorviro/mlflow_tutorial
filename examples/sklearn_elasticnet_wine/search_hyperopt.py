
import click

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import pandas as pd

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import mlflow



@click.command()
@click.argument("training_data", default="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv")

def train(training_data):

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

    def objective(params):
        regressors_type = params['type']
        del params['type']
        if regressors_type == 'ElasticNet':
            clf = ElasticNet(**params)
        else:
            return 0
        accuracy = cross_val_score(clf, X_train_scaled, y_train).mean()
        return {'loss': -accuracy, 'status': STATUS_OK}

    search_space = hp.choice('regressor_type', [
        {
            'type': 'ElasticNet',
            'alpha': hp.uniform('alpha', 0, 1.0),
            'l1_ratio': hp.uniform('l1_ratio', 0, 1.0)
        },
    ])

    algo=tpe.suggest

    with mlflow.start_run(experiment_id=2) as child_run:
        best_result = fmin(
            fn=objective, 
            space=search_space,
            algo=algo,
            max_evals=16)
        print(f'best parameters: {best_result}')
        alpha, l1_ratio = best_result["alpha"], best_result["l1_ratio"]

        p = mlflow.projects.run(
            uri=".",
            entry_point="train",
            run_id=child_run.info.run_id,
            parameters={
                "training_data": training_data,
                "alpha": alpha,
                "l1_ratio": l1_ratio
            },
            use_conda=False,  # We are already in the environment
            synchronous=False,  # Allow the run to fail if a model is not properly created
        )
                


if __name__ == "__main__":
    train()





