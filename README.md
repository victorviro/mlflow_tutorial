
# MLflow [Tutorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html): 


- Clone the repository via   ` git clone https://github.com/victorviro/mlflow_wine_regression_tutorial.git`
- Install MLflow and scikit-learn:
```
python3 -m venv venv
source venv/bin/activate
pip install mlflow
pip install scikit-learn
pip install boto3
```
- Install conda

- `cd` into the `examples` directory 

## Traininig the model:


The first thing we’ll do is train a linear regression model which takes two hyperparameters: `alpha` and `l1_ratio`.

The code which we will use is located at `examples/sklearn_elasticnet_wine/train.py`.
In this code, we create a simple ML model. We also use the MLflow tracking API to log
information about each training run, like the hyperparameters and metrics which we will use to evaluate the model. In addition, we serialize the model which we produced in a format that MLflow knows how to deploy (`mlflow.sklearn.log_model()`). We also generate a figure to visualize the importance of the features in the model and track it like an artifact.

To run this example we can execute:

```
python sklearn_elasticnet_wine/train.py
```

We try out some other values for `alpha` and `l1_ratio` by passing them as arguments:

```
python sklearn_elasticnet_wine/train.py --alpha 0.0098 --l1_ratio 0.4
python sklearn_elasticnet_wine/train.py --alpha 0.8 --l1_ratio 0.1
```

After running this, MLflow has logged information about our experiment runs and they are stored locally in the directory called `mlruns`. Note that MLflow runs can be recorded to local files, to a SQLAlchemy compatible database, or remotely to a tracking server. By default, the MLflow Python API logs runs locally to files in the `mlruns` directory we mentioned. 

To log runs remotely, we can set the `MLFLOW_TRACKING_URI` environment variable to a tracking server’s URI or call `mlflow.set_tracking_uri()` in the code. For instance, if we want store runs locally but we prefer to save it to a designated location, we could specify it at the beginning of the program as follows:
```
mlflow.set_tracking_uri('file:/home/viro/mlrun_store')
```
**Note**: Specified folder needs to be created during execution when referenced for the first time. If it is created upfront, the script terminates with a error message.

For more info see [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html).

### Experiments

We can optionally organize runs into *experiments*, which group together runs for a specific task. We can create an experiment using the mlflow experiments Command Line Interface (CLI), or using `mlflow.create_experiment()` in the code, or using the corresponding REST parameters. Let's create a new experiment with the CLI:

```
mlflow experiments create -n new_experiment
```
> Created experiment 'new_experiment' with id 1

A new dir folder `1` is added into the `mlruns` directory where runs in this experiment will be tracked.

We can specify in the code the experiment where we want to save the run: `mlflow.start_run(experiment_id=1)`.

If we record runs in an MLflow Project we can run the training remotely specifying the experiment as we will see later.


## Comparing the Models:

Next we will use the MLflow User Interface (UI) to compare the models which we have produced. 
Run 

```
mlflow ui
```

in the same current working directory as the one which contains the mlruns directory and we navigate our browser to http://localhost:5000.

On this page, we can see the metrics we can use to compare our models. We can use the Search Runs feature to filter out many models. For example, `metrics.rmse < 0.8`. We can see the runs of other experiments clicking in the experiment (tab at the left side) (if we do not specify a experiment id as we did, the runs will be tracked in the experiment created initially by default (Experiment ID: 0))


To display the recorded data in the browser in the case we have specified a designated location where store the run, we need to pass the address of our storage location to mlflow ui.

```
mlflow ui --backend-store-uri file:/home/viro/mlrun_store 
```

We can press Ctrl + C to stop the UI.


## Packaging Training Code in a Conda Environment

We can package the training code so that others can easily reuse the model, or so that we can run the training remotely.

We do this by using [MLflow Projects](https://www.mlflow.org/docs/latest/projects.html) conventions to
specify the dependencies and entry points to our code. The `sklearn_elasticnet_wine/MLproject` is a configuration file which
specifies that the project has the library dependencies located in a Conda environment file called `conda.yaml` and it has two entry points. The first one consists of a command and a set of configurable parameters (`alpha` and `l1_ratio` in this case) to pass to that command at runtime.

When we run a MLflow project with parameters, all of these parameters are automatically logged to the tracking service, that is we do not need to use the logging function `mlflow.log_param()` in the training code.

The Conda file (in `sklearn_elasticnet_wine/conda.yaml`) lists the dependencies.

The MLflow CLI can be used to run this project:
```
mlflow run sklearn_elasticnet_wine -P alpha=0.4011
```



After running this command, MLflow runs our training code in a new Conda environment with the dependencies specified in `conda.yaml` (a new run is recoded in `mlruns` directory). 

If we go to the UI we can see this run is added, but the source is `sklearn_elasticnet_wine` (instead of `train.py` like in the previous runs). Note that, in this case, the artifacts are not recorded.

Alternatively, we can run the project specifying the experiment (in this case, the artifacts are recorded). The experiment created by default has the ID 0. The second we have created has ID 1 so we can run the project with the command:

```
mlflow run -e train --experiment-id 1 sklearn_elasticnet_wine -P alpha=0.11
```


**Note**: If the repository has an MLproject file in the root we can also run a project directly from GitHub. 
For example, https://github.com/mlflow/mlflow-example . Run:
```
mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5
```
MLflow will automatically clone that repository, create a conda environment, install the dependencies and execute the code.

## Serving the Model
Now when we have packaged our model using the MLproject convention and have identified the best model,
we can deploy it using MLflow Models. An MLflow Model is a standard format for packaging ML models
that can be used in a variety of downstream tools — for example, real-time serving through a REST API or batch inference on Apache Spark.

In our example, in `train.py` we used `mlflow.sklearn.log_model(model, "model")` after training the linear regression model, so that in MLflow  was saved the model as an artifact within the run. To see this artifact, we can use the UI again (click in the date of the run to open a new window, the artifacts view is at the bottom).

The `mlflow.sklearn.log_model(model, "model")` produced two files in
`./mlruns/0/1ca86d9ac6e8480c951e7d849ddfdd34/artifacts/model/` (the full path in the view of that artifact in the UI) (the directory `1ca86d9ac6e8480c951e7d849ddfdd34` depicts the run_id, it will be different for you):

- MLmodel, is a metadata file that tells MLflow how to load the model. 
- model.pkl, is a serialized version of the linear regression model that we trained.

In this example, we can use this MLmodel format with MLflow to deploy a local REST server that
can serve predictions. To deploy the server, run (replace the path with your model’s actual path):
```
mlflow models serve -m ./mlruns/1/9e731d22515a453892948a063740fe7d/artifacts/model -p 1236
```


Once we have deployed the server (it's running), we can open another terminal and pass it some sample data and see the predictions. The following example uses curl to send a JSON-serialized pandas DataFrame with the split
orientation to the model server. More information about the input data formats accepted by the model server, see the [MLflow deployment tools documentation](https://www.mlflow.org/docs/latest/models.html#local-model-deployment).

```
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1236/invocations
```
the server should respond with output similar to:
> [6.379428821398614]


## References and more resources

- [MLflow turorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)

- [MLflow: An Open Platform to Simplify the Machine Learning Lifecycle](https://youtu.be/859OxXrt_TI)

- [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)

- [MLflow Projects](https://www.mlflow.org/docs/latest/projects.html)

- [MLflow Models](https://www.mlflow.org/docs/latest/models.html).



## Hyperopt

This section extends the `train.py` script used in the previous section. We use a script called `search_hyperopt.py` which looks for the best hyperparameters for the regression model using the [hyperopt](https://github.com/hyperopt/hyperopt) library. Then it calls the `train.py` script for training the regression model with the hyperparameters found. Note that we need to extend the MLproject configuration file if we want to run this process in the project (the new `entry_point` in the MLproject file is called `hyperopt`). 

We create a new experiment called `Hyperparameters_optimized` via CLI:

```
mlflow experiments create -n Hyperparameters_optimized
```
Now, we can run the project setting the name of the entry_point and the name of the experiment
```
mlflow run -e hyperopt --experiment-name Hyperparameters_optimized sklearn_elasticnet_wine 
```













