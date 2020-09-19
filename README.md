
# MLflow [Tutorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html): 


- Clone the repository via  `git clone https://github.com/victorviro/mlflow_wine_regression_tutorial.git`
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
In this code, we create a simple ML model. We also use the MLflow tracking APIs to log
information about each training run, like the hyperparameters and metrics which we will use to evaluate the model. In addition, we serialize the model which we produced in a format that MLflow knows how to deploy (`mlflow.sklearn.log_model()`).

To run this example execute:

```
python sklearn_elasticnet_wine/train.py
```

Try out some other values for alpha and l1_ratio by passing them as arguments:

```
python sklearn_elasticnet_wine/train.py 0.1 0.8
python sklearn_elasticnet_wine/train.py 0.8 0.1
```

After running this, MLflow has logged information about our experiment runs in the directory called mlruns.

## Comparing the Models:

Next we will use the MLflow User Interface (UI) to compare the models which we have produced. 
Run mlflow ui in the same current working directory as the one which contains the mlruns directory
```
mlflow ui
```

and navigate our browser to http://localhost:5000.

On this page, we can see the metrics we can use to compare our models. We can use the Search Runs feature to filter out many models. For example, `metrics.rmse < 0.8`


## Packaging Training Code in a Conda Environment

We can package the training code so that others can easily reuse the model,  or so that we can run the training remotely.

We do this by using [MLflow Projects](https://www.mlflow.org/docs/latest/projects.html) conventions to
specify the dependencies and entry points to our code. The `sklearn_elasticnet_wine/MLproject` file 
specifies that the project has the dependencies located in a Conda environment file called `conda.yaml` and has one entry point that takes two parameters: alpha and l1_ratio.

The Conda file (in `sklearn_elasticnet_wine/conda.yaml`) lists the dependencies.

To run this project, invoke:
```
mlflow run sklearn_elasticnet_wine -P alpha=0.42
```

After running this command, MLflow runs our training code in a new Conda environment with the dependencies specified in conda.yaml (a new run is recoded in `mlruns` directory).

If we go to the UI we see this run is added, but the source is sklearn_elasticnet_wine (instead train.py like in the previous runs).


**Note**: If the repository has an MLproject file in the root we can also run a project directly from GitHub. 
For example, https://github.com/mlflow/mlflow-example . Run:
```
mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5
```

## Serving the Model
Now that we have packaged our model using the MLproject convention and have identified the best model,
we can to deploy it using MLflow Models. An MLflow Model is a standard format for packaging ML models
that can be used in a variety of downstream tools — for example, real-time serving through a REST API or batch inference on Apache Spark.

In our example, in `train.py` we used `mlflow.sklearn.log_model(model, "model")` after training the linear regression model, so that in MLflow  was saved the model as an artifact within the run. To see this artifact, we can use the UI again (click in the date of the run to open a new window, at the bottom there is the artifacts view).

The `mlflow.sklearn.log_model(model, "model")` produced two files in
`./mlruns/0/1ca86d9ac6e8480c951e7d849ddfdd34/artifacts/model/` (full path in the view of that artifact in UI) (the value `1ca86d9ac6e8480c951e7d849ddfdd34` is the run_id, it will be different for you):
- MLmodel, is a metadata file that tells MLflow how to load the model. 
- model.pkl, is a serialized version of the linear regression model that we trained.

In this example, we can use this MLmodel format with MLflow to deploy a local REST server that
can serve predictions. To deploy the server, run (replace the path with your model’s actual path):
```
mlflow models serve -m ./mlruns/0/1ca86d9ac6e8480c951e7d849ddfdd34/artifacts/model -p 1236
```

Once we have deployed the server (it's running),  we can pass it some sample data and see the predictions. 
The following example uses curl to send a JSON-serialized pandas DataFrame with the split
orientation to the model server. More information about the input data formats accepted by the model server, see the [MLflow deployment tools documentation](https://www.mlflow.org/docs/latest/models.html#local-model-deployment).

```
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1236/invocations
```
the server should respond with output similar to:
>>> [6.379428821398614]

## More resources

See [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html), [MLflow Projects](https://www.mlflow.org/docs/latest/projects.html), [MLflow Models](https://www.mlflow.org/docs/latest/models.html).











