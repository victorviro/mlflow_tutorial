
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import keras

from functools import partial
import click

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import mlflow
import mlflow.keras


@click.command()
@click.option("--epochs", type=click.INT, default=10, help="Number of iterations")
@click.option("--batch_size", type=click.INT, default=100, help="Size of the batch in the training process")
@click.option("--dropout", type=click.FLOAT, default=0.5)

def run(epochs, batch_size, dropout):

    # load data
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    print(X_train_full.shape)
    print(X_test.shape)
    print(y_train_full.shape)
    print(y_test.shape)

    # Keras requires an extra dimension in the end (channels)
    X_train_full = X_train_full[..., np.newaxis]
    print(X_train_full.shape)
    X_test = X_test[..., np.newaxis]
    print(X_test.shape)

    # How many images does dataset have for each digit?
    print(pd.Series(y_train_full).value_counts())

    # Normalize the data
    X_train_full, X_test = X_train_full / 255.0, X_test / 255.0

    # Encode target variable to one hot vectors
    y_train_full = to_categorical(y_train_full, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)


    # Split the data in traininig and validation sets for the fitting
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                      test_size = 0.15, random_state=2)


    with mlflow.start_run():
        # CNN architechture is [[Conv2D->relu]*2 -> MaxPool2D]*2 -> Flatten -> Dense -> Dropout -> Out
        DefaultConv2D = partial(Conv2D, kernel_size=3, activation='relu', padding="SAME")

        model = Sequential([
            DefaultConv2D(filters=16, kernel_size=5, input_shape=[28, 28, 1]),
            DefaultConv2D(filters=16, kernel_size=5),
            MaxPooling2D(pool_size=2),
            DefaultConv2D(filters=32),
            DefaultConv2D(filters=32),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(units=256, activation='relu'),
            Dropout(dropout),
            Dense(units=10, activation='softmax'),
        ])

        class LogMetricsCalback(keras.callbacks.Callback):
            def on_epoch_end(self,epoch, logs={}):
                mlflow.log_metric('training_loss', logs['loss'], epoch)
                mlflow.log_metric('training_accuracy', logs['accuracy'], epoch)
                mlflow.log_metric('validation_loss', logs['val_loss'], epoch)
                mlflow.log_metric('validation_accuracy', logs['val_accuracy'], epoch)

        # Compile the model
        model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

        history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
                            validation_data = (X_val, y_val), callbacks=[LogMetricsCalback()])

        # evaluation
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        mlflow.log_metric('test_loss', test_loss)
        mlflow.log_metric('test_accuracy', test_acc)

        mlflow.keras.log_model(model, artifact_path='keras_model')

        # Prediction for the test set
        y_pred_test = model.predict(X_test)

        # plot consusion matrix
        fig = plt.figure(figsize=(10,8))
        y_pred_classes = np.argmax(y_pred_test, axis = 1) 
        y_test_classes = np.argmax(y_test, axis = 1) 
        # compute the confusion matrix
        confusion_mtx = confusion_matrix(y_test_classes, y_pred_classes) 
        # plot the confusion matrix
        sns.heatmap(confusion_mtx, annot=True, fmt="d")
        # Save figure
        fig.savefig("Confusion_matrix.png")
        plt.close(fig)
        # Log artifacts (output files)
        mlflow.log_artifact("Confusion_matrix.png")

if __name__ == "__main__":
    run()
