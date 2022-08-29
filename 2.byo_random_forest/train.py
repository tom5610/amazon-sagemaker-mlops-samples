# Python Built-Ins:
import argparse
import os

# External Dependencies:
#Joblib is a set of tools to provide lightweight pipelining in Python.
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, 
# along with a large collection of high-level mathematical functions to operate on these arrays.

#  pandas is a software library written for the Python programming language for data manipulation and analysis. 
#In particular, it offers data structures and operations for manipulating numerical tables and time series.

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score


# inference functions ---------------
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

# model traing --------------
def train(args):

    print("loading training data")
    train_data_path = os.path.join(args.train_dir, "train.csv")
    train_data = pd.read_csv(train_data_path, index_col=None)
    X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    
    validation_data_path = os.path.join(args.validation_dir, "validation.csv")
    validation_data = pd.read_csv(validation_data_path, index_col=None)
    X_val, y_val = validation_data.iloc[:, 1:], validation_data.iloc[:, 0]

    print("model training")
    
    hyperparameters = {
        "bootstrap": [True],
        "max_depth": [12, 13],
        "max_features": [13, 14],
        "n_estimators": [100, 150]
    }

    model = RandomForestClassifier()

    grid_search = GridSearchCV(model, hyperparameters, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(best_model)
    
    
    # Evaluation on Validation data.
    y_pred = grid_search.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"accuracy: {accuracy}")
    precision = precision_score(y_val, y_pred)
    print(f"precision: {precision}")
    recall = recall_score(y_val, y_pred)
    print(f"recall: {recall}")
    f1 = f1_score(y_val, y_pred)
    print(f"f1: {f1}")
    
    
    # Saving model
    path = os.path.join(args.model_dir, "model.joblib")
    print("Saving model to {}".format(path))
    joblib.dump(best_model, path)
    
if __name__ == '__main__':

    #------------------------------- parsing input parameters (from command line)
    print('extracting arguments')
    parser = argparse.ArgumentParser()

    # RandomForest hyperparameters
    parser.add_argument('--n-estimators', type=int, nargs="+", default=[100, 150])
    parser.add_argument('--max-depth', type=int, nargs="+", default=[12, 13])
    parser.add_argument('--max-features', type=int, nargs="+", default=[13, 14])
    parser.add_argument('--bootstrap', type=bool, nargs="+", default=[True])

    # Data, model, and output directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation-dir', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))

    args, _ = parser.parse_known_args()
    
    print(f"Received arguments: {args}")    
    
    train(args)
