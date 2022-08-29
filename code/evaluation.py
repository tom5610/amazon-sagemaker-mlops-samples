
import os
import json
import logging
from pathlib import Path
import tarfile
import argparse
import joblib

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score, classification_report


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

PROCESSING_INPUT_DIR = '/opt/ml/processing/test'
PROCESSING_MODEL_DIR = '/opt/ml/processing/model'
PROCESSING_EVALUATION_DIR = '/opt/ml/processing/evaluation'

def postprocess(args):
    model_path = Path(args.processing_model_dir) / 'model.tar.gz'
    if model_path.exists():
        logger.info('Extracting model from path: {}'.format(model_path))
        with tarfile.open(model_path) as tar:
            tar.extractall(path=args.processing_model_dir)

        for file in os.listdir(args.processing_model_dir):
            logger.info(file)
        
    logger.debug("Loading the model.")
    model = joblib.load(os.path.join(args.processing_model_dir, "model.joblib"))

    logger.debug("Reading test data.")
    test_data_path = os.path.join(args.processing_input_dir, 'test.csv')
    
    # index_col is from the old version of pandas read_csv() api.
    test_data = pd.read_csv(test_data_path, index_col=None)
    print(test_data.head())
    
    logger.debug("Reading test data.")
    y_test = test_data.iloc[:, 0].values
    test_data.drop(test_data.columns[0], axis=1, inplace=True)
    X_test = test_data.values
    
    logger.info("Performing predictions against test data.")
    y_pred = model.predict(X_test) 
    
    logger.info(f"predictions: {y_pred}")

    print('Creating classification evaluation report')
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {accuracy}")
    precision = precision_score(y_test, y_pred)
    print(f"precision: {precision}")
    recall = recall_score(y_test, y_pred)
    print(f"recall: {recall}")
    f1 = f1_score(y_test, y_pred)
    print(f"f1: {f1}")
    
    report_dict = {
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
    }
    
    evaluation_output_path = os.path.join(args.processing_evaluation_dir, 'evaluation.json')
    print('Saving classification report to {}'.format(evaluation_output_path))

    with open(evaluation_output_path, 'w') as f:
        f.write(json.dumps(report_dict))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processing-input-dir", type=str, default=PROCESSING_INPUT_DIR)
    parser.add_argument("--processing-model-dir", type=str, default=PROCESSING_MODEL_DIR)
    parser.add_argument("--processing-evaluation-dir", type=str, default=PROCESSING_EVALUATION_DIR)
    args, _ = parser.parse_known_args()
    
    print(f"Received arguments: {args}")

    postprocess(args)
