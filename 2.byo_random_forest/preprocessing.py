
import argparse
import os

import pandas as pd
import numpy as np

PROCESSING_INPUT_DIR = "/opt/ml/processing/input"
PROCESSING_OUTPUT_DIR = "/opt/ml/processing/output"

def preprocess(args): 
    
    input_data_path = os.path.join(args.processing_input_dir, "bank-additional-full.csv")
    
    print(f"Reading input data from {input_data_path}")
    data = pd.read_csv(input_data_path)
    
    # Indicator variable to capture when pdays takes a value of 999
    data['no_previous_contact'] = np.where(data['pdays'] == 999, 1, 0)                                 

    # Indicator for individuals not actively employed
    data['not_working'] = np.where(np.in1d(data['job'], ['student', 'retired', 'unemployed']), 1, 0)   

    # remove unnecessary data
    data = data.drop(
        ['duration', 
         'emp.var.rate', 
         'cons.price.idx', 
         'cons.conf.idx', 
         'euribor3m', 
         'nr.employed'
        ], 
        axis=1)

    # Convert categorical variables to sets of indicators
    model_data = pd.get_dummies(data)                    

    # Replace "y_no" and "y_yes" with a single label column, and bring it to the front:
    model_data = pd.concat([model_data['y_yes'], model_data.drop(['y_no', 'y_yes'], axis=1)], axis=1)
    
    # Randomly sort the data then split 
    validation_test_split_ratio = args.train_test_split_ratio + (1 - args.train_test_split_ratio) * 0.8
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729), 
        [int(args.train_test_split_ratio * len(model_data)), 
         int(validation_test_split_ratio * len(model_data))]) 
    print(f"total dataset length:{len(model_data)}")
    print(f"train data length:{len(train_data)}")
    print(f"validation data length:{len(validation_data)}")
    print(f"test data length:{len(test_data)}")
    
    # output to local folder
    train_data_output_path = os.path.join(args.processing_output_dir, "train/train.csv")
    train_data.to_csv(train_data_output_path, header=True, index=False)

    validation_data_output_path = os.path.join(args.processing_output_dir, "validation/validation.csv")
    validation_data.to_csv(validation_data_output_path, header=True, index=False)
    
    test_data_output_path = os.path.join(args.processing_output_dir, "test/test.csv")
    test_data.to_csv(test_data_output_path, header=True, index=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.7)
    parser.add_argument("--processing-input-dir", type=str, default=PROCESSING_INPUT_DIR)
    parser.add_argument("--processing-output-dir", type=str, default=PROCESSING_OUTPUT_DIR)
    args, _ = parser.parse_known_args()
    
    print(f"Received arguments: {args}")

    preprocess(args)
    
