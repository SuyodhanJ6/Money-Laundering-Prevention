import os 

from script.constant.s3_bucket import PREDICTION_BUCKET_NAME, TRAINING_BUCKET_NAME

PREDICTION_DATA_BUCKET = PREDICTION_BUCKET_NAME

# PREDICTION_INPUT_FILE_NAME = "money_laund_pred_data.csv"

PREDICTION_OUTPUT_FILE_NAME = "money_laund_predictions.csv"

MODEL_BUCKET_NAME = TRAINING_BUCKET_NAME


import os

# Define the constant for the input file name
PREDICTION_INPUT_FILE_NAME = "dataset.csv"

# Define the path to the dataset folder
DATASET_FOLDER = "/home/suyodhan/Money-Laundering-Prevention/dataset"

# Complete the path to the input file within the dataset folder
PREDICTION_INPUT_FILE_PATH = os.path.join(DATASET_FOLDER, PREDICTION_INPUT_FILE_NAME)
