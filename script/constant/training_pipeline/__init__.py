import os 
from script.constant.s3_bucket import TRAINING_BUCKET_NAME
from datasets import load_dataset
import pandas as pd

SAVED_MODEL_DIR =os.path.join("saved_models")

# defining common constant variable for training pipeline
PIPELINE_NAME: str = "money"
ARTIFACT_DIR: str = "artifact"
FILE_NAME : str = "money_laund.csv"


TEST_FILE_NAME: str = "test.csv"
TRAIN_FILE_NAME: str = "train.csv"

# Set a temporary cache directory to avoid caching warnings
# cache_dir = "/tmp/huggingface_cache"

# Dataset Path
DATASET_PATH_CLASS = "/home/suyodhan/Money-Laundering-Prevention/dataset/elliptic_txs_classes.csv"
#load_dataset("SuodhanJ6/elliptic_txs_classes", split="train")
# DATASET_PATH_CLASS = pd.DataFrame(DATASET_PATH_CLASS)

DATASET_PATH_FEATURE = "/home/suyodhan/Money-Laundering-Prevention/dataset/elliptic_txs_features.csv"
#load_dataset("SuodhanJ6/elliptic_txs_features", split="train")
# DATASET_PATH_FEATURE = pd.DataFrame(DATASET_PATH_FEATURE)

DATASET_PATH_EDGE_ID = "/home/suyodhan/Money-Laundering-Prevention/dataset/elliptic_txs_edgelist.csv"
#load_dataset("SuodhanJ6/elliptic_txs_edgelist", split="train")
# DATASET_PATH_EDGE_ID = pd.DataFrame(DATASET_PATH_EDGE_ID)

PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"


# Target Column Name
TARGET_COLUMN = "class"


"""
Data Ingestation related constant start with DATA_INGESTATION var name 
"""

DATA_INGESTION__DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR_NAME: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2



"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"



"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"


"""
Model Trainer ralated constant start with MODE TRAINER VAR NAME
"""

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.9
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05


"""
Model Evalution ralated constant start with MODE EVALUTION VAR NAME
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME= "report.yaml"


"""
Model Pusher ralated constant start with MODE PUSHER VAR NAME
"""
MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = SAVED_MODEL_DIR

"""
Model Prediction ralated constant 
"""
MODEL_PRECTION_DIR_NAME = "model_prediction"
MODEL_PRECTION_FILE_NAME = "predicted_results"