import os,sys
from datetime import datetime

from script.constant import TIMESTAMP
from script.constant.training_pipeline import *
from script.exception import MoneyLaunderingException
from script.logger import logging

class TrainingPipelineConfig:
    def __init__(self, timestamp = TIMESTAMP):
        """
        Method Name: __init__
        Description: Initializes the TrainingPipelineConfig with default or provided settings.
        
        Input:
        - timestamp (str): Timestamp to identify the pipeline instance
        
        On Failure: Raises an exception if there's an error during initialization.
        
        Version: 1.0
        """
        try:
            # Initialize class attributes with default or provided settings
            self.timestamp: str = timestamp
            self.pipeline_name: str = PIPELINE_NAME
            self.artifact_dir: str = os.path.join(ARTIFACT_DIR, timestamp)
        
        except Exception as e:
            # If an exception occurs during initialization, raise it with error details
            raise MoneyLaunderingException(e)




class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Method Name: __init__
        Description: Initializes the DataIngestionConfig with paths and settings for data ingestion.
        
        Input:
        - training_pipeline_config (TrainingPipelineConfig): Configuration for the training pipeline
        
        On Failure: Raises an exception if there's an error during initialization.
        
        Version: 1.0
        """
        try:
            # Construct file paths and directories based on the provided training pipeline config
            self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, DATA_INGESTION__DIR_NAME)
            
            self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR_NAME, FILE_NAME)
            
            self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
            
            self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
            
            self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
        
        except Exception as e:
            # If an exception occurs during initialization, raise it with error details
            raise MoneyLaunderingException(e)


        


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Method Name: __init__
        Description: Initializes the DataValidationConfig class with paths to directories and files.
        
        Input:
        - training_pipeline_config: TrainingPipelineConfig object containing training pipeline configurations
        
        Version: 1.0
        """
        # Directory paths for data validation
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
        
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_VALID_DIR)
        
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_INVALID_DIR)
        
        # File paths for validated and invalid data
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, TRAIN_FILE_NAME)
        
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, TEST_FILE_NAME)
        
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, TRAIN_FILE_NAME)
        
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, TEST_FILE_NAME)
        
        # File path for drift report
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_DRIFT_REPORT_DIR,
            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )



class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                             TRAIN_FILE_NAME.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCSSING_OBJECT_FILE_NAME,)


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Method Name: __init__
        Description: Initializes the DataTransformationConfig class with paths to directories and files.
        
        Input:
        - training_pipeline_config: TrainingPipelineConfig object containing training pipeline configurations
        
        Version: 1.0
        """
        # Directory path for data transformation
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
        
        # File paths for transformed data and preprocessing object
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TRAIN_FILE_NAME.replace("csv", "npy"))
        
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir, 
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME.replace("csv", "npy"))
        
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCSSING_OBJECT_FILE_NAME)

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Method Name: __init__
        Description: Initializes the ModelTrainerConfig class with paths to directories and files, and other configurations.
        
        Input:
        - training_pipeline_config: TrainingPipelineConfig object containing training pipeline configurations
        
        Version: 1.0
        """
        # Directory path for model trainer
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, 
            MODEL_TRAINER_DIR_NAME)
        
        # File path for the trained model
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, 
            MODEL_TRAINER_TRAINED_MODEL_DIR, 
            MODEL_FILE_NAME)
        
        # Expected accuracy for the trained model
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        
        # Threshold for detecting overfitting or underfitting
        self.overfitting_underfitting_threshold = MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD



class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Method Name: __init__
        Description: Initializes the ModelEvaluationConfig class with paths to directories and files, and other configurations.
        
        Input:
        - training_pipeline_config: TrainingPipelineConfig object containing training pipeline configurations
        
        Version: 1.0
        """
        # Directory path for model evaluation
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, 
            MODEL_EVALUATION_DIR_NAME)
        
        # File path for the evaluation report
        self.report_file_path = os.path.join(
            self.model_evaluation_dir, 
            MODEL_EVALUATION_REPORT_NAME)
        
        # Threshold for considering a model change significant
        self.change_threshold = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE




class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Method Name: __init__
        Description: Initializes the ModelPusherConfig class with paths to directories and files, and other configurations.
        
        Input:
        - training_pipeline_config: TrainingPipelineConfig object containing training pipeline configurations
        
        Version: 1.0
        """
        # Directory path for model pushing
        self.model_pusher_dir : str  = os.path.join(
            training_pipeline_config.artifact_dir, 
            MODEL_PUSHER_DIR_NAME)
        
        # File path for the model
        self.model_file_path : str  = os.path.join(
            self.model_pusher_dir, 
            MODEL_FILE_NAME)

        # Generate a timestamp for saved model path
        timestamp : int = round(datetime.now().timestamp())

        # Path for saving the model
        self.saved_model_path = os.path.join(SAVED_MODEL_DIR, f"{timestamp}", MODEL_FILE_NAME)
