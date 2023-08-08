import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

from script.logger import logging
from script.exception import MoneyLaunderingException

from script.entity.config_enitity import DataTransformationConfig
from script.entity.artifact_enitity import DataValidationArtifact, DataTransformationArtifact
from script.ml.model.estimetor import TargetValueMapping
from script.utils.main_utils import save_numpy_array_data, save_object
from script.constant.training_pipeline import *

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        """
        Method Name: __init__
        Description: Initializes the DataTransformation class.
        
        Input: data_validation_artifact - DataValidationArtifact containing validation results
               data_transformation_config - DataTransformationConfig containing transformation configurations
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Initialize with provided validation artifact and transformation configuration
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            logging.info("Entered DataTransformation class.")

        except Exception as e:
            raise MoneyLaunderingException(e, sys)

        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Method Name: read_data
        Description: Reads a CSV file and returns its content as a DataFrame.
        
        Input: file_path - Path to the CSV file
        
        Output: DataFrame containing the data from the CSV file
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Read the CSV file and return its content as a DataFrame
            return pd.read_csv(file_path)

        except Exception as e:
            # Log the error and raise an exception
            logging.error("Error occurred while reading data from '%s': %s", file_path, str(e))
            raise MoneyLaunderingException(e, sys)


        
    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Method Name: get_data_transformer_object
        Description: Returns a data transformation pipeline for preprocessing input data.
        
        Output: A scikit-learn pipeline for data transformation
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            logging.info("Entering get_data_transformer_object method.")
            
            # Create instances of transformers
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            
            # Define the data transformation pipeline
            preprocessor = Pipeline(
                steps=[
                    ("Imputer", simple_imputer), # Replace missing values with zero
                    ("RobustScaler", robust_scaler) # Keep every feature in the same range and handle outliers
                ]
            )
            
            logging.info("Returning data transformation pipeline.")
            return preprocessor
        except Exception as e:
            # Log the error and raise an exception
            logging.error("Error occurred in get_data_transformer_object: %s", str(e))
            raise MoneyLaunderingException(e, sys)



    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Method Name: initiate_data_transformation
        Description: Initiates the data transformation process.
        
        Output: DataTransformationArtifact containing paths to transformed data and preprocessing object
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            logging.info("Starting data transformation")

            # Read train and test data
            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path
            train_df = DataTransformation.read_data(train_file_path)
            test_df = DataTransformation.read_data(test_file_path)  

            # Get data transformer object
            preprocessor = self.get_data_transformer_object()

            # Separate input and target features for training and testing
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(TargetValueMapping().to_dict())
            target_feature_train_df = target_feature_train_df.astype(int)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())
            target_feature_test_df = target_feature_test_df.astype(int)

            # Fit preprocessing object on training data
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Apply SMOTE Tomek resampling
            smt = SMOTETomek(sampling_strategy="minority")
            logging.info("Starting SMOTE Tomek resampling for training data")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )
            logging.info("Completed SMOTE Tomek resampling for training data")

            logging.info("Starting SMOTE Tomek resampling for testing data")
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )
            logging.info("Completed SMOTE Tomek resampling for testing data")

            # Create arrays and save data
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            # Prepare and return the transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            # Log the error and raise an exception
            logging.error("Error occurred during data transformation: %s", str(e))
            raise MoneyLaunderingException(e, sys)

