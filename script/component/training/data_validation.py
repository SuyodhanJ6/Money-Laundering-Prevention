import os, sys

import pandas as pd
from scipy.stats import ks_2samp
import shutil

from script.logger import logging
from script.exception import MoneyLaunderingException
from script.constant.training_pipeline import SCHEMA_FILE_PATH, TARGET_COLUMN
from script.component.training.data_ingestion import DataIngestionArtifact
from script.entity.config_enitity import DataValidationConfig, DataIngestionConfig
from script.entity.artifact_enitity import DataValidationArtifact
from script.utils.main_utils import read_yaml_file,write_yaml_file
from script.entity.config_enitity import TrainingPipelineConfig
from script.component.training.data_ingestion import DataIngestion



class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Method Name: __init__
        Description: Initializes the DataValidation class.
        
        Input:
        - data_ingestion_artifact: DataIngestionArtifact object
        - data_validation_config: DataValidationConfig object
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            logging.info("Entered DataValidation class.")

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
            logging.info("Reading data from '%s'", file_path)
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error("Error occurred while reading data from '%s': %s", file_path, str(e))
            raise MoneyLaunderingException(e, sys)

        
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Method Name: validate_number_of_columns
        Description: Validates whether the number of columns in the DataFrame matches the expected number from the schema.
        
        Input: dataframe - The DataFrame to be validated
        
        Output: True if the number of columns matches, False otherwise
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            number_of_columns = len(self._schema_config["columns"])
            
            if len(dataframe.columns) == number_of_columns:
                logging.info(f"Required number of columns: {number_of_columns}")
                logging.info(f"Data frame has columns: {len(dataframe.columns)}")
                logging.info("Number of columns validation passed.")
                return True
            else:
                logging.warning(f"Number of columns validation failed. Expected: {number_of_columns}, Actual: {len(dataframe.columns)}")
                return False

        except Exception as e:
            logging.error("Error occurred during number of columns validation: %s", str(e))
            raise MoneyLaunderingException(e, sys)

        
    def validate_numerical_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Method Name: validate_numerical_columns
        Description: Validates whether all specified numerical columns are present in the DataFrame.
        
        Input:
            dataframe (pd.DataFrame): The DataFrame to be validated
            
        Output:
            bool: True if all specified numerical columns are present, False otherwise
            
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Get the list of specified numerical columns from the schema config
            numeric_columns = self._schema_config["numerical_columns"]
            
            # Get the list of columns present in the DataFrame
            dataframe_columns = dataframe.columns
            
            # Initialize a flag to track the presence of all specified numerical columns
            numeric_columns_present = True
            
            # Initialize a list to track missing numerical columns
            missing_numerical_columns = []
            
            # Loop through each specified numerical column
            for num_cols in numeric_columns:
                if num_cols not in dataframe_columns:
                    # If a specified numerical column is missing, update the flag and add to the missing list
                    numeric_columns_present = False
                    missing_numerical_columns.append(num_cols)
            
            # Log whether all specified numerical columns are present or not
            if numeric_columns_present:
                logging.info("All specified numerical columns are present.")
            else:
                logging.warning("Some specified numerical columns are missing.")
                logging.warning(f"Missing numerical columns: {missing_numerical_columns}")
            
            return numeric_columns_present

        except Exception as e:
            # Log the error and raise an exception
            logging.error("Error occurred during numerical columns validation: %s", str(e))
            raise MoneyLaunderingException(e, sys)

    # def txId_has_unique_values(self, dataframe : pd.DataFrame) -> bool:
        # try:
        #     txId_unique = dataframe['txId'].nunique()
        #     logging.info(f"txId has unique values: {txId_unique}")
        #     total_records = dataframe.shape[0]
        #     logging.info(f"Total records: {total_records}")

        #     if txId_unique == total_records:
        #         return True
        #     else:
        #         return False

        # except Exception as e:
        #     raise MoneyLaunderingException(e, sys)
        

    def time_step_has_positive_values(self, dataframe) -> bool:
        """
        Method Name: time_step_has_positive_values
        Description: Validates whether all values in the 'Time_step' column are positive.
        
        Input:
            dataframe (pd.DataFrame): The DataFrame to be validated
            
        Output:
            bool: True if all values in 'Time_step' column are positive, False otherwise
            
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            logging.info("Starting 'Time_step' validation")
            
            # Check if all values in the 'Time_step' column are positive
            time_step_positive = (dataframe['Time_step'] > 0).all()
            
            if time_step_positive:
                logging.info("'Time_step' validation passed. All values are positive.")
            else:
                logging.warning("'Time_step' validation failed. Some values are not positive.")
            
            logging.info("Ending 'Time_step' validation")
            return time_step_positive

        except Exception as e:
            # Log the error and raise an exception
            logging.error("Error occurred during 'Time_step' validation: %s", str(e))
            raise MoneyLaunderingException(e, sys)

    
    def validate_class_labels(self, dataframe: pd.DataFrame) -> bool:
        """
        Method Name: validate_class_labels
        Description: Validates whether class labels in the DataFrame are within the allowed labels.
        
        Input:
            dataframe (pd.DataFrame): The DataFrame to be validated
            
        Output:
            bool: True if all class labels are within the allowed labels, False otherwise
            list: List of invalid class labels
            
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Start the class label validation
            logging.info("Starting class label validation")
            
            # Check if the target column exists in the DataFrame
            if TARGET_COLUMN not in dataframe.columns:
                raise MoneyLaunderingException(f"'{TARGET_COLUMN}' column not found in the DataFrame.")

            # Get allowed labels from schema and unique labels from DataFrame
            allowed_labels = self._schema_config["label_columns"][0].get(TARGET_COLUMN, [])
            unique_labels = dataframe[TARGET_COLUMN].unique()
            
            # Find invalid labels not in allowed labels
            invalid_labels = [label for label in unique_labels if label not in allowed_labels]
            
            # Check validation status and log information
            if not invalid_labels:
                logging.info("Class label validation passed. All class labels are within allowed labels.")
            else:
                logging.warning("Class label validation failed. Some class labels are not within allowed labels.")
                logging.warning(f"Invalid class labels: {invalid_labels}")

            # End the class label validation
            logging.info("Ending class label validation")
            
            # Return validation status and invalid labels
            if not invalid_labels:
                return True, []
            else:
                return False, invalid_labels

        except Exception as e:
            # Log the error and raise an exception
            logging.error("Error occurred during class label validation: %s", str(e))
            raise MoneyLaunderingException(e, sys)

        
    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        """
        Method Name: detect_dataset_drift
        Description: Detects dataset drift between two DataFrames based on the Kolmogorov-Smirnov test.
        
        Input:
            base_df (pd.DataFrame): The baseline DataFrame for comparison
            current_df (pd.DataFrame): The current DataFrame for comparison
            threshold (float, optional): The p-value threshold for significance
            
        Output:
            bool: True if no significant drift detected, False if drift detected
            
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Start the dataset drift detection
            logging.info('Starting dataset drift detection')
            
            # Initialize status and report dictionary
            status = True
            report = {}
            
            # Iterate through columns in the DataFrame
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                
                # Perform the Kolmogorov-Smirnov test
                is_same_dist = ks_2samp(d1, d2)
                
                # Check if the p-value is above the threshold
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                
                # Update the report dictionary with p-value and drift status
                report.update({
                    column: {
                        "p_value": float(is_same_dist.pvalue),
                        "drift_status": is_found
                    }
                })
            
            # Get the drift report file path from configuration
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Write the drift report to YAML file
            write_yaml_file(file_path=drift_report_file_path, content=report)
            
            # End the dataset drift detection
            logging.info('Ending dataset drift detection')
            
            # Return the overall drift detection status
            return status
            
        except Exception as e:
            # Log the error and raise an exception
            logging.error("Error occurred during dataset drift detection: %s", str(e))
            raise MoneyLaunderingException(e, sys)

        
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name: initiate_data_validation
        Description: Initiates the data validation component of the training pipeline.
        
        Output: DataValidationArtifact containing validation results
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            logging.info("Starting data validation process")
            error_message = ""  # Initialize an empty error message

            # Read train and test data from files
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            ## Validate number of columns
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message}Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message}Test dataframe does not contain all columns.\n"
        
            ## Validate numerical columns
            status = self.validate_numerical_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message}Train dataframe does not contain all numerical columns.\n"
            
            status = self.validate_numerical_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message}Test dataframe does not contain all numerical columns.\n"

        
            ## Validate time step has positive values 
            status = self.time_step_has_positive_values(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message} In Train dataframe time step does not contain positive values.\n"
            
            status = self.time_step_has_positive_values(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message} In Test dataframe time step does not contain positive values.\n"


            ## Validate class has label 1, 2, 3
            status = self.validate_class_labels(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message} In Train dataframe class does not contain label[1,2,3] values.\n"
            
            status = self.time_step_has_positive_values(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message} In Test dataframe class does not contain label[1,2,3] values.\n"


            if len(error_message) > 0:
                raise Exception(error_message)  # Raise an exception if there are validation errors

            # Check data drift
            status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)

            # Create DataValidationArtifact to store validation results
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path, 
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"data validation artifact: {data_validation_artifact}")
            logging.info("Data validation process completed.")
            return data_validation_artifact

        except Exception as e:
            logging.error("Error occurred during data validation: %s", str(e))
            raise MoneyLaunderingException(e, sys)


