import sys
import pandas as pd 
from typing import Tuple

from script.cloud_storage.s3_data_reader import read_s3_csv_to_dataframe
from script.constant.training_pipeline import *
from script.utils.main_utils import read_yaml_file, load_object
from script.constant.training_pipeline import SCHEMA_FILE_PATH 
from script.exception import MoneyLaunderingException
from script.logger import logging

class Money:
    def __init__(self, class_dataset_file_path=DATASET_PATH_CLASS,
                 feature_dataset_file_path=DATASET_PATH_FEATURE):
        """
        Method Name: __init__
        Description: Initializes the Money class with dataset file paths.
        
        Input:
        - class_dataset_file_path: Path to the class dataset file (default: DATASET_PATH_CLASS)
        - feature_dataset_file_path: Path to the feature dataset file (default: DATASET_PATH_FEATURE)
        
        On Failure: Raises an exception if there's an error during initialization.
        
        Version: 1.0
        """
        try:
            # Log the entry into the class
            logging.info("Entered Money class.")

            # Initialize class attributes with provided or default dataset file paths
            self.class_dataset_file_path = class_dataset_file_path
            self.feature_dataset_file_path = feature_dataset_file_path
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            # If an exception occurs during initialization, raise it with error details
            raise MoneyLaunderingException(e,sys)



    def save_csv_files(self,) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method Name: save_csv_files
        Description: Reads class and feature CSV files and returns their DataFrames.
        
        Input:
        - class_file_path: Path to the class dataset CSV file
        - feature_file_path: Path to the feature dataset CSV file
        
        Output: Tuple containing class DataFrame and feature DataFrame
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Log the entry into the method
            logging.info("Entered save_csv_files method.")

            # Read class and feature CSV files
            df_class = read_s3_csv_to_dataframe(bucket_name=dataset_bucket_name, file_name=DATASET_PATH_CLASS)
            df_feature = read_s3_csv_to_dataframe(bucket_name=dataset_bucket_name, file_name=DATASET_PATH_FEATURE)
            
            # Log successful read operation
            logging.info("CSV files read successfully.")

            return df_class, df_feature

        except Exception as e:
            # If an exception occurs during the operation, log it and raise an exception
            # #("Error occurred while reading CSV files: %s", str(e))
            raise MoneyLaunderingException(e, sys)

    
    def change_column_names(self, df_feature: pd.DataFrame) -> pd.DataFrame:
        """
        Method Name: change_column_names
        Description: Changes the column names of the feature DataFrame.
        
        Input:
        - df_feature: DataFrame containing feature data
        
        Output: DataFrame with updated column names
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Log the entry into the method
            logging.info("Entered change_column_names method.")

            # Call the save_csv_files method to read class and feature CSV files
            df_class, df_feature = self.save_csv_files()
            
            # Define the new column names
            colNames = ['txId', 'Time_step']
            colNames += [f'Local_feature_{i}' for i in range(1, 94)]
            colNames += [f'Aggregate_feature_{i}' for i in range(1, 73)]

            logging.info(f"df_feature head : {df_feature.head()}")
            # Rename the columns of the DataFrame
            df_feature.columns = colNames
            
            # Log successful column renaming
            logging.info("Column names changed successfully.")
            logging.info(f"df_feature head : {df_feature.head()}")
            return df_feature

        except Exception as e:
            raise MoneyLaunderingException(e, sys)


    def merge_dataset(self) -> pd.DataFrame:
        """
        Method Name: merge_dataset
        Description: Merges the class and feature datasets based on the 'txId' column.
        
        Output: Merged DataFrame containing class and feature data
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Log the entry into the method
            logging.info("Entered merge_dataset method.")

            # Call the save_csv_files method to read class and feature CSV files
            df_class, df_feature = self.save_csv_files()

            # Call the change_column_names method to change column names in the feature DataFrame
            df_feature = self.change_column_names(df_feature)

            # Merge the dataframes based on 'txId' column and select the first 20000 rows
            dataframe = df_feature.merge(df_class, on='txId', how='left')

            # dataframe = dataframe.drop(self.schema_config['drop_columns'], axis=1)

            dataframe = dataframe.head(20000)

            # Log successful dataset merging
            logging.info("Datasets merged successfully.")

            return dataframe
            
        except Exception as e:
            # # If an exception occurs during the operation, log it and raise an exception
            # #("Error occurred while merging datasets: %s", str(e))
            raise MoneyLaunderingException(e, sys)


# if __name__ == "__main__":
#     try:
#         money = Money()
#         merged_data = money.merge_dataset()
        
#         # The code beyond this point should handle the merged_data as needed
#         # Add your deployment logic here

#     except MoneyLaunderingException as mle:
#         # Handle custom exceptions gracefully
#         logging.error(f"Money Laundering Exception: {str(mle)}")
#     except Exception as e:
#         # Handle other exceptions gracefully
#         logging.error(f"An unexpected error occurred: {str(e)}")