import os
import pandas as pd 
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from script.logger import logging
from script.exception import MoneyLaunderingException
from script.constant.training_pipeline import *
from script.entity.config_enitity import DataIngestionConfig, TrainingPipelineConfig
from script.entity.artifact_enitity import DataIngestionArtifact
from script.data_access.money import Money
from script.utils.main_utils import read_yaml_file


class DataIngestion:
    """
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Method Name: __init__
        Description: Initializes the DataIngestion class.
        
        Input: data_ingestion_config - Configuration for data ingestion
        On Failure: Writes an exception log and then raises an exception.
        
        Version: 1.0
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("Entered DataIngestion class.")

        except Exception as e:
            raise MoneyLaunderingException(e)

    def export_local_dataset_to_feature_store(self) -> pd.DataFrame:
        """
        Method Name: export_local_dataset_to_feature_store
        Description: This method exports the local dataset as a DataFrame into the feature store.
        
        Output: DataFrame containing the exported dataset
        On Failure: Writes an exception log and then raises an exception.
        
        Version: 1.0
        """
        try:
            logging.info("Exporting local dataset to feature store")
            
            # Instantiate Money class to merge the dataset
            money_data = Money()
            dataframe = money_data.merge_dataset()
            
            # Get the feature store file path from the configuration
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            
            # Create the directory path if it does not exist
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save the DataFrame as CSV in the feature store file path
            logging.info("Saving DataFrame to feature store file path")
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            
            logging.info("Export completed successfully.")
            return dataframe
        
        except Exception as e:
            logging.error("Error occurred during dataset export: %s", str(e))
            raise MoneyLaunderingException(e)




    def split_data_as_train_test(self, dataframe : DataFrame) -> None:
        """
        Method Name: split_data_as_train_test
        Description: This method performs the train-test split on the provided dataframe.
        
        Input: dataframe - The dataframe to be split
        
        Output: None
        On Failure: Raises an exception.
        
        Version: 1.0
        """
        try:
            logging.info("Performing train-test split on the dataframe")
            
            # Perform train-test split
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Train-test split has been performed on the dataframe.")
            
            # Create the directory path for the output files
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Export train and test sets to CSV files
            logging.info("Exporting train and test sets to CSV files")
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
            logging.info("Train and test sets exported to CSV files.")
            
        except Exception as e:
            logging.error("Error occurred during train-test split: %s", str(e))
            raise MoneyLaunderingException(e)


    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name: initiate_data_ingestion
        Description: This method initiates the data ingestion component of the training pipeline.
        
        Output: training_set and testing_set
        On Failure: Writes an exception log and then raises an exception.
        
        Version: 1.0
        """
        try:
            logging.info("Starting data ingestion process")

            # Export local dataset to feature store
            dataframe = self.export_local_dataset_to_feature_store()
            logging.info("Local dataset exported to feature store.")

            # Read schema config
            _schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

            # Drop unnecessary columns based on schema
            dataframe = dataframe.drop(_schema_config[SCHEMA_DROP_COLS], axis=1)
            logging.info("Dropped unnecessary columns based on schema.")

            # Split data into train and test sets
            self.split_data_as_train_test(dataframe=dataframe)
            logging.info("Data split into train and test sets.")

            # Create DataIngestionArtifact
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info("Data ingestion process completed.")

            return data_ingestion_artifact

        except Exception as e:
            logging.error("Error occurred during data ingestion: %s", str(e))
            raise MoneyLaunderingException(e)

