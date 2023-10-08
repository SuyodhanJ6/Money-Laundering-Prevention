import os, sys
from pandas import DataFrame
import pandas as pd 
import numpy as np
import boto3

from script.logger import logging
from script.exception import MoneyLaunderingException
from script.entity.config_enitity import PredictionPipelineConfig
from script.utils.main_utils import read_yaml_file, load_object
from script.constant.training_pipeline import SCHEMA_FILE_PATH 
from script.constant.prediction_pipeline import PREDICTION_INPUT_FILE_PATH 
from script.ml.model.estimetor import ModelResolver, Money_Laundering, TargetValueMapping


class PredictionPipeline:
    def __init__(self, prediction_pipeline_config : PredictionPipelineConfig = PredictionPipelineConfig()):
        try:

            self.prediction_pipeline_config = prediction_pipeline_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        
        except Exception as e:
            raise MoneyLaunderingException(e, sys)
        

    def get_data(self, ) -> DataFrame:
        try:
            
            logging.info("Entered get_data method of PredictionPipeline class")

            prediction_df = pd.read_csv(PREDICTION_INPUT_FILE_PATH)

            prediction_df = prediction_df.head(20000)

            logging.info("Read prediction csv file Local Fodlder")

            prediction_df = prediction_df.drop(self.schema_config['drop_columns'], axis=1)

            logging.info("Dropped the required columns")

            logging.info("Exited the get_data method of PredictionPipeline class")

            return prediction_df


        except Exception as e:
            raise MoneyLaunderingException(e, sys)
        

    def predict(self, dataframe) -> np.array:
        try:

            logging.info("Entered predict method of PredictionPipeline class")

            model_resolver = ModelResolver()
            
            if not model_resolver.is_model_exists():
                raise "Model Not Exists !"
            
            best_model_path = model_resolver.get_best_model_path()

            model =  load_object(file_path=best_model_path)
            
            # dataframe : DataFrame = self.get_data()
            # y_pred = 

            # dataframe['predicted_column'] = y_pred
            # dataframe['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)

            return model.predict(dataframe)

        except Exception as e:
            raise MoneyLaunderingException(e, sys)
        


    def save_local(self, predicted_dataframe, file_path):
        try:
            predicted_dataframe.to_csv(file_path, index=False)
            logging.info(f"Saved prediction results to {file_path}")
        except Exception as e:
            raise MoneyLaunderingException(e, sys)
        

    def save_to_s3(self, predicted_dataframe, s3_bucket_name, s3_object_key):
        try:
            s3_client = boto3.client('s3')

            # Convert DataFrame to CSV as bytes
            csv_buffer = predicted_dataframe.to_csv(index=False).encode('utf-8')

            # Upload to S3
            s3_client.put_object(Bucket=s3_bucket_name, Key=s3_object_key, Body=csv_buffer)

            logging.info(f"Saved prediction results to S3 bucket: {s3_bucket_name}/{s3_object_key}")
        except Exception as e:
            raise MoneyLaunderingException(e, sys)

    def initiate_prediction(self) -> pd.DataFrame:
        try:
            dataframe = self.get_data()
            predicted_arr = self.predict(dataframe)
            prediction = pd.DataFrame(list(predicted_arr))
            prediction.columns = ["class"]
            prediction.replace(TargetValueMapping().reverse_mapping(), inplace=True)
            predicted_dataframe = pd.concat([dataframe, prediction], axis=1)
            
            # Save locally
            local_file_path = "predicted_results.csv"
            self.save_local(predicted_dataframe, local_file_path)

            # Save to S3
            s3_bucket_name = "money-laund-datasource"
            s3_object_key = "predicted_results.csv"
            self.save_to_s3(predicted_dataframe, s3_bucket_name, s3_object_key)

            return predicted_dataframe

        except Exception as e:
            raise MoneyLaunderingException(e, sys)

    # def initiate_prediction(self,) -> None:
    #     try:
    #         dataframe = self.get_data()

    #         predicted_arr = self.predict(dataframe)
            
    #         prediction = pd.DataFrame(list(predicted_arr))

    #         prediction.columns = ["class"]

    #         prediction.replace(TargetValueMapping().reverse_mapping(), inplace=True)

    #         predicted_dataframe = pd.concat([dataframe, prediction], axis=1)
            
    #         logging.info("Uploaded artifacts folder to s3 bucket_name")

    #         return predicted_dataframe

    #     except Exception as e:
    #         raise MoneyLaunderingException(e, sys)

if __name__ == "__main__":
    prediction_pipeline = PredictionPipeline()

    prediction_pipeline.initiate_prediction()