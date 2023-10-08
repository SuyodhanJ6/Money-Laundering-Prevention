import os, sys

from script.entity.config_enitity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from script.entity.artifact_enitity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact
from script.component.training.data_ingestion import DataIngestion
from script.component.training.data_validation import DataValidation
from script.component.training.data_transformation import DataTransformation
from script.component.training.model_trainer import ModelTrainer
from script.component.training.model_evalution import ModelEvaluation
from script.component.training.model_pusher import ModelPusher
from script.cloud_storage.s3_syncer import S3Sync
from script.constant.s3_bucket import TRAINING_BUCKET_NAME
from script.constant.training_pipeline import SAVED_MODEL_DIR

from script.logger import logging
from script.exception import MoneyLaunderingException

class TrainPipeline:
    is_pipeline_running = False
    
    def __init__(self):
        """
        Method Name: __init__
        Description: Initializes the TrainPipeline class.
        
        On Failure: Raises an exception if there's an error during initialization.
        
        Version: 1.0
        """
        try:
            # Initialize the training pipeline configuration
            self.training_pipeline_config = TrainingPipelineConfig()
            self.s3_sync = S3Sync()
        except Exception as e:
            # If an exception occurs during initialization, raise it with error details
            raise MoneyLaunderingException(e, sys)

 
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name: start_data_ingestion
        Description: Starts the data ingestion process.
        
        Output: DataIngestionArtifact containing paths to ingested data.
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Initialize the data ingestion configuration
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            
            # Log the start of data ingestion process
            logging.info("Starting data ingestion")
            
            # Create a DataIngestion object and initiate the data ingestion process
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            
            # Log the completion of data ingestion process and return the artifact
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            # If an exception occurs, raise it with error details
            raise MoneyLaunderingException(e, sys)

    
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Method Name: start_data_validation
        Description: Starts the data validation process.
        
        Input:
        - data_ingestion_artifact: DataIngestionArtifact containing paths to ingested data.
        
        Output: DataValidationArtifact containing paths to validated data.
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Initialize the data validation configuration
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            
            # Log the start of data validation process
            logging.info("Starting data validation")
            
            # Create a DataValidation object and initiate the data validation process
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            
            # Log the completion of data validation process and return the artifact
            logging.info(f"Data validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            # If an exception occurs, raise it with error details
            raise MoneyLaunderingException(e, sys)

        
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Method Name: start_data_transformation
        Description: Starts the data transformation process.
        
        Input:
        - data_validation_artifact: DataValidationArtifact containing paths to validated data.
        
        Output: DataTransformationArtifact containing paths to transformed data and preprocessing object.
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Initialize the data transformation configuration
            data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            
            # Log the start of data transformation process
            logging.info("Starting data transformation")
            
            # Create a DataTransformation object and initiate the data transformation process
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            
            # Log the completion of data transformation process and return the artifact
            logging.info(f"Data transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            # If an exception occurs, raise it with error details
            raise MoneyLaunderingException(e, sys)



    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Method Name: start_model_training
        Description: Starts the model training process.
        
        Input:
        - data_transformation_artifact: DataTransformationArtifact containing paths to transformed data and preprocessing object.
        
        Output: ModelTrainerArtifact containing paths to the trained model and metrics.
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Initialize the model trainer configuration
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            
            # Create a ModelTrainer object and initiate the model training process
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            # Return the trained model artifact
            return model_trainer_artifact
        except Exception as e:
            # If an exception occurs, raise it with error details
            raise MoneyLaunderingException(e, sys)



    def start_model_evaluation(self, data_validation_artifact: DataValidationArtifact,
                           model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        Method Name: start_model_evaluation
        Description: Starts the model evaluation process.
        
        Input:
        - data_validation_artifact: DataValidationArtifact containing paths to validated data.
        - model_trainer_artifact: ModelTrainerArtifact containing paths to the trained model and metrics.
        
        Output: ModelEvaluationArtifact containing evaluation results.
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Initialize the model evaluation configuration
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            
            # Create a ModelEvaluation object and initiate the model evaluation process
            model_eval = ModelEvaluation(model_eval_config=model_eval_config,
                                        data_validation_artifact=data_validation_artifact,
                                        model_trainer_artifact=model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            
            # Return the model evaluation artifact
            return model_eval_artifact

        except Exception as e:
            # If an exception occurs, raise it with error details
            raise MoneyLaunderingException(e, sys)
        


    def start_model_pusher(self, model_eval_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        Method Name: start_model_pusher
        Description: Starts the model pusher process.
        
        Input:
        - model_eval_artifact: ModelEvaluationArtifact containing evaluation results.
        
        Output: ModelPusherArtifact containing information about the saved model.
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Initialize the model pusher configuration
            model_pusher_config = ModelPusherConfig(self.training_pipeline_config)
            
            # Create a ModelPusher object and initiate the model pusher process
            model_push = ModelPusher(model_pusher_config=model_pusher_config, model_eval_artifact=model_eval_artifact)
            model_push_artifact = model_push.initiate_model_pusher()
            
            # Return the model pusher artifact
            return model_push_artifact

        except Exception as e:
            # If an exception occurs, raise it with error details
            raise MoneyLaunderingException(e, sys)


    def sync_artifact_dir_to_s3(self):
        """
        Method Name: sync_artifact_dir_to_s3
        Description: Syncs the local artifact directory to an Amazon S3 bucket.
        
        Output: None
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Construct the AWS bucket URL
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            
            # Sync the local artifact directory to S3 bucket
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
        
        except Exception as e:
            # If an exception occurs during synchronization, raise it with error details
            raise MoneyLaunderingException(e, sys)

        

    def sync_saved_model_dir_to_s3(self):
        """
        Method Name: sync_saved_model_dir_to_s3
        Description: Syncs the local saved model directory to an Amazon S3 bucket.
        
        Output: None
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Construct the AWS bucket URL
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/{SAVED_MODEL_DIR}"
            
            # Sync the local saved model directory to S3 bucket
            self.s3_sync.sync_folder_to_s3(folder=SAVED_MODEL_DIR, aws_bucket_url=aws_bucket_url)

        except Exception as e:
            # If an exception occurs during synchronization, raise it with error details
            raise MoneyLaunderingException(e, sys)



    def run_pipeline(self):
        """
        Method Name: run_pipeline
        Description: Runs the complete training pipeline.
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            TrainPipeline.is_pipeline_running = True
            # Start the data ingestion process
            data_ingestion_artifact = self.start_data_ingestion()

            # Start the data validation process using the data ingestion artifact
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact)

            # Start the data transformation process using the data validation artifact
            data_transformation_artifact = self.start_data_transformation(
                data_validation_artifact=data_validation_artifact)

            # Start the model training process using the data transformation artifact
            model_trainer_artifact = self.start_model_training(
                data_transformation_artifact=data_transformation_artifact)

            # Start the model evaluation process using data validation and model trainer artifacts
            model_eval_artifact = self.start_model_evaluation(data_validation_artifact=data_validation_artifact, 
                                                            model_trainer_artifact=model_trainer_artifact)

            if not model_eval_artifact.is_model_accepted:
                raise Exception("Trained model is not better than the best model")
            
            # Start the model pusher process using the model evaluation artifact
            model_pusher_artifact = self.start_model_pusher(
                model_eval_artifact=model_eval_artifact)
                
            TrainPipeline.is_pipeline_running=False

            # Sync the entire artifact directory to the designated Amazon S3 bucket.
            self.sync_artifact_dir_to_s3()

            # Sync the saved model directory to the specified Amazon S3 bucket.
            self.sync_saved_model_dir_to_s3()


            
        except Exception as e:
            self.sync_artifact_dir_to_s3()
            TrainPipeline.is_pipeline_running=False 
            # If an exception occurs, raise it with error details
            raise MoneyLaunderingException(e, sys)

# if __name__ == "__main__":
#      train_pipeline = TrainPipeline() 
#      train_pipeline.run_pipeline()