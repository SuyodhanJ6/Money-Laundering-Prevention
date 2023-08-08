import os, sys
import shutil

from script.logger import logging
from script.exception import MoneyLaunderingException

from script.entity.artifact_enitity import ModelPusherArtifact, ModelEvaluationArtifact
from script.entity.config_enitity import ModelPusherConfig


class ModelPusher:
    """
    Class for pushing the trained model to a deployment environment.
    """
    def __init__(self, model_pusher_config: ModelPusherConfig, 
                 model_eval_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        Method Name: __init__
        Description: Initializes the ModelPusher class with required artifacts and configurations.
        
        Input:
        - model_pusher_config: ModelPusherConfig object containing model pusher configurations
        - model_eval_artifact: ModelEvaluationArtifact object containing model evaluation results
        
        Output: ModelPusherArtifact object containing information about the pushed model
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Initialize class attributes with provided inputs
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
            
            # Log the entry into the class
            logging.info("Entered ModelPusher class.")
        
        except Exception as e:
            # If an exception occurs during initialization, raise an exception with error details
            raise MoneyLaunderingException(e, sys)

        

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name: initiate_model_pusher
        Description: Initiates the model pushing process by copying the trained model to deployment locations.
        
        Output: ModelPusherArtifact containing paths to the saved model files
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        logging.info("Entered initiate_model_pusher method of ModelPusher class")
        try:
            # Get the path of the trained model from the model evaluation artifact
            trained_model_path: str = self.model_eval_artifact.trained_model_path

            # Log the paths for the model file and deployment locations
            logging.info(f"Trained model path: {trained_model_path}")
            logging.info(f"Model file path: {self.model_pusher_config.model_file_path}")
            logging.info(f"Saved model path: {self.model_pusher_config.saved_model_path}")

            # Create directories to save the model files
            model_file_path: str = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

            # Copy the trained model to the specified model file path
            shutil.copy(src=trained_model_path, dst=model_file_path)
            logging.info("Trained model copied to model file path")

            # Create directories to save the model files for deployment
            save_model_path: str = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

            # Copy the trained model to the deployment location
            shutil.copy(src=trained_model_path, dst=save_model_path)
            logging.info("Trained model copied to saved model path")

            # Prepare the model pusher artifact with the saved model paths
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=save_model_path,
                                                        model_file_path=model_file_path)

            # Log the artifact information
            logging.info(f"Model pusher artifact: {model_pusher_artifact}")

            # Return the model pusher artifact
            return model_pusher_artifact

        except Exception as e:
            # If an exception occurs during model pushing, log the error and raise an exception
            logging.error("Error occurred during model pushing: %s", str(e))
            raise MoneyLaunderingException(e, sys)
