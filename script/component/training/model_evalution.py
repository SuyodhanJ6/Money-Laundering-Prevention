import os, sys
from pandas import DataFrame
import pandas as pd 
from script.logger import logging
from script.exception import MoneyLaunderingException

from script.entity.config_enitity import ModelEvaluationConfig
from script.entity.artifact_enitity import ModelTrainerArtifact, DataValidationArtifact, ModelEvaluationArtifact
from script.constant.training_pipeline import TARGET_COLUMN
from script.ml.model.estimetor import TargetValueMapping, ModelResolver
from script.utils.main_utils import write_yaml_file, load_object
from script.ml.metric.classification_metric import get_classification_score


class ModelEvaluation:
    """
    Class for evaluating the performance of a trained model.
    """
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        Method Name: __init__
        Description: Initializes the ModelEvaluation class with required artifacts and configurations.
        
        Input:
        - model_eval_config: ModelEvaluationConfig object containing model evaluation configurations
        - data_validation_artifact: DataValidationArtifact object containing paths to validated data
        - model_trainer_artifact: ModelTrainerArtifact object containing paths to the trained model and metrics
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Initialize class attributes with provided inputs
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            
            # Log the entry into the class
            logging.info("Entered ModelEvaluation class.")
        
        except Exception as e:
            # If an exception occurs during initialization, raise an exception with error details
            raise MoneyLaunderingException(e, sys)

        
    @staticmethod
    def read_csv(file_path: str) -> DataFrame:
        """
        Method Name: read_csv
        Description: Reads a CSV file and returns its content as a DataFrame.
        
        Input:
        - file_path: Path to the CSV file
        
        Output: DataFrame containing the data from the CSV file
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            # Use pandas to read the CSV file and return the DataFrame
            return pd.read_csv(file_path)
        except Exception as e:
            # If an exception occurs during reading, raise an exception with error details
            raise MoneyLaunderingException(e, sys)

        
    def initiate_model_evaluation(self, ) -> ModelEvaluationArtifact:
        """
        Method Name: initiate_model_evaluation
        Description: Initiates the model evaluation process.
        
        Output: ModelEvaluationArtifact containing evaluation results and model paths
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        logging.info("Entered initiate_model_evaluation method of ModelEvaluation class")
        try:
            validation_train_file_path = self.data_validation_artifact.valid_train_file_path
            validation_test_file_path = self.data_validation_artifact.valid_test_file_path

            # Logging to indicate reading validation data
            logging.info("Reading validation data")

            train_df = ModelEvaluation.read_csv(validation_train_file_path)
            test_df = ModelEvaluation.read_csv(validation_test_file_path)

            # Concatenate train and test data
            df = pd.concat([train_df, test_df])

            # Get true labels
            y_true = df[TARGET_COLUMN]
            y_true = y_true.replace(TargetValueMapping().to_dict())  # Apply label mapping
            df.drop(TARGET_COLUMN, axis=1, inplace=True)  # Drop the target column

            # Get paths to trained and latest models
            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()

            is_model_accepted = True

            # Check if the best model exists
            if not model_resolver.is_model_exists():
                # Logging to indicate no best model found
                logging.info("No best model found.")

                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=None,
                    best_model_path=None,
                    trained_model_path=train_model_file_path,
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact,
                    best_model_metric_artifact=None)
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            y_trained_pred = train_model.predict(df)
            y_latest_pred = latest_model.predict(df)

            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

            improved_accuracy = trained_metric.f1_score - latest_metric.f1_score

            # Logging to indicate model evaluation results
            logging.info("Model evaluation results:")
            logging.info(f"Improved Accuracy: {improved_accuracy}")
            logging.info(f"Change Threshold: {self.model_eval_config.change_threshold}")

            if self.model_eval_config.change_threshold < improved_accuracy:
            # If the improvement in accuracy is greater than the specified threshold,
            # the model is considered accepted.
                is_model_accepted = True
            else:
                # If the improvement in accuracy is not greater than the specified threshold,
                # the model is not considered accepted.
                is_model_accepted = False

            # Create ModelEvaluationArtifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                improved_accuracy=improved_accuracy,
                best_model_path=latest_model_path,
                trained_model_path=train_model_file_path,
                train_model_metric_artifact=trained_metric,
                best_model_metric_artifact=latest_metric)

            # Create a report dictionary and save it
            model_eval_report = model_evaluation_artifact.__dict__
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            logging.error("An error occurred during model evaluation: %s", str(e))
            raise MoneyLaunderingException(e, sys)
