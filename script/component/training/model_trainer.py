import os, sys
from sklearn.ensemble import RandomForestClassifier

from script.logger import logging
from script.exception import MoneyLaunderingException

from script.entity.artifact_enitity import DataTransformationArtifact, ModelTrainerArtifact
from script.entity.config_enitity import ModelTrainerConfig
from script.utils.main_utils import load_numpy_array_data, load_object, save_object
from script.ml.metric.classification_metric import get_classification_score 
from script.ml.model.estimetor import Money_Laundering

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Class constructor to initialize the ModelTrainer.

        Input:
        - model_trainer_config: Configuration for the model trainer.
        - data_transformation_artifact: DataTransformationArtifact containing paths to transformed data.

        The constructor initializes the ModelTrainer object by storing the provided configuration
        and data transformation artifact. It also adds a log statement to indicate when the class
        is being entered.

        Version: 1.0
        """
        try:
            # Store the provided configuration and data transformation artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            
            # Log that the ModelTrainer class has been entered
            logging.info("Entered ModelTrainer class.")

        except Exception as e:
            # In case of any exception during initialization, raise a MoneyLaunderingException
            raise MoneyLaunderingException(e, sys)

        
    
    def train_model(self, x_train, y_train) -> RandomForestClassifier:
        """
        Method Name: train_model
        Description: Trains a Random Forest classifier on the provided training data.
        
        Input:
            x_train (numpy.ndarray): Features of the training data.
            y_train (numpy.ndarray): Target labels of the training data.
        
        Output:
            random_forest (RandomForestClassifier): Trained Random Forest classifier model.
        
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        try:
            logging.info("Entering train_model method")
            
            # Create a Random Forest classifier
            random_forest = RandomForestClassifier()
            
            # Fit the classifier on the training data
            random_forest.fit(x_train, y_train)
            
            logging.info("Exiting train_model method")
            return random_forest
        
        except Exception as e:
            logging.error("Error occurred in train_model method: %s", str(e))
            raise MoneyLaunderingException(e, sys)


    
    
    def initiate_model_trainer(self, ) -> ModelTrainerArtifact:
        """
        Method Name: initiate_model_trainer
        Description: Initiates the model training process.
        
        Output: ModelTrainerArtifact containing paths to the trained model, training, and testing metrics
        On Failure: Writes an exception log and raises an exception.
        
        Version: 1.0
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            # Load transformed train and test arrays
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            # Split data into features and target labels
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Train the model
            model = self.train_model(x_train=x_train, y_train=y_train)
            y_train_pred = model.predict(x_train)

            # Calculate classification metrics for training data
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            # Check if the trained model meets the expected accuracy
            if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                logging.info("Trained model is not good to provide expected accuracy")
                raise Exception("Trained model is not good to provide expected accuracy")

            # Predict on test data and calculate classification metrics
            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Calculate difference in F1 scores between training and testing
            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)

            # Compare the difference with the overfitting_underfitting_threshold
            # if diff > self.model_trainer_config.overfitting_underfitting_threshold:
            #     logging.info("Model is not good; try to do more experimentation.")
            #     raise Exception("Model is not good; try to do more experimentation.")

            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # Create Money Laundering model object
            money_laun_model = Money_Laundering(preprocessing_object=preprocessing_obj, trained_model_object=model)
            logging.info("Created Money Laundering truck model object with preprocessor and model")

            # Create directory for the trained model file
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            # Save the trained model object
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=money_laun_model)
            logging.info("Created best model file path.")

            # Prepare ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact

        except Exception as e:
            logging.error("Error occurred in initiate_model_trainer method: %s", str(e))
            raise MoneyLaunderingException(e, sys)

        

    