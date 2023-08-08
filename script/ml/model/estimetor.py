import os, sys 
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from script.logger import logging
from script.exception import MoneyLaunderingException
from script.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME


class TargetValueMapping:
    """
    Class for mapping target values to their corresponding labels.
    """
    def __init__(self):
        """
        Method Name: __init__
        Description: Initializes the TargetValueMapping class with mapping values.
        
        Input: None
        
        Version: 1.0
        """
        self.unknown: int = 3

    def to_dict(self):
        """
        Method Name: to_dict
        Description: Returns the mapping values as a dictionary.
        
        Output: Dictionary containing the mapping values.
        
        Version: 1.0
        """
        return self.__dict__

    def reverse_mapping(self):
        """
        Method Name: reverse_mapping
        Description: Returns a reverse mapping of the values.
        
        Output: Dictionary containing the reverse mapping.
        
        Version: 1.0
        """
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))



class Money_Laundering:
    """
    Class representing the Money Laundering model.
    """
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        Method Name: __init__
        Description: Initializes the Money_Laundering class with preprocessing and trained model objects.
        
        Input:
        - preprocessing_object: Preprocessing object (Pipeline) used for feature transformation
        - trained_model_object: Trained model object
        
        Version: 1.0
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object


    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Method Name: predict
        Description: Predicts the class labels using the trained model.
        
        Input:
        - dataframe: Input DataFrame containing features for prediction
        
        Output: DataFrame containing predicted class labels
        
        Version: 1.0
        """
        logging.info("Entered predict method of Money_Laundering class")

        try:
            # Transform the input data using the preprocessing object
            logging.info("Using the trained model to get predictions")
            x_transform = self.preprocessing_object.transform(dataframe)
            
            # Predict class labels using the trained model
            y_hat = self.trained_model_object.predict(x_transform)

            logging.info("Used the trained model to get predictions")

            return y_hat

        except Exception as e:
            raise MoneyLaunderingException(e, sys)


    def __repr__(self):
        """
        Method Name: __repr__
        Description: Returns a string representation of the Money_Laundering object.
        
        Output: String representation of the trained model object's class name
        
        Version: 1.0
        """
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        """
        Method Name: __str__
        Description: Returns a string representation of the Money_Laundering object.
        
        Output: String representation of the trained model object's class name
        
        Version: 1.0
        """
        return f"{type(self.trained_model_object).__name__}()"

    


class ModelResolver:
    def __init__(self, model_dir=SAVED_MODEL_DIR):
        """
        Method Name: __init__
        Description: Initializes the ModelResolver class with the directory path for saved models.
        
        Input:
        - model_dir: Directory path for saved models
        
        On Failure: Raises an exception if there's an error during initialization.
        
        Version: 1.0
        """
        logging.info("Entering ModelResolver class")
        try:
            # Initialize class attributes with provided inputs
            self.model_dir = model_dir

        except Exception as e:
            # If an exception occurs during initialization, raise an exception with error details
            raise MoneyLaunderingException(e, sys)

        

    def get_best_model_path(self) -> str:
        """
        Method Name: get_best_model_path
        Description: Retrieves the file path of the latest saved model.
        
        Output: File path of the latest saved model.
        
        Version: 1.0
        """
        try:
            # Get timestamps from the subdirectories containing saved models
            timestamps = list(map(int, os.listdir(self.model_dir)))
            # Find the latest timestamp
            latest_timestamp = max(timestamps)
            # Construct the file path of the latest saved model
            latest_model_path = os.path.join(self.model_dir, f"{latest_timestamp}", MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            # If an exception occurs, raise it with error details
            raise MoneyLaunderingException(e, sys)

    def is_model_exists(self) -> bool:
        """
        Method Name: is_model_exists
        Description: Checks whether any saved model exists.
        
        Output: True if saved model(s) exist, False otherwise.
        
        Version: 1.0
        """
        try:
            # Check if the model directory exists
            if not os.path.exists(self.model_dir):
                return False

            # Get the list of timestamps (subdirectories) within the model directory
            timestamps = os.listdir(self.model_dir)
            
            # If there are no timestamps (no saved models), return False
            if len(timestamps) == 0:
                return False
            
            # Get the file path of the latest saved model
            latest_model_path = self.get_best_model_path()

            # Check if the file path of the latest model exists
            if not os.path.exists(latest_model_path):
                return False

            # If all checks pass, return True indicating that saved model(s) exist
            return True
        except Exception as e:
            # If an exception occurs, raise it with error details
            raise MoneyLaunderingException(e, sys)
