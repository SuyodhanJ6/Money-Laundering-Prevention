from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Data class representing the file paths for trained and test data used in the data ingestion process.

    Attributes:
        trained_file_path (str): The file path for the trained data.
        test_file_path (str)   : The file path for the test data.
    """

    trained_file_path: str
    test_file_path: str


    
@dataclass
class DataValidationArtifact:
    """
    Data class representing the results and file paths from the data validation process.

    Attributes:
        validation_status (bool)     : Whether the validation was successful (True) or not (False).
        valid_train_file_path (str)  : File path of the validated training data.
        valid_test_file_path (str)   : File path of the validated test data.
        invalid_train_file_path (str): File path of the invalid training data (if applicable).
        invalid_test_file_path (str) : File path of the invalid test data (if applicable).
        drift_report_file_path (str) : File path of the dataset drift report.
    """

    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str



@dataclass
class DataTransformationArtifact:
    """
    Data class representing the results and file paths from the data transformation process.

    Attributes:
        transformed_object_file_path (str): File path of the saved data transformation object.
        transformed_train_file_path (str) : File path of the transformed training data.
        transformed_test_file_path (str)  : File path of the transformed test data.
    """

    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str





@dataclass
class ClassificationMetricArtifact:
    """
    Data class representing classification performance metrics.

    Attributes:
        f1_score (float)        : F1 score metric.
        precision_score (float) : Precision score metric.
        recall_score (float)    : Recall score metric.
    """

    f1_score: float
    precision_score: float
    recall_score: float



@dataclass
class ModelTrainerArtifact:
    """
    Data class representing artifacts related to model training.

    Attributes:
        trained_model_file_path (str): File path to the trained model.

        train_metric_artifact (ClassificationMetricArtifact): Metrics of the model on the training data.
        test_metric_artifact (ClassificationMetricArtifact) : Metrics of the model on the test data.
    """

    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact



@dataclass
class ModelEvaluationArtifact:
    """
    Data class representing artifacts related to model evaluation.

    Attributes:
        is_model_accepted (bool) : Indicates whether the model is accepted based on evaluation.
        improved_accuracy (float): Improvement in accuracy compared to a previous model.
        best_model_path (str)    : File path to the best model.
        trained_model_path (str) : File path to the trained model.

        train_model_metric_artifact (ClassificationMetricArtifact):
            Metrics of the model on the training data.
        best_model_metric_artifact (ClassificationMetricArtifact):
            Metrics of the best model on the test data.
    """

    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: ClassificationMetricArtifact
    best_model_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelPusherArtifact:
    """
    Data class representing artifacts related to model pushing.

    Attributes:
        saved_model_path (str): File path where the model is saved for backup.
        model_file_path (str) : File path where the model is pushed for deployment.
    """

    saved_model_path: str
    model_file_path: str

@dataclass
class ModelPredictionArtifact:
    model_prediction_path: str
    predction_file_name: str