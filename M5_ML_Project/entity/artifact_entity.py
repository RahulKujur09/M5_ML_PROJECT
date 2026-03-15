from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path : str
    test_file_path : str
    feature_store_path : str

@dataclass
class DataValidationartifacts:
    drift_report_file_path : str
    valid_train_set_file_path : str
    valid_test_set_file_path : str
    invalid_train_set_file_path : str
    invalid_test_set_file_path : str