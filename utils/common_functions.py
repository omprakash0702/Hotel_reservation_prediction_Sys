import os
import pandas
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info(f"YAML file {file_path} read successfully.")
            return config
    except Exception as e:
        logger.error(f"Error reading YAML file")
        raise CustomException("Failed to read YAML file", e)

def load_data(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")
        df = pandas.read_csv(path)
        logger.info(f"Data loaded successfully from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {path}")
        raise CustomException("Failed to load data", e)