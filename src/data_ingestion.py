import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config['data_ingestion']
        self.bucket_name = self.config['bucket_name']
        self.bucket_file_name = self.config['bucket_file_name']
        self.train_test_ratio = self.config['train_ratio']
        
        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"DataIngestion started with bucket: {self.bucket_name} and file: {self.bucket_file_name}")

 
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            blob.download_to_filename(RAW_FILE_PATH)
            logger.info(f"Downloaded {self.bucket_file_name} from GCS bucket {self.bucket_name} to {RAW_FILE_PATH}")
        except Exception as e:
            logger.error("Error downloading file from GCS")
            raise CustomException("Failed to download file from GCS", e)
    
    def split_data(self):
        try:
            df = pd.read_csv(RAW_FILE_PATH)
            train_df, test_df = train_test_split(df, train_size=self.train_test_ratio, random_state=42)
            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Data split into train and test sets with ratio {self.train_test_ratio}")
        except Exception as e:
            logger.error("Error splitting data")
            raise CustomException("Failed to split data", e)
    
    def run(self):
        try:
            logger.info("Starting data ingestion process")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully.")
        except Exception as e:
            logger.error("Data ingestion failed")
        
        finally:
            logger.info("Data Ingestion process finished.")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

