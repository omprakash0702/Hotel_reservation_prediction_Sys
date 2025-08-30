from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessing   
from src.model_training import ModelTraining
from config.paths_config import *
from utils.common_functions import read_yaml 


if __name__ == "__main__":
    
    ### 1. DATA INGESTION
    
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    ### 2. DATA PROCESSING
    
    data_processor = DataProcessing(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config=CONFIG_PATH
    )
    data_processor.process()
    
    
    ### 3. MODEL TRAINING
    
    trainer = ModelTraining(PROCESSED_TRAIN_FILE_PATH,PROCESSED_TEST_FILE_PATH,MODEL_OUTPUT_PATH)
    trainer.run()
