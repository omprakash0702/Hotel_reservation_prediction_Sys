import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.logger import get_logger
from src.custom_exception import CustomException    
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from imblearn.over_sampling import SMOTE    



logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, train_path, test_path, processed_dir, config):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        
        self.config = read_yaml(config)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def preprocess_data(self, df):
        try:
            logger.info("Starting data preprocessing")

            logger.info("Dropping the columns")

            df.drop(columns=['Booking_ID'], inplace=True)

            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_features"]
            num_cols = self.config["data_processing"]["numerical_features"]

            logger.info("Encoding categorical features")
            le = LabelEncoder()
            mapping = {}
            for col in cat_cols:
                df[col] = le.fit_transform(df[col])
                mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            
            logger.info("label Mappingd is as follows:")
            for col, map_dict in mapping.items():
                logger.info(f"{col}: {mapping}") 
            
            logger.info("Skewness correction using log1p")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            skewness = df[num_cols].skew().abs()
            for col in num_cols:
                if skewness[col] > skew_threshold:
                    df[col] = np.log1p(df[col])
                    logger.info(f"Applied log1p transformation on {col} due to high skewness of {skewness[col]}")
            return df
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def balance_data(self, df):
        try:
            logger.info("Handling data imbalance using SMOTE")
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)

            balacnced_df = pd.DataFrame(X_res, columns=X.columns)
            balacnced_df['booking_status'] = y_res
            logger.info("Data balancing completed")
            return balacnced_df
        except Exception as e:
            logger.error(f"Error in balancing data: {e}")
            raise CustomException(e, sys) from e 
    
    def select_features(self, df):
        try:
            logger.info("Starting feature selection step") 
            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

            feature_importances = rf.feature_importances_

            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
            top_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            num_features_to_select = self.config['data_processing']['num_of_features']
            top_10_features = top_feature_importance_df['Feature'].head(num_features_to_select).values

            logger.info(f"Top {num_features_to_select} features selected: {top_10_features}")
            top_10_df = df[top_10_features.tolist() + ['booking_status']]

            logger.info("Feature selection completed")
            return top_10_df
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            raise CustomException(e, sys) from e        

    def save_data(self, df, filename):
        try:
            file_path = os.path.join(self.processed_dir, filename)
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise CustomException(e, sys) from e
    def process(self):
        try:
            logger.info("Loading training and testing data")
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            logger.info("Preprocessing training data")
            train_df = self.preprocess_data(train_df)

            logger.info("Preprocessing testing data")
            test_df = self.preprocess_data(test_df)

            logger.info("Balancing training data")
            train_df = self.balance_data(train_df)

            logger.info("Balancing testing data")
            test_df = self.balance_data(test_df)

            logger.info("Selecting features from training data")
            train_df = self.select_features(train_df)

            logger.info("Selecting features from testing data")
            test_df = self.select_features(test_df)

            logger.info("Saving processed training data")
            self.save_data(train_df, 'processed_train.csv')

            logger.info("Saving processed testing data")
            self.save_data(test_df, 'processed_test.csv')

            logger.info("Data processing completed successfully")

        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            raise CustomException(e, sys) from e

if __name__ == "__main__":
    data_processor = DataProcessing(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config=CONFIG_PATH
    )
    data_processor.process()