import pandas as pd
import os
from ml_pipeline import AnomalyDetectionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_models(data_path='your_data.csv'):
    try:

        logger.info("Loading training data...")
        data = pd.read_csv(data_path)
        

        logger.info("Initializing ML pipeline...")
        pipeline = AnomalyDetectionPipeline()
        

        logger.info("Preprocessing data and training models...")
        processed_data = pipeline.preprocess_data(data)
        pipeline.train_isolation_forest(processed_data)
        pipeline.train_lstm(processed_data)
        

        os.makedirs('models', exist_ok=True)
        

        logger.info("Saving trained models...")
        pipeline.save_models('models')
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_models()
