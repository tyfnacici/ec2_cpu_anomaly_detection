import pandas as pd
import numpy as np
from ml_pipeline import AnomalyDetectionPipeline
import logging
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from typing import Tuple, Dict, Any
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_sequences(data: np.ndarray, sequence_length: int = 10) -> np.ndarray:
    """
    Create sequences for LSTM processing
    """
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)

def prepare_data_for_evaluation(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare data and create synthetic anomaly labels for evaluation
    """

    if 'value' not in data.columns:
        if len(data.columns) == 1:
            data = data.rename(columns={data.columns[0]: 'value'})
        else:
            raise ValueError("Data must contain a 'value' column")
    

    numeric_data = pd.to_numeric(data['value'], errors='coerce')
    

    if 'timestamp' not in data.columns:
        logger.info("Creating synthetic timestamp column...")
        base_timestamp = datetime.now()
        timestamps = [base_timestamp + timedelta(hours=i) for i in range(len(numeric_data))]
        data['timestamp'] = timestamps
    else:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    

    data = data.dropna(subset=['value'])
    values = data['value'].values.reshape(-1, 1)
    

    mean = np.mean(values)
    std = np.std(values)
    true_anomalies = np.abs(values - mean) > 3 * std
    
    return values, true_anomalies, data[['timestamp', 'value']]

def calculate_metrics(true_anomalies: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    """
    return {
        'precision': float(precision_score(true_anomalies, predictions)),
        'recall': float(recall_score(true_anomalies, predictions)),
        'f1': float(f1_score(true_anomalies, predictions))
    }

def evaluate_models(data_path: str = 'your_data.csv', models_dir: str = 'models') -> Dict[str, Any]:
    """
    Evaluate trained anomaly detection models using test data
    """
    try:

        logger.info("Loading test data...")
        data = pd.read_csv(data_path)
        
        logger.info("Preparing data for evaluation...")
        if data.empty:
            raise ValueError("The loaded data is empty")
            

        values, true_anomalies, processed_df = prepare_data_for_evaluation(data)
        

        logger.info("Loading trained models...")
        pipeline = AnomalyDetectionPipeline()
        pipeline.load_models(models_dir)
        

        logger.info("Processing data and generating predictions...")
        processed_data = pipeline.preprocess_data(processed_df)
        

        logger.info("Detecting anomalies with Isolation Forest...")
        if_predictions = (pipeline.isolation_forest.predict(processed_data) == -1)
        

        logger.info("Detecting anomalies with LSTM...")

        sequences = prepare_sequences(processed_data)
        lstm_reconstructions = pipeline.lstm_model.predict(sequences)
        reconstruction_errors = np.mean(np.square(sequences - lstm_reconstructions), axis=(1, 2))
        lstm_predictions = reconstruction_errors > pipeline.reconstruction_threshold
        

        min_length = min(len(true_anomalies), len(if_predictions), len(lstm_predictions))
        true_anomalies = true_anomalies[:min_length]
        if_predictions = if_predictions[:min_length]
        lstm_predictions = lstm_predictions[:min_length]
        

        metrics = {
            'isolation_forest': calculate_metrics(true_anomalies.ravel(), if_predictions),
            'lstm': calculate_metrics(true_anomalies.ravel(), lstm_predictions)
        }
        

        os.makedirs('evaluation_results', exist_ok=True)
        

        plt.figure(figsize=(15, 8))
        plt.plot(processed_df['timestamp'][:min_length], values[:min_length], 
                label='Original Data', alpha=0.5)
        

        if np.any(if_predictions):
            anomaly_idx = np.where(if_predictions)[0]
            plt.scatter(processed_df['timestamp'].iloc[anomaly_idx], 
                       values[anomaly_idx], 
                       color='red', 
                       label='Isolation Forest Anomalies')
        

        if np.any(lstm_predictions):
            anomaly_idx = np.where(lstm_predictions)[0]
            plt.scatter(processed_df['timestamp'].iloc[anomaly_idx], 
                       values[anomaly_idx], 
                       color='green', 
                       label='LSTM Anomalies')
        
        plt.title('Anomaly Detection Results')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('evaluation_results/anomaly_detection_results.png')
        plt.close()
        

        with open('evaluation_results/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("Evaluation completed successfully!")
        logger.info("\nModel Metrics:")
        logger.info(json.dumps(metrics, indent=2))
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_models()