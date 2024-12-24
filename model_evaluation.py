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

def calculate_reconstruction_threshold(errors: np.ndarray, n_sigma: float = 3.0) -> float:
    """
    Calculate threshold for anomaly detection based on reconstruction errors
    """
    mean = np.mean(errors)
    std = np.std(errors)
    return mean + n_sigma * std

def evaluate_models(data_path: str = 'your_data.csv', models_dir: str = 'models') -> Dict[str, Any]:
    """
    Evaluate trained anomaly detection models using test data
    """
    try:
        logger.info("Loading test data...")
        data = pd.read_csv(data_path)
        
        if data.empty:
            raise ValueError("The loaded data is empty")

        # Prepare data
        if 'value' not in data.columns:
            if len(data.columns) == 1:
                data = data.rename(columns={data.columns[0]: 'value'})
            else:
                raise ValueError("Data must contain a 'value' column")

        if 'timestamp' not in data.columns:
            base_timestamp = datetime.now()
            data['timestamp'] = [base_timestamp + timedelta(hours=i) for i in range(len(data))]
        else:
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Create synthetic anomaly labels (ground truth)
        values = data['value'].values
        mean = np.mean(values)
        std = np.std(values)
        true_anomalies = np.abs(values - mean) > 3 * std

        # Load and prepare pipeline
        logger.info("Loading trained models...")
        pipeline = AnomalyDetectionPipeline()
        pipeline.load_models(models_dir)

        # Preprocess data
        logger.info("Preprocessing data...")
        processed_data = pipeline.preprocess_data(data)
        
        # Isolation Forest predictions
        logger.info("Detecting anomalies with Isolation Forest...")
        if_predictions = (pipeline.isolation_forest.predict(processed_data.values) == -1)
        
        # LSTM predictions
        logger.info("Detecting anomalies with LSTM...")
        sequences = prepare_sequences(processed_data.values, pipeline.sequence_length)
        
        # Debug LSTM sequence shapes
        logger.info(f"Sequences shape: {sequences.shape}")
        
        # Get LSTM predictions
        lstm_reconstructions = pipeline.lstm_model.predict(sequences)
        reconstruction_errors = np.mean(np.abs(sequences[:, -1] - lstm_reconstructions), axis=1)
        
        # Debug reconstruction errors
        logger.info(f"Reconstruction errors stats:")
        logger.info(f"Mean: {np.mean(reconstruction_errors)}")
        logger.info(f"Std: {np.std(reconstruction_errors)}")
        logger.info(f"Min: {np.min(reconstruction_errors)}")
        logger.info(f"Max: {np.max(reconstruction_errors)}")
        
        # Calculate threshold dynamically
        threshold = calculate_reconstruction_threshold(reconstruction_errors)
        logger.info(f"Calculated threshold: {threshold}")
        
        lstm_predictions = reconstruction_errors > threshold
        
        # Align predictions lengths
        min_length = min(len(true_anomalies), len(if_predictions), len(lstm_predictions))
        true_anomalies = true_anomalies[:min_length]
        if_predictions = if_predictions[:min_length]
        lstm_predictions = lstm_predictions[:min_length]
        
        # Debug prediction distributions
        logger.info(f"Number of true anomalies: {np.sum(true_anomalies)}")
        logger.info(f"Number of IF predictions: {np.sum(if_predictions)}")
        logger.info(f"Number of LSTM predictions: {np.sum(lstm_predictions)}")

        # Calculate metrics
        metrics = {
            'isolation_forest': calculate_metrics(true_anomalies, if_predictions),
            'lstm': calculate_metrics(true_anomalies, lstm_predictions)
        }

        # Save results
        os.makedirs('evaluation_results', exist_ok=True)
        
        # Plot results
        plt.figure(figsize=(15, 8))
        plt.plot(data['timestamp'][:min_length], values[:min_length], 
                label='Original Data', alpha=0.5)
        
        if np.any(if_predictions):
            anomaly_idx = np.where(if_predictions)[0]
            plt.scatter(data['timestamp'].iloc[anomaly_idx], 
                       values[anomaly_idx], 
                       color='red', 
                       label='Isolation Forest Anomalies')
        
        if np.any(lstm_predictions):
            anomaly_idx = np.where(lstm_predictions)[0]
            plt.scatter(data['timestamp'].iloc[anomaly_idx], 
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