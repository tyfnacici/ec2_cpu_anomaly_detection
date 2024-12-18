import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
import json
import os

    
class ModelEvaluator:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = joblib.load('models/scaler.joblib')
        self.isolation_forest = joblib.load('models/isolation_forest.joblib')
        self.lstm_autoencoder = load_model('models/lstm_autoencoder.keras')  # Updated extension
        self.reconstruction_threshold = np.load('models/reconstruction_threshold.npy')
 
    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def detect_anomalies_isolation_forest(self, data):
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        predictions = self.isolation_forest.predict(scaled_data)
        return predictions == -1
    
    def detect_anomalies_lstm(self, data, threshold=None):
        if threshold is None:
            threshold = self.reconstruction_threshold
        
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        sequences = self.create_sequences(scaled_data)
        reconstructions = self.lstm_autoencoder.predict(sequences)
        mse = np.mean(np.square(sequences - reconstructions), axis=(1, 2))
        return mse > threshold   

 
    def evaluate_models(self, data, true_anomalies=None):
        # If no true anomalies provided, create synthetic ones based on statistical methods
        if true_anomalies is None:
            mean = np.mean(data)
            std = np.std(data)
            true_anomalies = np.abs(data - mean) > 3 * std
        
        # Get predictions from both models
        if_predictions = self.detect_anomalies_isolation_forest(data)
        lstm_predictions = self.detect_anomalies_lstm(data)
        
        # Adjust lengths to match
        min_length = min(len(true_anomalies[self.sequence_length-1:]), 
                        len(if_predictions[self.sequence_length-1:]),
                        len(lstm_predictions))
        
        true_anomalies = true_anomalies[self.sequence_length-1:self.sequence_length-1+min_length]
        if_predictions = if_predictions[self.sequence_length-1:self.sequence_length-1+min_length]
        lstm_predictions = lstm_predictions[:min_length]
        
        # Calculate metrics for Isolation Forest
        if_metrics = {
            'precision': float(precision_score(true_anomalies, if_predictions)),
            'recall': float(recall_score(true_anomalies, if_predictions)),
            'f1': float(f1_score(true_anomalies, if_predictions))
        }
        
        # Calculate metrics for LSTM Autoencoder
        lstm_metrics = {
            'precision': float(precision_score(true_anomalies, lstm_predictions)),
            'recall': float(recall_score(true_anomalies, lstm_predictions)),
            'f1': float(f1_score(true_anomalies, lstm_predictions))
        }
        
        return {
            'isolation_forest': if_metrics,
            'lstm_autoencoder': lstm_metrics,
            'predictions': {
                'isolation_forest': if_predictions.tolist(),
                'lstm_autoencoder': lstm_predictions.tolist(),
                'true_anomalies': true_anomalies.tolist()
            }
        }
    
    def plot_results(self, data, evaluation_results, save_path='evaluation_results'):
        """Plot and save visualization of results"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Plot 1: Time series with anomalies
        plt.figure(figsize=(15, 8))
        plt.plot(data, label='Original Data', alpha=0.5)
        
        # Plot anomalies detected by each model
        if_anomalies = np.array(evaluation_results['predictions']['isolation_forest'])
        lstm_anomalies = np.array(evaluation_results['predictions']['lstm_autoencoder'])
        
        if len(if_anomalies) > 0:
            plt.scatter(np.where(if_anomalies)[0], 
                       data[np.where(if_anomalies)[0]], 
                       color='red', 
                       label='Isolation Forest Anomalies')
        
        if len(lstm_anomalies) > 0:
            plt.scatter(np.where(lstm_anomalies)[0], 
                       data[np.where(lstm_anomalies)[0]], 
                       color='green', 
                       label='LSTM Autoencoder Anomalies')
        
        plt.title('Anomaly Detection Results')
        plt.legend()
        plt.savefig(f'{save_path}/anomaly_detection_results.png')
        plt.close()
        
        # Plot 2: Confusion matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Isolation Forest confusion matrix
        cm_if = confusion_matrix(
            evaluation_results['predictions']['true_anomalies'],
            evaluation_results['predictions']['isolation_forest']
        )
        sns.heatmap(cm_if, annot=True, fmt='d', ax=ax1)
        ax1.set_title('Isolation Forest Confusion Matrix')
        
        # LSTM Autoencoder confusion matrix
        cm_lstm = confusion_matrix(
            evaluation_results['predictions']['true_anomalies'],
            evaluation_results['predictions']['lstm_autoencoder']
        )
        sns.heatmap(cm_lstm, annot=True, fmt='d', ax=ax2)
        ax2.set_title('LSTM Autoencoder Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrices.png')
        plt.close()
        
        # Save metrics to JSON
        with open(f'{save_path}/metrics.json', 'w') as f:
            json.dump({
                'isolation_forest': evaluation_results['isolation_forest'],
                'lstm_autoencoder': evaluation_results['lstm_autoencoder']
            }, f, indent=4)

def main():
    try:
        # Load your data
        df = pd.read_csv('your_data.csv')
        data = df['value'].values
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate models
        evaluation_results = evaluator.evaluate_models(data)
        
        # Plot and save results
        evaluator.plot_results(data, evaluation_results)
        
        # Print results
        print("\nIsolation Forest Metrics:")
        print(json.dumps(evaluation_results['isolation_forest'], indent=2))
        print("\nLSTM Autoencoder Metrics:")
        print(json.dumps(evaluation_results['lstm_autoencoder'], indent=2))
        
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
