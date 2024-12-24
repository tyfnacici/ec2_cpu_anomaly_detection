import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import json

class ModelTrainer:
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.scaler = RobustScaler()
        self.isolation_forest = None
        self.lstm_model = None

    def create_sequences(self, data):
        X = []
        y = []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def detect_statistical_anomalies(self, data, threshold=3):
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / std)
        return z_scores > threshold

    def prepare_data(self, data):
        df = pd.DataFrame(data, columns=['value'])
        df['rolling_mean'] = df['value'].rolling(window=5).mean()
        df['rolling_std'] = df['value'].rolling(window=5).std()
        df['rolling_max'] = df['value'].rolling(window=5).max()
        df['rolling_min'] = df['value'].rolling(window=5).min()
        df['value_diff'] = df['value'].diff()
        df = df.fillna(method='bfill')
        
        scaled_data = self.scaler.fit_transform(df)
        
        value_anomalies = self.detect_statistical_anomalies(df['value'].values)
        diff_anomalies = self.detect_statistical_anomalies(df['value_diff'].values)
        anomalies = np.logical_or(value_anomalies, diff_anomalies)
        
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        test_anomalies = anomalies[train_size:]
        
        X_train_lstm, y_train_lstm = self.create_sequences(train_data)
        X_test_lstm, y_test_lstm = self.create_sequences(test_data)
        
        return {
            'if': (train_data, test_data, test_anomalies),
            'lstm': (X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, test_anomalies)
        }

    def train_isolation_forest(self, train_data):
        self.isolation_forest = IsolationForest(
            contamination=0.1, 
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        self.isolation_forest.fit(train_data)

    def train_lstm(self, X_train, y_train):
        self.lstm_model = Sequential([
            LSTM(64, activation='relu', input_shape=(self.sequence_length, X_train.shape[2]), 
                 return_sequences=True),
            Dropout(0.3),
            LSTM(32, activation='relu'),
            Dropout(0.3),
            Dense(X_train.shape[2])
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='huber')
        
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        return history

    def evaluate_models(self, if_data, lstm_data):
        _, test_data_if, true_anomalies = if_data
        

        if_scores = self.isolation_forest.score_samples(test_data_if)
        if_threshold = np.percentile(if_scores, 10) 
        if_pred_binary = (if_scores < if_threshold).astype(int)
        

        _, X_test_lstm, _, _, _ = lstm_data
        lstm_pred = self.lstm_model.predict(X_test_lstm)
        reconstruction_error = np.mean(np.abs(lstm_pred - X_test_lstm[:, -1, :]), axis=1)
        lstm_threshold = np.percentile(reconstruction_error, 90)
        lstm_pred_binary = (reconstruction_error > lstm_threshold).astype(int)
        

        if_metrics = {
            'precision': float(precision_score(true_anomalies[:-len(lstm_pred_binary)], if_pred_binary[:-len(lstm_pred_binary)], zero_division=1)),
            'recall': float(recall_score(true_anomalies[:-len(lstm_pred_binary)], if_pred_binary[:-len(lstm_pred_binary)], zero_division=1)),
            'f1': float(f1_score(true_anomalies[:-len(lstm_pred_binary)], if_pred_binary[:-len(lstm_pred_binary)], zero_division=1))
        }
        
        lstm_metrics = {
            'precision': float(precision_score(true_anomalies[-len(lstm_pred_binary):], lstm_pred_binary, zero_division=1)),
            'recall': float(recall_score(true_anomalies[-len(lstm_pred_binary):], lstm_pred_binary, zero_division=1)),
            'f1': float(f1_score(true_anomalies[-len(lstm_pred_binary):], lstm_pred_binary, zero_division=1))
        }
        
        metrics = {
            'isolation_forest': if_metrics,
            'lstm': lstm_metrics
        }
        
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(metrics, 'models/metrics.joblib')
        
        print("\nIsolation Forest Details:")
        print(f"Number of anomalies detected: {np.sum(if_pred_binary)}")
        print(f"Percentage of anomalies: {(np.sum(if_pred_binary) / len(if_pred_binary)) * 100:.2f}%")
        
        return metrics

    def save_models(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(self.isolation_forest, 'models/isolation_forest.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        self.lstm_model.save('models/lstm_model.keras')

def main():
    df = pd.read_csv('your_data.csv')
    data = df['value'].values
    
    trainer = ModelTrainer()
    prepared_data = trainer.prepare_data(data)
    
    trainer.train_isolation_forest(prepared_data['if'][0])
    history = trainer.train_lstm(prepared_data['lstm'][0], prepared_data['lstm'][2])
    
    metrics = trainer.evaluate_models(prepared_data['if'], prepared_data['lstm'])
    trainer.save_models()
    
    print("\nFinal Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()