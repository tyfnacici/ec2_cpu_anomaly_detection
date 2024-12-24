import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import precision_recall_fscore_support
import joblib
import logging
import tensorflow as tf
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetectionPipeline:
    def __init__(self, sequence_length=10):
        self.isolation_forest = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        
    def preprocess_data(self, df):
        """Preprocess the data for both models."""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        features = ['value', 'hour', 'day_of_week']
        scaled_data = self.scaler.fit_transform(df[features])
        
        return pd.DataFrame(scaled_data, columns=features)
    
    def create_sequences(self, data):
        """Create sequences for LSTM model."""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            target = data[i + self.sequence_length]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def train_isolation_forest(self, data, contamination=0.1):
        """Train Isolation Forest model."""
        logger.info("Training Isolation Forest model...")
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(data)
        logger.info("Isolation Forest training completed")
        
    def train_lstm(self, data, epochs=50, batch_size=32):
        """Train LSTM model."""
        logger.info("Preparing data for LSTM...")
        X, y = self.create_sequences(data.values)
        
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
        
        logger.info("Creating LSTM model...")
        self.lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, data.shape[1]), 
                 return_sequences=True),
            Dropout(0.2),
            LSTM(30, activation='relu'),
            Dropout(0.2),
            Dense(data.shape[1])
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        logger.info("Training LSTM model...")
        self.lstm_model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        logger.info("LSTM training completed")
        
    def detect_anomalies(self, data, threshold=3.0):
        """Detect anomalies using both models."""
        processed_data = self.preprocess_data(data.copy())
        
        if_predictions = self.isolation_forest.predict(processed_data)
        if_anomalies = if_predictions == -1
        
        X, _ = self.create_sequences(processed_data.values)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
        lstm_predictions = self.lstm_model.predict(X)
        
        reconstruction_error = np.mean(np.abs(lstm_predictions - X[:, -1, :]), axis=1)
        lstm_anomalies = reconstruction_error > threshold * np.std(reconstruction_error)
        
        combined_anomalies = np.logical_or(
            if_anomalies[self.sequence_length:],
            lstm_anomalies
        )
        
        return pd.DataFrame({
            'timestamp': data['timestamp'].iloc[self.sequence_length:].values,
            'value': data['value'].iloc[self.sequence_length:].values,
            'is_anomaly': combined_anomalies,
            'if_anomaly': if_anomalies[self.sequence_length:],
            'lstm_anomaly': lstm_anomalies,
            'reconstruction_error': reconstruction_error
        })
    
    def save_models(self, path):
        """Save trained models."""
        logger.info("Saving models...")
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.isolation_forest, f"{path}/isolation_forest.joblib")
        joblib.dump(self.scaler, f"{path}/scaler.joblib")
        self.lstm_model.save(f"{path}/lstm_model.h5")
        
    def load_models(self, path):
        """Load trained models."""
        logger.info("Loading models...")
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model directory {path} does not exist")
                
            isolation_forest_path = f"{path}/isolation_forest.joblib"
            scaler_path = f"{path}/scaler.joblib"
            lstm_path = f"{path}/lstm_model.h5"
            
            if not all(os.path.exists(p) for p in [isolation_forest_path, scaler_path, lstm_path]):
                raise FileNotFoundError("One or more model files are missing")
                
            self.isolation_forest = joblib.load(isolation_forest_path)
            self.scaler = joblib.load(scaler_path)
            
            self.lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
            self.lstm_model.compile(optimizer='adam', loss='mse')
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.error(f"Model paths exist check:")
            logger.error(f"isolation_forest.joblib: {os.path.exists(f'{path}/isolation_forest.joblib')}")
            logger.error(f"scaler.joblib: {os.path.exists(f'{path}/scaler.joblib')}")
            logger.error(f"lstm_model.h5: {os.path.exists(f'{path}/lstm_model.h5')}")
            raise