import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

class ImprovedModelTrainer:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()  # Changed to MinMaxScaler for better LSTM performance
        
    def create_sequences(self, data):
        sequences = []
        labels = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            labels.append(data[i + self.sequence_length])
        return np.array(sequences), np.array(labels)
    
    def train_isolation_forest(self, data):
        """Train Isolation Forest with improved parameters"""
        print("Training Isolation Forest...")
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Improved parameters for Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=200,  # Increased number of trees
            contamination=0.05,  # Reduced contamination assumption
            max_samples='auto',
            random_state=42,
            verbose=1
        )
        self.isolation_forest.fit(scaled_data)
        
        # Save models
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(self.isolation_forest, 'models/isolation_forest.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        print("Isolation Forest training completed")
    
    def train_lstm_autoencoder(self, data):
        """Train LSTM Autoencoder with improved architecture"""
        print("Training LSTM Autoencoder...")
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        sequences, _ = self.create_sequences(scaled_data)
        
        # Build improved LSTM Autoencoder
        model = Sequential([
            # Encoder
            LSTM(128, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=False),
            Dropout(0.2),
            
            # Bottleneck
            Dense(32, activation='relu'),
            
            # Decoder
            RepeatVector(self.sequence_length),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(128, activation='relu', return_sequences=True),
            Dropout(0.2),
            TimeDistributed(Dense(1))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Add early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train with validation split
        history = model.fit(
            sequences, sequences,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save the model with .keras extension
        model.save('models/lstm_autoencoder.keras')  # Changed this line
        np.save('models/training_history.npy', history.history)
        print("LSTM Autoencoder training completed")
        
        # Calculate and save reconstruction error threshold
        reconstructions = model.predict(sequences)
        reconstruction_errors = np.mean(np.abs(sequences - reconstructions), axis=(1,2))
        threshold = np.percentile(reconstruction_errors, 95)  # 95th percentile
        np.save('models/reconstruction_threshold.npy', threshold)
        
    @staticmethod
    def load_and_prepare_data(file_path):
        """Load and prepare data with improved preprocessing"""
        df = pd.read_csv(file_path)
        
        # Remove outliers beyond 3 standard deviations for training
        values = df['value'].values
        mean = np.mean(values)
        std = np.std(values)
        filtered_values = values[np.abs(values - mean) <= 3 * std]
        
        return filtered_values

def main():
    # Initialize trainer
    trainer = ImprovedModelTrainer(sequence_length=10)
    
    # Load and prepare data
    print("Loading and preparing data...")
    data = ImprovedModelTrainer.load_and_prepare_data('your_data.csv')
    
    # Train both models
    print("Starting model training...")
    trainer.train_isolation_forest(data)
    trainer.train_lstm_autoencoder(data)
    print("Model training completed successfully!")

if __name__ == "__main__":
    main()
