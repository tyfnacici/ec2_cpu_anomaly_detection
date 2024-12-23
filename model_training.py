import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.random_forest = None
        self.lstm_model = None

    def create_sequences(self, data):
        """Create sequences for LSTM model"""
        X = []
        y = []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def prepare_data(self, data):
        """Prepare data for both models"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        # Split into train and test sets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # Create sequences for LSTM
        X_train_lstm, y_train_lstm = self.create_sequences(train_data)
        X_test_lstm, y_test_lstm = self.create_sequences(test_data)
        
        # Prepare data for Random Forest
        # Using a simple threshold-based approach to create labels
        threshold = np.percentile(train_data, 95)  # Top 5% as anomalies
        y_train_rf = (train_data > threshold).astype(int)
        y_test_rf = (test_data > threshold).astype(int)
        
        return {
            'rf': (train_data, test_data, y_train_rf, y_test_rf),
            'lstm': (X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm)
        }

    def train_random_forest(self, train_data, y_train):
        """Train Random Forest model"""
        print("Training Random Forest...")
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.random_forest.fit(train_data, y_train.ravel())

    def train_lstm(self, X_train, y_train):
        """Train LSTM model"""
        print("Training LSTM...")
        self.lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(30, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        
        history = self.lstm_model.fit(
            X_train,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        return history

    def evaluate_models(self, rf_data, lstm_data):
        """Evaluate both models and print metrics"""
        # Random Forest evaluation
        _, test_data_rf, _, y_test_rf = rf_data
        rf_pred = self.random_forest.predict(test_data_rf)
        
        print("\nRandom Forest Metrics:")
        print(f"Accuracy: {accuracy_score(y_test_rf, rf_pred):.4f}")
        print(f"Precision: {precision_score(y_test_rf, rf_pred):.4f}")
        print(f"Recall: {recall_score(y_test_rf, rf_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test_rf, rf_pred):.4f}")
        
        # LSTM evaluation
        _, X_test_lstm, _, y_test_lstm = lstm_data
        lstm_pred = self.lstm_model.predict(X_test_lstm)
        
        # Convert LSTM predictions to binary (anomaly/normal) using MAE
        reconstruction_error = np.mean(np.abs(lstm_pred - y_test_lstm), axis=1)
        threshold = np.percentile(reconstruction_error, 95)
        lstm_pred_binary = (reconstruction_error > threshold).astype(int)
        y_test_lstm_binary = (y_test_lstm > np.percentile(y_test_lstm, 95)).astype(int)
        
        print("\nLSTM Metrics:")
        print(f"Accuracy: {accuracy_score(y_test_lstm_binary, lstm_pred_binary):.4f}")
        print(f"Precision: {precision_score(y_test_lstm_binary, lstm_pred_binary):.4f}")
        print(f"Recall: {recall_score(y_test_lstm_binary, lstm_pred_binary):.4f}")
        print(f"F1 Score: {f1_score(y_test_lstm_binary, lstm_pred_binary):.4f}")
        
        # Save metrics to file
        metrics = {
            'random_forest': {
                'accuracy': accuracy_score(y_test_rf, rf_pred),
                'precision': precision_score(y_test_rf, rf_pred),
                'recall': recall_score(y_test_rf, rf_pred),
                'f1': f1_score(y_test_rf, rf_pred)
            },
            'lstm': {
                'accuracy': accuracy_score(y_test_lstm_binary, lstm_pred_binary),
                'precision': precision_score(y_test_lstm_binary, lstm_pred_binary),
                'recall': recall_score(y_test_lstm_binary, lstm_pred_binary),
                'f1': f1_score(y_test_lstm_binary, lstm_pred_binary)
            }
        }
        
        # Save metrics to file
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(metrics, 'models/metrics.joblib')

    def save_models(self):
        """Save trained models"""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # Save Random Forest and scaler
        joblib.dump(self.random_forest, 'models/random_forest.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        
        # Save LSTM model with .keras extension
        self.lstm_model.save('models/lstm_model.keras')
        print("Models saved successfully")

    def plot_results(self, history):
        """Plot training results"""
        plt.figure(figsize=(12, 4))
        
        # Plot LSTM training history
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('LSTM Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.close()

def main():
    # Load your data
    df = pd.read_csv('your_data.csv')
    data = df['value'].values
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data for both models
    prepared_data = trainer.prepare_data(data)
    
    # Train Random Forest
    trainer.train_random_forest(prepared_data['rf'][0], prepared_data['rf'][2])
    
    # Train LSTM
    history = trainer.train_lstm(prepared_data['lstm'][0], prepared_data['lstm'][2])
    
    # Evaluate models
    trainer.evaluate_models(prepared_data['rf'], prepared_data['lstm'])
    
    # Save models
    trainer.save_models()
    
    # Plot results
    trainer.plot_results(history)

if __name__ == "__main__":
    main()
