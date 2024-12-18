import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Handle missing values
    df['value'].fillna(df['value'].mean(), inplace=True)
    
    # Calculate rolling statistics for anomaly detection
    df['rolling_mean'] = df['value'].rolling(window=12).mean()
    df['rolling_std'] = df['value'].rolling(window=12).std()
    
    # Calculate z-scores for anomaly detection
    df['z_score'] = (df['value'] - df['rolling_mean']) / df['rolling_std']
    
    # Mark anomalies (points beyond 3 standard deviations)
    df['is_anomaly'] = abs(df['z_score']) > 3
    
    return df

def visualize_data(df):
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Time series with anomalies
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df['value'], label='Value')
    anomalies = df[df['is_anomaly']]
    plt.scatter(anomalies.index, anomalies['value'], color='red', label='Anomaly')
    plt.title('Time Series Data with Anomalies')
    plt.legend()
    
    # Plot 2: Distribution of values
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='value', bins=30)
    plt.title('Distribution of Values')
    
    # Plot 3: Box plot
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, y='value')
    plt.title('Box Plot of Values')
    
    # Plot 4: Z-scores over time
    plt.subplot(2, 2, 4)
    plt.plot(df.index, df['z_score'])
    plt.axhline(y=3, color='r', linestyle='--')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.title('Z-scores Over Time')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png')
    plt.close()

if __name__ == "__main__":
    # Load and process the data
    df = load_and_preprocess_data('your_data.csv')
    
    # Create visualizations
    visualize_data(df)
    
    # Save processed data
    df.to_csv('processed_data.csv')
