import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):

    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    

    df.set_index('timestamp', inplace=True)
    

    df['value'].fillna(df['value'].mean(), inplace=True)
    

    df['rolling_mean'] = df['value'].rolling(window=12).mean()
    df['rolling_std'] = df['value'].rolling(window=12).std()
    

    df['z_score'] = (df['value'] - df['rolling_mean']) / df['rolling_std']
    

    df['is_anomaly'] = abs(df['z_score']) > 3
    
    return df

def visualize_data(df):

    plt.figure(figsize=(15, 10))
    

    plt.subplot(2, 2, 1)
    plt.plot(df.index, df['value'], label='Value')
    anomalies = df[df['is_anomaly']]
    plt.scatter(anomalies.index, anomalies['value'], color='red', label='Anomaly')
    plt.title('Time Series Data with Anomalies')
    plt.legend()
    

    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='value', bins=30)
    plt.title('Distribution of Values')
    

    plt.subplot(2, 2, 3)
    sns.boxplot(data=df, y='value')
    plt.title('Box Plot of Values')
    

    plt.subplot(2, 2, 4)
    plt.plot(df.index, df['z_score'])
    plt.axhline(y=3, color='r', linestyle='--')
    plt.axhline(y=-3, color='r', linestyle='--')
    plt.title('Z-scores Over Time')
    
    plt.tight_layout()
    plt.savefig('data_visualization.png')
    plt.close()

if __name__ == "__main__":

    df = load_and_preprocess_data('your_data.csv')
    

    visualize_data(df)
    

    df.to_csv('processed_data.csv')
