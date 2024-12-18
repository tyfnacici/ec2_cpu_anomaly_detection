from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
import os
import logging
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Update the model loading in the AnomalyDetector class initialization
class AnomalyDetector:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.scaler = joblib.load('models/scaler.joblib')
        self.isolation_forest = joblib.load('models/isolation_forest.joblib')
        self.lstm_autoencoder = load_model('models/lstm_autoencoder.keras')
	
    def create_sequence(self, value):
        # This is a simplified version for streaming - you'll need to maintain state
        # in a real application to create proper sequences
        return np.array([[value] * self.sequence_length])
    
    def detect_anomaly(self, value):
        # Scale the value
        scaled_value = self.scaler.transform([[value]])[0][0]
        
        # Isolation Forest detection
        if_prediction = self.isolation_forest.predict([[scaled_value]])[0]
        if_anomaly = if_prediction == -1
        
        # LSTM Autoencoder detection
        sequence = self.create_sequence(scaled_value)
        reconstruction = self.lstm_autoencoder.predict(sequence)
        reconstruction_error = np.mean(np.abs(sequence - reconstruction))
        lstm_anomaly = reconstruction_error > 0.5  # Threshold can be adjusted
        
        # Combine predictions (consider it an anomaly if either model detects it)
        return bool(if_anomaly or lstm_anomaly), float(reconstruction_error)

def create_spark_session():
    try:
        return SparkSession.builder \
            .appName("AnomalyDetection") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoints") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .master("local[*]") \
            .getOrCreate()
    except Exception as e:
        logger.error(f"Failed to create Spark session: {str(e)}")
        raise

def define_schema():
    return StructType([
        StructField("timestamp", TimestampType(), True),
        StructField("value", DoubleType(), True)
    ])

# Initialize anomaly detector
detector = AnomalyDetector()

# Create UDF for anomaly detection
@udf(StructType([
    StructField("is_anomaly", BooleanType(), True),
    StructField("reconstruction_error", DoubleType(), True)
]))
def detect_anomaly_udf(value):
    if value is None:
        return (False, 0.0)
    try:
        is_anomaly, reconstruction_error = detector.detect_anomaly(float(value))
        return (bool(is_anomaly), float(reconstruction_error))
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return (False, 0.0)

def process_stream(df):
    return df \
        .withColumn(
            "ml_detection",
            detect_anomaly_udf("value")
        ) \
        .select(
            "timestamp",
            "value",
            col("ml_detection.is_anomaly").alias("is_anomaly"),
            col("ml_detection.reconstruction_error").alias("reconstruction_error")
        )

def main():
    spark = None
    try:
        # Create Spark session
        logger.info("Creating Spark session...")
        spark = create_spark_session()
        
        # Ensure checkpoint directory exists
        checkpoint_dir = "/tmp/spark-checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info("Reading from Kafka...")
        # Read from Kafka
        input_df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "input_data") \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON data
        schema = define_schema()
        parsed_df = input_df \
            .select(from_json(
                col("value").cast("string"),
                schema
            ).alias("data")) \
            .select("data.*")
        
        logger.info("Processing stream with ML models...")
        # Process stream with ML models
        output_df = process_stream(parsed_df)
        
        logger.info("Starting console output stream...")
        # Write results to console for debugging
        console_query = output_df \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", "false") \
            .start()
        
        logger.info("Starting Kafka output stream...")
        # Write results back to Kafka
        kafka_query = output_df \
            .select(to_json(struct("*")).alias("value")) \
            .writeStream \
            .outputMode("append") \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("topic", "anomalies") \
            .option("checkpointLocation", os.path.join(checkpoint_dir, "kafka")) \
            .start()
        
        logger.info("Waiting for termination...")
        spark.streams.awaitAnyTermination()
        
    except Exception as e:
        logger.error(f"Error in Spark Streaming: {str(e)}")
        raise
    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")

if __name__ == "__main__":
    main()
