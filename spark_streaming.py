from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F
import os
import logging
from ml_pipeline import AnomalyDetectionPipeline
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def process_batch(batch_df, epoch_id, pipeline):
    try:
        if batch_df.isEmpty():
            return
        
        pandas_df = batch_df.toPandas()
        
        if 'timestamp' in pandas_df.columns:
            pandas_df['timestamp'] = pd.to_datetime(pandas_df['timestamp'], errors='coerce')
            pandas_df['timestamp'] = pandas_df['timestamp'].astype('datetime64[ns]')
        
        results = pipeline.detect_anomalies(pandas_df)
        
        logger.info(f"Batch {epoch_id}: Processed {len(results)} records")
        return results
    except Exception as e:
        logger.error(f"Error processing batch {epoch_id}: {str(e)}")
        logger.error(f"DataFrame columns: {batch_df.columns}")
        logger.error(f"DataFrame schema: {batch_df.schema}")
        raise

def main():
    spark = None
    try:
        logger.info("Creating Spark session...")
        spark = create_spark_session()

        logger.info("Initializing ML pipeline...")
        pipeline = AnomalyDetectionPipeline()
        
        if os.path.exists('models'):
            pipeline.load_models('models')
        else:
            logger.warning("No trained models found. Please run train_models.py first.")
            return

        checkpoint_dir = "/tmp/spark-checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info("Setting up Kafka stream...")
        input_df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "input_data") \
            .option("startingOffsets", "latest") \
            .load()

        schema = StructType([
            StructField("timestamp", StringType(), True),
            StructField("value", DoubleType(), True)
        ])

        parsed_df = input_df \
            .select(from_json(
                col("value").cast("string"),
                schema
            ).alias("data")) \
            .select("data.*") \
            .withColumn("timestamp", 
                when(
                    to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss").isNotNull(),
                    to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss")
                ).when(
                    to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss").isNotNull(),
                    to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mm:ss")
                ).otherwise(
                    lit(None).cast("timestamp")
                )
            )

        query = parsed_df \
            .writeStream \
            .foreachBatch(lambda df, epoch_id: process_batch(df, epoch_id, pipeline)) \
            .outputMode("update") \
            .start()

        logger.info("Streaming query started. Waiting for termination...")
        query.awaitTermination()

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise
    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")

if __name__ == "__main__":
    main()