from confluent_kafka import Producer, Consumer
import json
import time
import pandas as pd
from datetime import datetime
import threading


class DataProducer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = Producer({
            'bootstrap.servers': bootstrap_servers,
            'client.id': 'python-producer'
        })

    def delivery_report(self, err, msg):
        if err is not None:
            print(f'Message delivery failed: {err}')
        else:
            print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

    def send_data(self, topic, data):
        try:
            self.producer.produce(
                topic,
                json.dumps(data).encode('utf-8'),
                callback=self.delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            print(f"Error sending message: {e}")

    def close(self):
        self.producer.flush()


class DataConsumer:
    def __init__(self, topics, group_id='python-consumer', bootstrap_servers='localhost:9092'):
        self.consumer = Consumer({
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe(topics)

    def consume_messages(self, timeout=1.0):
        try:
            while True:
                msg = self.consumer.poll(timeout)
                if msg is None:
                    continue
                if msg.error():
                    print(f"Consumer error: {msg.error()}")
                    continue
                try:
                    data = json.loads(msg.value().decode('utf-8'))
                    print(f"Received: {data}")
                except Exception as e:
                    print(f"Error processing message: {e}")
        except KeyboardInterrupt:
            print("Stopping consumer...")
        finally:
            self.consumer.close()


def simulate_real_time_data(csv_file, producer):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        

        if 'timestamp' in df.columns:

            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            

            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
        for _, row in df.iterrows():

            timestamp_str = None
            if pd.notnull(row['timestamp']):

                timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            data = {
                'timestamp': timestamp_str,
                'value': float(row['value']) if 'value' in row and pd.notnull(row['value']) else None
            }
            producer.send_data('input_data', data)
            time.sleep(1)  
            
    except Exception as e:
        print(f"Error in data simulation: {e}")

        print(f"DataFrame dtypes: {df.dtypes}")
        print(f"Timestamp column sample: {df['timestamp'].head()}")
        raise
    finally:
        producer.close()
if __name__ == "__main__":
    try:

        producer = DataProducer()
        

        consumer = DataConsumer(['input_data', 'anomalies'])
        

        simulation_thread = threading.Thread(
            target=simulate_real_time_data,
            args=('processed_data.csv', producer)
        )
        simulation_thread.start()
        

        consumer.consume_messages()
    except Exception as e:
        print(f"Error in main: {e}")