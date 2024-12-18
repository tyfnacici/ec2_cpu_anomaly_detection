from confluent_kafka import Producer, Consumer
import json
import time
import pandas as pd
from datetime import datetime

# Kafka Producer
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

# Kafka Consumer
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
        df = pd.read_csv(csv_file)
        
        for _, row in df.iterrows():
            data = {
                'timestamp': str(row['timestamp']),
                'value': float(row['value'])
            }
            producer.send_data('input_data', data)
            time.sleep(1)  # Simulate real-time data by waiting 1 second between messages
    except Exception as e:
        print(f"Error in data simulation: {e}")
    finally:
        producer.close()

if __name__ == "__main__":
    try:
        # Start producer
        producer = DataProducer()
        
        # Start consumer
        consumer = DataConsumer(['input_data', 'anomalies'])
        
        # Run simulation in a separate thread
        import threading
        simulation_thread = threading.Thread(
            target=simulate_real_time_data,
            args=('processed_data.csv', producer)
        )
        simulation_thread.start()
        
        # Start consuming messages
        consumer.consume_messages()
        
    except Exception as e:
        print(f"Error in main: {e}")
