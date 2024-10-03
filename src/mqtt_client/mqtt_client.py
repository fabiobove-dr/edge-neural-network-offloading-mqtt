import json
import random
import time

import ntplib
import paho.mqtt.client as mqtt
from offloading_algo.offloading_algo import OffloadingAlgo

from src.commons import OffloadingDataFiles
from src.logger.log import logger
from src.mqtt_client.mqtt_configs import MqttClientConfig, Topics, DefaultMessages
from src.mqtt_client.mqtt_custom_message import MqttMessageData


class MqttClient:
    def __init__(
            self,
            broker_url: str = MqttClientConfig.broker_url,
            broker_port: int = MqttClientConfig.broker_port,
            client_id: str = MqttClientConfig.client_id,
            protocol: str = MqttClientConfig.protocol,
            subscribed_topics: list = None,
            ntp_server: str = MqttClientConfig.ntp_server
    ):
        self.broker_url = broker_url
        self.broker_port = broker_port
        self.client_id = client_id

        # Create the client with the specific MQTT protocol version
        self.client = mqtt.Client(client_id=client_id, protocol=protocol)

        # Attach callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Set up topics
        self.subscribed_topics = subscribed_topics

        # Set up NTP client
        self.ntp_client = ntplib.NTPClient()
        self.ntp_server = ntp_server
        self.start_timestamp = self.get_ntp_timestamp()

        # Stats
        self.layers_sizes = []
        self.edge_inference_times = []
        self.device_inference_times = []
        self.load_stats()

    @staticmethod
    def create_random_payload():
        """Creates a random payload for testing."""
        message = json.dumps({"id": random.randint(1, 1000)})
        return message

    def publish(self, topic: str, message: str, qos: int = 2, max_retries: int = 3):
        """Publishes a message to a topic."""
        logger.debug(f"Publishing message to {topic}: {message}")
        try:
            self.client.publish(topic, message, qos=qos, retain=False)
        except Exception as e:
            logger.debug(f"Error publishing message: {e}")

    def subscribe(self, topic: str):
        """Subscribes to a topic."""
        logger.debug(f"Subscribing to topic: {topic}")
        self.client.subscribe(topic)

    def run(self):
        """Connect to the broker and start the MQTT client loop."""
        self.client.connect(self.broker_url, self.broker_port, 60)
        self.client.loop_forever()

    def stop(self):
        """Stops the MQTT client loop and disconnects."""
        logger.debug("Disconnecting MQTT client")
        self.client.disconnect()

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.debug(f"Connected to {self.broker_url}:{self.broker_port} with client ID {self.client_id}")
            for topic in self.subscribed_topics:
                self.subscribe(topic)
            logger.debug(f"Initial NTP timestamp from NTP server {self.ntp_server}: {self.start_timestamp}")
        else:
            logger.debug(f"Connection failed with code {rc}")

    def get_ntp_timestamp(self) -> float:
        ntp_timestamp = None
        while ntp_timestamp is None:
            try:
                response = self.ntp_client.request(self.ntp_server)
                # Get the current NTP time (seconds since 1900)
                ntp_timestamp = response.tx_time
            except ntplib.NTPException as _:
                time.sleep(1)
        return str(ntp_timestamp)

    def on_message(self, client, userdata, message):

        # obtain message data if the message is JSON valid
        try:
            received_timestamp = self.get_ntp_timestamp()
            message_data = MqttMessageData.from_raw(message.topic, message.payload)
        except json.JSONDecodeError:
            logger.error(f"Received non-JSON message from {message.topic}: {message.payload.decode()}")
            return

        # check if the message is valid - sent after the edge mqtt client is started
        if float(message_data.timestamp) <= float(self.start_timestamp):
            return
        logger.debug(f"Received a valid message")

        # Extend message data
        message_data = self.extend_message_data(message_data, received_timestamp)
        # Save message data to file
        MqttMessageData.save_to_file(OffloadingDataFiles.evaluation_file_path, message_data.to_dict())

        # run offloading algorithm and ask for prediction after the device sends the registration message
        if message_data.topic == Topics.registration.value:
            # run offloading algorithm
            offloading_algo = OffloadingAlgo(
                avg_speed=message_data.avg_speed,
                num_layers=len(self.layers_sizes) - 1,
                layers_sizes=list(self.layers_sizes),
                inference_time_device=list(self.device_inference_times),
                inference_time_edge=list(self.edge_inference_times)
            )
            best_offloading_layer = offloading_algo.static_offloading()
            # ask for prediction
            self.ask_for_prediction(
                ask_device_id=message_data.device_id,
                message_id=message_data.message_id,
                best_offloading_layer=best_offloading_layer,
            )

        # ends the computation after receiving the inference result
        if message_data.topic == Topics.device_inference_result.value:
            # update device inference time
            with open(OffloadingDataFiles.data_file_path_device, 'r') as f:
                device_inference_times = json.load(f)
            for l_id, inference_time in enumerate(message_data.device_layers_inference_time):
                device_inference_times[f"layer_{l_id}"] = inference_time
            with open(OffloadingDataFiles.data_file_path_device, 'w') as f:
                json.dump(device_inference_times, f)
            # end the computation
            self.end_computation(ask_device_id=message_data.device_id, message_id=message_data.message_id)

    def ask_for_prediction(self, ask_device_id, message_id, best_offloading_layer: int):
        logger.debug(f"Sending inference request to {ask_device_id}")
        message_data = DefaultMessages.ask_for_inference_msg
        message_data["timestamp"] = self.get_ntp_timestamp()
        message_data['message_id'] = message_id
        message_data['offloading_layer_index'] = best_offloading_layer
        self.publish(Topics.device_inference.value, json.dumps(message_data))

    def end_computation(self, ask_device_id, message_id):
        logger.debug(f"Sending end computation to {ask_device_id}")
        message_data = DefaultMessages.end_computation_msg
        message_data["timestamp"] = self.get_ntp_timestamp()
        message_data['message_id'] = message_id
        self.publish(Topics.end_computation.value, json.dumps(message_data))

    def load_stats(self):
        """ Loads the offloading stats from the JSON files """
        with open(OffloadingDataFiles.data_file_path_device, 'r') as file:
            self.device_inference_times = json.load(file)
            self.device_inference_times = list({k: v for k, v in self.device_inference_times.items()}.values())

        with open(OffloadingDataFiles.data_file_path_edge, 'r') as file:
            self.edge_inference_times = json.load(file)
            self.edge_inference_times = list({k: v for k, v in self.edge_inference_times.items()}.values())

        with open(OffloadingDataFiles.data_file_path_sizes, 'r') as file:
            self.layers_sizes = json.load(file)
            self.layers_sizes = list({k: v for k, v in self.layers_sizes.items()}.values())

        logger.debug(f"Loaded stats data")

    @staticmethod
    def extend_message_data(message_data: MqttMessageData, received_timestamp: float) -> MqttMessageData:
        """Extend the message data with additional information.

        Args:
            message_data (MqttMessageData): The message data to extend.
            received_timestamp (float): The timestamp of the message reception.

        Returns:
            MqttMessageData: The extended message data.
        """
        # update stats info
        message_data.received_timestamp = received_timestamp
        message_data.payload_size = MqttMessageData.get_bytes_size(message_data.payload)
        message_data.synthetic_latency = MqttMessageData.get_synthetic_latency()
        message_data.latency = MqttMessageData.get_latency(message_data.received_timestamp, message_data.timestamp)
        message_data.avg_speed = MqttMessageData.get_avg_speed(
            message_data.payload_size,
            message_data.latency,
            message_data.synthetic_latency
        )
        # update offloading info
        (
            message_data.offloading_layer_index,
            message_data.layer_output,
            message_data.device_layers_inference_time
        ) = MqttMessageData.get_offloading_info(message_data.message_content)
        return message_data
