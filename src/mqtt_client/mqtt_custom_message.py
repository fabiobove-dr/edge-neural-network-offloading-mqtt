import json
import os
import random
from dataclasses import dataclass

import pandas as pd

from src.logger.log import logger


@dataclass
class MqttMessageData:
    topic: str
    payload: str
    device_id: int
    message_id: int
    message_content: str
    timestamp: str

    received_timestamp = None
    avg_speed = None
    latency = None
    synthetic_latency = None
    payload_size = None
    offloading_layer_index = None
    layer_output = None
    device_layers_inference_time = None

    @staticmethod
    def from_raw(topic: str, payload: bytes):
        """Parse the raw message payload into a MessageDataInput instance."""
        try:
            # decode the payload from bytes to string and parse as JSON
            message_data = json.loads(payload.decode())
            message_content = message_data["message_content"]
            # return an instance of MessageDataInput with extracted fields
            return MqttMessageData(
                topic=topic,
                payload=payload.decode(),
                device_id=message_data["device_id"],
                message_id=message_data["message_id"],
                message_content=message_content,
                timestamp=message_data["timestamp"],
            )
        except json.JSONDecodeError:
            # handles payload that cannot be parsed as JSON
            logger.error(f"Failed to decode JSON from payload on topic {topic}")
            return None

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def save_to_file(file_path: str, data_dict: dict):
        # check if the file already exists
        file_exists = os.path.isfile(file_path)
        try:
            # create a DataFrame from the data dictionary
            df = pd.DataFrame.from_dict([data_dict])
            # append to the CSV file; write header only if file does not exist
            df.to_csv(file_path, mode='a', header=not file_exists, index=False)
            logger.debug(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}")

    @staticmethod
    def get_latency(timestamp: str, recieved_timestamp: str) -> tuple[float, dict]:
        # NTP timestamps as strings (representing seconds since 1900)
        # convert the NTP timestamps from string to float
        ntp_timestamp_1 = float(timestamp)
        ntp_timestamp_2 = float(recieved_timestamp)
        # calculate the duration between the two NTP timestamps
        duration_seconds = ntp_timestamp_2 - ntp_timestamp_1
        # convert the duration to a readable format
        return duration_seconds

    @staticmethod
    def get_bytes_size(payload) -> int:
        return len(payload)

    @staticmethod
    def get_synthetic_latency() -> float:
        return random.uniform(1, 1111)

    @staticmethod
    def get_avg_speed(payload_size: float, latency: float, synthetic_latency: float) -> float:
        message_latency = latency * synthetic_latency
        try:
            avg_speed = payload_size / message_latency
        except ZeroDivisionError:
            avg_speed = 0
        return avg_speed

    @staticmethod
    def get_offloading_info(message_content: dict) -> tuple:
        # check if layer_output and offloading_layer_index exist in message_content
        try:
            layer_output = message_content.get("layer_output", None)
            offloading_layer_index = message_content.get("offloading_layer_index", None)
            device_layers_inference_time = message_content.get("layers_inference_time", None)
            return offloading_layer_index, layer_output, device_layers_inference_time
        except Exception as _:
            return None, None, None
