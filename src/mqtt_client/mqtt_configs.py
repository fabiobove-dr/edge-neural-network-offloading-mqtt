import enum
from dataclasses import dataclass

from paho.mqtt import client as mqtt


class Topics(enum.Enum):
    registration = "devices/"
    device_inference = "device_01/model_inference"
    device_inference_result = "device_01/model_inference_result"
    end_computation = "device_01/end_computation"


@dataclass
class MqttClientConfig:
    broker_url: str = "FABIO-PC.local"
    broker_port: int = 1883
    client_id: str = "edge"
    subscribe_topics: list = (
        Topics.registration.value,
        Topics.device_inference.value,
        Topics.device_inference_result.value,
        Topics.end_computation.value
    )
    protocol: mqtt.MQTTv311 = mqtt.MQTTv311
    ntp_server: str = "time.google.com"

@dataclass
class DefaultMessages:
    ask_for_inference_msg = {
        "device_id": "edge",
        "message_id": "edge",
        "timestamp": None,
        "message_content": "AskInference",
        "offloading_layer_index": None,
        "input_data": [
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            [255, 255, 0, 0, 0, 255, 255, 255, 255, 255],
            [255, 255, 0, 0, 255, 0, 0, 255, 255, 255],
            [255, 255, 0, 0, 255, 0, 0, 255, 255, 255],
            [255, 255, 0, 0, 255, 255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 0, 0, 255, 255, 255],
            [255, 255, 0, 0, 255, 0, 0, 255, 255, 255],
            [255, 255, 0, 0, 255, 0, 0, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        ]
    }

    end_computation_msg = {
        "device_id": "edge",
        "message_id": "edge",
        "timestamp": None,
        "message_content": "EndComputation"
    }
