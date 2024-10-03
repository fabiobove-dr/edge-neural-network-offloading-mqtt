from pytest import fixture

from src.mqtt_client.mqtt_client import MqttClient


@fixture
def mqtt_client_fixture():
    return MqttClient()


@fixture
def device_fixture(mqtt_client_fixture):
    return MqttClient()
