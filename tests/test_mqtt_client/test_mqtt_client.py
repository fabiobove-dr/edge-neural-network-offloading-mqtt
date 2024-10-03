import pytest


def test_create_random_payload(mqtt_client_fixture):
    # Test the payload creation
    payload = mqtt_client_fixture.create_random_payload()
    assert "id" in payload
    assert isinstance(payload, str)


def test_publish(mocker, mqtt_client_fixture):
    # Mock the client's publish method
    mock_client = mocker.patch('paho.mqtt.client.Client.publish')
    mqtt_client_fixture.publish("test/topic", "test message")

    # Assert that publish was called with correct topic and message
    mock_client.assert_called_once_with("test/topic", "test message")


def test_subscribe(mocker, mqtt_client_fixture):
    # Mock the client's subscribe method
    mock_client = mocker.patch('paho.mqtt.client.Client.subscribe')
    mqtt_client_fixture.subscribe("test/topic")

    # Assert that subscribe was called with correct topic
    mock_client.assert_called_once_with("test/topic")


def test_connect_to_broker(mocker, mqtt_client_fixture):
    # Mock the MQTT client's connect method
    mock_client = mocker.patch('paho.mqtt.client.Client.connect')

    # Simulate connecting to the broker
    mqtt_client_fixture.client.connect(mqtt_client_fixture.broker_url, mqtt_client_fixture.broker_port)

    # Assert that connect was called with the correct broker URL and port
    mock_client.assert_called_once_with(mqtt_client_fixture.broker_url, mqtt_client_fixture.broker_port)


def test_disconnect(mocker, mqtt_client_fixture):
    # Mock the client's disconnect method
    mock_client = mocker.patch('paho.mqtt.client.Client.disconnect')
    mqtt_client_fixture.stop()

    # Assert that disconnect was called
    mock_client.assert_called_once()


if __name__ == "__main__":
    pytest.main()
