from src.logger.log import logger

from src.mqtt_client.mqtt_client import MqttClient
from src.mqtt_client.mqtt_configs import MqttClientConfig

if __name__ == "__main__":
    logger.info("Starting the [EDGE] MQTT client")

    # start the MQTT client
    mqtt_client = MqttClient(
        broker_url=MqttClientConfig.broker_url,
        broker_port=MqttClientConfig.broker_port,
        client_id=MqttClientConfig.client_id,
        protocol=MqttClientConfig.protocol,
        subscribed_topics=MqttClientConfig.subscribe_topics
    )

    # run the MQTT client in loop
    mqtt_client.run()

    logger.info("Listening for messages...")
