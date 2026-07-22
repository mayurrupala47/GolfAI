import json
import logging
import time
from typing import Dict, Any
from ai.interfaces import IMqttPublisher

logger = logging.getLogger(__name__)


class MqttPublisher(IMqttPublisher):
    """
    Publishes stroke events to an MQTT broker using the Paho MQTT library.
    """
    def __init__(self, config: Dict[str, Any]):
        self.host = config.get("mqtt", {}).get("host", "localhost")
        self.port = config.get("mqtt", {}).get("port", 1883)
        self.topic = config.get("mqtt", {}).get("topic", "minigolf/stroke")
        self.client_id = config.get("mqtt", {}).get("client_id", "minigolf-ai-tracker")
        self.client = None
        self.connected = False

    def connect(self) -> None:
        """
        Connect to the MQTT broker. Fails gracefully if the broker is offline.
        """
        try:
            import paho.mqtt.client as mqtt
            # Determine API version based on paho-mqtt installation (v1 or v2 support)
            # In paho-mqtt v2, CallbackAPIVersion is required. We initialize safely.
            try:
                # paho-mqtt v2 check
                from paho.mqtt.enums import CallbackAPIVersion
                self.client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2, client_id=self.client_id)
            except (ImportError, AttributeError):
                # Fallback to paho-mqtt v1
                self.client = mqtt.Client(client_id=self.client_id)

            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect

            logger.info(f"Connecting to MQTT Broker at {self.host}:{self.port}...")
            # Set a connection timeout of 5 seconds to avoid hanging main thread
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}. Running in disconnected fallback state.")
            self.connected = False

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        # rc == 0 is successful connection
        # Check standard rc values (or ReasonCode in v2)
        status_code = rc if isinstance(rc, int) else getattr(rc, "value", -1)
        if status_code == 0:
            self.connected = True
            logger.info("Successfully connected to MQTT Broker.")
        else:
            self.connected = False
            logger.error(f"MQTT Connection failed with status code: {status_code}")

    def _on_disconnect(self, client, userdata, rc, properties=None) -> None:
        self.connected = False
        logger.warning("Disconnected from MQTT Broker.")

    def publish_stroke(self, camera_id: str, hole: int, ball_id: int, stroke_count: int, timestamp: str, ball_color: str = "unknown") -> None:
        """
        Publishes a stroke event in JSON format to the topic.
        """
        payload = {
            "camera_id": camera_id,
            "hole": hole,
            "ball_id": ball_id,
            "ball_color": ball_color,
            "stroke": stroke_count,
            "timestamp": timestamp
        }
        
        payload_str = json.dumps(payload)
        
        if self.client is not None and self.connected:
            try:
                res = self.client.publish(self.topic, payload_str, qos=1)
                # Check publish success
                if res.rc == 0: # MQTT_ERR_SUCCESS
                    logger.info(f"MQTT Published successfully to {self.topic}: {payload_str}")
                else:
                    logger.warning(f"MQTT Publish failed with return code {res.rc}. Payload: {payload_str}")
            except Exception as e:
                logger.error(f"Error publishing MQTT message: {e}")
        else:
            logger.info(f"[Offline Fallback] MQTT publish simulated (Broker offline): {payload_str}")

    def disconnect(self) -> None:
        """
        Clean up connection and stop the loop.
        """
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
                logger.info("Disconnected from MQTT broker loop.")
            except Exception as e:
                logger.error(f"Error disconnecting MQTT client: {e}")

    def publish_hole_complete(self, camera_id: str, hole: int, ball_id: int, stroke_count: int, timestamp: str, ball_color: str = "unknown") -> None:
        """
        Publishes a hole-completed event (ball entered the cup) in JSON format.
        """
        payload = {
            "event": "hole_complete",
            "camera_id": camera_id,
            "hole": hole,
            "ball_id": ball_id,
            "ball_color": ball_color,
            "total_strokes": stroke_count,
            "timestamp": timestamp
        }
        topic = self.topic.rsplit("/", 1)[0] + "/hole_complete"
        payload_str = json.dumps(payload)
        if self.client is not None and self.connected:
            try:
                res = self.client.publish(topic, payload_str, qos=1)
                if res.rc == 0:
                    logger.info(f"MQTT Published hole_complete to {topic}: {payload_str}")
                else:
                    logger.warning(f"MQTT Publish hole_complete failed rc={res.rc}")
            except Exception as e:
                logger.error(f"Error publishing hole_complete MQTT message: {e}")
        else:
            logger.info(f"[Offline Fallback] MQTT hole_complete simulated: {payload_str}")

    def publish_reset(self, camera_id: str, hole: int, ball_id: int, timestamp: str, ball_color: str = "unknown") -> None:
        """
        Publishes a tee-reset event (stroke counter cleared) in JSON format.
        """
        payload = {
            "event": "reset",
            "camera_id": camera_id,
            "hole": hole,
            "ball_id": ball_id,
            "ball_color": ball_color,
            "stroke": 0,
            "timestamp": timestamp
        }
        topic = self.topic.rsplit("/", 1)[0] + "/reset"
        payload_str = json.dumps(payload)
        if self.client is not None and self.connected:
            try:
                res = self.client.publish(topic, payload_str, qos=1)
                if res.rc == 0:
                    logger.info(f"MQTT Published reset to {topic}: {payload_str}")
                else:
                    logger.warning(f"MQTT Publish reset failed rc={res.rc}")
            except Exception as e:
                logger.error(f"Error publishing reset MQTT message: {e}")
        else:
            logger.info(f"[Offline Fallback] MQTT reset simulated: {payload_str}")


class MockMqttPublisher(IMqttPublisher):
    """
    Mock publisher that mimics MQTT publishes via python logging.
    Useful for local testing without an active broker.
    """
    def __init__(self, config: Dict[str, Any]):
        self.topic = config.get("mqtt", {}).get("topic", "minigolf/stroke")
        logger.info("Mock MQTT Publisher initialized.")

    def connect(self) -> None:
        logger.info("Mock MQTT connected (no-op).")

    def publish_stroke(self, camera_id: str, hole: int, ball_id: int, stroke_count: int, timestamp: str, ball_color: str = "unknown") -> None:
        payload = {
            "camera_id": camera_id,
            "hole": hole,
            "ball_id": ball_id,
            "ball_color": ball_color,
            "stroke": stroke_count,
            "timestamp": timestamp
        }
        logger.info(f"[Mock MQTT Publish] Topic: '{self.topic}', Payload: {json.dumps(payload)}")

    def publish_hole_complete(self, camera_id: str, hole: int, ball_id: int, stroke_count: int, timestamp: str, ball_color: str = "unknown") -> None:
        payload = {
            "event": "hole_complete",
            "camera_id": camera_id,
            "hole": hole,
            "ball_id": ball_id,
            "ball_color": ball_color,
            "total_strokes": stroke_count,
            "timestamp": timestamp
        }
        topic = self.topic.rsplit("/", 1)[0] + "/hole_complete"
        logger.info(f"[Mock MQTT Publish] Topic: '{topic}', Payload: {json.dumps(payload)}")

    def publish_reset(self, camera_id: str, hole: int, ball_id: int, timestamp: str, ball_color: str = "unknown") -> None:
        payload = {
            "event": "reset",
            "camera_id": camera_id,
            "hole": hole,
            "ball_id": ball_id,
            "ball_color": ball_color,
            "stroke": 0,
            "timestamp": timestamp
        }
        topic = self.topic.rsplit("/", 1)[0] + "/reset"
        logger.info(f"[Mock MQTT Publish] Topic: '{topic}', Payload: {json.dumps(payload)}")

    def disconnect(self) -> None:
        logger.info("Mock MQTT disconnected (no-op).")
