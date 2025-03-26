# This script publishes a previously captured image (from Script 1) as binary data to an MQTT topic.

import paho.mqtt.client as mqtt
import time

# MQTT Configuration
BROKER_ADDRESS = "localhost"       # Replace with your MQTT broker address
TOPIC = "camera/image"             # Topic to publish image data
IMAGE_PATH = "captured_image.jpg"  # Path to the image to be sent

# Initialize MQTT client and connect
client = mqtt.Client()
client.connect(BROKER_ADDRESS, 1883, 60)

# Read the image file in binary mode
try:
    with open(IMAGE_PATH, "rb") as f:
        image_data = f.read()
        # Publish the image data as binary payload
        client.publish(TOPIC, image_data)
        print(f"📤 Published image '{IMAGE_PATH}' to topic '{TOPIC}'")
except FileNotFoundError:
    print(f"❌ Image '{IMAGE_PATH}' not found. Please run the subscriber script to capture an image first.")

client.disconnect()

'''
1. Start an MQTT broker

2. Run subscriber_capture.py 

3. Publish "capture" to topic camera/trigger using any MQTT client (e.g. mosquitto_pub)
    mosquitto_pub -t camera/trigger -m "capture"

4. After image is saved, run publisher_image.py to publish it to camera/image
'''

# ======================== PERFORMANCE & USABILITY ENHANCEMENTS ============================

# 1. Add error handling for MQTT connection failure
# Prevents script from silently failing if the broker is unreachable
# try:
#     client.connect(BROKER_ADDRESS, 1883, 60)
# except Exception as e:
#     print(f"Could not connect to MQTT broker: {e}")
#     exit(1)

# ===========================================================================================

# 2. Use client.loop_start() to support async communication (if used in larger apps)
# This allows the MQTT client to handle background network traffic in threaded mode
# client.loop_start()
# ... perform publishing ...
# client.loop_stop()

# ===========================================================================================

# 3. Add confirmation callback to verify image was published successfully
# def on_publish(client, userdata, mid):
#     print("Publish confirmed.")
# client.on_publish = on_publish

# ===========================================================================================

# 4. Clean disconnect using client.disconnect()
# Already included — ensures graceful exit and clears retained sessions

# ===========================================================================================

# 5. Log payload size (helps debug large image handling)
# print(f"Image size: {len(image_data)} bytes")

# ===========================================================================================

# 6. Publish with QoS (Quality of Service) level if delivery reliability is needed
# QoS 0 = at most once (default), 1 = at least once, 2 = exactly once
# client.publish(TOPIC, image_data, qos=1)

# ===========================================================================================

# 7. Use Base64 encoding for safer transport over MQTT if binary corruption occurs
# import base64
# encoded = base64.b64encode(image_data)
# client.publish(TOPIC, encoded)
# This is useful if the MQTT broker struggles with raw binary or for cross-platform transmission.

# ===========================================================================================

# 8. Support for sending multiple images (loop through a folder)
# import os
# for filename in os.listdir("images"):
#     if filename.endswith(".jpg"):
#         with open(f"images/{filename}", "rb") as f:
#             client.publish(TOPIC, f.read())
#             print(f"Sent {filename}")

# ===========================================================================================

# 9. Add a dry-run mode to test without sending data
# useful for debugging file access
# dry_run = True
# if not dry_run:
#     client.publish(TOPIC, image_data)

# ===========================================================================================

# 10. Integrate into Flask or CLI for flexible usage
# You can turn this into a callable function in a larger app:
# def send_image(path, topic="camera/image"):
#     with open(path, "rb") as f:
#         client.publish(topic, f.read())

# ===========================================================================================
