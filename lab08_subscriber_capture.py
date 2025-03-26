# This script subscribes to a topic and captures an image from the webcam when it receives a "capture" command.

import cv2
import paho.mqtt.client as mqtt

# MQTT Configuration
BROKER_ADDRESS = "localhost"  # Replace with your MQTT broker address
TOPIC = "camera/trigger"      # Topic to listen for capture commands

# MQTT callback when a connection is established
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT broker with result code", rc)
    client.subscribe(TOPIC)

# MQTT callback when a message is received on a subscribed topic
def on_message(client, userdata, msg):
    command = msg.payload.decode()
    print(f"Received command: {command}")
    if command.lower() == "capture":
        capture_image()

# Function to capture image from webcam
def capture_image():
    print("Capturing image from webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access the webcam.")
        return
    ret, frame = cap.read()
    if ret:
        filename = "captured_image.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ Image captured and saved as '{filename}'")
    else:
        print("❌ Failed to capture image.")
    cap.release()

# Initialize and run MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER_ADDRESS, 1883, 60)

print("📡 Waiting for capture command...")
client.loop_forever()

# ====================== ⚙️ PERFORMANCE & FUNCTIONALITY ENHANCEMENTS ===========================

# 1. Improve camera access time by initializing the webcam only once
# This avoids reopening the webcam every time a "capture" command is received.
# Place cap = cv2.VideoCapture(0) at the global level and reuse it in capture_image().
# This also helps prevent issues with slow or unstable webcams.

# Example:
# global_cap = cv2.VideoCapture(0)
# def capture_image():
#     global global_cap
#     if not global_cap.isOpened():
#         print("Cannot access the webcam.")
#         return
#     ret, frame = global_cap.read()
#     ...

# ==============================================================================================

# 2. Gracefully handle program shutdown (Ctrl+C or SIGINT)
# This ensures the webcam is released and MQTT client is disconnected.

# import signal
# def graceful_exit(sig, frame):
#     print("Exiting...")
#     cap.release()
#     client.disconnect()
#     exit(0)
# signal.signal(signal.SIGINT, graceful_exit)

# ==============================================================================================

# 3. Validate MQTT payload content more strictly
# This prevents unintended behavior due to malformed or non-string payloads.

# if isinstance(msg.payload, bytes):
#     command = msg.payload.decode(errors="ignore").strip().lower()
#     if command == "capture":
#         capture_image()

# ==============================================================================================

# 4. Add publish acknowledgment to let the sender know image was captured
# client.publish("camera/ack", "image_captured")

# ==============================================================================================

# 5. Save images with timestamps for logging and tracking multiple events
# from datetime import datetime
# filename = f"captured_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
# cv2.imwrite(filename, frame)

# ==============================================================================================

# 6. Organize images into folders (e.g., by date or session)
# import os
# folder = f"captures/{datetime.now().strftime('%Y-%m-%d')}"
# os.makedirs(folder, exist_ok=True)
# cv2.imwrite(os.path.join(folder, filename), frame)

# ==============================================================================================

# 7. Run in threaded mode with `loop_start()` for non-blocking operation (advanced use)
# client.loop_start()
# This is useful if you want to integrate the subscriber into a GUI or a bigger app without freezing.

# ==============================================================================================

# 8. Secure communication by switching to TLS if MQTT broker requires it
# client.tls_set(ca_certs="ca.crt")
# client.username_pw_set("user", "pass")

# ==============================================================================================

# 9. Add debounce logic to ignore duplicate "capture" commands within short intervals
# import time
# last_capture_time = 0
# def on_message(...):
#     global last_capture_time
#     if time.time() - last_capture_time > 2:
#         capture_image()
#         last_capture_time = time.time()

# ==============================================================================================

# 10. Extend to capture and immediately publish the image (combined subscriber + publisher)
# Useful if you want everything handled in one process instead of two separate scripts.

# ==============================================================================================
