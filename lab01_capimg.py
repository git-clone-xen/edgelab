#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2  # Import OpenCV for image capture and processing

# Initialize the webcam (device index 0 = default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is successfully opened
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Capture a single frame from the webcam
ret, frame = cap.read()  # `ret` is a boolean indicating success, `frame` is the captured image

# Save the captured frame as an image file
cv2.imwrite('captured_image.jpg', frame)  # Writes the frame to a JPEG file in the current directory

# Release the webcam resource
cap.release()

# Print confirmation message
print("Image captured and saved!")

# ======================= Perfomance Enhancements =============================
# 1. Add a timestamp to the filename to avoid overwriting previous captures:
# import time
# timestamp = time.strftime("%Y%m%d-%H%M%S")
# filename = f"captured_image_{timestamp}.jpg"
# cv2.imwrite(filename, frame)

# ==============================================================================
# 2. Add frame capture error handling to avoid saving a blank image:
# if not ret:
#     print("Failed to capture image. Please try again.")
#     cap.release()
#     exit()

# ==============================================================================
# 3. Display the captured image to the user before saving (preview mode):
# cv2.imshow("Preview - Press any key to save", frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ==============================================================================
# 4. Convert the image to grayscale before saving:
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('captured_image_gray.jpg', gray)

# ==============================================================================
# 5. Save to a specific directory and ensure it exists:
# import os
# save_path = "captures"
# os.makedirs(save_path, exist_ok=True)
# cv2.imwrite(os.path.join(save_path, "captured_image.jpg"), frame)

# ==============================================================================
# 6. Apply a blur or filter to the image before saving (optional visual enhancement):
# blurred = cv2.GaussianBlur(frame, (7, 7), 0)
# cv2.imwrite('captured_image_blur.jpg', blurred)

# ==============================================================================
# 7. Capture multiple images in a burst loop (for animation or motion freeze):
# for i in range(5):
#     ret, frame = cap.read()
#     if ret:
#         cv2.imwrite(f"burst_{i}.jpg", frame)
#         time.sleep(0.5)  # Optional delay between captures

# ==============================================================================
# 8. Change the resolution of the webcam capture:
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ==============================================================================
# 9. Only save the image if a face is detected (using Haar Cascades):
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# if len(faces) > 0:
#     cv2.imwrite('captured_face.jpg', frame)
# else:
#     print("No face detected. Image not saved.")

# ==============================================================================
# 10. Trigger image capture from MQTT/Voice Command/etc. (IoT integration idea)
# You could subscribe to an MQTT topic and execute the `cap.read()` only when a message is received.

# Example pseudocode:
# def on_message(client, userdata, message):
#     if message.payload.decode() == "capture":
#         ret, frame = cap.read()
#         if ret:
#             cv2.imwrite("mqtt_capture.jpg", frame)

# ==============================================================================
