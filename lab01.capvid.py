#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np  # Not used directly here but often useful for image processing

# Initialize the webcam (device index 0 = default camera)
cap = cv2.VideoCapture(0)

# Read the first two frames from the webcam to use for motion comparison
_, frame1 = cap.read()
_, frame2 = cap.read()

while True:
    # === MOTION DETECTION PREPROCESSING ===

    # Compute the absolute difference between two frames
    diff = cv2.absdiff(frame1, frame2)  # Highlights pixel changes (i.e., motion)

    # Convert the diff image to grayscale (for simpler processing)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise and smooth it out
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a binary threshold to isolate significant differences
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded image to fill in holes and connect regions
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the dilated image (i.e., regions of movement)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # === DRAW DETECTIONS ON FRAME ===
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)  # Get bounding box of the contour

        # Ignore small areas (likely noise or minor flickers)
        if cv2.contourArea(contour) < 900:
            continue

        # Draw a green rectangle around the detected motion
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display "Movement" status text
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show the frame with detections in a window
    cv2.imshow("feed", frame1)

    # Prepare for next iteration by shifting frames
    frame1 = frame2
    _, frame2 = cap.read()  # Read a new frame for comparison

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(40) == ord('q'):
        break

# === CLEANUP ===
cap.release()            # Release webcam
cv2.destroyAllWindows()  # Close all OpenCV windows

# ============== ADDITIONAL Questions ==================
'''
Q1. Identify and explain the additional functionalities introduced
in Code #2. How do these changes transform the program from a
simple image capture to a movement detection system? 

========================================================

capimg.py:
- Captures a single image from the webcam.
- Saves the captured image as a .jpg file.
- No looping, no image processing, no interactivity beyond one shot.

capvid.py:
- Continously captures frames in a loop.
- Performs real-time motion detection using frame differencing.
- Compares frames, identifies regions of motion, and draws rectangles.
- Displays a live video feed with motion highlighted and status text overlaid.
'''
# ========================================================
'''
Q2. Several new OpenCV functions are used (like cv2.absdiff,
cv2.cvtColor, cv2.GaussianBlur, cv2.threshold, cv2.dilate, and
cv2.findContours). Research each of these functions and understand
their role in processing the video frames for movement detection. 

========================================================

cv2.absdiff(frame1, frame2)
- Calculates the absolute difference between two frames.
- Highlights what changed (motion).

cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
- Converts a color image to grayscale to simplify processing.

cv2.GaussianBlur(image, (5,5), 0)
- Applies a Gaussian blur to reduce noise and smooth the image for better thresholding.

cv2.treshold(img, thresh_val, max_val, type)
- Converts grayscale to binary image based on a threshold value.

cv2.dilate(img, None, iterations=3)
- Expands white areas in a binary image to fill gaps and join contours.

cv2.findContours(img, mode, method)
- Finds contours (i.e., boundaries of shapes) in a binary image to detect regions of interests.
'''
# ========================================================
'''
Q3. The program uses specific conditions (such as contour area) to
decide when to draw rectangles and indicate movement. Experiment
with these parameters to see how they affect the accuracy and
sensitivity of movement detection. 

========================================================
    
Parameters to Experiment With:

if cv2.contourArea(contour) < 900:
    continue

- Lower the threshold (e.g., 300): More sentitive, detects small movements (like
a waving hand or paper flutter).
- Increase the threshold (e.g., 1500-2000) : Less sensitive, only detects larger body
or object movements.

cv2.GaussianBlur(...)
- Change kernel size from (5,5) to (9,9) for more smoothing.

cv2.threshold(..., 20, 255, ...)
- Increase the 20 value to ignore smaller intensity changes (helps reduce false positives).

cv2.erode()
- Add cv2.erode() before cv2.dilate() for morphological opening to eliminate noise.
'''
# ========================================================
'''
Q4. Loop Mechanics and Video Processing: Analyze the role of the while
loop in the 2nd Code for continuous video capture and processing. How
does this looping mechanism differ from the single capture approach in
the 1st Code, especially in terms of real-time processing and movement detection? 

========================================================

capimg.py:
- Executes once and exits.
- Minimal processing.
- Good for capturing still images or debugging.

capvid.py:
- Uses a "while True" loop to capture and process frames in real-time.
- Each frame is:
    1. Compared with the next
    2. Pre-processed
    3. Analysed for motion
    4. Annotated with rectangles if movement is detected

Why This Matters:
- Enabled continous monitoring, like a surveillance camera.
- Provides immediate visual feedback.
- Processes input in near real-time (limited by system performance and webcam FPS).
'''
# ========================================================
'''
Q5. Consider aspects like improving the accuracy of movement detection,
optimizing performance, or adding new features (like recording video when
movement is detected).

========================================================

Accuracy Improvements:
- Background subtraction models (e.g. MOG2 or KNN) for more stable detection.
- Bounding box smoothing using temporal filtering to reduce flicker.
- Use object classification (e.g. detect only humans, not pets) with deep learning.

Performance Optimisations:
- Resize frames to a smaller resolution before processing.
- Run the frame comparison on a lower FPS to reduce CPU usage.
- Move from CPU to GPU-accelerated processing (if available)

New Features to Add:

1. Save video only when motion is detected
- Reduces disk space.
- USeful for surveillance systems.

2. Timestamp overlay
- Show exact time of motion for logs or debugging.

3. Integration with sound or alerts
- Send email/notification if movement is detected.

4. Take a burst of photos on motion
- Store evidence of motion-triggered events.

5. Motion-clip exporter
- Save each motion episde into a separate video file.
'''

# =================== PERFORMANCE & FEATURE ENHANCEMENTS ====================

# 1. Performance Optimization: Resize for Faster Processing
# frame1 = cv2.resize(frame1, (320, 240))
# frame2 = cv2.resize(frame2, (320, 240))
# - Reduces pixel count and speeds up motion detection.
# - Useful on low-power systems like Raspberry Pi.

# 2. Morphological Opening to Reduce Noise
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# dilated = cv2.dilate(opening, None, iterations=2)
# - Removes noise before dilation.

# 3. Use cv2.createBackgroundSubtractorMOG2() for adaptive background modeling:
# mog = cv2.createBackgroundSubtractorMOG2()
# fgmask = mog.apply(frame)
# - More stable under lighting changes or dynamic scenes.

# 4. Save Motion Events (Images or Videos)
# When motion is detected:
#    cv2.imwrite(f'motion_{time.time()}.jpg', frame1)
# OR
#    writer.write(frame1)
# - Helps archive motion triggers for review.

# 5. Trigger Alerts or Logging
# Play sound or log timestamp on motion detection:
# import datetime
# print(f"Motion detected at {datetime.datetime.now()}")

# 6. Dynamic Contour Area Filtering
# Instead of a hardcoded 900 value:
#     if 900 < cv2.contourArea(contour) < 10000:
# - Filters out both small noise and unusually large flickers (e.g., sudden light change)

# 7. Reduce Processing Rate
# Add time.sleep(0.1) or process only every nth frame to save CPU

# 8. Use threading for frame capture
# - Use one thread to continuously grab frames, and another to process.
# - Reduces lag and improves responsiveness.

# 9. Display FPS for debugging:
# import time; start = time.time(); ... end = time.time()
# print(f"FPS: {1/(end-start):.2f}")

# 10. Extend with GUI, Logging or MQTT/Socket Message
# - Notify a server when motion is detected
# - Stream alerts to another device
# - Integrate with cloud storage or dashboards

# ========================================================================
