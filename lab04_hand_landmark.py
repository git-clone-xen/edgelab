#%% Reference:
# https://github.com/googlesamples/mediapipe/tree/main/examples/hand_landmarker/raspberry_pi
# Model file can be downloaded using:
# wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#%% Parameters

numHands = 2  
# Maximum number of hands to detect in each frame.
# ↑ = detect more hands, but increases processing time.
# ↓ = faster but may miss multiple hands.

model = 'hand_landmarker.task'  
# Path to the downloaded TFLite model for hand landmark detection.
# Ensure the file exists in the directory or change the path accordingly.

minHandDetectionConfidence = 0.5  
# Minimum confidence threshold for hand detection.
# Range: 0.0 to 1.0
# ↑ = stricter hand detection (more confident but might miss some hands)
# ↓ = detects more hands but may result in false positives

minHandPresenceConfidence = 0.5  
# Threshold for hand **presence confidence** after detection.
# Ensures that once detected, the model is confident the hand remains in view.
# ↑ = more stable results, ↓ = might flicker on hand loss

minTrackingConfidence = 0.5  
# Confidence threshold for tracking the hand across frames.
# ↑ = smoother, more stable tracking (but slower)
# ↓ = less accurate tracking in motion or occlusion

frameWidth = 640   # Set the frame width of the webcam
frameHeight = 480  # Set the frame height of the webcam

# Visualization parameters
MARGIN = 10                  # Margin in pixels for drawing text/labels
FONT_SIZE = 1                # Font size used in overlay text
FONT_THICKNESS = 1           # Thickness of text drawn on the image
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # Text color (BGR) — vibrant green

#%% Create a HandLandmarker object

# Load model using MediaPipe's BaseOptions
base_options = python.BaseOptions(model_asset_path=model)

# Define options for the hand landmarker
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=numHands,  # How many hands to track
    min_hand_detection_confidence=minHandDetectionConfidence,
    min_hand_presence_confidence=minHandPresenceConfidence,
    min_tracking_confidence=minTrackingConfidence
)

# Instantiate the hand detector from the defined options
detector = vision.HandLandmarker.create_from_options(options)

#%% OpenCV Webcam Capture Setup
cap = cv2.VideoCapture(0)  # Open the default webcam (device index 0)

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)

# Validate webcam connection
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Main loop: runs until 'q' is pressed
while True:
    try:
        # Capture frame from webcam
        ret, frame = cap.read()

        # Flip the frame horizontally (mirror effect — more intuitive for users)
        frame = cv2.flip(frame, 1)

        # Convert image from BGR (OpenCV format) to RGB (TensorFlow Lite expects RGB)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Wrap RGB image in a MediaPipe format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run the detector to identify hands and landmarks
        detection_result = detector.detect(mp_image)

        # Extract the landmark coordinates of detected hands
        hand_landmarks_list = detection_result.hand_landmarks
        # handedness_list = detection_result.handedness  # Optionally used to determine left/right hand

        total_fingers_up = 0  # Accumulate total number of fingers raised across all hands

        # Process each detected hand
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # ---------------- [MODIFICATION] Show all 21 landmarks ----------------
            # Loop through all 21 landmark indices and draw a small blue dot at each location
            for i, landmark in enumerate(hand_landmarks):
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Blue dot

            # ---------------- Detect and Draw Thumb Tip ----------------
            # Index 4 corresponds to thumb tip as per MediaPipe's hand landmark guide
            x = int(hand_landmarks[4].x * frame.shape[1])
            y = int(hand_landmarks[4].y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # ---------------- Detect and Draw Index Finger Tip ----------------
            # Index 8 corresponds to the index finger tip
            x = int(hand_landmarks[8].x * frame.shape[1])
            y = int(hand_landmarks[8].y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # ---------------- Thumb Up Gesture Detection ----------------
            # Compare y-coordinates of thumb tip (index 4) and base (index 1)
            # If thumb tip is significantly above the base, we assume a "Thumb Up"
            threshold = 0.1  # Minimum vertical distance to consider as thumb up
            thumb_tip_y = hand_landmarks[4].y
            thumb_base_y = hand_landmarks[1].y

            # In image coordinates, higher y-value means lower in image, so we check if thumb_tip is "higher"
            thums_up = thumb_tip_y < (thumb_base_y - threshold)

            # Display label if thumb up is detected
            if thums_up:
                cv2.putText(frame, 'Thumb Up', (10, 30),
                            cv2.FONT_HERSHEY_DUPLEX,
                            FONT_SIZE,
                            HANDEDNESS_TEXT_COLOR,
                            FONT_THICKNESS,
                            cv2.LINE_AA)

            # ---------------- [MODIFICATION] Count raised fingers ----------------
            # Logic: Compare each fingertip with its PIP joint to determine if finger is "up"
            # Landmark pairs: (thumb: 4-2), (index: 8-6), (middle: 12-10), (ring: 16-14), (pinky: 20-18)
            tips_ids = [4, 8, 12, 16, 20]
            base_ids = [2, 6, 10, 14, 18]

            fingers_up = 0
            for tip_id, base_id in zip(tips_ids, base_ids):
                tip = hand_landmarks[tip_id]
                base = hand_landmarks[base_id]

                # For thumb, use x-axis since thumb sticks out sideways
                if tip_id == 4:
                    if tip.x > base.x:  # Adjust for right hand; may reverse for left
                        fingers_up += 1
                else:
                    if tip.y < base.y:  # Tip higher than base = finger raised
                        fingers_up += 1

            total_fingers_up += fingers_up  # Accumulate across all hands

        # ---------------- [MODIFICATION] Display total finger count ----------------
        cv2.putText(frame, f'Fingers: {total_fingers_up}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SIZE + 0.5,
                    (0, 255, 255), 2, cv2.LINE_AA)  # Yellow text

        # Show the annotated frame
        cv2.imshow('Annotated Image', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Graceful exit with Ctrl+C
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

# ========================== ENHANCEMENT & OPTIMIZATION COMMENTS ==============================

# 1. Save gesture sequences (e.g., finger counts) for ML training or logging:
# Useful for training a classifier, building datasets, or gesture logging.

# Example:
# import csv
# with open("finger_log.csv", "a") as f:
#     f.write(f"{time.time()},{total_fingers_up}\n")

# ==============================================================================================

# 2. Label individual hands as "Left" or "Right" using handedness info:
# Use detection_result.handedness to differentiate actions per hand.

# Example:
# handedness = detection_result.handedness[idx][0].category_name
# cv2.putText(frame, handedness, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

# ==============================================================================================

# 3. Add FPS display for real-time performance tracking:
# Helps you track inference speed and tweak thresholds or resolution accordingly.

# Example:
# prev_time = time.time()
# fps = 1 / (time.time() - prev_time)
# cv2.putText(frame, f'FPS: {fps:.1f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

# ==============================================================================================

# 4. Add gesture control actions (e.g., map number of fingers to app actions):
# Trigger system events (play/pause, volume up/down, etc.)

# Example:
# import pyautogui
# if total_fingers_up == 5:
#     pyautogui.press("playpause")

# ==============================================================================================

# 5. Apply median filter or sliding window smoothing to finger count:
# Prevents rapid fluctuation due to slight finger motion/noise.

# Example:
# from collections import deque
# finger_buffer = deque(maxlen=5)
# finger_buffer.append(total_fingers_up)
# smoothed_fingers = round(np.median(finger_buffer))

# ==============================================================================================

# 6. Reduce frame resolution (e.g., 320x240) for faster inference:
# Works especially well on resource-limited hardware like Raspberry Pi.

# Example:
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ==============================================================================================

# 7. Add gesture-based triggers (e.g., "Thumbs up" = capture photo):
# Can trigger save events, UI updates, etc.

# Example:
# if thumbs_up:
#     cv2.imwrite(f"thumb_{int(time.time())}.jpg", frame)

# ==============================================================================================

# 8. Draw bounding boxes around hands:
# Adds spatial context, especially useful for UIs or interactive overlays.

# Example:
# x_vals = [int(landmark.x * frame.shape[1]) for landmark in hand_landmarks]
# y_vals = [int(landmark.y * frame.shape[0]) for landmark in hand_landmarks]
# cv2.rectangle(frame, (min(x_vals), min(y_vals)), (max(x_vals), max(y_vals)), (0, 255, 0), 2)

# ==============================================================================================

# 9. Display a dynamic overlay hint (e.g., "Raise all fingers for menu"):
# Helpful for guiding users during gesture interaction.

# Example:
# if total_fingers_up == 0:
#     cv2.putText(frame, "Show all fingers to open menu", (50, frame.shape[0] - 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# ==============================================================================================

# 10. Train your own model with custom gestures (using Model Maker or TF Lite):
# Go beyond landmark-only processing and classify hand poses.

# Docs: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

# ==============================================================================================
