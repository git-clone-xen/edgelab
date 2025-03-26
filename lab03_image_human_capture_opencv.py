# For the code to work, the open-source Haar Cascade XML model must be downloaded.
# Download from:
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml
# Place it in the same folder as this script.

import cv2  # OpenCV library for computer vision and webcam processing

#%% Initialize Haar Cascade Face Detector

# Path to the Haar Cascade classifier file for detecting frontal faces
haarcascade = "haarcascade_frontalface_alt2.xml"

# Load the Haar Cascade model
detector = cv2.CascadeClassifier(haarcascade)

#%% Start Webcam Video Capture

# Open the default webcam (device index 0)
cap = cv2.VideoCapture(0)

# Validate webcam connection
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#%% Main Frame Processing Loop

# This loop will run until 'q' is pressed
while True:
    try:
        # Capture a single frame from the webcam
        ret, frame = cap.read()

        # OPTIONAL: Resize the image to a smaller size for faster processing
        # This helps speed up face detection
        frame = cv2.resize(frame, (256, 256))  # Try commenting this line to compare speed

        # Convert the color image to grayscale
        # Haar Cascades work on grayscale images for efficiency and simplicity
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run the face detector on the grayscale image
        # Returns a list of bounding boxes (x, y, width, height) for each detected face
        faces = detector.detectMultiScale(image_gray)

        # Loop through each detected face
        for face in faces:
            (x, y, w, d) = face

            # Draw a white rectangle around the detected face on the original frame
            # Parameters: top-left corner, bottom-right corner, color (white), thickness
            cv2.rectangle(frame, (x, y), (x + w, y + d), (255, 255, 255), 2)

        # OPTIONAL: Resize the frame to display a larger window (720x720 pixels)
        frame = cv2.resize(frame, (720, 720))

        # Show the processed frame in a window titled "frame"
        cv2.imshow("frame", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Allow exit using Ctrl+C
        break

#%% Cleanup After Exit

cap.release()              # Release the webcam
cv2.destroyAllWindows()    # Close any OpenCV GUI windows

'''
detectMultiScale() Parameters

faces = detector.detectMultiScale(
    image_gray,
    scaleFactor=1.1,       # How much the image size is reduced at each scale (lower = slower, better)
    minNeighbors=5,        # How many neighbors each rectangle should have to retain it
    minSize=(30, 30),      # Minimum face size to detect
    flags=cv2.CASCADE_SCALE_IMAGE
)
'''

# ======================== Enhancements & Optimization Suggestions ============================

# 1. Use full `detectMultiScale()` parameters for more control and stability:
# Helps fine-tune detection for accuracy and reduce false positives.

# Example:
# faces = detector.detectMultiScale(
#     image_gray,
#     scaleFactor=1.1,         # How much the image is scaled between detections (lower = finer detection)
#     minNeighbors=5,          # Number of surrounding rectangles to consider a detection valid
#     minSize=(30, 30),        # Skip objects smaller than this (reduces false positives)
#     flags=cv2.CASCADE_SCALE_IMAGE
# )

# ================================================================================================

# 2. Adjust `cv2.resize()` size for performance trade-off:
# A smaller frame (e.g., 128x128) will be faster but may miss small faces.
# A larger frame (e.g., 320x320) gives better detection but increases CPU load.

# Example:
# frame = cv2.resize(frame, (320, 320))

# ================================================================================================

# 3. Apply Histogram Equalization for better contrast in poor lighting:
# This improves the grayscale input and increases face detection success in shadows or uneven lighting.

# Example:
# image_gray = cv2.equalizeHist(image_gray)

# ================================================================================================

# 4. Save images when a face is detected:
# Useful for building datasets or logging who entered the frame.

# Example:
# if len(faces) > 0:
#     cv2.imwrite(f"face_detected_{time.time()}.jpg", frame)

# ================================================================================================

# 5. Display the number of detected faces:
# This gives visual feedback on how many faces were found in real-time.

# Example:
# cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# ================================================================================================

# 6. Use another Haar cascade (like for eyes or smiles) for multi-feature detection:
# You can cascade multiple detectors to enhance applications (e.g., filter only when both face + eyes exist).

# Example:
# eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
# eyes = eyes_cascade.detectMultiScale(image_gray)

# ================================================================================================

# 7. Add FPS counter to monitor performance:
# Helps track if detection is real-time or lagging.

# Example:
# import time
# prev_time = time.time()
# fps = 1 / (time.time() - prev_time)
# cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# ================================================================================================

# 8. Add background blur outside the face:
# Draw attention to detected face and hide background.

# Example (optional advanced):
# Use a mask to isolate and blur background using GaussianBlur()

# ================================================================================================

# 9. Save detections to video:
# Add a video writer to save the entire session with rectangles drawn.

# Example:
# out = cv2.VideoWriter('faces.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (720, 720))
# out.write(frame)

# ================================================================================================

# 10. Combine with motion detection or emotion recognition:
# You can trigger face detection only when motion is detected for efficiency.
# You can also pass the detected face ROI to a CNN for emotion classification.

# Example:
# face_img = frame[y:y+d, x:x+w]
# prediction = emotion_model.predict(preprocess(face_img))
