# Import required libraries
import cv2                      # OpenCV for video capture and image processing
import mediapipe as mp          # MediaPipe for face mesh and drawing utilities

#%% Initialize MediaPipe Face Mesh

# Load the FaceMesh module from MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Create a FaceMesh object with settings:
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,           # False = continuous video stream (not static images)
    max_num_faces=1,                   # Detect only one face (change to >1 to track multiple)
    min_detection_confidence=0.5,      # Minimum confidence for face detection
    min_tracking_confidence=0.5        # Minimum confidence for tracking facial landmarks
)

# Drawing utilities from MediaPipe to render mesh and contours
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#%% Initialize the webcam

cap = cv2.VideoCapture(0)  # Open the default webcam (device index 0)

# Check if webcam is accessible
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

#%% Main processing loop

while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the BGR image (OpenCV default) to RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe FaceMesh
    results = face_mesh.process(rgb_frame)

    # If facial landmarks were detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh tessellation (triangular structure across the face)
            mp_drawing.draw_landmarks(
                image=frame,  # Original frame (BGR)
                landmark_list=face_landmarks,  # Detected landmark coordinates
                connections=mp_face_mesh.FACEMESH_TESSELATION,  # Mesh triangles
                landmark_drawing_spec=None,  # Don't draw individual dots
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Draw contours (jawline, lips, eyebrows, etc.)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    # Show the annotated frame in a window
    cv2.imshow('Mediapipe Face Mesh', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

#%% Clean up

cap.release()              # Release the webcam
cv2.destroyAllWindows()    # Close all OpenCV windows

# ============================ Enhancements & Optimization Suggestions ============================

# 1. Resize frame before processing to improve FPS:
# Reduces computational load and increases speed, especially helpful on lower-end CPUs or Raspberry Pi.

# Example:
# frame = cv2.resize(frame, (640, 480))
# Optional: process a copy for mesh drawing, and show the full-sized original if needed

# =================================================================================================

# 2. Limit processing to every Nth frame:
# Face mesh detection is expensive. Skipping some frames improves speed with minimal visual drop.

# Example:
# frame_count = 0
# if frame_count % 3 == 0:
#     results = face_mesh.process(rgb_frame)

# =================================================================================================

# 3. Customize drawing styles (change mesh color, thickness, etc.)
# You can define your own drawing specs to improve clarity or aesthetics.

# Example:
# custom_style = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
# mp_drawing.draw_landmarks(..., landmark_drawing_spec=custom_style)

# =================================================================================================

# 4. Track and annotate landmark indices:
# Helpful for AR applications (e.g., detecting mouth, eyes, etc.)

# Example:
# for idx, landmark in enumerate(face_landmarks.landmark):
#     x = int(landmark.x * frame.shape[1])
#     y = int(landmark.y * frame.shape[0])
#     cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

# =================================================================================================

# 5. Export landmarks for analysis or ML:
# You can save the landmarks to a file (e.g., CSV or JSON) for emotion recognition, face tracking, etc.

# Example:
# landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
# np.savetxt("face_landmarks.csv", landmarks, delimiter=",")

# =================================================================================================

# 6. Add head pose estimation using selected landmark points:
# Use 3D model fitting with points like eyes, nose tip, chin for estimating tilt, pitch, and yaw.

# Libraries like OpenCV's `solvePnP` help with this:
# https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

# =================================================================================================

# 7. 👁Detect face direction (left, right, center):
# Use key landmarks like eye corners or nose tip to infer horizontal orientation.

# Example logic:
# if nose_x < frame_width * 0.4: face turned right
# if nose_x > frame_width * 0.6: face turned left

# =================================================================================================

# 8. Add a timeout or fallback if no face is detected:
# Avoid visual clutter by adding overlay text when no face is found.

# Example:
# if not results.multi_face_landmarks:
#     cv2.putText(frame, "No face detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

# =================================================================================================

# 9. Combine with audio (speech recognition, emotion via voice):
# Pair this with libraries like `speech_recognition` or `librosa` for multimodal interaction.

# =================================================================================================

# 10. Record annotated video with face mesh:
# Save output to file using OpenCV's `VideoWriter`.

# Example:
# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))
# out.write(frame)

# =================================================================================================

# 11. Enable/disable specific mesh layers (like irises, lips):
# Use FACEMESH_IRISES, FACEMESH_LIPS, FACEMESH_RIGHT_EYE, etc., selectively to reduce clutter.

# Example:
# mp_face_mesh.FACEMESH_IRISES, mp_face_mesh.FACEMESH_LIPS

# =================================================================================================

# 12. Use multiprocessing/threading for heavy real-time applications:
# If using this in a Flask server, PyQt GUI, or a ROS node, offload camera or inference to threads.

# =================================================================================================

# 13. FPS counter overlay:
# Measure and display how fast your mesh detection runs.

# Example:
# import time
# prev_time = time.time()
# fps = 1 / (time.time() - prev_time)
# cv2.putText(frame, f"FPS: {int(fps)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# =================================================================================================

# 14. Upgrade MediaPipe to use GPU if available:
# Install `mediapipe-nightly` or run via Android/iOS to utilize GPU acceleration for better FPS.

# =================================================================================================
