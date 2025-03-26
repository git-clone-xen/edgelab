import cv2
import mediapipe as mp
import time
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === CONFIGURATION ===
model_path = 'efficientdet.tflite'           # Path to the downloaded TFLite object detection model
target_object = 'cell phone'                 # Object name to filter for (case-insensitive)
input_video_path = 'input_video.mp4'         # Path to the input video to summarize
output_video_path = 'summary_output.mp4'     # Path to the summarized output video
save_frames_dir = 'filtered_frames'          # Optional folder to store individual matching frames as images

# === OBJECT DETECTION PARAMETERS ===
score_threshold = 0.3     # Minimum confidence score to count a detection as valid
max_results = 5           # Maximum number of objects to detect per frame

# === PREPARE OUTPUT FOLDER ===
os.makedirs(save_frames_dir, exist_ok=True)  # Create folder to save matched frames (if it doesn't exist)

# === SET UP DETECTOR ===
# Define model and detection options for MediaPipe
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,       # Non-live mode (frame-by-frame from video)
    max_results=max_results,
    score_threshold=score_threshold
)

# Create the object detector from the configured options
detector = vision.ObjectDetector.create_from_options(options)

# === OPEN INPUT VIDEO ===
cap = cv2.VideoCapture(input_video_path)         # Open video file for reading
if not cap.isOpened():
    raise IOError(f"Cannot open video: {input_video_path}")

# Retrieve video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))             # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Width of video frames
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Height of video frames

# === OUTPUT VIDEO WRITER ===
# Create a video writer to store the summary (only matched frames)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# === FRAME LOOP ===
frame_idx = 0       # Counter to track current frame index
matched_count = 0   # Counter to track how many frames matched the target object

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Stop if the video ends or there's an error

    # Convert frame from BGR (OpenCV) to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap the RGB image into MediaPipe's Image class
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run the object detection on the current frame
    detection_result = detector.detect(mp_image)

    # Check if any detection matches the target object
    match_found = False
    for detection in detection_result.detections:
        category = detection.categories[0]  # Take the top prediction for each detection
        if category.category_name.lower() == target_object.lower():
            match_found = True
            break

    if match_found:
        matched_count += 1

        # Write this frame to the summary video output
        out.write(frame)

        # Save the frame as a .jpg image (optional)
        filename = os.path.join(save_frames_dir, f'frame_{frame_idx:04d}.jpg')
        cv2.imwrite(filename, frame)

    frame_idx += 1  # Move to the next frame

# === CLEANUP ===
cap.release()   # Release the video input stream
out.release()   # Finalize the output video file

# Final report
print(f"✅ Summarization complete. {matched_count} matching frames saved.")
print(f"🎞️ Output video saved to: {output_video_path}")

# ========================== PERFORMANCE ENHANCEMENTS & FEATURE ADDITIONS ==========================

# 1. Skip frames to speed up processing:
# Instead of analyzing every frame, analyze every Nth frame (e.g., every 3rd frame).
# Speeds up processing and reduces output video size.
# Add at the top of the loop:
# if frame_idx % 3 != 0:
#     frame_idx += 1
#     continue
# This skips 2 out of every 3 frames — adjust based on use case.

# ================================================================================================

# 2. Save only one image per detection segment:
# Avoid writing 10 identical consecutive "cell phone" frames.
# Modify logic to store only the *first* matched frame in a segment:
# prev_match = False  # Add before the loop
# Then inside the loop:
# if match_found and not prev_match:
#     cv2.imwrite(filename, frame)
# prev_match = match_found

# ================================================================================================

# 3. Add bounding boxes around matched objects:
# Helps visualize *what* was detected in the frame.
# After detecting a match:
# for detection in detection_result.detections:
#     category = detection.categories[0]
#     if category.category_name.lower() == target_object.lower():
#         bbox = detection.bounding_box
#         start_point = (bbox.origin_x, bbox.origin_y)
#         end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
#         cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

# ================================================================================================

# 4. Save metadata about matches (e.g., frame numbers, timestamps):
# Useful for indexing or reporting.
# Add this inside the `if match_found:` block:
# with open("summary_log.txt", "a") as f:
#     timestamp = frame_idx / fps
#     f.write(f"Match @ Frame {frame_idx}, Time: {timestamp:.2f} sec\n")

# ================================================================================================

# 5. Add ETA/progress reporting during processing:
# Useful for long videos.
# Every N frames:
# if frame_idx % 100 == 0:
#     print(f"Processing frame {frame_idx}...")

# ================================================================================================

# 6. Add size constraints on detections (ignore tiny objects):
# Skip detections smaller than a threshold (e.g., 10% of frame width/height):
# if bbox.width < width * 0.1 or bbox.height < height * 0.1:
#     continue

# ================================================================================================

# 7. Use multiple target objects (not just one):
# Allow matching against a list of objects instead of one hardcoded name.
# Replace:
# target_object = 'cell phone'
# With:
# target_objects = ['cell phone', 'laptop', 'keyboard']
# Then check:
# if category.category_name.lower() in [obj.lower() for obj in target_objects]:

# ================================================================================================

# 8. Allow fuzzy matching using partial keywords:
# Useful if you're not sure of the exact category name used in the model.
# Replace:
# if category.category_name.lower() == target_object.lower():
# With:
# if target_object.lower() in category.category_name.lower():

# ================================================================================================

# 9. Display total video duration and summary video ratio:
# After processing:
# total_frames = frame_idx
# summary_ratio = matched_count / total_frames * 100
# print(f"Matched frame ratio: {summary_ratio:.2f}% of original video")

# ================================================================================================

# 10. Optional: overlay category label and confidence score on saved frames:
# Inside the match block:
# label = f"{category.category_name} ({category.score:.2f})"
# cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# ================================================================================================
