# Import required libraries
import cv2                    # OpenCV for video capture and image processing
import numpy as np            # NumPy for numerical operations
from skimage import feature   # For HOG feature extraction
from skimage import exposure  # For enhancing/normalizing image intensity

#%% OpenCV Webcam Capture

# Start video capture from default webcam (device index 0)
cap = cv2.VideoCapture(0)

# Check if webcam is available
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#%% Main loop to process webcam frames

# This loop will continue until the 'q' key is pressed
while True:
    try:
        # Read a single frame from the webcam
        ret, frame = cap.read()

        # OPTIONAL: Resize the image to 256x256 for faster processing
        # frame = cv2.resize(frame, (256, 256))

        # Convert the frame to grayscale
        # HOG in scikit-image only works on grayscale images
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === HOG Feature Extraction ===
        # Compute HOG features and a visualization image of gradients
        (H, hogImage) = feature.hog(image,
                                    orientations=9,              # Number of gradient bins
                                    pixels_per_cell=(8, 8),      # Size of each cell
                                    cells_per_block=(2, 2),      # Size of block of cells
                                    transform_sqrt=True,         # Normalize contrast by sqrt
                                    block_norm="L1",             # Block normalization method
                                    visualize=True)              # Generate image for visualization

        # === Enhance HOG visualization ===
        # Rescale intensity to the 0–255 range for display
        ## Note: Normalize the pixels to standard 0-255 range for proper display
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")

        # Convert the grayscale HOG image to 3-channel RGB so it can be concatenated
        hogImg = cv2.cvtColor(hogImage, cv2.COLOR_GRAY2RGB)
        ## Note: convert the hogImage to RGB image

        # Concatenate original frame and HOG visualization side by side
        catImg = cv2.hconcat([frame, hogImg])

        # Display the combined image
        cv2.imshow("HOG Image", catImg)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break  # Allow graceful exit with Ctrl+C

#%% Cleanup

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# =========================== Enhancements & Optimization Suggestions ===========================

# 1. Resize frames before processing for speed-up:
# Processing smaller frames improves FPS and reduces CPU usage. For example, 256x256 balances speed and clarity.

# Example:
# frame = cv2.resize(frame, (256, 256))

# ================================================================================================

# 2. Adjust HOG parameters for finer or coarser feature detection:
# You can tune the following to control how fine the edge gradients are:
# - orientations: more bins = finer angular resolution
# - pixels_per_cell: smaller cell = finer details
# - cells_per_block: larger block = more spatial context

# Example:
# Change orientations=9 to 6 for speed or 12 for more detail
# Change pixels_per_cell=(16, 16) for faster processing but coarser features

# ================================================================================================

# 3. Use HOG feature vector (`H`) for downstream ML tasks:
# The HOG descriptor `H` can be stored or fed into classifiers like SVM, KNN, or deep models.

# Example:
# from sklearn.svm import SVC
# clf = SVC()
# clf.fit([H], [label])  # Use H as input for classification

# ================================================================================================

# 4. Save feature vectors or HOG images to disk:
# Useful for building datasets for training classifiers or tracking changes over time.

# Example:
# np.save("hog_vector.npy", H)
# cv2.imwrite("hog_image.jpg", hogImage)

# ================================================================================================

# 5. Save output video with HOG overlay:
# Add functionality to export the combined frames to a video file for documentation or analysis.

# Example:
# writer = cv2.VideoWriter("hog_output.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width*2, height))
# writer.write(catImg)

# ================================================================================================

# 6. Add FPS counter overlay:
# Helps evaluate processing performance in real-time.

# Example:
# import time
# prev_time = time.time()
# fps = 1 / (time.time() - prev_time)
# cv2.putText(catImg, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

# ================================================================================================

# 7. Use edge detection as a pre-HOG filter (optional):
# Applying Sobel or Canny edges first may improve feature visibility in some cases.

# Example:
# edge = cv2.Canny(image, 100, 200)
# Then apply HOG to the edge image

# ================================================================================================

# 8. Use region-of-interest (ROI) cropping:
# Process only a portion of the frame (like the center or where movement is detected) to reduce overhead.

# Example:
# roi = frame[100:300, 100:300]
# Process `roi` instead of entire frame

# ================================================================================================

# 9. Add text overlay for instructions:
# Let users know how to exit the program or what’s being visualized.

# Example:
# cv2.putText(catImg, "Press 'q' to quit", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# ================================================================================================

# 10. Combine with motion detection or face detection:
# Run HOG only on detected motion areas or face regions for focused analysis.

# Example:
# Use OpenCV’s `cv2.CascadeClassifier` or a motion mask to localize input
