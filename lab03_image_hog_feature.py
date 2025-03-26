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
