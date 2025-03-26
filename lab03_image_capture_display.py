# Reference: https://pyimagesearch.com/2014/08/04/opencv-python-color-detection/
import cv2                  # OpenCV for computer vision and webcam capture
import numpy as np          # NumPy for numerical operations and masking

#%% Define RGB (actually BGR in OpenCV) color boundaries for segmentation
# These are the lower and upper bounds for detecting Red, Blue, and Green in an image
# Values are in BGR format because OpenCV uses BGR instead of RGB by default

boundaries = [
    ([17, 15, 100], [50, 56, 200]),   # Red range
    ([86, 31, 4], [220, 88, 50]),     # Blue range
    ([25, 90, 4], [62, 200, 50])      # Green range
    ([20, 100, 100], [30, 255, 255])  # Yellow range
]

#%% Utility function to normalize images for display

def normalizeImg(Img):
    """
    Normalize the input image to the range 0–255.
    This helps in enhancing the contrast of the masked output.

    Parameters:
        Img: Input image (NumPy array)

    Returns:
        norm_img: Normalized image
    """
    Img = np.float64(Img)  # Convert to float for safe division
    norm_img = (Img - np.min(Img)) / (np.max(Img) - np.min(Img) + 1e-6)  # Avoid division by zero
    norm_img = np.uint8(norm_img * 255.0)  # Scale to 8-bit range
    return norm_img

#%% Start capturing video from webcam

cap = cv2.VideoCapture(0)  # Device 0 = default webcam

# Check if webcam opened successfully
if not cap.isOpened():
    raise IOError("Cannot open webcam")

#%% Main loop for frame capture and processing

# Press 'q' to exit
while True:
    try:
        # Read one frame from webcam
        ret, frame = cap.read()
        output = []  # Store segmented images for each color

        # Loop over each defined BGR boundary
        for (lower, upper) in boundaries:
            # Convert lists to NumPy arrays
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # Create a binary mask where the colors within the range are white
            mask = cv2.inRange(frame, lower, upper)

            # Apply the mask to the original frame
            # This extracts only the pixels within the specified color range
            segmented = cv2.bitwise_and(frame, frame, mask=mask)
            output.append(segmented)

        # Normalize each color-segmented image to enhance visibility
        red_img = normalizeImg(output[0])
        green_img = normalizeImg(output[1])
        blue_img = normalizeImg(output[2])
        yellow_img = normalizeImg(output[3])

        # Concatenate original + segmented frames horizontally for visualization
        catImg = cv2.hconcat([frame, red_img, green_img, blue_img])

        # Display the concatenated image
        cv2.imshow("Images with Colours", catImg)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break

#%% Cleanup after exiting the loop

cap.release()             # Release the webcam
cv2.destroyAllWindows()   # Close any OpenCV windows
