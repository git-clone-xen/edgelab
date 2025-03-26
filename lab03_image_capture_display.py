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

# ============================ Enhancements & Performance Suggestions ============================

# 1. Convert to HSV color space for more robust color detection:
# The current method uses BGR ranges, which are less intuitive and more sensitive to lighting.
# HSV (Hue-Saturation-Value) makes color range detection more reliable across lighting conditions.

# Example:
# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# red_lower = np.array([0, 120, 70])
# red_upper = np.array([10, 255, 255])
# mask = cv2.inRange(hsv, red_lower, red_upper)

# =================================================================================================

# 2. Add trackbars to dynamically tune BGR or HSV thresholds:
# Helps in fine-tuning the color segmentation interactively without restarting the program.

# Example:
# def nothing(x): pass
# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("Low H", "Trackbars", 0, 179, nothing)
# cv2.createTrackbar("High H", "Trackbars", 179, 179, nothing)
# Then read values dynamically: lowH = cv2.getTrackbarPos("Low H", "Trackbars")

# =================================================================================================

# 3. Save segmented regions as images for logging or training:
# Use `cv2.imwrite("red_segment.jpg", red_img)` when motion or a key is triggered.
# Useful for collecting dataset for ML model training (e.g., color classification).

# =================================================================================================

# 4. Add contour detection around segmented areas:
# Helps draw outlines or bounding boxes around detected colored objects.

# Example:
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     if cv2.contourArea(cnt) > 500:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# =================================================================================================

# 5. Optimize performance by resizing input frame before processing:
# Resize to e.g., 320x240 for faster segmentation (at the cost of accuracy).

# Example:
# frame = cv2.resize(frame, (320, 240))

# =================================================================================================

# 6. Show live histogram of each color channel:
# Helps analyze color distribution in the frame and debug segmentation results.

# Example:
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([frame],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)

# =================================================================================================

# 7. Process every Nth frame instead of all:
# Useful when high frame rate isn't necessary (e.g., 1 in every 3 frames)

# Example:
# if frame_idx % 3 == 0:
#     process_frame()

# =================================================================================================

# 8. Extend to multiple color masks with more color boundaries:
# Add yellow, orange, cyan, etc., to detect more regions.

# Example:
# boundaries.append(([X, Y, Z], [A, B, C]))  # Add more custom color ranges

# =================================================================================================

# 9. Add keyboard shortcuts to toggle individual color views:
# Useful for isolating and troubleshooting specific segmentation channels.

# Example:
# Press 'r' to view only red_img, 'g' for green_img, etc.

# =================================================================================================

# 10. Add audio feedback or system alerts when a specific color is detected:
# Helpful for accessibility use cases or real-world event triggers.

# Example:
# if np.sum(mask) > threshold:
#     os.system('say "Red object detected"')  # macOS
#     winsound.Beep(1000, 300)                # Windows

# =================================================================================================

# 11. Use CLAHE or histogram equalization before segmentation:
# Improves contrast and helps reveal colors in poor lighting.

# Example:
# hsv[...,2] = cv2.equalizeHist(hsv[...,2]) or use cv2.createCLAHE()

# =================================================================================================

# 12. Display frames in separate windows for clarity:
# cv2.imshow("Red", red_img), etc., instead of using horizontal concat.

# Helpful if display resolution is limited or for presentations.

# =================================================================================================

# 13. Profile execution time using time.time() to find bottlenecks:
# Example:
# start = time.time()
# [segmentation code...]
# print("Frame time:", time.time() - start)

# =================================================================================================

# 14. Export segmented data to JSON, CSV, or NumPy for ML pipelines:
# Save mask positions, bounding boxes, or pixel counts as structured data.

# =================================================================================================
