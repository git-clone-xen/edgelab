#%% OpenCV based real-time optical flow estimation and tracking
# Ref: https://github.com/daisukelab/cv_opt_flow/tree/master
import numpy as np
import cv2

#%% Generic Parameters
# Create an array of 100 random RGB colors to visually differentiate motion vectors when drawing them
# Increasing this value allows more unique colors for more tracking points
color = np.random.randint(0, 255, (100, 3))

#%% Parameters for Lucas-Kanade Optical Flow (Sparse Tracking)
# Used for detecting strong feature points in the first frame using the Shi-Tomasi method
feature_params = dict(
    maxCorners=100,       # Max number of corner points to detect; ↑ = more points = better detail, but more CPU usage
    qualityLevel=0.3,     # Value between 0 and 1 indicating the minimum quality of corners to retain.
                          # ↑ = stricter selection (fewer, more reliable points); ↓ = looser (more but potentially noisy points)
    minDistance=7,        # Minimum pixel distance between two detected corners; ↓ = more packed features, ↑ = sparser selection
    blockSize=7           # Size of the averaging window used for corner detection; ↑ = smoother but less sensitive to small corners
)

# Parameters for Lucas-Kanade optical flow algorithm (cv2.calcOpticalFlowPyrLK)
lk_params = dict(
    winSize=(15, 15),     # Size of the window used to search for matching points in the next frame.
                          # ↑ = better for larger motion, but slower and can blur small motion
    maxLevel=2,           # Number of pyramid levels for multi-resolution analysis; ↑ = handles large motions better
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
        10,               # Maximum number of iterations to refine flow for each point
        0.03              # Epsilon — minimum error allowed to stop the iteration early
                          # ↓ = more precise flow, ↑ = faster convergence but may sacrifice accuracy
    )
)

#%% Setup function for first frame
def set1stFrame(frame):
    # Convert the input frame to grayscale — essential for most computer vision algorithms
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect strong corner features to track using Shi-Tomasi method with the parameters defined above
    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
    
    # Create a mask for drawing motion lines — same size as frame but initialized with zeros (black)
    mask = np.zeros_like(frame)
    
    return frame_gray, mask, p0

#%% Lucas-Kanade Optical Flow (Sparse)
def LucasKanadeOpticalFlow(frame, old_gray, mask, p0):
    # Convert current frame to grayscale for processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # In case no initial features were detected or lost, fallback to dummy default points
    if (p0 is None or len(p0) == 0):
        p0 = np.array([[50, 50], [100, 100]], dtype=np.float32).reshape(-1, 1, 2)

    # Calculate optical flow using Lucas-Kanade method (between old and current grayscale frame)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is not None:
        # Filter only successfully tracked points (status=1)
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
        # Draw motion vectors between previous and new positions
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            # Draw line from old to new position
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            # Draw a circle at the new position
            frame_gray = cv2.circle(frame_gray, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Overlay drawing mask on top of current frame
        img = cv2.add(frame, mask)

        # Update for next iteration
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    return img, old_gray, p0

#%% Farneback Dense Optical Flow
# This method computes motion across all pixels (not just sparse features)
step = 16  # Distance between sampled points for visualization
           # ↓ = more motion vectors drawn (denser visualization); ↑ = sparser and faster

def DenseOpticalFlowByLines(frame, old_gray):
    # Convert current frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get height and width of the frame
    h, w = frame_gray.shape[:2]

    # Generate grid of points sampled every 'step' pixels
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1)

    # Calculate optical flow between previous and current frame using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        old_gray, frame_gray, None,
        pyr_scale=0.5,   # ↓ = more details at each level; ↑ = faster, smoother pyramids (range 0.3–0.9)
        levels=3,        # ↑ = better for large movements, ↑ = slower
        winsize=15,      # Size of the window used to compute the flow for each pixel
                         # ↑ = smoother and more robust to noise; ↓ = faster, less smooth
        iterations=3,    # Number of iterations at each pyramid level
                         # ↑ = more accurate but slower
        poly_n=5,        # Size of the pixel neighborhood used to fit the polynomial expansion
                         # ↑ = better for large motions; ↓ = more responsive to noise
        poly_sigma=1.2,  # Standard deviation of the Gaussian used to smooth derivatives
                         # ↑ = smoother results; ↓ = more sensitive to local changes
        flags=0          # Additional flags; usually kept as 0
    )

    # Extract motion vectors at the sampled points
    fx, fy = flow[y, x].T

    # Stack origin and destination points and reshape into lines
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # Draw motion vectors as lines on top of the frame
    cv2.polylines(frame, lines, isClosed=False, color=(0, 255, 0))

    # Optionally draw a dot at the origin of each motion vector
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)

    return frame

#%% Initialize Video Capture
cap = cv2.VideoCapture(0)  # Start capturing from default webcam (device index 0)

# Ensure webcam is accessible
if not cap.isOpened():
    raise IOError("Cannot open webcam")

firstframeflag = 1  # Used to run setup only once on the first frame

# Main loop to continuously process frames from webcam
while True:
    try:
        if firstframeflag:
            # Read the first frame
            ret, frame = cap.read()

            # Initialize grayscale image, tracking points, and drawing mask
            old_gray, mask, p0 = set1stFrame(frame)

            # Reset the flag
            firstframeflag = 0

        # Read the next frame
        ret, frame = cap.read()

        # === USE DENSE OPTICAL FLOW ===
        img = DenseOpticalFlowByLines(frame, old_gray)

        # === TO USE SPARSE TRACKING (LUCAS-KANADE), UNCOMMENT BELOW ===
        # img, old_gray, p0 = LucasKanadeOpticalFlow(frame, old_gray, mask, p0)

        # Display the current frame with motion vectors drawn
        cv2.imshow("Optical Flow", img)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        # Graceful exit with Ctrl+C
        break

# Cleanup: release webcam and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# ========================== PERFORMANCE & FUNCTIONALITY ENHANCEMENTS ==========================

# 1. Improve frame rate with grayscale preview instead of full color:
# Dense flow modifies full-color frames which is computationally heavier.
# Replace in DenseOpticalFlowByLines:
# return frame
# With:
# return cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
# ➜ This will overlay flow vectors on a lightweight gray background.

# ==============================================================================================

# 2. Adaptive reinitialization of lost points (Lucas-Kanade):
# When too many points are lost or drift, re-detect features automatically.
# Replace this:
# if (p0 is None or len(p0) == 0):
#     p0 = np.array([[50, 50], [100, 100]], dtype=np.float32).reshape(-1, 1, 2)
# With:
# if p0 is None or len(p0) < 10:
#     p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
# ➜ This prevents "dumb" fallback and keeps tracking responsive.

# ==============================================================================================

# 3. Add performance monitor (frame rate estimator):
# At top: `import time`
# Then add before loop:
# prev_time = time.time()
# And at bottom inside the loop:
# fps = 1 / (time.time() - prev_time)
# prev_time = time.time()
# cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
# ➜ Real-time FPS display helps profile responsiveness.

# ==============================================================================================

# 4. Toggle between dense and sparse tracking using keyboard:
# Instead of editing code to switch modes, allow pressing 'd' or 's' keys:
# Add at top of loop:
# mode = 'dense'  # Global toggle
# Then inside loop, replace processing block with:
# if mode == 'dense':
#     img = DenseOpticalFlowByLines(frame, old_gray)
# elif mode == 'sparse':
#     img, old_gray, p0 = LucasKanadeOpticalFlow(frame, old_gray, mask, p0)
# Then handle key press:
# key = cv2.waitKey(1) & 0xFF
# if key == ord('d'):
#     mode = 'dense'
# elif key == ord('s'):
#     mode = 'sparse'
# elif key == ord('q'):
#     break
# ➜ Makes it interactive and lets you test both modes without restarting.

# ==============================================================================================

# 5. Save optical flow screenshots for documentation or analysis:
# Press 'c' to capture frame:
# Add under key check:
# elif key == ord('c'):
#     cv2.imwrite(f'flow_capture_{int(time.time())}.jpg', img)
# ➜ Helpful for saving test cases and visual results during runtime.

# ==============================================================================================

# 6. Reduce frame size for faster processing (especially for dense flow):
# Add before processing:
# frame = cv2.resize(frame, (320, 240))  # Or 480x360
# ➜ Much faster, especially on lower-end machines or when real-time speed matters.

# ==============================================================================================

# 7. Record dense flow as video:
# Useful for post-analysis or training datasets.
# At top: 
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('dense_output.mp4', fourcc, 20.0, (width, height))
# Then in loop: `out.write(img)`
# Finally in cleanup: `out.release()`

# ==============================================================================================

# 8. Add motion magnitude thresholding (dense flow):
# Only draw vectors with meaningful displacement:
# Replace flow extraction with:
# magnitude = np.sqrt(fx**2 + fy**2)
# mask = magnitude > 1.0  # Minimum movement threshold
# x, y, fx, fy = x[mask], y[mask], fx[mask], fy[mask]
# ➜ Prevents clutter and highlights real motion.

# ==============================================================================================
