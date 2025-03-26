#%% Import the required libraries

import pyaudio  # For audio input/output (real-time access to the microphone)
# See: https://people.csail.mit.edu/hubert/pyaudio/

import struct  # For converting raw audio byte data into numeric values (int16)
# See: https://docs.python.org/3/library/struct.html

import numpy as np  # For numerical operations and handling audio samples as arrays
import matplotlib.pyplot as plt  # For plotting waveforms and spectrum
from scipy.fftpack import fft, fftfreq  # For performing Fast Fourier Transform
# See: https://docs.scipy.org/doc/scipy/tutorial/fft.html

import time  # For tracking time taken to compute FFT

#%% Parameters for audio stream and visualization

BUFFER = 1024 * 16           # Number of audio samples per frame (larger = more smoothing, more latency)
FORMAT = pyaudio.paInt16     # 16-bit audio format (standard for most microphones)
CHANNELS = 1                 # Mono audio input (one microphone)
RATE = 44100                 # Sample rate in Hz (CD quality audio)
RECORD_SECONDS = 30          # Duration of audio capture in seconds

#%% Set up initial plots for waveform and frequency spectrum

fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))  # Two subplots: waveform (top) and spectrum (bottom)

# X-axis for waveform plot: sample indices (0 to BUFFER with step of 2 since int16 = 2 bytes)
x = np.arange(0, 2 * BUFFER, 2)

# X-axis for spectrum plot: frequency bins (from 0 Hz to Nyquist frequency)
xf = fftfreq(BUFFER, 1 / RATE)[:BUFFER // 2]

# Create placeholder lines for waveform and spectrum (initialized with random values)
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)
line_fft, = ax2.plot(xf, np.random.rand(BUFFER // 2), '-', lw=2)

# === Format Waveform Plot (Top) ===
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-5000, 5000)   # Adjust based on expected volume amplitude range
ax1.set_xlim(0, BUFFER)

# === Format Spectrum Plot (Bottom) ===
ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Log Magnitude')
ax2.set_ylim(0, 1000)       # Adjust this based on expected intensity
ax2.set_xlim(0, RATE / 2)   # Nyquist frequency (half the sampling rate)

# Show the plots non-blocking (allows real-time updating)
plt.show(block=False)

#%% Initialize PyAudio input stream

audio = pyaudio.PyAudio()  # Create a PyAudio instance

## Note: Capturing audio
# Open a stream to access the microphone
stream = audio.open(
    format=FORMAT,             # 16-bit format
    channels=CHANNELS,         # Mono channel
    rate=RATE,                 # Sampling rate
    input=True,                # Enable input (microphone)
    output=True,               # Enable output (optional for monitoring)
    frames_per_buffer=BUFFER   # How many samples per audio chunk
)

print('stream started')

# Initialize list to store execution time of each FFT (for performance profiling)
exec_time = []

# Loop for the total number of chunks to be captured within the recording duration
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):

    # === Read audio chunk from microphone ===
    data = stream.read(BUFFER)  # Raw binary data

    # === Convert binary audio data to integers ===
    # 'h' means 16-bit signed integers. We unpack BUFFER samples
    data_int = struct.unpack(str(BUFFER) + 'h', data)
    ## Note: Read a chunk of audio data from the mic, but the data is in raw binary data (bytes)
    ##   so convert the raw bytes into sequence of numerical values

    # === Perform FFT to extract frequency components ===
    start_time = time.time()     # Record time before FFT
    yf = fft(data_int)           # Fast Fourier Transform (complex output)
    ## Note: Take in a NumPy array of integer audio samples

    # Record execution time of FFT and store it
    exec_time.append(time.time() - start_time)

    # === Update plots in real-time ===
    
    # Update waveform plot
    line.set_ydata(data_int)

    # Update spectrum plot (log magnitude of FFT)
    line_fft.set_ydata(2.0 / BUFFER * np.abs(yf[0:BUFFER // 2]))

    # Redraw the updated canvas
    fig.canvas.draw()
    fig.canvas.flush_events()

# Once done, stop audio stream
audio.terminate()

# Print completion message
print('stream stopped')

# Print average FFT computation time
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))

# ====================== Performance Tips & Feature Enhancements ============================

# 1. Reduce BUFFER size for lower latency (important for real-time responsiveness):
# BUFFER = 1024 * 4  # Smaller buffer → faster response, but slightly noisier spectrum
# Adjust RATE // BUFFER * RECORD_SECONDS accordingly if you change BUFFER

# ===========================================================================================

# 2. Normalize input data for consistent FFT scaling and avoid clipping:
# data_norm = np.array(data_int) / 32768.0  # Convert from int16 to float [-1, 1]
# yf = fft(data_norm)

# ===========================================================================================

# 3. Use log scaling for better spectrum visibility (especially for quiet sounds):
# spectrum = 2.0 / BUFFER * np.abs(yf[0:BUFFER // 2])
# log_spectrum = 10 * np.log10(spectrum + 1e-8)  # Avoid log(0) error
# line_fft.set_ydata(log_spectrum)
# ax2.set_ylabel("Log Power (dB)")  # Update label

# ===========================================================================================

# 4. Plot the dominant frequency for pitch or tone tracking:
# dominant_freq = xf[np.argmax(spectrum)]
# print(f"Dominant Frequency: {dominant_freq:.1f} Hz")

# ===========================================================================================

# 5. Visualize audio in decibel (dB) scale for clearer interpretation:
# Convert waveform to dB (approximate RMS-based loudness)
# rms = np.sqrt(np.mean(np.square(data_int)))
# db = 20 * np.log10(rms + 1e-10)
# print(f"Estimated Loudness: {db:.2f} dB")

# ===========================================================================================

# 6. Save waveform and FFT to file for offline analysis:
# np.save("waveform.npy", data_int)
# np.save("fft_magnitude.npy", spectrum)

# ===========================================================================================

# 7. Apply a windowing function before FFT to reduce spectral leakage:
# window = np.hanning(BUFFER)
# windowed = window * np.array(data_int)
# yf = fft(windowed)

# ===========================================================================================

# 8. Export FFT computation time statistics:
# np.savetxt("fft_times.csv", np.array(exec_time) * 1000, delimiter=",")
# Helps visualize performance over time

# ===========================================================================================

# 9. Add spectrogram-style history (like a scrolling heatmap):
# Collect all FFT slices → stack → use imshow() for 2D spectrogram plot

# ===========================================================================================

# 10. Detect presence of specific frequency ranges:
# For example, detect a 1kHz tone:
# freq_bin = np.argmin(np.abs(xf - 1000))
# if spectrum[freq_bin] > threshold:
#     print("1kHz tone detected!")

# ===========================================================================================

# 11. Audio monitoring (loopback output):
# stream.write(data)  # Play back audio in real-time if needed

# ===========================================================================================

# 12. Plot phase response (optional for signal analysis):
# phase = np.angle(yf[:BUFFER // 2])
# plt.plot(xf, phase); plt.title("Phase Spectrum"); plt.show()

# ===========================================================================================

# 13. Downsample input signal before FFT for faster processing (low-frequency focus):
# from scipy.signal import decimate
# downsampled = decimate(np.array(data_int), 2)
# yf = fft(downsampled)

# ===========================================================================================

# 14. Add automatic gain control (AGC) for stable amplitude scaling:
# data_int = data_int / np.max(np.abs(data_int)) * 5000  # Normalize for plotting range

# ===========================================================================================

# 15. For interactive analysis, log and visualize FFT magnitude over time:
# magnitude_over_time.append(spectrum)
# At end, use `imshow()` to display as heatmap (rows = frames, columns = frequency bins)

# ===========================================================================================
