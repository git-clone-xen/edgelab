#%% Import the required libraries

import sounddevice as sd  # For audio input/output using Python (easy and cross-platform)
# See: https://python-sounddevice.readthedocs.io/en/0.4.6/

import numpy as np  # For numerical processing and FFT computation
import matplotlib.pyplot as plt  # For real-time plotting of waveform and spectrum
import time  # For timing the FFT computation

#%% Parameters for audio and recording

BUFFER = 1024 * 16           # Number of samples per frame (chunk size). Larger = smoother but more delay
CHANNELS = 1                 # Mono recording (1 microphone)
RATE = 44100                 # Audio sample rate (samples per second)
RECORD_SECONDS = 30          # Duration of the recording session (in seconds)

#%% Setup matplotlib figure and line plots

# Create a figure with 2 vertically stacked subplots: waveform and frequency spectrum
fig, (ax1, ax2) = plt.subplots(2, figsize=(4, 4))

# Generate X-axis values for waveform: sample indices (every 2 for 16-bit audio)
x = np.arange(0, 2 * BUFFER, 2)

# Generate X-axis values for frequency spectrum: frequency bins (up to Nyquist freq)
xf = np.fft.fftfreq(BUFFER, 1 / RATE)[:BUFFER // 2]

# Initialize line plots with random data (placeholders)
line, = ax1.plot(x, np.random.rand(BUFFER), '-', lw=2)            # Waveform
line_fft, = ax2.plot(xf, np.random.rand(BUFFER // 2), '-', lw=2)  # Spectrum

# === Configure waveform axis ===
ax1.set_title('AUDIO WAVEFORM')
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set_ylim(-5000, 5000)  # Adjust based on expected volume range
ax1.set_xlim(0, BUFFER)

# === Configure frequency spectrum axis ===
ax2.set_title('SPECTRUM')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('Log Magnitude')
ax2.set_ylim(0, 1000)      # Adjust based on expected intensity
ax2.set_xlim(0, RATE / 2)  # Show up to Nyquist frequency

# Display the figure without blocking (so it can update in real-time)
plt.show(block=False)

#%% Start recording loop and update plots with real-time audio

exec_time = []  # List to record execution times for FFT processing

# Determine number of chunks to process based on total duration and buffer size
for _ in range(0, RATE // BUFFER * RECORD_SECONDS):

    # === Record a single chunk of audio ===
    # `blocking=True` ensures the data is fully captured before moving on
    data = sd.rec(frames=BUFFER, samplerate=RATE, channels=CHANNELS, dtype='int16', blocking=True)
    data = np.squeeze(data)  # Remove unnecessary dimensions (shape: [BUFFER])

    # === Perform FFT to get frequency domain representation ===
    start_time = time.time()  # Start timer to measure FFT performance
    fft_data = np.fft.fft(data)  # Compute FFT of the audio chunk (complex output)
    fft_data = np.abs(fft_data[:BUFFER // 2])  # Get magnitude of positive frequencies only

    # Record FFT processing time
    exec_time.append(time.time() - start_time)

    # === Update the plots in real-time ===

    # Update waveform with new audio data
    line.set_ydata(data)

    # Update spectrum with scaled FFT magnitude
    line_fft.set_ydata(2.0 / BUFFER * fft_data)

    # Redraw the updated figure
    fig.canvas.draw()
    fig.canvas.flush_events()

#%% Post-recording summary and cleanup

print('stream stopped')  # Indicate end of recording
print('average execution time = {:.0f} milli seconds'.format(np.mean(exec_time) * 1000))  # Print average FFT time

# ====================== Performance Tips & Enhancement Ideas ===========================

# 1. ✂️ Trim silence before/after recording to reduce noise and FFT artifacts:
# from librosa.effects import trim
# y_trimmed, _ = librosa.effects.trim(data.astype(float), top_db=30)

# ➤ This helps focus the spectrum on actual sound content.

# =======================================================================================

# 2. 🔊 Normalize waveform before plotting for consistent scale across recordings:
# data = data / np.max(np.abs(data)) * 5000  # Rescale between -5000 and +5000

# ➤ Ensures amplitude plots remain visually consistent even if volume varies.

# =======================================================================================

# 3. 🔍 Apply windowing before FFT to reduce spectral leakage:
# window = np.hanning(BUFFER)
# fft_data = np.fft.fft(data * window)

# ➤ Helps isolate frequency peaks more clearly, especially in music or tonal analysis.

# =======================================================================================

# 4. 🎚️ Convert FFT to log scale (decibels) for better human interpretation:
# log_spectrum = 10 * np.log10(2.0 / BUFFER * fft_data + 1e-10)
# line_fft.set_ydata(log_spectrum)
# ax2.set_ylabel("Magnitude (dB)")

# ➤ Human hearing is logarithmic; this makes plots more intuitive.

# =======================================================================================

# 5. 📈 Add rolling average or median filtering to stabilize spectrum readings:
# smoothed = np.convolve(fft_data, np.ones(3)/3, mode='same')
# line_fft.set_ydata(2.0 / BUFFER * smoothed)

# ➤ Reduces jitter and makes plots smoother in noisy environments.

# =======================================================================================

# 6. 🕒 Plot dominant frequency (peak detection):
# dominant_freq = xf[np.argmax(fft_data)]
# print(f"Dominant frequency: {dominant_freq:.2f} Hz")

# ➤ Useful for pitch tracking, frequency recognition, or tone analysis.

# =======================================================================================

# 7. 🎼 Show pitch estimation with librosa or custom zero-crossing-based method:
# import librosa
# pitch = librosa.yin(data.astype(float), fmin=50, fmax=2000, sr=RATE)
# plt.plot(pitch); plt.title("Estimated Pitch (Hz)"); plt.show()

# ➤ Good for voice or music pitch detection tasks.

# =======================================================================================

# 8. ⏱️ Real-time spectrogram:
# Instead of updating one FFT, store a rolling list of FFT frames:
# spectrogram.append(fft_data)
# Then use imshow() to display as a scrolling 2D spectrogram heatmap

# ➤ Adds memory of past frequencies, useful in musical or speech visualization.

# =======================================================================================

# 9. 💾 Save waveform/spectrum data to file for ML or offline analysis:
# np.save("raw_waveform.npy", data)
# np.save("fft_magnitude.npy", fft_data)

# ➤ Enables integration with AI pipelines or batch processing.

# =======================================================================================

# 10. 🧠 Extract MFCC features from live audio (with librosa):
# import librosa
# mfcc = librosa.feature.mfcc(y=data.astype(float), sr=RATE, n_mfcc=13)
# librosa.display.specshow(mfcc, x_axis='time')

# ➤ Widely used in speech emotion detection and audio classification tasks.

# =======================================================================================

# 11. 🔔 Trigger event when a specific frequency or volume threshold is exceeded:
# if fft_data[freq_bin] > threshold:
#     print("High-frequency event detected!")

# ➤ Great for surveillance or environmental monitoring.

# =======================================================================================

# 12. ⚡ Accelerate FFT with GPU (requires CuPy or PyTorch):
# import cupy as cp
# data_gpu = cp.asarray(data)
# fft_data_gpu = cp.abs(cp.fft.fft(data_gpu))

# ➤ Useful when processing high-volume audio or multi-stream input.

# =======================================================================================

# 13. 🧼 Apply pre-emphasis to highlight higher frequencies:
# data[1:] = data[1:] - 0.97 * data[:-1]

# ➤ Enhances features for voice and speech analysis.

# =======================================================================================

# 14. 🧠 Add pitch tracking and beat detection:
# import librosa
# tempo, _ = librosa.beat.beat_track(y=data.astype(float), sr=RATE)
# print(f"Estimated tempo: {tempo:.2f} BPM")

# ➤ Supports music analysis and tempo synchronization tasks.

# =======================================================================================
