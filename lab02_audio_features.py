#%% Import the required libraries

import numpy as np                      # For numerical operations
import matplotlib.pyplot as plt         # For visualizing plots
import librosa                          # Core library for audio feature extraction
import librosa.display                  # For advanced audio-specific plotting

#%% Load an audio file

# Load audio from 'test.wav'. sr=None keeps the original sample rate of the file
y, sr = librosa.load("test.wav", sr=None)

# y: audio time series (1D NumPy array)
# sr: sampling rate of the loaded audio

#%% Compute the Spectrogram

# Compute the complex Short-Time Fourier Transform (STFT)
# `S_full`: magnitude of spectrogram
# `phase`: phase component (discarded in visualizations)
S_full, phase = librosa.magphase(librosa.stft(y))

#%% Plot Waveform and Spectrogram (Time-Frequency Representation)

fig, (ax1, ax2) = plt.subplots(2, figsize=(7, 7))

# === Time-domain waveform plot ===
ax1.plot(y)
ax1.set_xlabel('samples')
ax1.set_ylabel('volume')
ax1.set(title='Time Series')

# === Spectrogram plot ===
# Convert amplitude to decibels and display using logarithmic frequency scale
img = librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
                               y_axis='log', x_axis='time', sr=sr, ax=ax2)
fig.colorbar(img, ax=ax2)
ax2.set(title='Spectrogram')

# Show both plots
plt.show()

#%% Chroma Estimation (Pitch Class Energy over Time)

# Chroma represents intensity for each of 12 pitch classes (C, C#, ..., B)
# Useful for chord detection, harmony analysis, key estimation

# Compute power spectrogram with large FFT window
S = np.abs(librosa.stft(y, n_fft=4096))**2

# Extract chromagram from power spectrogram
chroma = librosa.feature.chroma_stft(S=S, sr=sr)

# Plot power spectrogram and chroma
fig, ax = plt.subplots(nrows=2, sharex=True)

# === Power spectrogram ===
img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                               y_axis='log', x_axis='time', ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].label_outer()
ax[0].set(title='Power Spectrogram')

# === Chroma plot ===
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='Chromogram')

plt.show()

#%% Compute Mel-Spectrogram

# Mel spectrogram compresses frequency axis to match human perception (Mel scale)
# Common in audio classification, music/speech modeling

S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Convert to dB scale for visualization
S_mel_dB = librosa.power_to_db(S_mel, ref=np.max)

# Plot Mel spectrogram
fig, ax = plt.subplots()
img = librosa.display.specshow(S_mel_dB, x_axis='time',
                               y_axis='mel', sr=sr,
                               fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()

#%% Compute MFCC (Mel-Frequency Cepstral Coefficients)

# MFCCs represent short-term spectral shape and are widely used in speech recognition

# Compute MFCCs from audio signal (default 20, here we use 40)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

# Also compute mel spectrogram for reference in same plot
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

# Plot both Mel spectrogram and MFCCs
fig, ax = plt.subplots(nrows=2, sharex=True)

# === Mel spectrogram ===
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()

# === MFCC plot ===
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')

plt.show()


'''
Use Cases:
- Speech recognition and emotion detection
- Music genre classification
- Audio fingerprinting
- Deep learning audio preprocessing

Possible Additions:
- Add pitch estimation: librosa.yin()
- Add tempo estimation: librosa.beat.tempo()
- Save features as .npy or .csv for training ML models
- Segment features using windowing for real-time classification
'''

# ====================== Performance Enhancements & Advanced Features ==============================

# 1. Trim silence from beginning and end of audio (reduces noise and improves efficiency):
#    This removes long silences from the start and end that are irrelevant for most analyses.
# y, _ = librosa.effects.trim(y)

# ================================================================================================

# 2. Apply pre-emphasis filter (boosts higher frequencies):
#    High-frequency components are often weaker and more important in speech/audio recognition.
#    A simple high-pass filter like this helps highlight details like consonants or cymbals.
# y = np.append(y[0], y[1:] - 0.97 * y[:-1])  # Pre-emphasis with α = 0.97

# ================================================================================================

# 3. Save extracted features to disk:
#    Useful when training models so you don’t recompute features every run.
#    Reduces runtime and improves reproducibility in machine learning pipelines.
# np.save("mfcc.npy", mfccs)
# np.save("chroma.npy", chroma)

# ================================================================================================

# 4. Segment audio into short-time frames (e.g., 2048 samples with 512 hop):
#    This simulates real-time feature extraction and enables time-based analysis.
# frame_length = 2048
# hop_length = 512
# mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

# ================================================================================================

# 5. Add pitch estimation:
#    Extract the fundamental pitch frequency over time using piptrack.
#    Useful for melody analysis, singing voice detection, and vocal intonation tracking.
# pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
# pitch_values = pitches[magnitudes > np.median(magnitudes)]
# plt.plot(pitch_values)
# plt.title("Pitch Estimation")
# plt.show()

# ================================================================================================

# 6. Estimate tempo:
#    Uses onset strength envelope to estimate global tempo (BPM) and beat positions.
#    Useful for music synchronization, beat tracking, and rhythmic pattern analysis.
# tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
# print(f"Estimated Tempo: {tempo:.2f} BPM")

# ================================================================================================

# 7. Normalize to fixed duration (ensures uniformity across inputs):
#    Helps when feeding into deep learning models with fixed input sizes.
#    Pads or truncates audio to a consistent duration.
# desired_duration = 5.0  # seconds
# y = librosa.util.fix_length(y, int(sr * desired_duration))

# ================================================================================================

# 8. Compute delta (Δ) and delta-delta (ΔΔ) of MFCCs:
#    Captures how spectral features change over time — great for capturing emotion, rhythm.
#    Common in speech emotion recognition and speaker identification.
# mfcc_delta = librosa.feature.delta(mfccs)
# mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

# ================================================================================================

# 9. Extract tonal centroid features (Tonnetz):
#    Useful in emotion-aware music systems or harmonic similarity applications.
#    Captures tonal movement (e.g., major vs minor, consonance vs dissonance).
# tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
# librosa.display.specshow(tonnetz, x_axis='time')
# plt.title("Tonal Centroids (Tonnetz)")
# plt.show()

# ================================================================================================

# 10. Compare audio similarity using Dynamic Time Warping (DTW):
#     Allows you to compare structure and timing between two audio samples.
#     Useful for query-by-example, cover song detection, speaker alignment, etc.
# y2, sr2 = librosa.load("test2.wav", sr=sr)  # Load second audio file
# D, wp = librosa.sequence.dtw(X=mfccs, Y=librosa.feature.mfcc(y=y2, sr=sr2))
# plt.imshow(D, cmap='gray_r', origin='lower')
# plt.title("DTW Cost Matrix")
# plt.xlabel("Audio 1")
# plt.ylabel("Audio 2")
# plt.show()

# ================================================================================================

# 11. Reduce dimensionality of extracted features using PCA:
#     Reduces noise and visualization complexity — especially useful for exploratory data analysis.
#     Use PCA on MFCC or chroma features before ML or clustering.
# from sklearn.decomposition import PCA
# mfcc_flat = mfccs.T  # Transpose to shape (frames, features)
# pca = PCA(n_components=2)
# pca_features = pca.fit_transform(mfcc_flat)
# plt.scatter(pca_features[:, 0], pca_features[:, 1])
# plt.title("PCA of MFCCs")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

# ================================================================================================
