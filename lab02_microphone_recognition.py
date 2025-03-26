#!/usr/bin/env python3
# Refer to: https://github.com/Uberi/speech_recognition
# This is a demonstration for speech recognition.
# You should speak clearly when prompted: "Say something"

#%% Import required libraries

import speech_recognition as sr  # Core library for speech-to-text
import time                      # For measuring how long recognition takes
import os                        # For clearing the terminal (optional)

#%% Define wake words

WAKE_WORDS = ["ok google", "alexa", "siri"]  # List of wake words to listen for
WAKE_WORD_DETECTED = False

#%% Recording from Microphone

# Create a Recognizer instance to handle speech recognition
r = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    # Adjust the recognizer to ignore ambient noise (e.g., fan, background chatter)
    # You should stay silent for 1-2 seconds during this
    r.adjust_for_ambient_noise(source)

    # Clear the terminal (for Unix-based systems, optional)
    os.system('clear')

    print("Say something!")  # Prompt the user to speak

    # Record audio from the microphone
    audio = r.listen(source)  # Blocks until speech is detected and ends automatically

#%% Recognize Speech using Google Web API (Online)

start_time = time.time()  # Start timer to measure recognition time

try:
    # Perform recognition using Google Web Speech API
    # This is an **online service** and requires an internet connection

    # If you have a Google API key, you can provide it:
    # r.recognize_google(audio, key="YOUR_API_KEY")

    # Convert audio to text
    print("Google Speech Recognition thinks you said: " + r.recognize_google(audio))

    # Check for wake words
    for wake_word in WAKE_WORDS:
        if wake_word in audio:
            print(f"Wake word '{wake_word}' detected by Google!")
            WAKE_WORD_DETECTED = True
            break # No need to check other wake words once one is found

except sr.UnknownValueError:
    # Raised when speech is unintelligible (e.g., mumbling, noise)
    print("Google Speech Recognition could not understand audio")

except sr.RequestError as e:
    # Raised when there's a problem connecting to the API
    print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Display how long the Google recognition took
print('Time for Google Speech Recognition = {:.0f} seconds'.format(time.time() - start_time))

#%% Recognize Speech using PocketSphinx (Offline)

start_time = time.time()  # Start timer again for Sphinx

try:
    # Perform recognition using CMU Sphinx (fully offline)
    # You must have PocketSphinx installed: `pip install pocketsphinx`
    print("Sphinx thinks you said: " + r.recognize_sphinx(audio))

    # Check for wake words
    for wake_word in WAKE_WORDS:
        if wake_word in audio:
            print(f"Wake word '{wake_word}' detected by Sphinx!")
            WAKE_WORD_DETECTED = True
            break # No need to check other wake words once one is found

except sr.UnknownValueError:
    # Raised when Sphinx cannot interpret the spoken words
    print("Sphinx could not understand audio")

except sr.RequestError as e:
    # Raised if Sphinx backend fails
    print("Sphinx error: {0}".format(e))

# Display how long the Sphinx recognition took
print('Time for Sphinx recognition = {:.0f} seconds'.format(time.time() - start_time))

# Indicate if any wake word was detected
if WAKE_WORD_DETECTED:
    print("System is ready for further commands.")
else:
    print("No wake word detected.")

# =================================================================

'''
    Metrics     Google API  vs. Sphinx
    Internet    Yes             No
    Accuracy    High            Lower
    Speed       Depends         Fast

- Google API good for production demos.
- Sphinx is a great fallback option / offline mode.
'''

# =========================== Enhancements & Performance Ideas ============================

# 1. Add audio duration limit to avoid overly long recordings:
# audio = r.listen(source, timeout=5, phrase_time_limit=10)
# ➤ `timeout=5`: wait max 5s for speech to start
# ➤ `phrase_time_limit=10`: max duration of captured speech
# Helps prevent hanging if user forgets to speak.

# ==========================================================================================

# 2. Save recorded audio for debugging or training:
# with open("recorded.wav", "wb") as f:
#     f.write(audio.get_wav_data())
# ➤ Useful for reviewing user inputs or building training datasets.

# ==========================================================================================

# 3. Use a loop to keep recognizing until user quits:
# while True:
#     with sr.Microphone() as source:
#         print("Say something (or 'exit' to stop)...")
#         audio = r.listen(source)
#         text = r.recognize_google(audio)
#         if 'exit' in text.lower():
#             break
#         print("You said:", text)

# ➤ Allows continuous interaction without restarting the script.

# ==========================================================================================

# 4. Add WER (Word Error Rate) comparison between Google and Sphinx:
# from jiwer import wer
# google_text = r.recognize_google(audio)
# sphinx_text = r.recognize_sphinx(audio)
# print("WER (Sphinx vs Google):", wer(google_text, sphinx_text))

# ➤ Quantifies offline model performance vs. online.

# ==========================================================================================

# 5. Use other online recognizers like Bing, IBM, or Wit.ai:
# print(r.recognize_ibm(audio, username='xxx', password='xxx'))
# ➤ Gives flexibility if Google's quota is exceeded or blocked.

# ==========================================================================================

# 6. Use pre-recorded audio instead of mic (for testing):
# with sr.AudioFile("test.wav") as source:
#     audio = r.record(source)
# ➤ Useful for unit testing, reproducibility, and dataset labeling.

# ==========================================================================================

# 7. Add real-time feedback with waveform visualizer:
# Use `sounddevice` or `pyaudio` + `matplotlib` to show mic waveform while recording.
# ➤ Improves UX by indicating when speech is detected.

# ==========================================================================================

# 8. Improve ambient noise handling by dynamic calibration:
# r.dynamic_energy_threshold = True
# r.energy_threshold = 300
# ➤ Enables the recognizer to adapt to noisy or quiet environments.

# ==========================================================================================

# 9. Use custom keyword detection before speech capture (hotword detection):
# E.g., only activate recognition if user says "Hey Assistant"
# ➤ Helps reduce accidental activation, improves intent flow.

# ==========================================================================================

# 10. Compare latency and accuracy across multiple recognizers:
# Run multiple recognizers in sequence and log their timing, output, and correctness.
# ➤ Helps pick the best recognizer for specific environments.

# ==========================================================================================

# 11. Optimize performance for real-time applications:
# Pre-load recognizer and perform `r.listen_in_background()` to handle audio asynchronously.
# ➤ Great for assistants and real-time bots.

# ==========================================================================================

# 12. Transcribe to file or overlay on UI:
# with open("transcription.txt", "w") as f:
#     f.write(google_text)
# ➤ Good for logs, accessibility, or transcripts.

# ==========================================================================================

# 13. Integrate NLP for downstream processing:
# import transformers, nltk, etc. to analyze text output (sentiment, intent, etc.)
# ➤ Enables chatbots, assistants, or voice-based interfaces.

# ==========================================================================================

# 14. Wrap into a function or CLI tool:
# def recognize_once(source="mic", method="google"): ...
# ➤ Modularizes and reuses code for other apps or scripts.

# ==========================================================================================
