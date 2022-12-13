import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import pyaudio
import wave
   
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLING_RATE = 16000
CHUNK = 1024

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Generate audio file from mic input
p = pyaudio.PyAudio()

stream = p.open(rate = SAMPLING_RATE, 
                channels=1, 
                format=pyaudio.paInt16, 
                input=True,
                frames_per_buffer=CHUNK)


while True:
    frames = []
    print("Starting Recording")
    for i in range(0, int(SAMPLING_RATE / CHUNK * 3)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.extend(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0 )

    # Audio file
    # AUDIO_PATH = "./test.wav"

    # waveform, _ = librosa.load(AUDIO_PATH, sr = SAMPLING_RATE)
    inputs = processor(frames, return_tensors="pt", sampling_rate = SAMPLING_RATE)
    input_features = inputs.input_features

    generated_ids = model.generate(inputs=input_features)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(transcription)

stream.close()
p.terminate()