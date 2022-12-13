import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import pyaudio
import wave

import time

# GLOBALS
SAMPLING_RATE = 16000
CHUNK = 1024
PASSIVE_TIME_SLICE = 3 # seconds
ACTIVE_TIME_SLICE = 5 # seconds

def main():  
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    from transformers import WhisperProcessor, WhisperForConditionalGeneration

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
        for i in range(0, int(SAMPLING_RATE / CHUNK * PASSIVE_TIME_SLICE)):
            data = stream.read(CHUNK, exception_on_overflow = False)
            frames.extend(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0 )

        inputs = processor(frames, return_tensors="pt", sampling_rate = SAMPLING_RATE)
        input_features = inputs.input_features
        generated_ids = model.generate(inputs=input_features)

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(transcription)

        if("hey toad" in transcription.lower().replace(',','')):
            print("Waiting for commands...")
            command_session(stream, processor, model)
        else:
            print("I am asleep")

    stream.close()
    p.terminate()

def command_session(stream, processor, model):
    while True:
        frames = []
        print("Starting Recording")
        for i in range(0, int(SAMPLING_RATE / CHUNK * ACTIVE_TIME_SLICE)):
            data = stream.read(CHUNK, exception_on_overflow = False)
            frames.extend(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0 )

        inputs = processor(frames, return_tensors="pt", sampling_rate = SAMPLING_RATE)
        input_features = inputs.input_features
        generated_ids = model.generate(inputs=input_features)

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if("calendar" in transcription.lower() or "schedule" in transcription.lower()):
            print("Retrieving calendar info.")
        elif("weather" in transcription.lower() or "temperature" in transcription.lower()):
            print("Retrieving weather info.")
        elif("bye" in transcription.lower() or "exit" in transcription.lower()):
            return
        else:
            print("I do not understand you")


if __name__ == "__main__":
    main()