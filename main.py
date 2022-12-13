import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import torchaudio
import librosa
   
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

from transformers import WhisperProcessor, WhisperForConditionalGeneration

SAMPLING_RATE = 16000

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Audio file
AUDIO_PATH = "./test.wav"

waveform, _ = librosa.load(AUDIO_PATH, sr = SAMPLING_RATE)
inputs = processor(waveform, return_tensors="pt", sampling_rate = SAMPLING_RATE)
input_features = inputs.input_features

generated_ids = model.generate(inputs=input_features)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)