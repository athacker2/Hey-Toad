import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
   
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# load dummy dataset and read soundfiles
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
input_features = processor(ds[0]["audio"]["array"], return_tensors="pt").input_features 

# Generate logits
logits = model(input_features, decoder_input_ids = torch.tensor([[50258]])).logits 
# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)

print(predicted_ids)
print(transcription)

