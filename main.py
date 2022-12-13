import torch
import numpy as np
import pyaudio

from transformers import WhisperProcessor, WhisperForConditionalGeneration

# GLOBALS
SAMPLING_RATE = 16000
CHUNK = 1024
PASSIVE_TIME_SLICE = 3 # seconds
ACTIVE_TIME_SLICE = 5 # seconds

def main():  
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    # Open mic audio stream
    p = pyaudio.PyAudio()
    stream = p.open(rate = SAMPLING_RATE, 
                    channels=1, 
                    format=pyaudio.paInt16, 
                    input=True,
                    frames_per_buffer=CHUNK)

    # Passively listen for 'Hey Toad' from mic
    while True:
        frames = retrieve_audio_input(stream, PASSIVE_TIME_SLICE)
        transcription = transcribe_audio(frames, model, processor)
        print(transcription)

        # switch to command mode if audio contains 'hey toad'
        if("hey toad" in transcription.lower().replace(',','')):
            print("Waiting for commands...")
            command_session(stream, processor, model)
        else:
            print("I am asleep")

    stream.close()
    p.terminate()

def command_session(stream, processor, model):
    # Actively listen for commands from mic
    while True:
        frames = retrieve_audio_input(stream, ACTIVE_TIME_SLICE)

        # Convert audio to text using whisper
        transcription = transcribe_audio(frames, model, processor)
        print(transcription)

        # handle each command accordingly
        if("calendar" in transcription.lower() or "schedule" in transcription.lower()):
            print("Retrieving calendar info.")
        elif("weather" in transcription.lower() or "temperature" in transcription.lower()):
            print("Retrieving weather info.")
        elif("bye" in transcription.lower() or "exit" in transcription.lower()):
            print("Going back to sleep")
            return
        else:
            print("I do not understand you")

def retrieve_audio_input(stream, seconds):
    # Read from mic input
    frames = []
    # print("Starting Recording")
    for i in range(0, int(SAMPLING_RATE / CHUNK * PASSIVE_TIME_SLICE)):
        data = stream.read(CHUNK, exception_on_overflow = False)
        frames.extend(np.frombuffer(data, np.int16).flatten().astype(np.float32) / 32768.0 )
    
    return frames

def transcribe_audio(audio, model, processor):
    # Convert audio to text using whisper
    inputs = processor(audio, return_tensors="pt", sampling_rate = SAMPLING_RATE)
    input_features = inputs.input_features
    generated_ids = model.generate(inputs=input_features)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

if __name__ == "__main__":
    main()