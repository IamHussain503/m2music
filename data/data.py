import os
import json
import soundfile as sf
import random
from datasets import load_dataset

# Load the dataset from Hugging Face
dataset = load_dataset("AudioSubnet/ttm_validation_dataset_10sec")

# Set directory paths
base_dir = "/root/m2music/data"
audio_dir = os.path.join(base_dir, "audio")
os.makedirs(audio_dir, exist_ok=True)

captions_file = os.path.join(base_dir, "captions.txt")

# Convert dataset to list and shuffle for randomness
full_data = list(dataset['train'])

random.shuffle(full_data)

# Split sizes (80%, 10%, 10%) if you still want to maintain splits
train_size = int(0.8 * len(full_data))
valid_size = int(0.1 * len(full_data))

train_data = full_data[:train_size]
valid_data = full_data[train_size:train_size + valid_size]
test_data = full_data[train_size + valid_size:]

# Combine all splits if you want a single dataset
combined_data = train_data + valid_data + test_data

# Function to convert FLAC to WAV (if needed)
def convert_flac_to_wav(input_path, output_path):
    audio_data, sample_rate = sf.read(input_path)
    sf.write(output_path, audio_data, sample_rate)

# Prepare captions list
captions = []

# Process data and save audio files
for idx, sample in enumerate(combined_data, start=1):
    # Debug: Print sample structure (optional)
    # print(sample)

    # Extract the audio file path from 'File_Path' -> 'path'
    audio_path = sample['File_Path']['path']  # Access the correct key

    # Generate the wav file path
    wav_filename = f"track{idx}.wav"
    wav_path = os.path.join(audio_dir, wav_filename)

    # If the file is FLAC, convert it to WAV
    if audio_path.endswith('.flac'):
        convert_flac_to_wav(audio_path, wav_path)
    else:
        # If the file is already in WAV format, copy or save it
        if not os.path.exists(wav_path):
            # If audio data is available in the 'array' field, use that to save as wav
            audio_data = sample['File_Path']['array']
            sample_rate = 32000  # Adjust if sample rate is available in the dataset
            sf.write(wav_path, audio_data, sample_rate)

    # Append the caption corresponding to this audio
    caption = sample.get('Prompts', '')  # Use 'Prompts' as caption; adjust if necessary
    captions.append(caption)

# Save all captions to captions.txt
with open(captions_file, "w", encoding="utf-8") as f:
    for caption in captions:
        f.write(caption.replace('\n', ' ') + '\n')  # Ensure each caption is on a new line

print("Dataset preparation completed.")