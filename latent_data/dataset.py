import os
import json
import soundfile as sf
import random
from datasets import load_dataset, Dataset

# Load the dataset from Hugging Face
dataset = load_dataset("AudioSubnet/ttm_validation_dataset_10sec")

# Set directory paths
base_dir = "/root/m2music/latent_data"
audio_dir = os.path.join(base_dir, "datasets", "audioset", "wav")
metadata_dir = os.path.join(base_dir, "metadata")
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(os.path.join(metadata_dir, "MusicSet", "datafiles"), exist_ok=True)

# Prepare metadata structure for dataset_root.json
dataset_root = {
    "MusicSet": "/data/dataset/audioset",
    "comments": {},
    "metadata": {
        "path": {
            "MusicSet": {
                "train": "/data/dataset/metadata/MusicSet/datafiles/train.json",
                "test": "/data/dataset/metadata/MusicSet/datafiles/test.json",
                "val": "/data/dataset/metadata/MusicSet/datafiles/valid.json",
                "class_label_indices": ""
            }
        }
    }
}

# Save dataset_root.json
with open(os.path.join(metadata_dir, "dataset_root.json"), "w") as f:
    json.dump(dataset_root, f, indent=4)

# Convert dataset to list and shuffle for randomness
full_data = list(dataset['train'])

random.shuffle(full_data)

# Split sizes (80%, 10%, 10%)
train_size = int(0.8 * len(full_data))
valid_size = int(0.1 * len(full_data))

train_data = full_data[:train_size]
valid_data = full_data[train_size:train_size + valid_size]
test_data = full_data[train_size + valid_size:]

# Function to convert FLAC to WAV (if needed)
def convert_flac_to_wav(input_path, output_path):
    audio_data, sample_rate = sf.read(input_path)
    sf.write(output_path, audio_data, sample_rate)

# Function to process data and create json files
def process_data(data, split_name):
    samples = []
    for idx, sample in enumerate(data):
        # Check the structure to find the correct field name
        print(sample)  # This will help us understand how the data is structured

        # Extract the audio file path from 'File_Path' -> 'path'
        audio_path = sample['File_Path']['path']  # Access the correct key

        # Generate the wav file path
        wav_path = os.path.join(audio_dir, f"{idx:08d}.wav")

        # If the file is FLAC, convert it to WAV (assuming audio file is in 'flac' format)
        if audio_path.endswith('.flac'):
            convert_flac_to_wav(audio_path, wav_path)
        else:
            # If the file is already in WAV format, just move it
            if not os.path.exists(wav_path):
                # If audio data is available in the 'array' field, use that to save as wav
                audio_data = sample['File_Path']['array']
                sample_rate = 32000  # or extract this if available in the dataset
                sf.write(wav_path, audio_data, sample_rate)

        # Prepare metadata for the json files
        samples.append({
            "wav": f"wav/{idx:08d}.wav",
            "seg_label": "",
            "labels": "",
            "caption": sample['Prompts']  # Use 'Prompts' as caption
        })

    # Save to json
    json_data = {"data": samples}
    json_file = os.path.join(metadata_dir, "MusicSet", "datafiles", f"{split_name}.json")
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)

# Process and save each split
process_data(train_data, "train")
process_data(valid_data, "valid")
process_data(test_data, "test")

print("Dataset preparation completed.")